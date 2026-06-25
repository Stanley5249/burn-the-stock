use miette::{Context, IntoDiagnostic, Result, bail, miette};
use scraper::{Element, Html, Selector};
use serde::Deserialize;
use std::str::FromStr;
use std::time::Duration;
use tokio_retry::RetryIf;
use tokio_retry::strategy::FixedInterval;
use tracing::instrument;
use url::Url;

use std::collections::HashMap;

use crate::types::{MarketType, Profile, UserStock};
use crate::urls::sim_stock as urls;

/// Per-request timeout for `sim_stock` when callers do not set one. The platform is flaky and a
/// hung request would otherwise block forever (`reqwest` has no default timeout).
pub const DEFAULT_TIMEOUT_MS: u64 = 10_000;

/// Fixed pause between retries, a property of the flaky platform shared by every endpoint.
pub const RETRY_DELAY_MS: u64 = 2_000;

/// Per-endpoint retry budgets against the flaky platform. Orders never retry, to avoid
/// duplicate fills.
pub const USER_STOCKS_RETRIES: usize = 3;
pub const LOGIN_RETRIES: usize = 3;
pub const PROFILE_RETRIES: usize = 3;
pub const STOCK_LIST_RETRIES: usize = 3;

/// Log and retry past any transient failure. The caller wraps only requests that are safe to
/// repeat (never orders).
fn retry_any(error: &reqwest::Error) -> bool {
    tracing::warn!(%error, "sim stock request failed, retrying");
    true
}

pub struct SimStockClient {
    pub client: reqwest::Client,
    pub base: Url,
    pub account: String,
    pub password: String,
}

impl SimStockClient {
    /// Build a client with a cookie-storing reqwest client (the login session needs it),
    /// `base` defaulting to [`urls::base`], and `timeout` defaulting to [`DEFAULT_TIMEOUT_MS`].
    ///
    /// # Errors
    /// If the client fails to build.
    pub fn new(
        base: Option<Url>,
        account: String,
        password: String,
        timeout: Option<Duration>,
    ) -> Result<Self> {
        let client = reqwest::Client::builder()
            .cookie_store(true)
            .timeout(timeout.unwrap_or(Duration::from_millis(DEFAULT_TIMEOUT_MS)))
            .build()
            .into_diagnostic()
            .wrap_err("build sim client")?;

        let base = base.unwrap_or(urls::base());

        Ok(Self {
            client,
            base,
            account,
            password,
        })
    }

    /// Load credentials from `STOCK_ACCOUNT` and `STOCK_PASSWORD`.
    ///
    /// # Errors
    /// If a credential env var is missing.
    pub fn from_env(base: Option<Url>, timeout: Option<Duration>) -> Result<Self> {
        let account = std::env::var("STOCK_ACCOUNT").into_diagnostic()?;
        let password = std::env::var("STOCK_PASSWORD").into_diagnostic()?;

        Self::new(base, account, password, timeout)
    }

    /// # Errors
    /// Network or deserialization failure, or if the platform rejects the request.
    #[instrument(skip_all, fields(holdings), err)]
    pub async fn user_stocks(&self) -> Result<Vec<UserStock>> {
        #[derive(Deserialize)]
        #[serde(tag = "result", deny_unknown_fields)]
        enum Response {
            #[serde(rename = "success")]
            Success {
                #[allow(dead_code)]
                status: String,
                data: Vec<UserStock>,
            },
            #[serde(rename = "failed")]
            Failed { status: String },
        }

        let url = self.base.join(urls::USER_STOCKS).into_diagnostic()?;
        let response: Response = RetryIf::start(
            FixedInterval::from_millis(RETRY_DELAY_MS).take(USER_STOCKS_RETRIES),
            || async {
                self.client
                    .post(url.clone())
                    .form(&[("account", &self.account), ("password", &self.password)])
                    .send()
                    .await?
                    .error_for_status()
            },
            retry_any,
        )
        .await
        .into_diagnostic()?
        .json()
        .await
        .into_diagnostic()
        .wrap_err("decode user_stocks")?;

        match response {
            Response::Success { data, .. } => {
                tracing::Span::current().record("holdings", data.len());
                Ok(data)
            }
            Response::Failed { status } => bail!("sim_stock rejected request: {status}"),
        }
    }

    /// Place a buy order. `lots` is in 張 (board lots of 1,000 shares, the platform's unit),
    /// `price` is per share.
    ///
    /// # Errors
    /// Network failure or if the server rejects the order.
    #[instrument(skip(self), err)]
    pub async fn buy(&self, code: &str, lots: u64, price: f64) -> Result<()> {
        self.order(urls::BUY, code, lots, price).await
    }

    /// Place a sell order. `lots` is in 張 (board lots of 1,000 shares), `price` is per share.
    ///
    /// # Errors
    /// Network failure or if the server rejects the order.
    #[instrument(skip(self), err)]
    pub async fn sell(&self, code: &str, lots: u64, price: f64) -> Result<()> {
        self.order(urls::SELL, code, lots, price).await
    }

    async fn order(&self, path: &str, code: &str, lots: u64, price: f64) -> Result<()> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Response {
            result: String,
            status: String,
        }

        let response: Response = self
            .client
            .post(self.base.join(path).into_diagnostic()?)
            .form(&[
                ("account", self.account.as_str()),
                ("password", self.password.as_str()),
                ("stock_code", code),
                ("stock_shares", &lots.to_string()),
                ("stock_price", &price.to_string()),
            ])
            .send()
            .await
            .into_diagnostic()?
            .error_for_status()
            .into_diagnostic()?
            .json()
            .await
            .into_diagnostic()
            .wrap_err("decode order")?;

        if response.result != "success" {
            bail!("sim_stock rejected order: {}", response.status);
        }

        Ok(())
    }

    /// Establish a session: fetch the login page for its csrf token, then POST credentials.
    /// The cookie store keeps the session for later scrapes. Both requests retry past the flaky
    /// platform's transient HTTP failures.
    ///
    /// # Errors
    /// Network failure, a missing csrf token, or a rejected login.
    #[instrument(skip_all, err)]
    pub async fn login(&self) -> Result<()> {
        let url = self.base.join(urls::LOGIN).into_diagnostic()?;

        let login_page = RetryIf::start(
            FixedInterval::from_millis(RETRY_DELAY_MS).take(LOGIN_RETRIES),
            || async {
                self.client
                    .get(url.clone())
                    .send()
                    .await?
                    .error_for_status()
            },
            retry_any,
        )
        .await
        .into_diagnostic()?
        .text()
        .await
        .into_diagnostic()?;

        let login_page = Html::parse_document(&login_page);

        let token =
            parse_csrf(&login_page).ok_or_else(|| miette!("login page missing csrf token"))?;

        // Django rejects the HTTPS POST without a Referer matching the host.
        RetryIf::start(
            FixedInterval::from_millis(RETRY_DELAY_MS).take(LOGIN_RETRIES),
            || async {
                self.client
                    .post(url.clone())
                    .header(reqwest::header::REFERER, url.as_str())
                    .form(&[
                        ("csrfmiddlewaretoken", token),
                        ("account", self.account.as_str()),
                        ("password", self.password.as_str()),
                        ("next", ""),
                    ])
                    .send()
                    .await?
                    .error_for_status()
            },
            retry_any,
        )
        .await
        .into_diagnostic()?;

        Ok(())
    }

    /// Scrape the account dashboard for cash and headline figures. The trading API has no
    /// balance endpoint, so this reads the `profile/` page. Call [`login`](Self::login) first
    /// to establish the session.
    ///
    /// # Errors
    /// Network failure or a missing field on the page.
    #[instrument(
        skip_all,
        fields(usable_cash, total_assets, cumulative_return, trade_count),
        err
    )]
    pub async fn profile(&self) -> Result<Profile> {
        let url = self.base.join(urls::PROFILE).into_diagnostic()?;
        let profile_page = RetryIf::start(
            FixedInterval::from_millis(RETRY_DELAY_MS).take(PROFILE_RETRIES),
            || async {
                self.client
                    .get(url.clone())
                    .send()
                    .await?
                    .error_for_status()
            },
            retry_any,
        )
        .await
        .into_diagnostic()?
        .text()
        .await
        .into_diagnostic()?;

        let profile_page = Html::parse_document(&profile_page);

        let profile = parse_profile(&profile_page)?;

        let span = tracing::Span::current();
        span.record("usable_cash", profile.usable_cash);
        span.record("total_assets", profile.total_assets);
        span.record("cumulative_return", profile.cumulative_return);
        span.record("trade_count", profile.trade_count);

        Ok(profile)
    }

    /// Fetch the full symbol universe. Unauthenticated, so it is a free function like
    /// [`crate::twse::fetch_holidays`] rather than a [`SimStockClient`] method.
    ///
    /// # Errors
    /// Network or deserialization failure.
    #[instrument(skip_all, fields(symbols), err)]
    pub async fn stock_list(&self) -> Result<HashMap<String, MarketType>> {
        // Each value is an object like {"name": "台積電", "type": "TWSE"}, so pull out the type.
        #[derive(Deserialize)]
        struct Entry {
            r#type: MarketType,
        }

        let url = self.base.join(urls::STOCK_LIST).into_diagnostic()?;

        let map: HashMap<String, Entry> = RetryIf::start(
            FixedInterval::from_millis(RETRY_DELAY_MS).take(STOCK_LIST_RETRIES),
            || async {
                self.client
                    .get(url.clone())
                    .send()
                    .await?
                    .error_for_status()
            },
            retry_any,
        )
        .await
        .into_diagnostic()?
        .json()
        .await
        .into_diagnostic()
        .wrap_err("decode sim stock list")?;

        tracing::Span::current().record("symbols", map.len());

        Ok(map
            .into_iter()
            .map(|(code, entry)| (code, entry.r#type))
            .collect())
    }
}

fn parse_csrf(html: &Html) -> Option<&str> {
    let selector = Selector::parse(r#"input[name="csrfmiddlewaretoken"]"#).ok()?;

    html.select(&selector)
        .find_map(|input| input.value().attr("value"))
}

fn parse_profile(html: &Html) -> Result<Profile> {
    Ok(Profile {
        usable_cash: field_after(html, "可用餘額")?,
        total_assets: field_after(html, "資產總額")?,
        cumulative_return: field_after(html, "累積報酬率")?,
        trade_count: field_after(html, "交易成功次數")?,
    })
}

/// Read the number in the element that follows the one whose text is `label`. The dashboard
/// renders each metric as a label div immediately followed by a value div, with no stable ids.
fn field_after<T: FromStr>(html: &Html, label: &str) -> Result<T> {
    let text =
        label_sibling_text(html, label).ok_or_else(|| miette!("field not found: {label}"))?;

    // Drop everything but the number: currency `$`, thousands commas, `%`, and surrounding
    // whitespace and newlines the page wraps values in.
    let cleaned: String = text
        .chars()
        .filter(|character| character.is_ascii_digit() || *character == '.' || *character == '-')
        .collect();

    cleaned
        .parse()
        .map_err(|_| miette!("field {label}: {text:?} is not a number"))
}

fn label_sibling_text(html: &Html, label: &str) -> Option<String> {
    let div = Selector::parse("div").ok()?;

    html.select(&div).find_map(|element| {
        if element.text().collect::<String>().trim() != label {
            return None;
        }

        let text = element.next_sibling_element()?.text().collect();

        Some(text)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_base_joins_to_full_endpoints() {
        let base = urls::base();
        assert_eq!(
            base.join(urls::USER_STOCKS).unwrap().as_str(),
            "https://ciot.imis.ncku.edu.tw/stock/trading_api/get_user_stocks"
        );
        assert_eq!(
            base.join(urls::BUY).unwrap().as_str(),
            "https://ciot.imis.ncku.edu.tw/stock/trading_api/buy"
        );
        assert_eq!(
            base.join(urls::LOGIN).unwrap().as_str(),
            "https://ciot.imis.ncku.edu.tw/stock/login/"
        );
        assert_eq!(
            base.join(urls::PROFILE).unwrap().as_str(),
            "https://ciot.imis.ncku.edu.tw/stock/profile/"
        );
    }

    #[test]
    fn parse_csrf_reads_hidden_input() {
        let html = Html::parse_document(
            r#"<input type="hidden" name="csrfmiddlewaretoken" value="abc123">"#,
        );
        assert_eq!(parse_csrf(&html), Some("abc123"));
    }

    #[test]
    fn parse_profile_extracts_all_fields() {
        let html = Html::parse_document(
            r"<div><div>可用餘額</div><div>$22,606,117</div></div>
            <div><div>資產總額</div><div>$ 101,906,017</div></div>
            <div><div>累積報酬率</div><div>1.906%</div></div>
            <div><div>交易成功次數</div><div>61</div></div>",
        );
        let profile = parse_profile(&html).unwrap();
        assert!((profile.usable_cash - 22_606_117.0).abs() < 1e-6);
        assert!((profile.total_assets - 101_906_017.0).abs() < 1e-6);
        assert!((profile.cumulative_return - 1.906).abs() < 1e-9);
        assert_eq!(profile.trade_count, 61);
    }
}
