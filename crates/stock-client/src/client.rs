use crate::error::{Error, Result};
use crate::types::{ApiResponse, Market, UserStock};
use crate::urls;

pub struct StockClient {
    http: reqwest::Client,
    account: String,
    password: String,
}

impl StockClient {
    /// Load credentials from `STOCK_ACCOUNT` and `STOCK_PASSWORD` env vars.
    /// Call `dotenvy::dotenv().ok()` before this if using a `.env` file.
    ///
    /// # Errors
    ///
    /// Returns an error if either env var is missing.
    pub fn from_env() -> Result<Self> {
        let account = std::env::var("STOCK_ACCOUNT")?;
        let password = std::env::var("STOCK_PASSWORD")?;
        Ok(Self {
            http: reqwest::Client::new(),
            account,
            password,
        })
    }

    /// # Errors
    ///
    /// Returns an error on network or deserialization failure.
    pub async fn stock_list(&self) -> Result<Vec<String>> {
        let map: serde_json::Map<String, serde_json::Value> = self
            .http
            .get(format!("{}/stock_list", urls::TRADING_API_BASE))
            .send()
            .await?
            .json()
            .await?;
        Ok(map.into_iter().map(|(k, _v)| k).collect())
    }

    /// # Errors
    ///
    /// Returns an error on network, deserialization, or unknown market type.
    pub async fn stock_market(&self, code: &str) -> Result<Market> {
        #[derive(serde::Deserialize)]
        struct Response {
            #[serde(rename = "type")]
            kind: String,
        }
        let response: Response = self
            .http
            .get(format!("{}/stock_type", urls::TRADING_API_BASE))
            .query(&[("stock_code", code)])
            .send()
            .await?
            .json()
            .await?;
        match response.kind.as_str() {
            "TWSE" | "ETF" => Ok(Market::Twse),
            "OTC" => Ok(Market::Tpex),
            "ESB" => Ok(Market::Esb),
            other => Err(Error::UnknownMarket(other.to_owned())),
        }
    }

    /// # Errors
    ///
    /// Returns an error on network or deserialization failure.
    pub async fn user_stocks(&self) -> Result<Vec<UserStock>> {
        let response: serde_json::Value = self
            .http
            .post(format!("{}/get_user_stocks", urls::TRADING_API_BASE))
            .form(&[("account", &self.account), ("password", &self.password)])
            .send()
            .await?
            .json()
            .await?;
        if response["result"] != "success" {
            return Ok(vec![]);
        }
        Ok(serde_json::from_value(response["data"].clone())?)
    }

    /// # Errors
    ///
    /// Returns an error on network failure or if the server rejects the order.
    pub async fn buy(&self, code: &str, shares: u64, price: f64) -> Result<bool> {
        self.order("buy", code, shares, price).await
    }

    /// # Errors
    ///
    /// Returns an error on network failure or if the server rejects the order.
    pub async fn sell(&self, code: &str, shares: u64, price: f64) -> Result<bool> {
        self.order("sell", code, shares, price).await
    }

    async fn order(&self, action: &str, code: &str, shares: u64, price: f64) -> Result<bool> {
        let response: ApiResponse = self
            .http
            .post(format!("{}/{}", urls::TRADING_API_BASE, action))
            .form(&[
                ("account", self.account.as_str()),
                ("password", self.password.as_str()),
                ("stock_code", code),
                ("stock_shares", &shares.to_string()),
                ("stock_price", &price.to_string()),
            ])
            .send()
            .await?
            .json()
            .await?;

        if response.result != "success" {
            return Err(Error::ApiFailure {
                status: response.status,
            });
        }
        Ok(true)
    }
}
