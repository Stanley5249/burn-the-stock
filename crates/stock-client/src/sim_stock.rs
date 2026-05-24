use crate::error::{Error, Result};
use crate::types::{MarketType, StockInfo, UserStock};
use crate::urls;
use serde::Deserialize;
use std::collections::HashMap;

pub struct SimStockClient {
    http: reqwest::Client,
    account: String,
    password: String,
}

impl SimStockClient {
    #[must_use]
    pub fn http(&self) -> &reqwest::Client {
        &self.http
    }

    /// Load credentials from `STOCK_ACCOUNT` and `STOCK_PASSWORD` env vars.
    /// Call `dotenvy::dotenv().ok()` before this if using a `.env` file.
    ///
    /// # Errors
    ///
    /// Returns an error if either env var is missing.
    pub fn from_env(http: reqwest::Client) -> Result<Self> {
        let account = std::env::var("STOCK_ACCOUNT")?;
        let password = std::env::var("STOCK_PASSWORD")?;
        Ok(Self {
            http,
            account,
            password,
        })
    }

    /// # Errors
    ///
    /// Returns an error on network or deserialization failure.
    pub async fn stock_list(&self) -> Result<HashMap<String, StockInfo>> {
        let list = self
            .http
            .get(format!("{}/stock_list", urls::SIM_STOCK_API_BASE))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;

        tracing::info!("stock_list");

        Ok(list)
    }

    /// # Errors
    ///
    /// Returns an error on network or deserialization failure.
    pub async fn stock_market(&self, code: &str) -> Result<MarketType> {
        #[derive(Debug, Deserialize)]
        #[serde(tag = "result", deny_unknown_fields)]
        enum Response {
            #[serde(rename = "success")]
            Success {
                #[allow(dead_code)]
                stock_code: String,
                r#type: MarketType,
            },
            #[serde(rename = "failed")]
            Failed { status: String },
        }

        let response: Response = self
            .http
            .get(format!("{}/stock_type", urls::SIM_STOCK_API_BASE))
            .query(&[("stock_code", code)])
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;

        tracing::info!(?response, "stock_type");

        match response {
            Response::Success { r#type, .. } => Ok(r#type),
            Response::Failed { status } => Err(Error::Api { status }),
        }
    }

    /// # Errors
    ///
    /// Returns an error on network or deserialization failure.
    pub async fn user_stocks(&self) -> Result<Vec<UserStock>> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Response {
            result: String,
            status: String,
            data: Option<Vec<UserStock>>,
        }

        let response: Response = self
            .http
            .post(format!("{}/get_user_stocks", urls::SIM_STOCK_API_BASE))
            .form(&[("account", &self.account), ("password", &self.password)])
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;

        tracing::info!(response.result, response.status, "get_user_stocks");

        response.data.ok_or_else(|| Error::Api {
            status: response.status,
        })
    }

    /// # Errors
    ///
    /// Returns an error on network failure or if the server rejects the order.
    pub async fn buy(&self, code: &str, shares: u64, price: f64) -> Result<()> {
        self.order("buy", code, shares, price).await
    }

    /// # Errors
    ///
    /// Returns an error on network failure or if the server rejects the order.
    pub async fn sell(&self, code: &str, shares: u64, price: f64) -> Result<()> {
        self.order("sell", code, shares, price).await
    }

    async fn order(&self, action: &str, code: &str, shares: u64, price: f64) -> Result<()> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Response {
            result: String,
            status: String,
        }

        let response: Response = self
            .http
            .post(format!("{}/{}", urls::SIM_STOCK_API_BASE, action))
            .form(&[
                ("account", self.account.as_str()),
                ("password", self.password.as_str()),
                ("stock_code", code),
                ("stock_shares", &shares.to_string()),
                ("stock_price", &price.to_string()),
            ])
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;

        tracing::info!(response.result, response.status, "{action}");

        if response.result != "success" {
            return Err(Error::Api {
                status: response.status,
            });
        }

        Ok(())
    }
}
