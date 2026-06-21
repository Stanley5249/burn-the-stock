use crate::error::{Error, Result};
use crate::types::{MarketType, StockInfo, UserStock};
use crate::urls;
use serde::Deserialize;
use std::collections::HashMap;

pub struct SimStockClient {
    http: reqwest::Client,
    base: String,
    account: String,
    password: String,
}

impl SimStockClient {
    #[must_use]
    pub fn http(&self) -> &reqwest::Client {
        &self.http
    }

    /// Load credentials from `STOCK_ACCOUNT` and `STOCK_PASSWORD`, and the trading API base
    /// from `SIM_STOCK_BASE` (default [`urls::SIM_STOCK_API_BASE`]). Call
    /// `dotenvy::dotenv().ok()` first if using a `.env` file.
    ///
    /// # Errors
    /// If either credential env var is missing.
    pub fn from_env(http: reqwest::Client) -> Result<Self> {
        let account = std::env::var("STOCK_ACCOUNT")?;
        let password = std::env::var("STOCK_PASSWORD")?;
        let base = std::env::var("SIM_STOCK_BASE")
            .unwrap_or_else(|_| urls::SIM_STOCK_API_BASE.to_string());
        Ok(Self {
            http,
            base,
            account,
            password,
        })
    }

    /// # Errors
    /// Network or deserialization failure.
    pub async fn stock_list(&self) -> Result<HashMap<String, StockInfo>> {
        let list = self
            .http
            .get(format!("{}/stock_list", self.base))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;

        tracing::info!("stock_list");

        Ok(list)
    }

    /// # Errors
    /// Network or deserialization failure.
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
            .get(format!("{}/stock_type", self.base))
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
    /// Network or deserialization failure.
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
            .post(format!("{}/get_user_stocks", self.base))
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
    /// Network failure or if the server rejects the order.
    pub async fn buy(&self, code: &str, shares: u64, price: f64) -> Result<()> {
        self.order("buy", code, shares, price).await
    }

    /// # Errors
    /// Network failure or if the server rejects the order.
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
            .post(format!("{}/{}", self.base, action))
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
