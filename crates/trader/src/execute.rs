//! Place the planned orders against `sim_stock`. Recording the fills into the cash ledger
//! happens in `main`, once the network has settled.

use futures::stream::{StreamExt, TryStreamExt};
use miette::{Context, Result};
use stock_client::sim_stock::SimStockClient;

use crate::plan::{Buy, Sell};

/// Place every sell, up to `concurrency` orders in flight at once. Orders are independent, so
/// the first rejection fails the batch.
///
/// # Errors
/// If the platform rejects any sell.
#[tracing::instrument(skip_all, fields(orders = sells.len()))]
pub async fn place_sells(sim: &SimStockClient, sells: &[Sell], concurrency: usize) -> Result<()> {
    futures::stream::iter(sells)
        .map(async |sell| {
            sim.sell(&sell.code, sell.lots, sell.price)
                .await
                .wrap_err_with(|| format!("sell {}", sell.code))
        })
        .buffer_unordered(concurrency)
        .try_collect::<()>()
        .await
}

/// Place every buy, up to `concurrency` orders in flight at once. Orders are independent, so
/// the first rejection fails the batch.
///
/// # Errors
/// If the platform rejects any buy.
#[tracing::instrument(skip_all, fields(orders = buys.len()))]
pub async fn place_buys(sim: &SimStockClient, buys: &[Buy], concurrency: usize) -> Result<()> {
    futures::stream::iter(buys)
        .map(async |buy| {
            sim.buy(&buy.code, buy.lots, buy.price)
                .await
                .wrap_err_with(|| format!("buy {}", buy.code))
        })
        .buffer_unordered(concurrency)
        .try_collect::<()>()
        .await
}
