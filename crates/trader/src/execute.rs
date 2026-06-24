//! Place the planned orders against sim stock. Recording the fills into the cash ledger
//! happens in `main`, once the network has settled.

use miette::{Context, Result};
use stock_client::sim_stock::SimStockClient;

use crate::plan::{Buy, Sell};

/// Place every sell, one at a time since the sim server is single-threaded. Orders are
/// independent, so the first rejection fails the batch.
///
/// # Errors
/// If the platform rejects any sell.
#[tracing::instrument(skip_all, fields(orders = sells.len()))]
pub async fn place_sells(sim: &SimStockClient, sells: &[Sell]) -> Result<()> {
    for sell in sells {
        sim.sell(&sell.code, sell.lots, sell.price)
            .await
            .wrap_err_with(|| format!("sell {}", sell.code))?;
    }
    Ok(())
}

/// Place every buy, one at a time since the sim server is single-threaded. Orders are
/// independent, so the first rejection fails the batch.
///
/// # Errors
/// If the platform rejects any buy.
#[tracing::instrument(skip_all, fields(orders = buys.len()))]
pub async fn place_buys(sim: &SimStockClient, buys: &[Buy]) -> Result<()> {
    for buy in buys {
        sim.buy(&buy.code, buy.lots, buy.price)
            .await
            .wrap_err_with(|| format!("buy {}", buy.code))?;
    }
    Ok(())
}
