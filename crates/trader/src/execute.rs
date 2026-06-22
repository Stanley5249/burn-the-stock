//! Place the planned orders against `sim_stock` and update the cash ledger.

use chrono::{Duration, NaiveDate};
use miette::{Context, IntoDiagnostic, Result};
use stock_client::sim_stock::SimStockClient;

use crate::plan::{Buy, Sell};
use crate::state::LiveState;

/// Place the sells then the buys, recording each in `state`. Sale proceeds settle after
/// `settle_lag` days, so they do not fund today's buys.
///
/// # Errors
/// If the platform rejects any order.
pub async fn execute(
    sim: &SimStockClient,
    sells: &[Sell],
    buys: &[Buy],
    state: &mut LiveState,
    today: NaiveDate,
    settle_lag: i64,
) -> Result<()> {
    let settle_date = today + Duration::days(settle_lag);

    for sell in sells {
        sim.sell(&sell.code, sell.lots, sell.price)
            .await
            .into_diagnostic()
            .wrap_err_with(|| format!("sell {}", sell.code))?;
        state.record_sell(sell.proceeds, settle_date);
    }

    for buy in buys {
        sim.buy(&buy.code, buy.lots, buy.price)
            .await
            .into_diagnostic()
            .wrap_err_with(|| format!("buy {}", buy.code))?;
        state.record_buy(buy.cost);
    }

    Ok(())
}
