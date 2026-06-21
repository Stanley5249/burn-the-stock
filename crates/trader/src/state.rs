//! The live trader's own cash ledger. `sim_stock` exposes positions but no balance, so
//! cash is tracked here across runs. Positions are reconciled from the API each run, so
//! this file holds only cash and sale proceeds not yet settled.

use std::path::Path;

use chrono::NaiveDate;
use miette::{IntoDiagnostic, Result};
use serde::{Deserialize, Serialize};

/// Sale proceeds that become spendable on `available_date`. Sells do not free cash the
/// same day, so the budget never counts today's proceeds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pending {
    pub amount: f64,
    pub available_date: NaiveDate,
}

/// Persisted cash state. Buys spend `settled_cash`; sells push into `pending` until their
/// settlement date passes.
#[derive(Debug, Serialize, Deserialize)]
pub struct LiveState {
    pub settled_cash: f64,
    pub pending: Vec<Pending>,
    pub last_run: Option<NaiveDate>,
}

impl LiveState {
    fn seed(starting_cash: f64) -> Self {
        Self {
            settled_cash: starting_cash,
            pending: Vec::new(),
            last_run: None,
        }
    }

    /// Load the ledger from `path`, or seed a fresh one at `starting_cash` when the file
    /// does not exist yet.
    ///
    /// # Errors
    /// If the file exists but cannot be read or parsed.
    pub fn load_or_seed(path: &Path, starting_cash: f64) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::seed(starting_cash));
        }
        let text = std::fs::read_to_string(path).into_diagnostic()?;
        serde_json::from_str(&text).into_diagnostic()
    }

    /// Write the ledger to `path` as pretty JSON.
    ///
    /// # Errors
    /// If the file cannot be written.
    pub fn save(&self, path: &Path) -> Result<()> {
        let text = serde_json::to_string_pretty(self).into_diagnostic()?;
        std::fs::write(path, text).into_diagnostic()
    }

    /// Move proceeds whose settlement date has arrived into spendable cash. Call once at
    /// the start of a run.
    pub fn settle(&mut self, today: NaiveDate) {
        let (matured, still_pending): (Vec<_>, Vec<_>) = self
            .pending
            .drain(..)
            .partition(|entry| entry.available_date <= today);
        self.settled_cash += matured.iter().map(|entry| entry.amount).sum::<f64>();
        self.pending = still_pending;
    }

    /// Spend cash on a buy.
    pub fn record_buy(&mut self, cost: f64) {
        self.settled_cash -= cost;
    }

    /// Park sale proceeds until they settle.
    pub fn record_sell(&mut self, proceeds: f64, available_date: NaiveDate) {
        self.pending.push(Pending {
            amount: proceeds,
            available_date,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn date(day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 6, day).unwrap()
    }

    #[test]
    fn settle_releases_only_matured_proceeds() {
        let mut state = LiveState::seed(100.0);
        state.record_buy(40.0);
        state.record_sell(30.0, date(10));
        state.record_sell(20.0, date(20));

        // Day 10: the first sale settles, the second is still pending.
        state.settle(date(10));

        assert!((state.settled_cash - 90.0).abs() < 1e-9);
        assert_eq!(state.pending.len(), 1);
        assert!((state.pending[0].amount - 20.0).abs() < 1e-9);
    }
}
