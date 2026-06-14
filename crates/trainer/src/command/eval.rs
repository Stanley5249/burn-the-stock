use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn::config::Config;
use chrono::Duration;
use miette::{IntoDiagnostic, Result};
use stock_model::inference::Predictor;

use crate::cli::EvalArgs;
use crate::report::{self, EvalReport};
use crate::store::TickerStore;
use crate::training::TrainingConfig;

type InferenceBackend = Wgpu;

/// Backtest a trained model over the held-out split: replay it across every window
/// in the validation period, apply the same long-only position map and fee the
/// Sharpe metric trains on, and report realized performance. Unlike a live trader
/// this places no orders; it measures how the policy would have done.
pub fn run(args: &EvalArgs) -> Result<()> {
    let device = WgpuDevice::default();

    // The full training config carries the barrier and fee knobs the Predictor's
    // inference subset omits, so reload it to rebuild the same labels and split.
    let config = TrainingConfig::load(args.artifact_dir.join("config.json")).into_diagnostic()?;

    let predictor = Predictor::<InferenceBackend>::load(&args.artifact_dir, device)?;

    let store = TickerStore::load(
        &args.data,
        config.take_profit,
        config.stop_loss,
        config.label_horizon,
    )
    .into_diagnostic()?;

    // Reproduce the training split: the last `valid_days` validate and everything
    // earlier trains, so the backtest scores only data the model never fit.
    let max_date = store
        .max_date()
        .expect("loaded data should have at least one dated row");
    let cutoff = max_date - Duration::days(args.valid_days);
    let (_, valid) = store
        .train_valid_split(cutoff, config.steps)
        .into_diagnostic()?;

    let (windows, rewards) = valid.backtest_windows(config.steps);
    let predictions = predictor.predict(&windows);

    let report = EvalReport::aggregate(&predictions, &rewards, config.fee, args.min_position);
    report::render(&report);

    Ok(())
}
