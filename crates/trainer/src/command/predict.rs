use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use miette::{IntoDiagnostic, Result};
use stock_model::inference::Predictor;

use crate::cli::PredictArgs;
use crate::report;
use crate::store::TickerStore;

type InferenceBackend = Wgpu;

/// Offline inference: read the most recent `steps` bars per ticker from the parquet
/// snapshot, run them through the exact training feature pipeline, and report the
/// actions a trader would take. Live fetching and order placement live in the
/// `trader` bin.
pub fn run(args: &PredictArgs) -> Result<()> {
    let device = WgpuDevice::default();

    let predictor = Predictor::<InferenceBackend>::load(&args.artifact_dir, device)?;

    let windows =
        TickerStore::load_inference_windows(&args.data, predictor.steps()).into_diagnostic()?;

    let predictions = predictor.predict(&windows);

    report::render(&predictions, args.min_position);

    Ok(())
}
