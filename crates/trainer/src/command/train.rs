use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use miette::Result;

use crate::cli::TrainArgs;
use crate::training::{link, train};

type TrainBackend = Autodiff<Wgpu>;

/// Train a model from the parsed flags and write the run's artifacts, then refresh
/// the `latest` link so predict and other tools find this run.
pub fn run(args: &TrainArgs) -> Result<()> {
    let device = WgpuDevice::default();
    let config = args.training_config();
    let options = args.run_options();

    train::<TrainBackend>(&device, &args.data, &args.artifact_dir, &config, options)?;

    link::refresh_latest(&args.artifact_dir);

    Ok(())
}
