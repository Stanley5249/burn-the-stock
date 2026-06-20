mod cli;
mod command;
mod label;
mod logging;
mod portfolio;
mod training;

use clap::Parser;
use miette::Result;

use crate::cli::{Cli, Command};

fn main() -> Result<()> {
    dotenvy::dotenv_override().ok();

    // Install the global subscriber first; training redirects it to its artifact dir.
    logging::install();

    match Cli::parse().command {
        Command::Train(args) => command::train::run(&args),
        Command::Backtest(args) => command::backtest::run(&args),
    }
}
