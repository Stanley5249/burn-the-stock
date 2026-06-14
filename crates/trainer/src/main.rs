mod cli;
mod command;
mod data;
mod logging;
mod portfolio;
mod training;

use clap::Parser;
use miette::Result;

use crate::cli::{Cli, Command};

fn main() -> Result<()> {
    // Install the global subscriber before anything runs so every command logs
    // through it; training redirects the writer to its artifact dir once known.
    logging::install();

    match Cli::parse().command {
        Command::Train(args) => command::train::run(&args),
        Command::Backtest(args) => command::backtest::run(&args),
    }
}
