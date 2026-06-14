mod cli;
mod command;
mod data;
mod link;
mod logging;
mod portfolio;
mod training;

use clap::Parser;
use miette::Result;

use crate::cli::{Cli, Command};

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Train(args) => command::train::run(&args),
        Command::Backtest(args) => command::backtest::run(&args),
    }
}
