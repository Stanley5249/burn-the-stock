mod cli;
mod command;
mod label;
mod link;
mod logging;
mod portfolio;
mod report;
mod store;
mod training;

use clap::Parser;
use miette::Result;

use crate::cli::{Cli, Command};

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Train(args) => command::train::run(&args),
        Command::Eval(args) => command::eval::run(&args),
        Command::Backtest(args) => command::backtest::run(&args),
    }
}
