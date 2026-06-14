mod cli;
mod command;
mod label;
mod link;
mod logging;
mod report;
mod store;
mod training;

use clap::Parser;
use miette::Result;

use crate::cli::{Cli, Command};

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Train(args) => command::train::run(&args),
        Command::Predict(args) => command::predict::run(&args),
    }
}
