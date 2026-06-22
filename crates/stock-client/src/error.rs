use miette::Diagnostic;
use thiserror::Error;

#[derive(Debug, Error, Diagnostic)]
pub enum Error {
    #[error(transparent)]
    #[diagnostic(code(stock::http))]
    Http(#[from] reqwest::Error),

    #[error(transparent)]
    #[diagnostic(code(stock::env))]
    Env(#[from] std::env::VarError),

    #[error(transparent)]
    #[diagnostic(code(stock::url))]
    Url(#[from] url::ParseError),

    #[error(transparent)]
    #[diagnostic(code(stock::header))]
    Header(#[from] reqwest::header::InvalidHeaderValue),

    #[error("API returned failure: {status}")]
    #[diagnostic(code(stock::api))]
    Api { status: String },

    #[error("invalid row: {0}")]
    #[diagnostic(code(stock::invalid_row))]
    InvalidRow(String),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
