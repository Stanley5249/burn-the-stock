use miette::Diagnostic;
use thiserror::Error;

#[derive(Debug, Error, Diagnostic)]
pub enum Error {
    #[error(transparent)]
    #[diagnostic(code(stock::http))]
    Http(#[from] reqwest::Error),

    #[error(transparent)]
    #[diagnostic(code(stock::json))]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    #[diagnostic(code(stock::env))]
    Env(#[from] std::env::VarError),

    #[error("API returned failure: {status}")]
    #[diagnostic(code(stock::api))]
    ApiFailure { status: String },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
