use miette::Diagnostic;
use thiserror::Error;

#[derive(Debug, Error, Diagnostic)]
pub enum Error {
    #[error(transparent)]
    #[diagnostic(code(stock::http))]
    Http(#[from] reqwest::Error),

    #[error(transparent)]
    #[diagnostic(code(stock::url))]
    Url(#[from] url::ParseError),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
