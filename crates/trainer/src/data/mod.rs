//! The trainer's data layer on top of `stock_model::data::TickerFrames`: triple-barrier
//! labeling of the standardized frames. Loading and splitting live in the shared store.

pub mod label;

/// `n_tickers` tickers of `rows` rows each, for the dataset and batcher tests. Row `i`
/// fills every feature slot with `ticker * 1000 + i`, so a window's value identifies its
/// ticker and row; labels cycle 0/1/2.
#[cfg(test)]
pub(crate) fn synthetic(n_tickers: i16, rows: i16) -> stock_model::data::TickerFrames {
    use polars::prelude::*;
    use stock_model::data::{LABEL, TickerFrames};
    use stock_model::features::{FEATURE, FEATURE_NAMES, TICKER, feature_array};

    let height = usize::try_from(rows).unwrap();

    let frames = (0..n_tickers)
        .map(|ticker| {
            let base = f32::from(ticker) * 1000.0;
            let value: Vec<f32> = (0..rows).map(|i| base + f32::from(i)).collect();

            let mut columns = vec![
                Column::new(TICKER, vec![format!("t{ticker}"); height]),
                Column::new(
                    LABEL,
                    (0..rows)
                        .map(|i| u8::try_from(i % 3).unwrap())
                        .collect::<Vec<_>>(),
                ),
            ];
            for feature in FEATURE_NAMES {
                columns.push(Column::new(feature, value.clone()));
            }

            DataFrame::new(height, columns)
                .unwrap()
                .lazy()
                .with_column(feature_array().alias(FEATURE))
                .select([col(TICKER), col(FEATURE), col(LABEL)])
                .collect()
                .unwrap()
        })
        .collect();

    TickerFrames { frames }
}
