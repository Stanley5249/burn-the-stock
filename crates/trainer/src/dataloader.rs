use crate::label::{LABEL_THRESHOLD, compute_labels};
use burn::data::dataloader::{DataLoader, DataLoaderIterator, Progress};
use burn::prelude::*;
use chrono::NaiveDate;
use fastrand::Rng;
use polars::prelude::*;
use std::ops::Range;
use std::sync::Arc;

const SEP: PlSmallStr = PlSmallStr::from_static("_");

const MARKET: PlSmallStr = PlSmallStr::from_static("market");
const CODE: PlSmallStr = PlSmallStr::from_static("code");
const TICKER: PlSmallStr = PlSmallStr::from_static("ticker");
const DATE: PlSmallStr = PlSmallStr::from_static("date");
const FEATURE: PlSmallStr = PlSmallStr::from_static("feature");
const LABEL: PlSmallStr = PlSmallStr::from_static("label");

const OPEN: PlSmallStr = PlSmallStr::from_static("open");
const HIGH: PlSmallStr = PlSmallStr::from_static("high");
const LOW: PlSmallStr = PlSmallStr::from_static("low");
const CLOSE: PlSmallStr = PlSmallStr::from_static("close");
const VOLUME: PlSmallStr = PlSmallStr::from_static("volume");

const FEATURE_NAMES: [PlSmallStr; 5] = [OPEN, HIGH, LOW, CLOSE, VOLUME];

const fn col(name: PlSmallStr) -> Expr {
    Expr::Column(name)
}

pub struct StockBatch<B: Backend> {
    /// Shape `[batch_size, steps, ohlcv_features]`.
    pub technical: Tensor<B, 3>,
    /// Shape `[batch_size, ticker_features]`.
    pub ticker: Tensor<B, 2>,
    /// Shape `[batch_size]` — class index 0/1/2.
    pub label: Tensor<B, 1, Int>,
}

/// Per-ticker dataloader.
///
/// `frames` holds one `(name, dense DataFrame)` per ticker, each frame with
/// columns `date`, `feature` (Array<f32, 5>) and `label` (u8), sorted by date.
/// Because a single ticker trades on contiguous rows, any `steps`-length window
/// is null-free, so a batch always has exactly `n_tickers` rows with no gaps to
/// drop.
///
/// Each batch draws `n_tickers` tickers and an independent random window per
/// ticker, so samples are not date-aligned across the batch. That is fine for
/// the per-sample classification this model does.
#[derive(Clone)]
struct StockDataLoader<B: Backend> {
    frames: Vec<(PlSmallStr, DataFrame)>,
    steps: usize,
    n_tickers: usize,
    seed: Option<u64>,
    device: B::Device,
    index_range: Range<usize>,
}

impl<B: Backend> StockDataLoader<B> {
    /// Load the parquet file into one dense frame per ticker.
    ///
    /// Tickers with fewer than `steps` usable rows (after dropping the final
    /// label-less row) are discarded. `epoch_size` sets how many batches one
    /// iteration pass yields.
    pub fn load(
        path: PlRefPath,
        steps: usize,
        n_tickers: usize,
        epoch_size: usize,
        seed: Option<u64>,
        device: B::Device,
    ) -> PolarsResult<Self> {
        let schema = Some(Arc::new(Schema::from_iter([
            (MARKET, DataType::String),
            (CODE, DataType::String),
            (DATE, DataType::Date),
            (OPEN, DataType::Float32),
            (HIGH, DataType::Float32),
            (LOW, DataType::Float32),
            (CLOSE, DataType::Float32),
            (VOLUME, DataType::Float32),
        ])));

        let args = ScanArgsParquet {
            schema,
            ..Default::default()
        };

        let ticker_expr = concat_str([col(MARKET), col(CODE)], &SEP, false);

        let feature_expr = concat_arr(FEATURE_NAMES.map(col).to_vec()).unwrap();

        let long = LazyFrame::scan_parquet(path, args)?
            .select([
                ticker_expr.alias(TICKER),
                col(DATE),
                feature_expr.alias(FEATURE),
                col(CLOSE),
            ])
            .sort([TICKER, DATE], SortMultipleOptions::new())
            .collect()?;

        let groups = long.partition_by_stable([TICKER], true)?;

        let mut frames = Vec::with_capacity(groups.len());

        for group in groups {
            let height = group.height();

            // The last row has no forward window, so a usable frame needs more
            // than `steps` rows to keep at least `steps` after trimming it.
            if height <= steps {
                continue;
            }

            let name: PlSmallStr = group.column(&TICKER)?.str()?.get(0).unwrap().into();

            let labels = compute_labels(group.column(&CLOSE)?, LABEL, LABEL_THRESHOLD)?;

            let mut frame = group.select([DATE, FEATURE])?.head(Some(height - 1));

            frame.with_column(labels)?;

            frames.push((name, frame));
        }

        Ok(Self {
            frames,
            steps,
            n_tickers,
            seed,
            device,
            index_range: 0..epoch_size,
        })
    }

    /// Split every ticker frame at `cutoff` into an earlier train loader and a
    /// later valid loader. Tickers whose train or valid side has fewer than
    /// `steps` rows are dropped from that side. Both loaders share the same
    /// config; errors if either side ends up empty.
    pub fn train_valid_split(&self, cutoff: NaiveDate) -> PolarsResult<(Self, Self)> {
        let mut train_frames = Vec::with_capacity(self.frames.len());
        let mut valid_frames = Vec::with_capacity(self.frames.len());

        for (name, frame) in &self.frames {
            let split = frame
                .clone()
                .lazy()
                .filter(col(DATE).lt(lit(cutoff)))
                .collect()?
                .height();

            let (train, valid) = frame.split_at(i64::try_from(split).unwrap());

            if train.height() >= self.steps {
                train_frames.push((name.clone(), train));
            }
            if valid.height() >= self.steps {
                valid_frames.push((name.clone(), valid));
            }
        }

        polars_ensure!(
            !train_frames.is_empty() && !valid_frames.is_empty(),
            NoData: "train/valid split left one side empty; check cutoff and steps"
        );

        let train = self.with_frames(train_frames);
        let valid = self.with_frames(valid_frames);

        Ok((train, valid))
    }

    fn with_frames(&self, frames: Vec<(PlSmallStr, DataFrame)>) -> Self {
        Self {
            frames,
            ..self.clone()
        }
    }

    fn epoch_size(&self) -> usize {
        self.index_range.end - self.index_range.start
    }

    /// Assemble one [`StockBatch`] from `n_tickers` random tickers, each over
    /// its own random `steps`-length window. The whole batch is a pure function
    /// of `seed + index`, so iterating the same index range twice yields the
    /// same data.
    fn batch(&self, index: usize) -> PolarsResult<StockBatch<B>> {
        let mut rng = match self.seed {
            Some(seed) => Rng::with_seed(seed.wrapping_add(u64::try_from(index).unwrap())),
            None => Rng::new(),
        };

        let count = self.n_tickers.min(self.frames.len());
        let chosen = rng.choose_multiple(0..self.frames.len(), count);

        let mut technical_data = Vec::with_capacity(count * self.steps * FEATURE_NAMES.len());

        let mut label_data = Vec::with_capacity(count);

        for &ticker_index in &chosen {
            let dataframe = &self.frames[ticker_index].1;

            let end = i64::try_from(dataframe.height() - self.steps).unwrap();
            let start = rng.i64(0..=end);

            let window = dataframe.slice(start, self.steps);

            let features = window.column(&FEATURE)?;

            technical_data.extend(features.array()?.get_inner().f32()?.into_no_null_iter());

            let label = window.column(&LABEL)?.u8()?.last().unwrap();

            label_data.push(i32::from(label));
        }

        let technical = Tensor::from_data(
            TensorData::new(technical_data, [count, self.steps, FEATURE_NAMES.len()]),
            &self.device,
        );

        let label = Tensor::from_data(TensorData::new(label_data, [count]), &self.device);

        // TODO: No ticker-level features yet; placeholder empty second dimension.
        let ticker: Tensor<B, 2> = Tensor::zeros([count, 0], &self.device);

        Ok(StockBatch {
            technical,
            ticker,
            label,
        })
    }
}

struct StockIterator<'a, B: Backend> {
    loader: &'a StockDataLoader<B>,
    index: usize,
}

impl<B: Backend> Iterator for StockIterator<'_, B> {
    type Item = StockBatch<B>;

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.loader.index_range.end - self.index;
        (remaining, Some(remaining))
    }

    fn next(&mut self) -> Option<StockBatch<B>> {
        if self.index >= self.loader.index_range.end {
            return None;
        }

        let batch = self.loader.batch(self.index).ok()?;

        self.index += 1;

        Some(batch)
    }
}

impl<B: Backend> DataLoaderIterator<StockBatch<B>> for StockIterator<'_, B> {
    fn progress(&self) -> Progress {
        Progress {
            items_processed: self.index - self.loader.index_range.start,
            items_total: self.loader.epoch_size(),
        }
    }
}

impl<B: Backend> DataLoader<B, StockBatch<B>> for StockDataLoader<B> {
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<StockBatch<B>> + 'a> {
        Box::new(StockIterator {
            loader: self,
            index: self.index_range.start,
        })
    }

    fn num_items(&self) -> usize {
        self.epoch_size()
    }

    fn to_device(&self, device: &B::Device) -> Arc<dyn DataLoader<B, StockBatch<B>>> {
        Arc::new(Self {
            device: device.clone(),
            ..self.clone()
        })
    }

    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<B, StockBatch<B>>> {
        assert!(
            start <= end && end <= self.epoch_size(),
            "slice [{start}, {end}) out of bounds for epoch size {}",
            self.epoch_size()
        );

        let base = self.index_range.start;

        Arc::new(Self {
            index_range: base + start..base + end,
            ..self.clone()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::flex::{Flex, FlexDevice};

    type TestBackend = Flex;

    /// Build one dense ticker frame of `rows` rows with a `feature` `Array<f32,
    /// 5>` column and a `u8` `label` column. `base` offsets the values so
    /// frames differ.
    fn make_frame(base: f32, rows: i16) -> DataFrame {
        let dates: Vec<i32> = (0..i32::from(rows)).collect();
        let values: Vec<f32> = (0..rows).map(|i| base + f32::from(i)).collect();
        let labels: Vec<u8> = (0..rows).map(|i| u8::try_from(i % 3).unwrap()).collect();

        let df = df!(
            "date" => dates,
            "open" => values.clone(),
            "high" => values.clone(),
            "low" => values.clone(),
            "close" => values.clone(),
            "volume" => values,
            "label" => labels,
        )
        .unwrap();

        df.lazy()
            .select([
                col(DATE).cast(DataType::Date),
                concat_arr(FEATURE_NAMES.map(col).to_vec())
                    .unwrap()
                    .alias(FEATURE),
                col(LABEL),
            ])
            .collect()
            .unwrap()
    }

    fn make_loader(
        n_frames: i16,
        rows: i16,
        steps: usize,
        n_tickers: usize,
        seed: Option<u64>,
        epoch_size: usize,
    ) -> StockDataLoader<TestBackend> {
        let frames: Vec<(PlSmallStr, DataFrame)> = (0..n_frames)
            .map(|t| {
                (
                    format!("t{t}").into(),
                    make_frame(f32::from(t) * 1000.0, rows),
                )
            })
            .collect();

        StockDataLoader {
            frames,
            steps,
            n_tickers,
            seed,
            device: FlexDevice,
            index_range: 0..epoch_size,
        }
    }

    #[test]
    fn batch_is_reproducible_and_shaped() {
        let loader = make_loader(5, 20, 4, 3, Some(42), 8);

        let first = loader.batch(0).unwrap();
        let again = loader.batch(0).unwrap();

        assert_eq!(first.technical.dims(), [3, 4, 5]);
        assert_eq!(first.label.dims(), [3]);

        first
            .technical
            .to_data()
            .assert_eq(&again.technical.to_data(), true);
        first
            .label
            .to_data()
            .assert_eq(&again.label.to_data(), true);
    }

    #[test]
    fn slice_narrows_index_range() {
        let loader = make_loader(5, 20, 4, 3, Some(1), 10);

        let shard = loader.slice(2, 7);

        assert_eq!(shard.num_items(), 5);
    }

    #[test]
    fn train_valid_split_partitions_rows() {
        let loader = make_loader(3, 20, 4, 3, Some(0), 8);
        let cutoff = NaiveDate::from_ymd_opt(1970, 1, 11).unwrap();

        let (train, valid) = loader.train_valid_split(cutoff).unwrap();

        assert_eq!(train.frames.len(), 3);
        assert_eq!(valid.frames.len(), 3);
        assert_eq!(train.frames[0].1.height(), 10);
        assert_eq!(valid.frames[0].1.height(), 10);
    }
}
