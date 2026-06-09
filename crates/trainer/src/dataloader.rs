use crate::label::{LABEL_THRESHOLD, compute_labels};
use burn::data::dataloader::{DataLoader, DataLoaderIterator, Progress};
use burn::prelude::*;
use chrono::NaiveDate;
use fastrand::Rng;
use polars::prelude::*;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

const SEP: PlSmallStr = PlSmallStr::from_static("_");

const MARKET: PlSmallStr = PlSmallStr::from_static("market");
const CODE: PlSmallStr = PlSmallStr::from_static("code");
const TICKER: PlSmallStr = PlSmallStr::from_static("ticker");
const DATE: PlSmallStr = PlSmallStr::from_static("date");
const FEATURE: PlSmallStr = PlSmallStr::from_static("feature");
const LABEL: PlSmallStr = PlSmallStr::from_static("label");
const INDUSTRY: PlSmallStr = PlSmallStr::from_static("industry");

const OPEN: PlSmallStr = PlSmallStr::from_static("open");
const HIGH: PlSmallStr = PlSmallStr::from_static("high");
const LOW: PlSmallStr = PlSmallStr::from_static("low");
const CLOSE: PlSmallStr = PlSmallStr::from_static("close");
const VOLUME: PlSmallStr = PlSmallStr::from_static("volume");

const FEATURE_NAMES: [PlSmallStr; 5] = [OPEN, HIGH, LOW, CLOSE, VOLUME];

/// Offsets into a flattened OHLCV row. OHLC occupy the first four slots, so the
/// close sits at index 3 and volume, the lone non-price feature, sits last.
const CLOSE_OFFSET: usize = 3;
const VOLUME_OFFSET: usize = 4;

const fn col(name: PlSmallStr) -> Expr {
    Expr::Column(name)
}

/// Normalize one flattened `[steps * 5]` OHLCV window in place.
///
/// A GRU should learn the shape of a window, not the price level it happens to
/// sit at, so OHLC are divided by the window's last close and land near 1.0.
/// Volume lives on a wildly different scale, so it is `log1p` compressed and
/// then z-scored within the window to match the price ratios.
#[allow(clippy::cast_precision_loss)] // `steps` is a small window length
fn normalize_window(window: &mut [f32], steps: usize) {
    let stride = FEATURE_NAMES.len();

    let last_close = window[(steps - 1) * stride + CLOSE_OFFSET];

    for row in window.chunks_mut(stride) {
        if last_close != 0.0 {
            for price in &mut row[..VOLUME_OFFSET] {
                *price /= last_close;
            }
        }
        row[VOLUME_OFFSET] = row[VOLUME_OFFSET].max(0.0).ln_1p();
    }

    let volumes = || window.iter().skip(VOLUME_OFFSET).step_by(stride);

    let mean = volumes().sum::<f32>() / steps as f32;
    let variance = volumes().map(|v| (v - mean).powi(2)).sum::<f32>() / steps as f32;
    let std = variance.sqrt();

    for row in window.chunks_mut(stride) {
        row[VOLUME_OFFSET] = if std > f32::EPSILON {
            (row[VOLUME_OFFSET] - mean) / std
        } else {
            0.0
        };
    }
}

/// Build the `ticker name -> industry index` map from a frame with `ticker`
/// and `industry` string columns. Distinct industries are indexed in sorted
/// order for a stable encoding, and the returned width includes one extra
/// bucket for tickers with an unknown industry.
fn index_industries(frame: &DataFrame) -> PolarsResult<(HashMap<PlSmallStr, usize>, usize)> {
    let names = frame.column(&TICKER)?.str()?;
    let industries = frame.column(&INDUSTRY)?.str()?;

    let mut distinct: Vec<&str> = industries.into_iter().flatten().collect();
    distinct.sort_unstable();
    distinct.dedup();

    let index_of: HashMap<&str, usize> = distinct
        .iter()
        .enumerate()
        .map(|(index, industry)| (*industry, index))
        .collect();

    let mut codes = HashMap::new();
    for (name, industry) in names.into_iter().zip(industries) {
        if let (Some(name), Some(industry)) = (name, industry) {
            codes.insert(name.into(), index_of[industry]);
        }
    }

    // The trailing bucket catches tickers whose industry is null or missing.
    Ok((codes, distinct.len() + 1))
}

pub struct StockBatch<B: Backend> {
    /// Shape `[batch_size, steps, ohlcv_features]`.
    pub technical: Tensor<B, 3>,
    /// Shape `[batch_size, ticker_features]`.
    pub ticker: Tensor<B, 2>,
    /// Shape `[batch_size]` — class index 0/1/2.
    pub label: Tensor<B, 1, Int>,
}

/// How a loader picks the windows that make up each batch.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Sampling {
    /// Random tickers and random windows per batch, reseeded by batch index.
    /// Used for training, where stochastic coverage is what we want.
    Random,
    /// Deterministic sweep over every window of every ticker, in order. Used
    /// for validation, so metrics are stable from one epoch to the next.
    Sequential,
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
    sampling: Sampling,
    /// Every `(ticker_index, window_start)` pair, only populated in
    /// [`Sampling::Sequential`]. Shared behind an `Arc` so cloning a loader for
    /// `slice`/`to_device` stays cheap.
    windows: Arc<Vec<(usize, i64)>>,
    /// Maps a ticker name (`market_code`) to its industry index. Empty until
    /// [`Self::attach_industries`] runs, in which case [`Self::assemble`] leaves
    /// the `ticker` tensor width-0.
    industry_codes: Arc<HashMap<PlSmallStr, usize>>,
    /// One-hot width for the industry feature: distinct industries plus a final
    /// bucket for tickers with no known industry. Zero means no categorical
    /// feature is attached.
    n_industries: usize,
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
            sampling: Sampling::Random,
            windows: Arc::new(Vec::new()),
            industry_codes: Arc::new(HashMap::new()),
            n_industries: 0,
        })
    }

    /// Attach the industry categorical feature from a `tickers.parquet` written
    /// by the `tickers` prefetch bin (columns `market`, `code`, `industry`).
    ///
    /// Distinct industries get stable indices in sorted order, and a final
    /// bucket absorbs every ticker whose industry is null or absent from the
    /// file. Must run before [`Self::train_valid_split`] so the mapping, which
    /// is keyed by ticker name, propagates to both sides.
    pub fn attach_industries(mut self, path: PlRefPath) -> PolarsResult<Self> {
        let frame = LazyFrame::scan_parquet(path, ScanArgsParquet::default())?
            .select([
                concat_str([col(MARKET), col(CODE)], &SEP, false).alias(TICKER),
                col(INDUSTRY).cast(DataType::String),
            ])
            .collect()?;

        let (industry_codes, n_industries) = index_industries(&frame)?;

        self.industry_codes = Arc::new(industry_codes);
        self.n_industries = n_industries;

        Ok(self)
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
        let valid = self.with_frames(valid_frames).into_sequential();

        Ok((train, valid))
    }

    fn with_frames(&self, frames: Vec<(PlSmallStr, DataFrame)>) -> Self {
        Self {
            frames,
            ..self.clone()
        }
    }

    /// Switch the loader to a deterministic full sweep. Enumerates every
    /// `steps`-length window of every ticker and sizes the epoch to cover them
    /// all in batches of `n_tickers`, so the final batch may be short.
    fn into_sequential(mut self) -> Self {
        let mut windows = Vec::new();

        for (ticker_index, (_, frame)) in self.frames.iter().enumerate() {
            let last_start = i64::try_from(frame.height() - self.steps).unwrap();
            for start in 0..=last_start {
                windows.push((ticker_index, start));
            }
        }

        let batches = windows.len().div_ceil(self.n_tickers);

        self.sampling = Sampling::Sequential;
        self.windows = Arc::new(windows);
        self.index_range = 0..batches;
        self
    }

    fn epoch_size(&self) -> usize {
        self.index_range.end - self.index_range.start
    }

    /// Assemble one [`StockBatch`] for batch `index`. The windows come from the
    /// active [`Sampling`] mode, then share the same extraction and tensor
    /// packing.
    fn batch(&self, index: usize) -> PolarsResult<StockBatch<B>> {
        let selection = match self.sampling {
            Sampling::Random => self.random_selection(index),
            Sampling::Sequential => self.sequential_selection(index),
        };

        self.assemble(&selection)
    }

    /// Pick `n_tickers` random tickers, each over its own random window. The
    /// choice is a pure function of `seed + index`, so the same index yields the
    /// same windows.
    fn random_selection(&self, index: usize) -> Vec<(usize, i64)> {
        let mut rng = match self.seed {
            Some(seed) => Rng::with_seed(seed.wrapping_add(u64::try_from(index).unwrap())),
            None => Rng::new(),
        };

        let count = self.n_tickers.min(self.frames.len());
        let chosen = rng.choose_multiple(0..self.frames.len(), count);

        chosen
            .into_iter()
            .map(|ticker_index| {
                let height = self.frames[ticker_index].1.height();
                let end = i64::try_from(height - self.steps).unwrap();
                (ticker_index, rng.i64(0..=end))
            })
            .collect()
    }

    /// Take the `index`-th contiguous slice of `n_tickers` windows from the
    /// precomputed sweep. The last batch may be short.
    fn sequential_selection(&self, index: usize) -> Vec<(usize, i64)> {
        let start = index * self.n_tickers;
        let end = (start + self.n_tickers).min(self.windows.len());

        self.windows[start..end].to_vec()
    }

    /// Slice each chosen window, normalize it, and pack the batch into tensors.
    fn assemble(&self, selection: &[(usize, i64)]) -> PolarsResult<StockBatch<B>> {
        let count = selection.len();

        let mut technical_data = Vec::with_capacity(count * self.steps * FEATURE_NAMES.len());

        let mut label_data = Vec::with_capacity(count);

        for &(ticker_index, start) in selection {
            let window = self.frames[ticker_index].1.slice(start, self.steps);

            let features = window.column(&FEATURE)?;

            let mut flat: Vec<f32> = features
                .array()?
                .get_inner()
                .f32()?
                .into_no_null_iter()
                .collect();

            normalize_window(&mut flat, self.steps);

            technical_data.extend(flat);

            let label = window.column(&LABEL)?.u8()?.last().unwrap();

            label_data.push(i32::from(label));
        }

        let technical = Tensor::from_data(
            TensorData::new(technical_data, [count, self.steps, FEATURE_NAMES.len()]),
            &self.device,
        );

        let label = Tensor::from_data(TensorData::new(label_data, [count]), &self.device);

        let ticker = self.one_hot_industries(selection);

        Ok(StockBatch {
            technical,
            ticker,
            label,
        })
    }

    /// One-hot encode the industry of each selected ticker into a
    /// `[count, n_industries]` tensor. With no industries attached this stays a
    /// width-0 placeholder. Tickers absent from the map fall into the trailing
    /// unknown bucket.
    fn one_hot_industries(&self, selection: &[(usize, i64)]) -> Tensor<B, 2> {
        let count = selection.len();

        if self.n_industries == 0 {
            return Tensor::zeros([count, 0], &self.device);
        }

        let unknown = self.n_industries - 1;

        let mut data = vec![0.0f32; count * self.n_industries];

        for (row, &(ticker_index, _)) in selection.iter().enumerate() {
            let name = &self.frames[ticker_index].0;
            let industry = self.industry_codes.get(name).copied().unwrap_or(unknown);
            data[row * self.n_industries + industry] = 1.0;
        }

        Tensor::from_data(
            TensorData::new(data, [count, self.n_industries]),
            &self.device,
        )
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
            sampling: Sampling::Random,
            windows: Arc::new(Vec::new()),
            industry_codes: Arc::new(HashMap::new()),
            n_industries: 0,
        }
    }

    #[test]
    fn normalize_window_scales_price_and_volume() {
        // Three rows of (open, high, low, close, volume), last close is 16.
        let mut window = vec![
            10.0, 11.0, 9.0, 10.0, 100.0, //
            12.0, 13.0, 11.0, 12.0, 200.0, //
            14.0, 15.0, 13.0, 16.0, 300.0, //
        ];

        normalize_window(&mut window, 3);

        // OHLC are divided by the window's last close, so it lands exactly on 1.
        assert!((window[2 * 5 + CLOSE_OFFSET] - 1.0).abs() < 1e-6);
        assert!((window[3] - 10.0 / 16.0).abs() < 1e-6);

        // Volume is z-scored, so it has near-zero mean and stays monotonic.
        let volumes = [window[4], window[9], window[14]];
        let mean = volumes.iter().sum::<f32>() / 3.0;
        assert!(mean.abs() < 1e-6);
        assert!(volumes[0] < volumes[1] && volumes[1] < volumes[2]);
    }

    #[test]
    fn index_industries_encodes_and_buckets_unknown() {
        let frame = df!(
            "ticker" => ["tse_2330", "otc_1240", "tse_2317", "tse_9999"],
            "industry" => [Some("24"), Some("16"), Some("24"), None],
        )
        .unwrap();

        let (codes, n_industries) = index_industries(&frame).unwrap();

        // Two distinct industries ("16", "24") plus the unknown bucket.
        assert_eq!(n_industries, 3);
        // Sorted order: "16" -> 0, "24" -> 1.
        assert_eq!(codes[&PlSmallStr::from_static("otc_1240")], 0);
        assert_eq!(codes[&PlSmallStr::from_static("tse_2330")], 1);
        assert_eq!(codes[&PlSmallStr::from_static("tse_2317")], 1);
        // The null-industry ticker is left out of the map entirely.
        assert!(!codes.contains_key(&PlSmallStr::from_static("tse_9999")));
    }

    #[test]
    fn industry_feature_is_one_hot() {
        let mut loader = make_loader(3, 20, 4, 3, Some(7), 4);

        // make_loader names tickers t0, t1, t2; leave t2 to the unknown bucket.
        let codes = HashMap::from([
            (PlSmallStr::from_static("t0"), 0),
            (PlSmallStr::from_static("t1"), 1),
        ]);
        loader.industry_codes = Arc::new(codes);
        loader.n_industries = 3;

        let batch = loader.batch(0).unwrap();

        assert_eq!(batch.ticker.dims(), [3, 3]);

        // Every row is one-hot, so each row sums to exactly one.
        let row_sums = batch.ticker.sum_dim(1).into_data();
        for value in row_sums.to_vec::<f32>().unwrap() {
            assert!((value - 1.0).abs() < 1e-6);
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
    fn sequential_sweeps_every_window_once() {
        // 3 tickers of 20 rows, window of 4, two windows per batch.
        let loader = make_loader(3, 20, 4, 2, None, 8).into_sequential();

        // Each ticker yields 20 - 4 + 1 = 17 windows, so 51 across 26 batches.
        let windows_per_ticker = 20 - 4 + 1;
        let total_windows: usize = windows_per_ticker * 3;
        assert_eq!(loader.num_items(), total_windows.div_ceil(2));

        let swept: usize = (0..loader.epoch_size())
            .map(|index| loader.batch(index).unwrap().label.dims()[0])
            .sum();

        assert_eq!(swept, total_windows);
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
