use crate::label::{LABEL_THRESHOLD, compute_labels};
use burn::data::dataloader::{DataLoader, DataLoaderIterator, Progress};
use burn::prelude::*;
use chrono::NaiveDate;
use fastrand::Rng;
use polars::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

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

/// Enumerate every `steps`-length window of every frame as a
/// `(ticker_index, window_start)` pair. This is the pool both sampling modes
/// draw from, so it is rebuilt whenever `frames` change.
fn enumerate_windows(frames: &[(PlSmallStr, DataFrame)], steps: usize) -> Vec<(u32, u32)> {
    let mut windows = Vec::new();

    for (ticker_index, (_, frame)) in frames.iter().enumerate() {
        let ticker_index = u32::try_from(ticker_index).unwrap();
        let last_start = frame.height() - steps;
        for start in 0..=u32::try_from(last_start).unwrap() {
            windows.push((ticker_index, start));
        }
    }

    windows
}

/// Mix a base seed and an epoch number into a well-distributed per-epoch seed,
/// so each epoch reshuffles differently while the whole run stays reproducible.
/// This is the `SplitMix64` finalizer.
fn splitmix64(seed: u64, epoch: u64) -> u64 {
    let mut z = seed.wrapping_add(epoch.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub struct StockBatch<B: Backend> {
    /// Shape `[batch_size, steps, ohlcv_features]`.
    pub technical: Tensor<B, 3>,
    /// Shape `[batch_size, ticker_features]`.
    pub ticker: Tensor<B, 2>,
    /// Shape `[batch_size]` — class index 0/1/2.
    pub label: Tensor<B, 1, Int>,
}

/// How a loader orders the window pool into batches.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Sampling {
    /// Reshuffle the whole pool every epoch and walk it in `batch_size` chunks,
    /// sampling without replacement. Used for training, so each epoch sees
    /// different batches in a different order.
    Shuffle,
    /// Walk the pool in its fixed order. Used for validation, so the metric is
    /// stable from one epoch to the next.
    Fixed,
}

/// Per-ticker dataloader.
///
/// `frames` holds one `(name, dense DataFrame)` per ticker, each frame with
/// columns `date`, `feature` (Array<f32, 5>) and `label` (u8), sorted by date.
/// Because a single ticker trades on contiguous rows, any `steps`-length window
/// is null-free.
///
/// Every window of every ticker is enumerated once into `windows`. A batch is
/// `batch_size` windows taken from that pool in the order the active [`Sampling`]
/// mode produces, so a batch can hold several windows of the same ticker. That
/// is fine for the per-sample classification this model does.
#[derive(Clone)]
pub(crate) struct StockDataLoader<B: Backend> {
    pub(crate) frames: Vec<(PlSmallStr, DataFrame)>,
    steps: usize,
    batch_size: usize,
    seed: Option<u64>,
    device: B::Device,
    sampling: Sampling,
    /// Every `(ticker_index, window_start)` pair, the pool both modes draw from.
    /// Shared behind an `Arc` so cloning a loader for `slice`/`to_device` stays
    /// cheap, and rebuilt whenever `frames` change.
    windows: Arc<Vec<(u32, u32)>>,
    /// Virtual-epoch cap in batches. `None` is one full pass over `windows`;
    /// `Some(k)` emits `k` (reshuffled) batches per epoch, which controls the
    /// validation cadence since burn only validates between epochs.
    max_batches: Option<usize>,
    /// First batch to emit, counted over `windows`. Non-zero only after
    /// [`Self::slice`].
    batch_offset: usize,
    /// Per-epoch counter driving the [`Sampling::Shuffle`] reseed. Shared across
    /// clones so one logical loader advances a single epoch sequence.
    epoch: Arc<AtomicU64>,
    /// Maps a ticker name (stock `code`) to its industry index. Empty until
    /// [`Self::attach_industries`] runs; while it is empty [`Self::assemble`]
    /// leaves the `ticker` tensor width-0.
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
    /// label-less row) are discarded. `max_batches` caps how many batches one
    /// epoch yields, or `None` for a full pass over every window.
    pub fn load(
        path: &str,
        steps: usize,
        batch_size: usize,
        max_batches: Option<usize>,
        seed: Option<u64>,
        device: B::Device,
    ) -> PolarsResult<Self> {
        let feature_expr = concat_arr(
            FEATURE_NAMES
                .map(|name| col(name).cast(DataType::Float32))
                .to_vec(),
        )
        .unwrap();

        let long = LazyFrame::scan_parquet(PlRefPath::new(path), ScanArgsParquet::default())?
            .select([
                col(CODE).cast(DataType::String).alias(TICKER),
                col(DATE).cast(DataType::Date),
                feature_expr.alias(FEATURE),
                col(CLOSE).cast(DataType::Float32),
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

        let windows = Arc::new(enumerate_windows(&frames, steps));

        Ok(Self {
            frames,
            steps,
            batch_size,
            seed,
            device,
            sampling: Sampling::Shuffle,
            windows,
            max_batches,
            batch_offset: 0,
            epoch: Arc::new(AtomicU64::new(0)),
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
    pub fn attach_industries(mut self, path: &str) -> PolarsResult<Self> {
        let frame = LazyFrame::scan_parquet(PlRefPath::new(path), ScanArgsParquet::default())?
            .select([
                col(CODE).cast(DataType::String).alias(TICKER),
                col(INDUSTRY).cast(DataType::String),
            ])
            .collect()?;

        let (industry_codes, n_industries) = index_industries(&frame)?;

        self.industry_codes = Arc::new(industry_codes);
        self.n_industries = n_industries;

        Ok(self)
    }

    /// Randomly keep `count` tickers, chosen with `seed` so the subset is
    /// reproducible. Asking for at least as many tickers as exist is a no-op.
    /// Used to carve a small subset for overfit diagnostics, where we want a
    /// representative random sample rather than the first `count` by sort order.
    pub fn sample_tickers(mut self, count: usize, seed: u64) -> Self {
        if count >= self.frames.len() {
            return self;
        }

        let mut rng = Rng::with_seed(seed);
        let indices = rng.choose_multiple(0..self.frames.len(), count);

        let frames: Vec<_> = indices
            .into_iter()
            .map(|index| self.frames[index].clone())
            .collect();
        self.windows = Arc::new(enumerate_windows(&frames, self.steps));
        self.frames = frames;
        self
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
        let valid = self.with_frames(valid_frames).into_fixed();

        Ok((train, valid))
    }

    fn with_frames(&self, frames: Vec<(PlSmallStr, DataFrame)>) -> Self {
        let windows = Arc::new(enumerate_windows(&frames, self.steps));
        Self {
            frames,
            windows,
            ..self.clone()
        }
    }

    /// One-hot width of the industry feature, needed to size the model's
    /// categorical branch.
    pub fn n_industries(&self) -> usize {
        self.n_industries
    }

    /// Rebuild the loader on a different backend. The frames are backend-free,
    /// so this only swaps the device and tensor type. Used to lift the train
    /// split onto the autodiff backend while validation stays on the inner one.
    pub fn to_backend<B2: Backend>(&self, device: B2::Device) -> StockDataLoader<B2> {
        StockDataLoader {
            frames: self.frames.clone(),
            steps: self.steps,
            batch_size: self.batch_size,
            seed: self.seed,
            device,
            sampling: self.sampling,
            windows: self.windows.clone(),
            max_batches: self.max_batches,
            batch_offset: self.batch_offset,
            epoch: Arc::new(AtomicU64::new(0)),
            industry_codes: self.industry_codes.clone(),
            n_industries: self.n_industries,
        }
    }

    /// Switch the loader to a deterministic fixed sweep over the whole window
    /// pool, used for validation so the metric is stable across epochs. Clears
    /// any virtual-epoch cap so one pass covers every window.
    fn into_fixed(mut self) -> Self {
        // `windows` already matches `frames` from the preceding `with_frames`,
        // so only the sweep policy needs to change here.
        self.sampling = Sampling::Fixed;
        self.max_batches = None;
        self.batch_offset = 0;
        self
    }

    /// Shuffle the window pool once with a fixed `seed` and keep the first
    /// `max_batches` batches as a representative validation subsample. A full
    /// sweep over two years of every ticker is far too large to validate every
    /// epoch, and capping the natural order instead would bias the sample to the
    /// earliest tickers and dates. The pool is shuffled once, not per epoch, so
    /// the metric stays comparable across epochs.
    pub fn into_subsample(mut self, max_batches: Option<usize>, seed: u64) -> Self {
        let mut windows = (*self.windows).clone();
        Rng::with_seed(seed).shuffle(&mut windows);

        self.windows = Arc::new(windows);
        self.sampling = Sampling::Fixed;
        self.max_batches = max_batches;
        self.batch_offset = 0;
        self
    }

    /// Number of batches one epoch yields, honoring both the `batch_offset` from
    /// a `slice` and the `max_batches` virtual-epoch cap.
    fn batch_count(&self) -> usize {
        let total = self.windows.len().div_ceil(self.batch_size);
        let available = total.saturating_sub(self.batch_offset);
        self.max_batches.map_or(available, |cap| cap.min(available))
    }

    /// Build the window order for one epoch. [`Sampling::Shuffle`] advances the
    /// epoch counter and reshuffles the pool from a per-epoch seed, so each pass
    /// differs while a fixed base seed keeps the whole run reproducible.
    /// [`Sampling::Fixed`] returns the pool in its natural order.
    fn epoch_order(&self) -> Vec<u32> {
        let mut order: Vec<u32> = (0..u32::try_from(self.windows.len()).unwrap()).collect();

        if self.sampling == Sampling::Shuffle {
            let epoch = self.epoch.fetch_add(1, Ordering::Relaxed);
            let mut rng = match self.seed {
                Some(seed) => Rng::with_seed(splitmix64(seed, epoch)),
                None => Rng::new(),
            };
            rng.shuffle(&mut order);
        }

        order
    }

    /// Slice each chosen window, normalize it, and pack the batch into tensors.
    fn assemble(&self, selection: &[(u32, u32)]) -> PolarsResult<StockBatch<B>> {
        let count = selection.len();

        let mut technical_data = Vec::with_capacity(count * self.steps * FEATURE_NAMES.len());

        let mut label_data = Vec::with_capacity(count);

        for &(ticker_index, start) in selection {
            let window = self.frames[ticker_index as usize]
                .1
                .slice(i64::from(start), self.steps);

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
    fn one_hot_industries(&self, selection: &[(u32, u32)]) -> Tensor<B, 2> {
        let count = selection.len();

        if self.n_industries == 0 {
            return Tensor::zeros([count, 0], &self.device);
        }

        let unknown = self.n_industries - 1;

        let mut data = vec![0.0f32; count * self.n_industries];

        for (row, &(ticker_index, _)) in selection.iter().enumerate() {
            let name = &self.frames[ticker_index as usize].0;
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
    /// Window indices into `loader.windows` for this epoch, already ordered by
    /// the active sampling mode.
    order: Vec<u32>,
    /// Batch cursor in `[0, num_batches)`.
    batch: usize,
    num_batches: usize,
}

impl<B: Backend> Iterator for StockIterator<'_, B> {
    type Item = StockBatch<B>;

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.num_batches - self.batch;
        (remaining, Some(remaining))
    }

    fn next(&mut self) -> Option<StockBatch<B>> {
        if self.batch >= self.num_batches {
            return None;
        }

        let batch_size = self.loader.batch_size;
        let order_start = (self.loader.batch_offset + self.batch) * batch_size;
        let order_end = (order_start + batch_size).min(self.order.len());

        let selection: Vec<(u32, u32)> = self.order[order_start..order_end]
            .iter()
            .map(|&window| self.loader.windows[window as usize])
            .collect();

        self.batch += 1;

        self.loader.assemble(&selection).ok()
    }
}

impl<B: Backend> DataLoaderIterator<StockBatch<B>> for StockIterator<'_, B> {
    fn progress(&self) -> Progress {
        Progress {
            items_processed: self.batch,
            items_total: self.num_batches,
        }
    }
}

impl<B: Backend> DataLoader<B, StockBatch<B>> for StockDataLoader<B> {
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<StockBatch<B>> + 'a> {
        Box::new(StockIterator {
            loader: self,
            order: self.epoch_order(),
            batch: 0,
            num_batches: self.batch_count(),
        })
    }

    fn num_items(&self) -> usize {
        self.batch_count()
    }

    fn to_device(&self, device: &B::Device) -> Arc<dyn DataLoader<B, StockBatch<B>>> {
        Arc::new(Self {
            device: device.clone(),
            ..self.clone()
        })
    }

    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<B, StockBatch<B>>> {
        assert!(
            start <= end && end <= self.batch_count(),
            "slice [{start}, {end}) out of bounds for batch count {}",
            self.batch_count()
        );

        Arc::new(Self {
            batch_offset: self.batch_offset + start,
            max_batches: Some(end - start),
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
        batch_size: usize,
        seed: Option<u64>,
        max_batches: usize,
    ) -> StockDataLoader<TestBackend> {
        let frames: Vec<(PlSmallStr, DataFrame)> = (0..n_frames)
            .map(|t| {
                (
                    format!("t{t}").into(),
                    make_frame(f32::from(t) * 1000.0, rows),
                )
            })
            .collect();

        let windows = Arc::new(enumerate_windows(&frames, steps));

        StockDataLoader {
            frames,
            steps,
            batch_size,
            seed,
            device: FlexDevice,
            sampling: Sampling::Shuffle,
            windows,
            // A test passes 0 to mean "no cap", matching the production default.
            max_batches: (max_batches != 0).then_some(max_batches),
            batch_offset: 0,
            epoch: Arc::new(AtomicU64::new(0)),
            industry_codes: Arc::new(HashMap::new()),
            n_industries: 0,
        }
    }

    /// Drain the first batch of a fresh epoch.
    fn first_batch(loader: &StockDataLoader<TestBackend>) -> StockBatch<TestBackend> {
        loader.iter().next().unwrap()
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

        let batch = first_batch(&loader);

        assert_eq!(batch.ticker.dims(), [3, 3]);

        // Every row is one-hot, so each row sums to exactly one.
        let row_sums = batch.ticker.sum_dim(1).into_data();
        for value in row_sums.to_vec::<f32>().unwrap() {
            assert!((value - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn first_epoch_is_reproducible_and_shaped() {
        // Two fresh loaders with the same seed start at epoch 0, so their first
        // epoch's first batch is identical. Reproducibility is now per-epoch via
        // the counter, not a pure function of the batch index.
        let first = first_batch(&make_loader(5, 20, 4, 3, Some(42), 8));
        let again = first_batch(&make_loader(5, 20, 4, 3, Some(42), 8));

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
    fn shuffle_varies_across_epochs() {
        // The same loader reshuffles on each `iter`, so consecutive epochs draw
        // a different first batch. make_frame offsets each ticker by 1000, so
        // different windows give different technical data.
        let loader = make_loader(5, 30, 4, 4, Some(1), 0);

        let epoch_one = first_batch(&loader).technical.to_data();
        let epoch_two = first_batch(&loader).technical.to_data();

        assert!(
            epoch_one.to_vec::<f32>().unwrap() != epoch_two.to_vec::<f32>().unwrap(),
            "consecutive epochs should reshuffle to a different first batch"
        );
    }

    #[test]
    fn fixed_sweeps_every_window_once() {
        // 3 tickers of 20 rows, window of 4, two windows per batch.
        let loader = make_loader(3, 20, 4, 2, None, 0).into_fixed();

        // Each ticker yields 20 - 4 + 1 = 17 windows, so 51 across 26 batches.
        let windows_per_ticker = 20 - 4 + 1;
        let total_windows: usize = windows_per_ticker * 3;
        assert_eq!(loader.num_items(), total_windows.div_ceil(2));

        let swept: usize = loader.iter().map(|batch| batch.label.dims()[0]).sum();

        assert_eq!(swept, total_windows);
    }

    #[test]
    fn slice_narrows_batch_range() {
        let loader = make_loader(5, 20, 4, 3, Some(1), 10);

        let shard = loader.slice(2, 7);

        assert_eq!(shard.num_items(), 5);
    }

    #[test]
    fn subsample_is_capped_and_stable() {
        // 4 tickers of 30 rows, window 4, three windows per batch.
        let loader = make_loader(4, 30, 4, 3, None, 0).into_subsample(Some(2), 99);

        // The cap holds regardless of how many windows the pool actually has.
        assert_eq!(loader.num_items(), 2);

        // Fixed mode does not advance the epoch counter, so repeated passes see
        // the same subsample in the same order.
        let first = first_batch(&loader).technical.to_data();
        let again = first_batch(&loader).technical.to_data();

        assert_eq!(
            first.to_vec::<f32>().unwrap(),
            again.to_vec::<f32>().unwrap()
        );
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
