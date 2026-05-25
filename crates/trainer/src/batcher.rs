use crate::dataset::StockItem;
use crate::model::StockBatch;
use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;

/// Converts a vec of `StockItem`s into a `StockBatch` tensor pair.
///
/// Not generic on the backend so it can be reused for both the training
/// (autodiff) loader and the validation (inner) loader.
#[derive(Clone, Debug)]
pub struct StockBatcher;

impl<B: Backend> Batcher<B, StockItem, StockBatch<B>> for StockBatcher {
    fn batch(&self, _items: Vec<StockItem>, _device: &B::Device) -> StockBatch<B> {
        // TODO: stack features and targets into tensors on `device`.
        todo!("convert Vec<StockItem> into StockBatch tensors")
    }
}
