//! The model's output classes and their fixed index order, shared by the labeler that
//! produces them and the inference path that reads them back. Plain data with no burn
//! or polars, so every layer names the same Sell/Hold/Buy contract.

/// One of the model's three classes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Sell,
    Hold,
    Buy,
}

/// Class indices in the model's output order, derived from the enum discriminants so
/// callers index probabilities and labels without respelling the order.
pub const SELL: usize = Action::Sell as usize;
pub const HOLD: usize = Action::Hold as usize;
pub const BUY: usize = Action::Buy as usize;

/// Number of classes the model scores.
pub const NUM_CLASSES: usize = 3;

impl Action {
    /// The class at a model output index, or `None` when the index is out of range.
    #[must_use]
    pub const fn from_class(index: usize) -> Option<Self> {
        match index {
            SELL => Some(Action::Sell),
            HOLD => Some(Action::Hold),
            BUY => Some(Action::Buy),
            _ => None,
        }
    }

    /// This class's index in the model's output order, the compact label form.
    #[must_use]
    pub const fn class(self) -> u8 {
        self as u8
    }

    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Action::Sell => "Sell",
            Action::Hold => "Hold",
            Action::Buy => "Buy",
        }
    }
}
