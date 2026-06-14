use stock_model::inference::{Action, Prediction};

/// Print the predictions, then the buy candidates sorted by how strong the long
/// signal is. This is an offline diagnostic, so it reports the actions a trader
/// would take without placing any order.
pub fn print(predictions: &[Prediction], min_position: f32) {
    if predictions.is_empty() {
        println!("No tickers had enough history to fill the model's window.");
        return;
    }

    let as_of = predictions.iter().map(|p| p.date).max().expect("non-empty");
    let mut counts = [0usize; 3];
    for prediction in predictions {
        counts[prediction.action as usize] += 1;
    }
    println!(
        "Predictions as of {as_of} ({} tickers): Sell {}  Hold {}  Buy {}",
        predictions.len(),
        counts[Action::Sell as usize],
        counts[Action::Hold as usize],
        counts[Action::Buy as usize],
    );

    let mut candidates: Vec<&Prediction> = predictions
        .iter()
        .filter(|p| p.position > min_position)
        .collect();
    candidates.sort_by(|left, right| right.position.total_cmp(&left.position));

    println!("\nBuy candidates (position > {min_position:.2}):");
    if candidates.is_empty() {
        println!("  none above threshold; staying flat.");
        return;
    }

    println!(
        "  {:<8} {:<6} {:>7} {:>7} {:>7} {:>9}",
        "TICKER", "ACTION", "P(Sell)", "P(Hold)", "P(Buy)", "POSITION"
    );
    for candidate in &candidates {
        let [sell, hold, buy] = candidate.probabilities;
        println!(
            "  {:<8} {:<6} {sell:>7.3} {hold:>7.3} {buy:>7.3} {:>9.3}",
            candidate.ticker,
            candidate.action.as_str(),
            candidate.position,
        );
    }
}
