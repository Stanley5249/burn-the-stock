use crate::error::Result;
use crate::types::{Market, OhlcvRow};
use crate::urls;
use chrono::{Datelike, NaiveDate};

/// Fetch OHLCV rows for `code` over `[start, end]` from the appropriate exchange.
///
/// # Errors
///
/// Returns an error on network or deserialization failure.
pub async fn fetch_stock_data(
    http: &reqwest::Client,
    code: &str,
    start: NaiveDate,
    end: NaiveDate,
    market: Market,
) -> Result<Vec<OhlcvRow>> {
    let mut rows = Vec::new();

    for month in month_starts(start, end) {
        let batch = match market {
            Market::Twse => fetch_twse(http, code, month).await?,
            Market::Tpex => fetch_tpex(http, code, month).await?,
            Market::Esb => fetch_esb(http, code, month).await?,
        };
        rows.extend(batch);
    }

    rows.retain(|r| {
        NaiveDate::parse_from_str(&r.date, "%Y-%m-%d").is_ok_and(|d| d >= start && d <= end)
    });

    rows.sort_by(|a, b| a.date.cmp(&b.date));

    Ok(rows)
}

fn month_starts(start: NaiveDate, end: NaiveDate) -> Vec<NaiveDate> {
    let mut months = Vec::new();
    let mut current = NaiveDate::from_ymd_opt(start.year(), start.month(), 1).unwrap();
    let end_month = NaiveDate::from_ymd_opt(end.year(), end.month(), 1).unwrap();
    while current <= end_month {
        months.push(current);
        current = if current.month() == 12 {
            NaiveDate::from_ymd_opt(current.year() + 1, 1, 1).unwrap()
        } else {
            NaiveDate::from_ymd_opt(current.year(), current.month() + 1, 1).unwrap()
        };
    }
    months
}

async fn fetch_twse(http: &reqwest::Client, code: &str, month: NaiveDate) -> Result<Vec<OhlcvRow>> {
    let response_json: serde_json::Value = http
        .get(urls::TWSE_STOCK_DAY)
        .query(&[
            ("response", "json"),
            ("date", &month.format("%Y%m%d").to_string()),
            ("stockNo", code),
        ])
        .send()
        .await?
        .json()
        .await?;

    if response_json["stat"] != "OK" {
        return Ok(vec![]);
    }

    let Some(data) = response_json["data"].as_array() else {
        return Ok(vec![]);
    };

    let mut rows = Vec::with_capacity(data.len());
    for row in data {
        let columns = str_cols(row);
        if columns.len() < 9 {
            continue;
        }
        rows.push(OhlcvRow {
            date: roc_to_iso(columns[0]),
            stock_code_id: code.to_owned(),
            capacity: clean_f64(columns[1]),
            turnover: clean_f64(columns[2]),
            open: Some(clean_f64(columns[3])),
            high: Some(clean_f64(columns[4])),
            low: Some(clean_f64(columns[5])),
            close: Some(clean_f64(columns[6])),
            change: Some(clean_f64(columns[7])),
            transaction_volume: clean_f64(columns[8]),
        });
    }
    Ok(rows)
}

async fn fetch_tpex(http: &reqwest::Client, code: &str, month: NaiveDate) -> Result<Vec<OhlcvRow>> {
    let response_json: serde_json::Value = http
        .get(urls::TPEX_TRADING_STOCK)
        .query(&[
            ("code", code),
            ("date", &month.format("%Y/%m/%d").to_string()),
            ("id", ""),
            ("response", "json"),
        ])
        .send()
        .await?
        .json()
        .await?;

    let stat = response_json["stat"].as_str().unwrap_or("");
    if !stat.eq_ignore_ascii_case("ok") {
        return Ok(vec![]);
    }

    let Some(data) = response_json["tables"][0]["data"].as_array() else {
        return Ok(vec![]);
    };

    let mut rows = Vec::with_capacity(data.len());
    for row in data {
        let columns = str_cols(row);
        // columns: date, capacity(lots), turnover(thousands), open, high, low, close, change, txn
        if columns.len() < 9 {
            continue;
        }
        rows.push(OhlcvRow {
            date: roc_to_iso(columns[0]),
            stock_code_id: code.to_owned(),
            capacity: clean_f64(columns[1]) * 1000.0,
            turnover: clean_f64(columns[2]) * 1000.0,
            open: Some(clean_f64(columns[3])),
            high: Some(clean_f64(columns[4])),
            low: Some(clean_f64(columns[5])),
            close: Some(clean_f64(columns[6])),
            change: Some(clean_f64(columns[7])),
            transaction_volume: clean_f64(columns[8]),
        });
    }
    Ok(rows)
}

async fn fetch_esb(http: &reqwest::Client, code: &str, month: NaiveDate) -> Result<Vec<OhlcvRow>> {
    let response_json: serde_json::Value = http
        .get(urls::TPEX_EMERGING_HISTORICAL)
        .query(&[
            ("type", "Monthly"),
            ("date", &month.format("%Y/%m/%d").to_string()),
            ("code", code),
            ("id", ""),
            ("response", "json"),
        ])
        .send()
        .await?
        .json()
        .await?;

    let stat = response_json["stat"].as_str().unwrap_or("");
    if !stat.eq_ignore_ascii_case("ok") {
        return Ok(vec![]);
    }

    let Some(data) = response_json["tables"][0]["data"].as_array() else {
        return Ok(vec![]);
    };

    let mut rows = Vec::with_capacity(data.len());
    for row in data {
        let columns = str_cols(row);
        // 13 columns: date, capacity1, turnover1, high1, low1, avg1, transaction1, capacity2, turnover2, high2, low2, avg2, transaction2
        if columns.len() < 13 {
            continue;
        }
        let capacity_1 = clean_f64(columns[1]);
        let turnover_1 = clean_f64(columns[2]);
        let high_1 = clean_f64(columns[3]);
        let low_1 = clean_f64(columns[4]);
        let transaction_1 = clean_f64(columns[6]);
        let capacity_2 = clean_f64(columns[7]);
        let turnover_2 = clean_f64(columns[8]);
        let high_2 = clean_f64(columns[9]);
        let low_2 = clean_f64(columns[10]);
        let transaction_2 = clean_f64(columns[12]);

        let capacity = capacity_1 + capacity_2;
        let turnover = turnover_1 + turnover_2;
        // weighted average price as close proxy; no open available
        let close = (capacity > 0.0).then(|| turnover / capacity);
        let high = nonzero_max(high_1, high_2);
        let low = nonzero_min(low_1, low_2);

        rows.push(OhlcvRow {
            date: roc_to_iso(columns[0]),
            stock_code_id: code.to_owned(),
            capacity,
            turnover,
            open: None,
            high,
            low,
            close,
            change: None,
            transaction_volume: transaction_1 + transaction_2,
        });
    }
    Ok(rows)
}

fn nonzero_max(a: f64, b: f64) -> Option<f64> {
    match (a > 0.0, b > 0.0) {
        (true, true) => Some(a.max(b)),
        (true, false) => Some(a),
        (false, true) => Some(b),
        (false, false) => None,
    }
}

fn nonzero_min(a: f64, b: f64) -> Option<f64> {
    match (a > 0.0, b > 0.0) {
        (true, true) => Some(a.min(b)),
        (true, false) => Some(a),
        (false, true) => Some(b),
        (false, false) => None,
    }
}

fn str_cols(row: &serde_json::Value) -> Vec<&str> {
    row.as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default()
}

fn roc_to_iso(s: &str) -> String {
    let mut parts = s.split('/');
    let (Some(y), Some(m), Some(d)) = (parts.next(), parts.next(), parts.next()) else {
        return s.to_owned();
    };
    let year: u32 = y.trim().parse().unwrap_or(0) + 1911;
    format!("{year}-{m}-{d}")
}

fn clean_f64(s: &str) -> f64 {
    s.replace([',', 'X'], "").trim().parse().unwrap_or(0.0)
}
