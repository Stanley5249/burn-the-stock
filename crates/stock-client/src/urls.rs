pub mod sim_stock {
    use url::Url;

    /// # Panics
    /// Never panics if `BASE` is a valid URL.
    #[must_use]
    pub fn base() -> Url {
        Url::parse(BASE).expect("invalid sim stock base url")
    }

    pub const BASE: &str = "https://ciot.imis.ncku.edu.tw/stock/";

    pub const USER_STOCKS: &str = "trading_api/get_user_stocks";
    pub const BUY: &str = "trading_api/buy";
    pub const SELL: &str = "trading_api/sell";

    pub const LOGIN: &str = "login/";
    pub const PROFILE: &str = "profile/";
}

pub mod twse {
    pub const HOLIDAY_SCHEDULE: &str =
        "https://openapi.twse.com.tw/v1/holidaySchedule/holidaySchedule";
}

pub mod yahoo {
    /// Chart API base; the symbol is appended.
    pub const CHART_BASE: &str = "https://query1.finance.yahoo.com/v8/finance/chart/";

    /// Priming GET seeds the consent cookie some regions require.
    pub const CONSENT: &str = "https://fc.yahoo.com/consent";

    /// Desktop browser UA avoids trivial bot blocking.
    pub const USER_AGENT: &str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36";
}

pub mod fugle {
    use url::Url;

    /// # Panics
    /// Never panics if `BASE` is a valid URL.
    #[must_use]
    pub fn base() -> Url {
        Url::parse(BASE).expect("invalid fugle base url")
    }

    pub const BASE: &str = "https://api.fugle.tw/marketdata/v1.0/stock/";

    pub const INTRADAY_TICKER: &str = "intraday/ticker";
    pub const INTRADAY_QUOTE: &str = "intraday/quote";
}
