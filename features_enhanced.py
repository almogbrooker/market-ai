
from typing import Optional
import pandas as pd
import numpy as np

def _tz_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Try common date column forms to set index if needed
    for c in ["Date", "date", "datetime", "timestamp"]:
        if c in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
                df = df.set_index(c)
                break
            except Exception:
                pass
    # Ensure DateTimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df.sort_index()

def add_technical_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    df = _tz_utc_index(prices)
    # Expect columns case-insensitively
    cols = {c.lower(): c for c in df.columns}
    close = cols.get("close", None)
    high  = cols.get("high", None)
    low   = cols.get("low", None)
    vol   = cols.get("volume", None)

    if close is None:
        raise ValueError("Expected a Close column in price data")

    # SMAs
    df["sma_10"] = df[close].rolling(10, min_periods=1).mean()
    df["sma_50"] = df[close].rolling(50, min_periods=1).mean()

    # RSI 14
    delta = df[close].diff()
    up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    down = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
    rs = up / (down + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # ATR 14
    if high and low:
        tr = pd.concat([
            (df[high] - df[low]).abs(),
            (df[high] - df[close].shift(1)).abs(),
            (df[low]  - df[close].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["atr_14"] = tr.ewm(alpha=1/14, adjust=False).mean()
    else:
        df["atr_14"] = np.nan

    return df

def add_market_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """Add a few generic market-wide indicators (placeholders)."""
    df = _tz_utc_index(prices)
    cols = {c.lower(): c for c in df.columns}
    close = cols.get("close", None)
    if close is None:
        return df
    # Daily returns and rolling volatility
    df["ret_1d"] = df[close].pct_change()
    df["vol_10d"] = df["ret_1d"].rolling(10).std()
    df["mom_20d"] = df[close] / df[close].shift(20) - 1
    return df

def merge_news_to_prices(prices: pd.DataFrame, news: pd.DataFrame,
                         tolerance: str = "3D", agg_by: Optional[str] = None) -> pd.DataFrame:
    p = _tz_utc_index(prices)
    n = news.copy()

    # normalize published column
    if "published" not in n.columns:
        for c in ["published", "published_at", "time", "timestamp", "date", "publishedAt"]:
            if c in n.columns:
                n = n.rename(columns={c: "published"})
                break
    n["published"] = pd.to_datetime(n["published"], errors="coerce", utc=True)
    n = n.dropna(subset=["published"]).sort_values("published")

    # Ensure sentiment numeric columns exist (fill if missing)
    for col in ["sentiment_neg", "sentiment_neu", "sentiment_pos", "sentiment_compound"]:
        if col not in n.columns:
            n[col] = np.nan

    if agg_by:
        n["_bin"] = n["published"].dt.to_period(agg_by).dt.start_time
        n = n.groupby("_bin")[["sentiment_neg","sentiment_neu","sentiment_pos","sentiment_compound"]].mean().reset_index()
        n = n.rename(columns={"_bin": "published"})

    merged = pd.merge_asof(
        p.reset_index().rename(columns={"index": "ts"}).sort_values("ts"),
        n.sort_values("published"),
        left_on="ts",
        right_on="published",
        direction="backward",
        tolerance=pd.Timedelta(tolerance),
    ).set_index("ts")

    return merged
