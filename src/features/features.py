import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict
from data_validation import DataValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_technical_indicators(df):
    df = df.copy()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["Return_1"] = df["Close"].pct_change()
    df["Volatility_10"] = df["Return_1"].rolling(10).std()
    df["Momentum_10"] = df["Close"] / df["Close"].shift(10) - 1
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(14).mean()
    ma_down = down.rolling(14).mean()
    rs = ma_up / (ma_down + 1e-8)
    df["RSI_14"] = 100 - (100 / (1 + rs))
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df = df.fillna(0)
    return df

def merge_news_to_prices(stock_df, news_df):
    nd = news_df.copy()
    nd["date"] = pd.to_datetime(nd["publishedAt"]).dt.date
    agg = nd.groupby(["ticker","date"])["sentiment_score"].mean().reset_index()
    agg.rename(columns={"date":"Date"}, inplace=True)
    agg["Date"] = pd.to_datetime(agg["Date"])
    merged = stock_df.merge(agg, left_on=["Ticker","Date"], right_on=["ticker","Date"], how="left")
    merged["sentiment_score"] = merged["sentiment_score"].fillna(0.0)
    merged.drop(columns=["ticker"], inplace=True, errors="ignore")
    return merged
