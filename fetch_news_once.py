#!/usr/bin/env python3
import os
import sys
import pandas as pd

# unbuffered prints
os.environ.setdefault("PYTHONUNBUFFERED","1")
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from data_loader_hf import load_news_hf
from sentiment import analyze_news_sentiment
from config import DATA_DIR

CACHE_CSV = os.path.join(DATA_DIR, "news.csv")
CACHE_PARQUET = os.path.join(DATA_DIR, "news.parquet")

def ensure_text(df: pd.DataFrame) -> pd.DataFrame:
    if "text" not in df.columns:
        title = df.get("title", "").fillna("")
        desc  = df.get("description", "").fillna("")
        df["text"] = (title.astype(str) + ". " + desc.astype(str)).str.strip()
    return df

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"[News ingest] Target cache: {CACHE_CSV}", flush=True)

    # 1) Pull a manageable chunk; adjust sample as you like
    print("[News ingest] Loading from HF…", flush=True)
    news = load_news_hf(
        use_config_dates=True,   # respects START_DATE/END_DATE from config if set
        sample=50_000,           # smaller = faster; can raise later
        batch_size=20_000,
        verbose=True,
        progress_every=1
    )

    if news.empty:
        print("[News ingest] No rows loaded — nothing to cache.", flush=True)
        return

    # 2) Build text, run sentiment
    news = ensure_text(news)
    print("[News ingest] Running sentiment (FinBERT)…", flush=True)
    news = analyze_news_sentiment(news)

    # 3) Harmonize columns for training pipeline
    if "publishedAt" not in news.columns:
        news["publishedAt"] = pd.to_datetime(news.get("published"), errors="coerce", utc=True)
    if "sentiment_score" not in news.columns:
        news["sentiment_score"] = news.get("sentiment_compound", 0).fillna(0)

    keep_cols = ["publishedAt", "text", "ticker", "sentiment_score"]
    for c in keep_cols:
        if c not in news.columns:
            news[c] = pd.NA
    news = news[keep_cols].dropna(subset=["publishedAt"]).reset_index(drop=True)

    # 4) Save cache (csv + parquet if available)
    news.to_csv(CACHE_CSV, index=False)
    try:
        news.to_parquet(CACHE_PARQUET, index=False)
    except Exception:
        pass

    print(f"[News ingest] Cached {len(news):,} rows → {CACHE_CSV}", flush=True)
    if os.path.exists(CACHE_PARQUET):
        print(f"[News ingest] Also saved Parquet → {CACHE_PARQUET}", flush=True)

if __name__ == "__main__":
    main()
