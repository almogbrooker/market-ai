# data_loader_hf.py
import pandas as pd
import re
from typing import Optional, List, Any, Dict, Union

# Try to import HF datasets
try:
    from datasets import load_dataset
    _HAS_HF = True
except Exception:
    _HAS_HF = False

# Expanded set of common names for published time columns
_DATE_CANDIDATES = [
    "published", "published_at", "publishedAt",
    "time", "timestamp", "date", "datetime",
    "time_published", "publish_time", "created_utc", "created_at"
]

# Compact format often used in finance news: 20240118T152300
_COMPACT_TS_RE = re.compile(r"^\d{8}T\d{6}$")  # YYYYMMDDTHHMMSS


def _parse_ts_utc(x: Union[str, pd.Timestamp, int, float, None]) -> Optional[pd.Timestamp]:
    if x is None:
        return None
    try:
        # First try flexible pandas parsing
        ts = pd.to_datetime(x, format="%Y-%m-%d %H:%M:%S",
                            errors="coerce", utc=True)
        if pd.notna(ts):
            return ts if pd.notna(ts) else None
        # Then try compact format string like 20240118T152300
        if isinstance(x, str) and _COMPACT_TS_RE.match(x.strip()):
            ts2 = pd.to_datetime(x.strip(), format="%Y%m%dT%H%M%S", utc=True, errors="coerce")
            if pd.notna(ts2):
                return ts2
        # Then try epoch seconds / ms if numeric
        if isinstance(x, (int, float)):
            for unit in ("s", "ms"):
                ts3 = pd.to_datetime(x, unit=unit, errors="coerce", utc=True)
                if pd.notna(ts3):
                    return ts3
    except Exception:
        pass
    return None


def _ensure_text_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Map/ensure title + description columns so we can later form a 'text' column."""
    if "title" not in df.columns:
        for c in ["headline", "head", "Title"]:
            if c in df.columns:
                df = df.rename(columns={c: "title"})
                break
    if "title" not in df.columns:
        df["title"] = ""

    if "description" not in df.columns:
        for c in ["summary", "body", "text", "content", "lead", "article"]:
            if c in df.columns:
                df = df.rename(columns={c: "description"})
                break
    if "description" not in df.columns:
        df["description"] = ""
    return df


def _parse_compact_time_published(s: pd.Series) -> pd.Series:
    """Parse strings like 20240118T152300 into UTC timestamps."""
    def _p(x):
        if not isinstance(x, str):
            return pd.NaT
        x = x.strip()
        if not _COMPACT_TS_RE.match(x):
            return pd.NaT
        try:
            return pd.to_datetime(x, format="%Y%m%dT%H%M%S", utc=True)
        except Exception:
            return pd.NaT
    return s.apply(_p)


def _best_datetime_column(df: pd.DataFrame) -> Optional[str]:
    """
    Try all columns; pick the one with the most valid timestamps
    using multiple parsing strategies.
    """
    best_col, best_valid = None, -1
    for c in df.columns:
        s = df[c]
        valid = 0

        # 1) pandas flexible
        try:
            ts = pd.to_datetime(s, errors="coerce", utc=True)
            valid = max(valid, ts.notna().sum())
        except Exception:
            pass

        # 2) compact YYYYMMDDTHHMMSS
        try:
            ts2 = _parse_compact_time_published(s.astype(str))
            valid = max(valid, ts2.notna().sum())
        except Exception:
            pass

        # 3) epoch secs/ms
        if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
            try:
                ts3 = pd.to_datetime(s, unit="s", errors="coerce", utc=True)
                valid = max(valid, ts3.notna().sum())
                ts4 = pd.to_datetime(s, unit="ms", errors="coerce", utc=True)
                valid = max(valid, ts4.notna().sum())
            except Exception:
                pass

        if valid > best_valid:
            best_col, best_valid = c, valid

    return best_col if best_valid > 0 else None


def _ensure_ticker_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize ticker column:
    - prefer existing 'ticker'
    - else try 'symbol', 'symbols', 'tickers', 'company'
    - explode if list-like
    """
    if "ticker" in df.columns:
        col = "ticker"
    else:
        col = None
        for c in ["symbol", "symbols", "tickers", "company", "Ticker"]:
            if c in df.columns:
                col = c
                break
        if col is None:
            df["ticker"] = None
            return df

    s = df[col]
    # explode list-like
    if s.apply(lambda x: isinstance(x, (list, tuple, set))).any():
        df = df.explode(col)

    if col != "ticker":
        df = df.rename(columns={col: "ticker"})

    # normalize to string upper (but keep None)
    df["ticker"] = df["ticker"].apply(lambda x: None if pd.isna(x) else str(x).strip().upper())
    return df


def _normalize_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Unify to 'published' (UTC), parse deterministically, and drop out-of-range rows."""
    # 1) Rename any known datetime column to 'published'
    if "published" not in df.columns:
        for c in _DATE_CANDIDATES:
            if c in df.columns:
                df = df.rename(columns={c: "published"})
                break

    # Auto-detect a best guess if still missing
    if "published" not in df.columns:
        guess = _best_datetime_column(df)
        if guess:
            df = df.rename(columns={guess: "published"})
        else:
            df["published"] = pd.NaT

    s = df["published"]

    # 2) Deterministic, fast parses first (no dateutil warnings)
    # Try a small set of explicit formats in order, filling only the NaTs each time.
    # Add/remove formats to match your dataset.
    explicit_formats = [
        "%Y-%m-%d %H:%M:%S%z",  # e.g., 2020-06-10 06:33:52+0000
        "%Y-%m-%d %H:%M:%S",    # e.g., 2020-06-10 06:33:52
        "%Y-%m-%dT%H:%M:%S%z",  # e.g., 2020-06-10T06:33:52+00:00
        "%Y-%m-%dT%H:%M:%S",    # e.g., 2020-06-10T06:33:52
        "%Y-%m-%d",             # e.g., 2020-06-10
    ]

    ts = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

    # Work on a string view for the explicit string formats
    s_str = s.astype(str)

    for fmt in explicit_formats:
        mask = ts.isna()
        if not mask.any():
            break
        try:
            parsed = pd.to_datetime(
                s_str.where(mask),
                format=fmt,
                errors="coerce",
                utc=True
            )
            ts = ts.where(~mask, parsed)
        except Exception:
            # If a format bombs (rare), just continue to next
            pass

    # 3) Compact finance format (e.g., 20240118T152300)
    mask = ts.isna()
    if mask.any():
        try:
            compact = _parse_compact_time_published(s_str.where(mask))
            ts = ts.where(~mask, compact)
        except Exception:
            pass

    # 4) Epoch seconds / milliseconds if remaining values are numeric-like
    mask = ts.isna()
    if mask.any():
        s_num = s.where(mask)
        # seconds
        try:
            parsed_s = pd.to_datetime(s_num, unit="s", errors="coerce", utc=True)
            ts = ts.where(~mask, parsed_s)
        except Exception:
            pass
        # milliseconds (only where still NaT)
        mask = ts.isna()
        if mask.any():
            try:
                parsed_ms = pd.to_datetime(s_num, unit="ms", errors="coerce", utc=True)
                ts = ts.where(~mask, parsed_ms)
            except Exception:
                pass

    # 5) Final ultra-fallback: flexible parser (dateutil) just for anything still NaT
    # (This is what caused warnings before; we now call it only on the leftovers.)
    mask = ts.isna()
    if mask.any():
        try:
            parsed_flex = pd.to_datetime(s_str.where(mask), errors="coerce", utc=True)
            ts = ts.where(~mask, parsed_flex)
        except Exception:
            pass

    # Assign and clean
    df["published"] = ts
    df = df.dropna(subset=["published"])

    # 6) Very wide, sane bounds
    start = pd.Timestamp("1980-01-01", tz="UTC")
    end = pd.Timestamp.utcnow() + pd.Timedelta(days=1)
    df = df[(df["published"] >= start) & (df["published"] <= end)]

    # 7) Ensure text & ticker normalization for downstream pipeline
    df = _ensure_text_cols(df)
    df = _ensure_ticker_column(df)
    return df


def load_news_hf(
    dataset: str = "Zihan1004/FNSPID",
    split: str = "train",
    streaming: bool = True,
    batch_size: int = 5000,
    sample: Optional[int] = None,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    use_config_dates: bool = False,
    verbose: bool = False,                 # NEW: progress prints
    progress_every: int = 5                # NEW: print every N flushes
) -> pd.DataFrame:
    """
    Load a news dataset from HF with robust datetime/ticker handling and optional date-range filtering.
    Returns columns: published(UTC), title, description, ticker (string/None).
    """
    if not _HAS_HF:
        raise RuntimeError("Please `pip install datasets` to use Hugging Face loading.")

    # Pull dates from config if requested
    if use_config_dates and (start_date is None or end_date is None):
        try:
            import config
            if start_date is None and hasattr(config, "START_DATE"):
                start_date = config.START_DATE
            if end_date is None and hasattr(config, "END_DATE"):
                end_date = config.END_DATE
        except Exception:
            pass

    start_ts = _parse_ts_utc(start_date) if start_date is not None else None
    end_ts = _parse_ts_utc(end_date) if end_date is not None else None

    if verbose:
        rng = f"[{start_ts.date() if start_ts is not None else '1980-01-01'} → {end_ts.date() if end_ts is not None else 'now'}]"
        print(f"[HF] Loading dataset='{dataset}', split='{split}', streaming={streaming}, "
              f"batch_size={batch_size}, sample={sample} {rng}")

    ds = load_dataset(dataset, split=split, streaming=streaming)

    rows: List[pd.DataFrame] = []
    kept = 0
    flush_count = 0

    def _post_filter(df: pd.DataFrame) -> pd.DataFrame:
        nonlocal kept
        if start_ts is not None:
            df = df[df["published"] >= start_ts]
        if end_ts is not None:
            df = df[df["published"] <= end_ts]
        if df is not None and not df.empty:
            kept += len(df)
        return df

    if streaming:
        buf: Dict[str, List[Any]] = {}

        def flush():
            nonlocal buf, rows, flush_count
            if not buf:
                return
            flush_count += 1
            df = pd.DataFrame(buf)
            buf = {}

            if df.empty:
                if verbose and (flush_count % progress_every == 0):
                    print(f"[HF] flush={flush_count:4d} | kept_total={kept:10d} | last=n/a", flush=True)
                return

            df = _normalize_datetime(df)
            df = _post_filter(df)
            if df is not None and not df.empty:
                rows.append(df)

            if verbose and (flush_count % progress_every == 0):
                last_ts = df["published"].max() if df is not None and not df.empty else None
                last_str = str(last_ts) if pd.notna(last_ts) else "n/a"
                print(f"[HF] flush={flush_count:4d} | kept_total={kept:10d} | last={last_str}", flush=True)

        for ex in ds:
            for k, v in ex.items():
                # HF streaming examples can have nested lists; keep raw, pandas will handle
                buf.setdefault(k, []).append(v)
            # approximate threshold by any column length
            if buf and len(next(iter(buf.values()))) >= batch_size:
                flush()
                if sample is not None and kept >= sample:
                    break
        # final flush
        flush()

        news = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
            columns=["published", "title", "description", "ticker"]
        )
        if sample is not None and len(news) > sample:
            news = news.head(sample)
    else:
        df = ds.to_pandas()
        df = _normalize_datetime(df)
        df = _post_filter(df)
        news = df if df is not None else pd.DataFrame(
            columns=["published", "title", "description", "ticker"]
        )
        if sample is not None and len(news) > sample:
            news = news.head(sample)

    return news.reset_index(drop=True)


# Backward-compat alias so old code keeps working:
def load_news_from_huggingface(*args, **kwargs) -> pd.DataFrame:
    """Alias to load_news_hf(...) for compatibility."""
    return load_news_hf(*args, **kwargs)
