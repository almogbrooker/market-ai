import requests
import pandas as pd
import time
import os
from config import NEWS_API_KEY, DATA_DIR
from tqdm import tqdm

def fetch_news_for_ticker(ticker, from_date, to_date, api_key=NEWS_API_KEY, page_size=100, max_pages=5):
    articles = []
    url = "https://newsapi.org/v2/everything"
    for page in range(1, max_pages+1):
        params = {
            "q": ticker,
            "from": from_date,
            "to": to_date,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "page": page,
            "apiKey": api_key
        }
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            break
        data = r.json()
        articles.extend(data.get("articles", []))
        total = data.get("totalResults", 0)
        if total <= page * page_size:
            break
        time.sleep(0.2)
    df = pd.DataFrame(articles)
    if not df.empty:
        df = df[["publishedAt", "title", "description", "source"]]
        df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
    return df

def collect_all_news(tickers=None, out_path=None, from_date=None, to_date=None):
    tickers = tickers or []
    out_path = out_path or os.path.join(DATA_DIR, "news_raw.parquet")
    frames = []
    for t in tqdm(tickers):
        try:
            df = fetch_news_for_ticker(t, from_date, to_date)
            if not df.empty:
                df["ticker"] = t
                frames.append(df)
        except Exception:
            continue
    if frames:
        all_news = pd.concat(frames, ignore_index=True)
        all_news.to_parquet(out_path, index=False)
        return all_news
    return pd.DataFrame()

if __name__ == "__main__":
    collect_all_news(TICKERS, out_path=f"{DATA_DIR}/news_raw.parquet", from_date=START_DATE, to_date=END_DATE)
