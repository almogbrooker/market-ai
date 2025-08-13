import yfinance as yf
import pandas as pd
import os
from config import START_DATE, END_DATE, TICKERS, DATA_DIR
from tqdm import tqdm
import json

def fetch_and_save_universe(tickers=None, start=START_DATE, end=END_DATE, out_dir=DATA_DIR):
    tickers = tickers or TICKERS
    os.makedirs(out_dir, exist_ok=True)
    
    # --- Fetch Price Data ---
    print("Fetching price data...")
    panel = yf.download(tickers, start=start, end=end, threads=True, progress=True)
    if panel.empty:
        raise RuntimeError("yfinance returned empty price data")
    
    all_data = {}
    for ticker in tickers:
        try:
            df = panel.loc[:, (slice(None), ticker)].copy()
            df.columns = df.columns.droplevel(1)
            df.dropna(inplace=True)
            df.index.name = "Date"
            all_data[ticker] = df
            df.to_csv(os.path.join(out_dir, f"{ticker}.csv"))
        except Exception as e:
            print(f"Could not process price data for {ticker}: {e}")
            continue
    
    # --- Fetch Fundamental Data ---
    print("\nFetching fundamental data...")
    all_fundamentals = {}
    for ticker in tqdm(tickers, desc="Fetching Fundamentals"):
        try:
            stock_info = yf.Ticker(ticker).info
            # Select a subset of useful fundamental data
            fundamentals = {
                'marketCap': stock_info.get('marketCap'),
                'trailingPE': stock_info.get('trailingPE'),
                'forwardPE': stock_info.get('forwardPE'),
                'priceToBook': stock_info.get('priceToBook'),
                'enterpriseToRevenue': stock_info.get('enterpriseToRevenue'),
                'enterpriseToEbitda': stock_info.get('enterpriseToEbitda'),
                'profitMargins': stock_info.get('profitMargins'),
                'heldPercentInsiders': stock_info.get('heldPercentInsiders'),
                'heldPercentInstitutions': stock_info.get('heldPercentInstitutions'),
                'shortRatio': stock_info.get('shortRatio'),
                'beta': stock_info.get('beta'),
                'averageVolume': stock_info.get('averageVolume')
            }
            all_fundamentals[ticker] = fundamentals
        except Exception as e:
            print(f"Could not fetch fundamental data for {ticker}: {e}")
            continue
            
    # Save fundamentals to a single JSON file
    fundamentals_path = os.path.join(out_dir, "fundamentals.json")
    with open(fundamentals_path, 'w') as f:
        json.dump(all_fundamentals, f, indent=4)
        
    print(f"\nFundamental data saved to {fundamentals_path}")
    
    return all_data, all_fundamentals

if __name__ == "__main__":
    fetch_and_save_universe()