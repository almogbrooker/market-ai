#!/usr/bin/env python3
"""
DATA EXPLORER
=============
Fresh start - explore and understand the training data
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def explore_training_data():
    """Explore the training dataset to understand structure"""
    print("🔍 EXPLORING TRAINING DATA")
    print("=" * 50)
    
    # Load training data
    df = pd.read_parquet("../artifacts/ds_train.parquet")
    
    print(f"📊 Dataset Shape: {df.shape}")
    print(f"📅 Columns: {len(df.columns)}")
    
    # Basic info
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"📆 Date Range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"⏰ Time Span: {(df['Date'].max() - df['Date'].min()).days} days")
    
    if 'Ticker' in df.columns:
        tickers = df['Ticker'].nunique()
        print(f"🏢 Unique Tickers: {tickers}")
        print(f"📊 Avg rows per ticker: {len(df)/tickers:.1f}")
        
        # Show sample tickers
        sample_tickers = df['Ticker'].unique()[:10]
        print(f"📋 Sample tickers: {list(sample_tickers)}")
    
    # Target analysis
    target_cols = [col for col in df.columns if 'target' in col.lower()]
    print(f"\n🎯 TARGET COLUMNS: {target_cols}")
    
    for target_col in target_cols:
        target_data = df[target_col].dropna()
        print(f"   {target_col}:")
        print(f"     Mean: {target_data.mean():.6f}")
        print(f"     Std: {target_data.std():.4f}")
        print(f"     Range: [{target_data.min():.4f}, {target_data.max():.4f}]")
        print(f"     Count: {len(target_data):,}")
    
    # Feature categories
    print(f"\n📊 FEATURE ANALYSIS:")
    
    # Identify feature types
    feature_categories = {
        'Price/Return': [col for col in df.columns if any(x in col.lower() for x in ['price', 'return', 'close', 'open', 'high', 'low'])],
        'Volume': [col for col in df.columns if 'volume' in col.lower()],
        'Technical': [col for col in df.columns if any(x in col.lower() for x in ['rsi', 'macd', 'bb', 'sma', 'ema'])],
        'Fundamental': [col for col in df.columns if any(x in col.lower() for x in ['pe', 'pb', 'roe', 'roa', 'debt', 'eps'])],
        'Macro': [col for col in df.columns if any(x in col.lower() for x in ['vix', 'treasury', 'yield', 'gdp', 'cpi'])],
        'ML/Alpha': [col for col in df.columns if any(x in col.lower() for x in ['ml_', 'alpha', 'signal', 'score'])],
        'Rank/ZScore': [col for col in df.columns if any(x in col.lower() for x in ['rank', 'zscore', 'percentile'])],
    }
    
    for category, features in feature_categories.items():
        if features:
            print(f"   {category}: {len(features)} features")
            # Show first few features
            print(f"     Examples: {features[:5]}")
    
    # Missing data analysis
    print(f"\n❓ MISSING DATA ANALYSIS:")
    missing_pct = (df.isnull().sum() / len(df) * 100).round(1)
    
    # Group by missing data levels
    no_missing = missing_pct[missing_pct == 0]
    low_missing = missing_pct[(missing_pct > 0) & (missing_pct <= 10)]
    medium_missing = missing_pct[(missing_pct > 10) & (missing_pct <= 50)]
    high_missing = missing_pct[missing_pct > 50]
    
    print(f"   No missing: {len(no_missing)} features")
    print(f"   Low missing (0-10%): {len(low_missing)} features")
    print(f"   Medium missing (10-50%): {len(medium_missing)} features")
    print(f"   High missing (>50%): {len(high_missing)} features")
    
    if len(high_missing) > 0:
        print(f"   High missing features: {list(high_missing.head(10).index)}")
    
    # Data types
    print(f"\n📋 DATA TYPES:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   {dtype}: {count} columns")
    
    # Sample data
    print(f"\n📊 SAMPLE DATA (first 3 rows):")
    # Show key columns only
    key_cols = ['Date', 'Ticker']
    if target_cols:
        key_cols.extend(target_cols[:2])  # First 2 target columns
    
    # Add some feature examples
    available_cols = [col for col in key_cols if col in df.columns]
    sample_features = [col for col in df.columns if col not in available_cols][:5]
    available_cols.extend(sample_features)
    
    print(df[available_cols].head(3).to_string(index=False))
    
    return df

def main():
    """Main exploration function"""
    df = explore_training_data()
    
    print(f"\n✅ DATA EXPLORATION COMPLETE")
    print(f"📊 Ready to build institutional-grade system")
    
    return df

if __name__ == "__main__":
    df = main()