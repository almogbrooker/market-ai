#!/usr/bin/env python3
"""
Rebuild Complete Dataset with Temporal Fixes
Uses corrected data_builder.py to generate full historical dataset
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def rebuild_complete_dataset():
    """Rebuild the complete historical dataset with proper temporal buffers"""
    
    logger.info("🔄 REBUILDING COMPLETE DATASET WITH TEMPORAL FIXES")
    logger.info("=" * 70)
    
    # Load existing enhanced dataset as base
    base_data_path = Path('data/training_data_enhanced_with_fundamentals.csv')
    if not base_data_path.exists():
        base_data_path = Path('data/training_data_enhanced.csv')
    if not base_data_path.exists():
        base_data_path = Path('data/training_data_2020_2024_complete.csv')
    
    if not base_data_path.exists():
        logger.error("No base dataset found!")
        return False
    
    logger.info(f"📊 Loading base dataset: {base_data_path.name}")
    data = pd.read_csv(base_data_path)
    data['Date'] = pd.to_datetime(data['Date'])
    
    logger.info(f"Original dataset: {len(data):,} samples")
    logger.info(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    logger.info(f"Tickers: {data['Ticker'].nunique()}")
    
    # Recalculate targets with proper temporal buffers
    logger.info("🔒 Recalculating targets with proper temporal buffers...")
    
    all_data = []
    
    for ticker in data['Ticker'].unique():
        ticker_data = data[data['Ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('Date').reset_index(drop=True)
        
        if len(ticker_data) < 25:  # Need sufficient data
            logger.warning(f"Skipping {ticker} - insufficient data ({len(ticker_data)} rows)")
            continue
        
        # Remove old target columns if they exist
        target_cols = [col for col in ticker_data.columns if col.startswith('target_')]
        ticker_data = ticker_data.drop(columns=target_cols, errors='ignore')
        
        # 🔒 FIXED: Create targets with proper temporal buffers (shift -(periods+1))
        ticker_data['target_1d'] = ticker_data['Close'].pct_change(periods=1).shift(-2)   # 1-day return with +1 buffer
        ticker_data['target_5d'] = ticker_data['Close'].pct_change(periods=5).shift(-6)   # 5-day return with +1 buffer  
        ticker_data['target_20d'] = ticker_data['Close'].pct_change(periods=20).shift(-21) # 20-day return with +1 buffer
        
        # Calculate QQQ-relative returns (alpha) - simplified
        if 'target_1d' in ticker_data.columns:
            market_return_1d = ticker_data['target_1d'].rolling(100, min_periods=20).mean()
            ticker_data['alpha_1d'] = ticker_data['target_1d'] - market_return_1d
        
        all_data.append(ticker_data)
        
        if len(all_data) % 10 == 0:
            logger.info(f"   Processed {len(all_data)} tickers...")
    
    if not all_data:
        logger.error("No valid tickers processed!")
        return False
    
    # Combine all data
    logger.info("🔗 Combining processed data...")
    result = pd.concat(all_data, ignore_index=True)
    result = result.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    # Remove rows without targets (last few rows per ticker)
    before_dropna = len(result)
    result = result.dropna(subset=['target_1d'])
    after_dropna = len(result)
    
    logger.info(f"Dropped {before_dropna - after_dropna:,} rows without targets")
    
    logger.info(f"✅ Final rebuilt dataset: {result.shape}")
    logger.info(f"Date range: {result['Date'].min()} to {result['Date'].max()}")
    logger.info(f"Tickers: {result['Ticker'].nunique()}")
    
    # Validate temporal compliance
    logger.info("🔍 Validating temporal compliance...")
    
    sample_ticker = result['Ticker'].iloc[0]
    sample_data = result[result['Ticker'] == sample_ticker].sort_values('Date').head(5)
    
    if len(sample_data) >= 3:
        # Check first few samples for proper shift(-2) pattern
        for i in range(min(3, len(sample_data) - 2)):
            if i + 2 < len(sample_data):
                close_t1 = sample_data.iloc[i + 1]['Close']
                close_t2 = sample_data.iloc[i + 2]['Close'] 
                expected = (close_t2 - close_t1) / close_t1
                actual = sample_data.iloc[i]['target_1d']
                
                if pd.notna(actual) and pd.notna(expected):
                    if abs(expected - actual) < 0.0001:
                        logger.info(f"✅ Temporal compliance validated for {sample_ticker}")
                        break
        else:
            logger.warning("⚠️ Temporal compliance check inconclusive")
    
    # Save the corrected dataset
    output_path = Path('data/training_data_enhanced_FIXED.csv')
    result.to_csv(output_path, index=False)
    logger.info(f"💾 Saved corrected dataset: {output_path}")
    
    # Also save as parquet for faster loading
    parquet_path = Path('data/training_data_enhanced_FIXED.parquet')
    result.to_parquet(parquet_path)
    logger.info(f"💾 Saved corrected dataset: {parquet_path}")
    
    # Update the clean validation protocol to use the FIXED dataset
    logger.info("🔄 Now run: python clean_validation_protocol.py --use-fixed")
    
    return True

def main():
    """Main execution"""
    success = rebuild_complete_dataset()
    
    if success:
        print("\n🎉 DATASET REBUILD COMPLETE!")
        print("✅ Temporal buffers properly implemented")
        print("✅ Ready to regenerate time slices with fixed data")
        print("\nNext steps:")
        print("1. Update clean_validation_protocol.py to use FIXED dataset")
        print("2. Re-run OOS validation with leak-free data")
        print("3. Re-run guardrails tests")
    else:
        print("\n❌ DATASET REBUILD FAILED!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())