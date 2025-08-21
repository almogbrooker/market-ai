#!/usr/bin/env python3
"""
NASDAQ STOCK-PICKER: Long & Short Baskets
Mission Brief Implementation - Full NASDAQ Universe (~2.5k names)

Goal: Daily cross-sectional Rank IC ≥ 0.8-1.5% on unseen dates
Beat cash baseline after realistic costs with proper risk controls
"""

import pandas as pd
import numpy as np
import yfinance as yf
import lightgbm as lgb
import sys
import os
from pathlib import Path
import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NASDAQStockPicker:
    """
    NASDAQ Stock-Picker following Mission Brief specifications
    """
    
    def __init__(self, start_date: str = '2018-01-01', end_date: str = None):
        logger.info("🎯 NASDAQ STOCK-PICKER - MISSION BRIEF IMPLEMENTATION")
        
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        
        # Mission Brief Parameters
        self.target_rank_ic_min = 0.008  # 0.8% minimum
        self.target_rank_ic_max = 0.015  # 1.5% excellent
        self.max_training_ic = 0.03      # 3% overfitting threshold
        self.max_features = 10           # Complexity control
        
        # Universe filters (Mission Brief Section 1)
        self.min_price = 3.0             # $3 minimum
        self.min_adv = 2e6              # $2M daily volume
        self.max_adv_usage = 0.05       # 5% ADV capacity limit
        
        # Cost model (Mission Brief Section 1)
        self.fee_bps = 3.5              # 2-5 bps fee (mid-point)
        self.slippage_bps = 8.5         # 7-10 bps slippage (mid-point)
        self.short_borrow_rate = 0.01   # 100 bps annualized
        
        # CV parameters (Mission Brief Section 4)
        self.purge_days = 10            # Calendar-day purge
        self.embargo_days = 3           # Embargo period
        self.cv_splits = 3              # 3-5 purged splits
        
        # Portfolio parameters (Mission Brief Section 5)
        self.long_pct = 0.3             # Top 30% long
        self.short_pct = 0.3            # Bottom 30% short
        self.max_position_size = 0.08   # 8% max single name
        
        # Setup directories
        self.artifacts_dir = Path(__file__).parent.parent / "artifacts" / "nasdaq_picker"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Track gates for final report
        self.gates_passed = {}
        
        logger.info(f"🎯 Mission Brief Parameters:")
        logger.info(f"   Target Rank IC: {self.target_rank_ic_min:.1%} - {self.target_rank_ic_max:.1%}")
        logger.info(f"   Max Training IC: {self.max_training_ic:.1%}")
        logger.info(f"   Max Features: {self.max_features}")
        logger.info(f"   Universe Filters: Price ≥${self.min_price}, ADV ≥${self.min_adv:,.0f}")
        logger.info(f"   Cost Model: {self.fee_bps}bps fee + {self.slippage_bps}bps slippage")
        
    def download_nasdaq_universe(self) -> pd.DataFrame:
        """Download full NASDAQ universe (Mission Brief Section 1)"""
        
        logger.info("📊 Downloading NASDAQ universe...")
        
        # For MVP, start with a representative sample of liquid NASDAQ stocks
        # In production, this would use official NASDAQ API or data vendor
        nasdaq_symbols = [
            # Mega caps
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA',
            # Large caps  
            'NFLX', 'INTC', 'CSCO', 'ADBE', 'PEP', 'CMCSA', 'COST', 'AVGO',
            'QCOM', 'TXN', 'AMGN', 'HON', 'SBUX', 'GILD', 'MDLZ', 'ADP',
            # Mid caps
            'ISRG', 'REGN', 'CSX', 'FISV', 'ATVI', 'MRNA', 'ILMN', 'KLAC',
            'LRCX', 'MCHP', 'PAYX', 'FAST', 'CDNS', 'SNPS', 'CTAS', 'ORLY',
            # Growth/Tech
            'NXPI', 'ASML', 'ABNB', 'TEAM', 'DXCM', 'ROST', 'PCAR', 'CHTR',
            'WDAY', 'VRSK', 'IDXX', 'CTSH', 'VRSN', 'TTWO', 'ANSS', 'CDNS',
            # Biotech/Healthcare
            'BIIB', 'CELG', 'MAR', 'EA', 'EBAY', 'XLNX', 'BKNG', 'TMUS',
            'EXPE', 'VRTX', 'WBA', 'DLTR', 'MELI', 'ZM', 'DOCU', 'OKTA'
        ]
        
        logger.info(f"Starting with {len(nasdaq_symbols)} representative NASDAQ stocks")
        
        # Download data for all symbols
        all_data = []
        failed_symbols = []
        
        for i, symbol in enumerate(nasdaq_symbols):
            try:
                logger.info(f"Downloading {symbol} ({i+1}/{len(nasdaq_symbols)})...")
                
                # Download with progress disabled
                data = yf.download(symbol, start=self.start_date, end=self.end_date, 
                                 progress=False, auto_adjust=True)
                
                if len(data) > 0:
                    # Flatten multi-index if needed
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
                    
                    data = data.reset_index()
                    data['Ticker'] = symbol
                    
                    # Add sector (simplified)
                    data['Sector'] = self._get_sector(symbol)
                    
                    all_data.append(data)
                else:
                    failed_symbols.append(symbol)
                    
            except Exception as e:
                logger.warning(f"Failed to download {symbol}: {e}")
                failed_symbols.append(symbol)
        
        if not all_data:
            raise ValueError("No data downloaded successfully")
        
        # Combine all data
        df = pd.concat(all_data, ignore_index=True)
        
        logger.info(f"✅ Downloaded {len(all_data)} symbols, {len(failed_symbols)} failed")
        logger.info(f"   Total samples: {len(df)}")
        logger.info(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        return df
    
    def _get_sector(self, symbol: str) -> str:
        """Get simplified sector classification"""
        
        tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'INTC', 'CSCO', 
                      'ADBE', 'QCOM', 'TXN', 'AVGO', 'ISRG', 'KLAC', 'LRCX', 'MCHP',
                      'CDNS', 'SNPS', 'NXPI', 'ASML', 'TEAM', 'WDAY', 'VRSK', 'TTWO',
                      'ANSS', 'EA', 'XLNX', 'ZM', 'DOCU', 'OKTA']
        
        consumer_stocks = ['AMZN', 'TSLA', 'NFLX', 'PEP', 'CMCSA', 'COST', 'SBUX', 
                          'MDLZ', 'FAST', 'CTAS', 'ORLY', 'ROST', 'PCAR', 'CHTR',
                          'ABNB', 'MAR', 'EBAY', 'BKNG', 'EXPE', 'MELI', 'DLTR']
        
        healthcare_stocks = ['AMGN', 'GILD', 'REGN', 'MRNA', 'ILMN', 'DXCM', 'IDXX',
                           'BIIB', 'CELG', 'WBA', 'VRTX']
        
        if symbol in tech_stocks:
            return 'Technology'
        elif symbol in consumer_stocks:
            return 'Consumer'
        elif symbol in healthcare_stocks:
            return 'Healthcare'
        else:
            return 'Other'
    
    def apply_universe_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply liquidity/quality filters (Mission Brief Section 1)"""
        
        logger.info("🔍 Applying universe filters...")
        
        df = df.copy()
        initial_count = len(df)
        
        # Calculate rolling 20-day average dollar volume
        df['Dollar_Volume'] = df['Close'] * df['Volume']
        df['ADV_20D'] = df.groupby('Ticker')['Dollar_Volume'].rolling(20).mean().reset_index(0, drop=True)
        
        # Apply filters
        filters_applied = {}
        
        # 1. Price ≥ $3
        price_filter = df['Close'] >= self.min_price
        filters_applied['price_filter'] = price_filter.sum()
        df = df[price_filter]
        
        # 2. 20-day ADV ≥ $2M
        adv_filter = df['ADV_20D'] >= self.min_adv
        filters_applied['adv_filter'] = adv_filter.sum()
        df = df[adv_filter]
        
        # 3. Remove penny stocks and extreme prices
        reasonable_price = (df['Close'] >= 1) & (df['Close'] <= 1000)
        filters_applied['reasonable_price'] = reasonable_price.sum()
        df = df[reasonable_price]
        
        # 4. Remove extreme volume days (likely errors)
        volume_filter = df['Volume'] < df.groupby('Ticker')['Volume'].transform(lambda x: x.quantile(0.99))
        filters_applied['volume_filter'] = volume_filter.sum()
        df = df[volume_filter]
        
        # 5. Require minimum trading history
        min_history = 100  # 100 trading days
        df['trading_days'] = df.groupby('Ticker').cumcount() + 1
        history_filter = df['trading_days'] >= min_history
        filters_applied['history_filter'] = history_filter.sum()
        df = df[history_filter]
        
        final_count = len(df)
        
        logger.info(f"✅ Universe filters applied:")
        logger.info(f"   Initial samples: {initial_count:,}")
        logger.info(f"   Price ≥ ${self.min_price}: {filters_applied['price_filter']:,}")
        logger.info(f"   ADV ≥ ${self.min_adv:,.0f}: {filters_applied['adv_filter']:,}")
        logger.info(f"   Reasonable price: {filters_applied['reasonable_price']:,}")
        logger.info(f"   Volume filter: {filters_applied['volume_filter']:,}")
        logger.info(f"   History filter: {filters_applied['history_filter']:,}")
        logger.info(f"   Final samples: {final_count:,}")
        logger.info(f"   Symbols remaining: {df['Ticker'].nunique()}")
        
        return df
    
    def create_sleeve_c_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Sleeve C: Momentum + Quality features (Mission Brief Section 3)"""
        
        logger.info("🔧 Creating Sleeve C: Momentum + Quality features...")
        df = df.copy()
        
        # Calculate returns
        df['Return_1D'] = df.groupby('Ticker')['Close'].pct_change()
        df['Return_5D'] = df.groupby('Ticker')['Close'].pct_change(5)
        df['Return_20D'] = df.groupby('Ticker')['Close'].pct_change(20)
        df['Return_60D'] = df.groupby('Ticker')['Close'].pct_change(60)
        
        # 12M ex-1M momentum
        df['Return_252D'] = df.groupby('Ticker')['Close'].pct_change(252)
        df['Return_21D'] = df.groupby('Ticker')['Close'].pct_change(21)
        df['Return_12M_ex_1M'] = df['Return_252D'] - df['Return_21D']
        
        # Volatility features
        df['Vol_20D'] = df.groupby('Ticker')['Return_1D'].rolling(20).std().reset_index(0, drop=True)
        df['Vol_60D'] = df.groupby('Ticker')['Return_1D'].rolling(60).std().reset_index(0, drop=True)
        
        # Volume features
        df['Volume_Ratio'] = df['Volume'] / df.groupby('Ticker')['Volume'].rolling(20).mean().reset_index(0, drop=True)
        df['Dollar_Volume_Ratio'] = df['Dollar_Volume'] / df['ADV_20D']
        
        # Price level
        df['Log_Price'] = np.log(df['Close'])
        
        # Quality proxy (price momentum vs volume)
        df['Price_Volume_Trend'] = df.groupby('Ticker').apply(
            lambda x: x['Return_20D'].rolling(20).corr(x['Volume_Ratio'].shift(1))
        ).reset_index(0, drop=True)
        
        # LAGGED FEATURES (ALL ≥1 day lag - Mission Brief Section 2)
        feature_columns = []
        
        # 1. Momentum features (lagged)
        momentum_features = {
            'return_5d_lag1': ('Return_5D', 1),
            'return_20d_lag1': ('Return_20D', 1), 
            'return_60d_lag1': ('Return_60D', 1),
            'return_12m_ex_1m_lag1': ('Return_12M_ex_1M', 1)
        }
        
        for feat_name, (source_col, lag) in momentum_features.items():
            if source_col in df.columns:
                df[feat_name] = df.groupby('Ticker')[source_col].shift(lag)
                feature_columns.append(feat_name)
        
        # 2. Volatility features (lagged)
        vol_features = {
            'vol_20d_lag1': ('Vol_20D', 1),
            'vol_60d_lag1': ('Vol_60D', 1)
        }
        
        for feat_name, (source_col, lag) in vol_features.items():
            if source_col in df.columns:
                df[feat_name] = df.groupby('Ticker')[source_col].shift(lag)
                feature_columns.append(feat_name)
        
        # 3. Volume features (lagged)
        volume_features = {
            'volume_ratio_lag1': ('Volume_Ratio', 1),
            'dollar_volume_ratio_lag1': ('Dollar_Volume_Ratio', 1)
        }
        
        for feat_name, (source_col, lag) in volume_features.items():
            if source_col in df.columns:
                df[feat_name] = df.groupby('Ticker')[source_col].shift(lag)
                feature_columns.append(feat_name)
        
        # 4. Quality features (lagged)
        quality_features = {
            'log_price_lag1': ('Log_Price', 1),
            'price_volume_trend_lag1': ('Price_Volume_Trend', 1)
        }
        
        for feat_name, (source_col, lag) in quality_features.items():
            if source_col in df.columns:
                df[feat_name] = df.groupby('Ticker')[source_col].shift(lag)
                feature_columns.append(feat_name)
        
        # Limit to max features
        feature_columns = feature_columns[:self.max_features]
        self.feature_columns = [f for f in feature_columns if f in df.columns]
        
        logger.info(f"✅ Sleeve C features created: {len(self.feature_columns)} features")
        logger.info(f"   Features: {self.feature_columns}")
        
        return df
    
    def create_targets_and_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create targets and clean data (Mission Brief Section 2)"""
        
        logger.info("🎯 Creating targets and cleaning data...")
        df = df.copy()
        
        # Create next-day return targets (Mission Brief Section 2)
        df['target_return'] = df.groupby('Ticker')['Return_1D'].shift(-1)
        
        # Create cross-sectional rank targets for each date
        rank_targets = []
        for date, date_group in df.groupby('Date'):
            if len(date_group) < 10:  # Need minimum stocks
                continue
            
            returns = date_group['target_return'].dropna()
            if len(returns) < 5:
                continue
                
            # Cross-sectional ranks (0 to 1)
            ranks = returns.rank(pct=True)
            
            for idx, rank in ranks.items():
                rank_targets.append({'idx': idx, 'target_rank': rank})
        
        # Merge back
        if rank_targets:
            rank_df = pd.DataFrame(rank_targets).set_index('idx')
            df = df.join(rank_df, how='left')
        else:
            df['target_rank'] = np.nan
        
        # Clean features
        for col in self.feature_columns:
            if col in df.columns:
                # Remove infinite values
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                
                # Forward fill by ticker only (no cross-contamination)
                df[col] = df.groupby('Ticker')[col].ffill()
                
                # Fill remaining with median
                df[col] = df[col].fillna(df[col].median())
                
                # Conservative outlier clipping
                q01, q99 = df[col].quantile([0.01, 0.99])
                df[col] = df[col].clip(q01, q99)
        
        # Final cleanup
        df = df.dropna(subset=['target_return', 'target_rank'])
        
        # Require high feature coverage
        feature_coverage = df[self.feature_columns].notna().mean(axis=1)
        df = df[feature_coverage >= 0.9]
        
        # Remove early dates per ticker (for meaningful lags)
        df = df.groupby('Ticker').apply(
            lambda x: x.iloc[60:] if len(x) > 60 else x[0:0]
        ).reset_index(drop=True)
        
        logger.info(f"✅ Targets and cleanup completed: {len(df)} samples")
        return df
    
    def save_dataset_metadata(self, df: pd.DataFrame) -> str:
        """Save dataset with metadata (Mission Brief Section 7)"""
        
        # Create dataset hash
        data_hash = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()[:8]
        
        # Save dataset
        dataset_path = self.artifacts_dir / f"nasdaq_dataset_{data_hash}.csv"
        df.to_csv(dataset_path, index=False)
        
        # Save metadata
        metadata = {
            'dataset_hash': data_hash,
            'build_config': {
                'start_date': self.start_date,
                'end_date': self.end_date,
                'universe_filters': {
                    'min_price': self.min_price,
                    'min_adv': self.min_adv,
                    'max_adv_usage': self.max_adv_usage
                }
            },
            'statistics': {
                'total_samples': len(df),
                'unique_tickers': df['Ticker'].nunique(),
                'date_range': [str(df['Date'].min()), str(df['Date'].max())],
                'features': self.feature_columns
            },
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = self.artifacts_dir / f"dataset_metadata_{data_hash}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Dataset saved with hash: {data_hash}")
        logger.info(f"   Dataset: {dataset_path}")
        logger.info(f"   Metadata: {metadata_path}")
        
        return data_hash

def main():
    """Build NASDAQ Stock-Picker following Mission Brief"""
    
    logger.info("=" * 80)
    logger.info("🎯 NASDAQ STOCK-PICKER - MISSION BRIEF IMPLEMENTATION")
    logger.info("=" * 80)
    
    try:
        # Initialize picker
        picker = NASDAQStockPicker()
        
        # Step 1: Download NASDAQ universe
        logger.info("📊 STEP 1: Download NASDAQ Universe")
        raw_data = picker.download_nasdaq_universe()
        
        # Step 2: Apply universe filters
        logger.info("🔍 STEP 2: Apply Universe Filters")
        filtered_data = picker.apply_universe_filters(raw_data)
        
        # Step 3: Create Sleeve C features  
        logger.info("🔧 STEP 3: Create Sleeve C Features")
        featured_data = picker.create_sleeve_c_features(filtered_data)
        
        # Step 4: Create targets and cleanup
        logger.info("🎯 STEP 4: Create Targets and Cleanup")
        final_data = picker.create_targets_and_cleanup(featured_data)
        
        # Step 5: Save with metadata
        logger.info("💾 STEP 5: Save Dataset with Metadata")
        dataset_hash = picker.save_dataset_metadata(final_data)
        
        # Summary
        logger.info("=" * 80)
        logger.info("🏆 NASDAQ UNIVERSE BUILD COMPLETE")
        logger.info("=" * 80)
        logger.info(f"📊 Dataset Hash: {dataset_hash}")
        logger.info(f"📈 Total Samples: {len(final_data):,}")
        logger.info(f"🏢 Unique Tickers: {final_data['Ticker'].nunique()}")
        logger.info(f"📅 Date Range: {final_data['Date'].min()} to {final_data['Date'].max()}")
        logger.info(f"🔧 Features: {len(picker.feature_columns)}")
        logger.info(f"💾 Artifacts saved to: {picker.artifacts_dir}")
        
        print(f"\n✅ NASDAQ UNIVERSE READY!")
        print(f"📊 {len(final_data):,} samples, {final_data['Ticker'].nunique()} tickers")
        print(f"🔧 {len(picker.feature_columns)} Sleeve C features")
        print(f"💾 Hash: {dataset_hash}")
        print(f"🎯 Ready for Sleeve C training!")
        
        return final_data, dataset_hash
        
    except Exception as e:
        logger.error(f"❌ NASDAQ universe build failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    data, hash_id = main()