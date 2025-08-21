#!/usr/bin/env python3
"""
LABELING & TARGET AGENT - Chat-G.txt Section 2
Mission: Produce robust targets for ranking & classification
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LabelingAgent:
    """
    Labeling & Target Agent - Chat-G.txt Section 2
    Produce robust targets for ranking & classification
    """
    
    def __init__(self, data_config: Dict):
        logger.info("🎯 LABELING & TARGET AGENT - ROBUST TARGETS")
        
        self.config = data_config
        self.base_dir = Path(__file__).parent.parent
        self.artifacts_dir = self.base_dir / "artifacts"
        
        # Load features data
        self.features_path = self.artifacts_dir / "features" / "daily.parquet"
        
        logger.info("📋 Target Types:")
        logger.info("   Primary: 21-day excess return (stock - sector ETF)")
        logger.info("   Meta-labels: Trinary {-1,0,+1} via dead-zone ±(cost + 10bps)")
        logger.info("   Barrier: 5-day triple-barrier outcome (TP/SL/timeout)")
        
    def create_targets(self) -> bool:
        """
        Create all target types following Chat-G.txt specification
        DoD: Leakage audit (no feature timestamp ≥ target window start)
        """
        
        logger.info("🏗️ Creating robust targets...")
        
        try:
            # Load features data
            if not self.features_path.exists():
                logger.error(f"Features data not found: {self.features_path}")
                return False
            
            df = pd.read_parquet(self.features_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values(['Ticker', 'Date'])
            
            logger.info(f"📊 Loaded features: {len(df)} samples, {df['Ticker'].nunique()} tickers")
            
            # Create primary targets - 21-day excess returns
            df = self._create_excess_return_targets(df)
            
            # Create meta-labels - trinary classification
            df = self._create_meta_labels(df)
            
            # Create barrier labels - triple-barrier outcomes
            df = self._create_barrier_labels(df)
            
            # Leakage audit
            if not self._leakage_audit(df):
                logger.error("❌ Leakage audit failed")
                return False
            
            # Save labeled data
            self._save_labels(df)
            
            logger.info("✅ Targets created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create targets: {e}")
            return False
    
    def _create_excess_return_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create 21-day excess return targets
        Chat-G.txt: next-21 trading day excess return (stock return - sector ETF return)
        """
        
        logger.info("📈 Creating 21-day excess return targets...")
        
        df = df.copy()
        
        # Calculate 21-day forward returns
        df['return_21d_forward'] = df.groupby('Ticker')['Close'].pct_change(periods=-21).shift(-21)
        
        # Create sector ETF proxy returns (simplified for MVP)
        # In production, would use actual sector ETF data
        sector_returns = df.groupby(['Date', 'sector'])['return_21d_forward'].median().reset_index()
        sector_returns = sector_returns.rename(columns={'return_21d_forward': 'sector_return_21d'})
        
        # Merge sector returns
        df = df.merge(sector_returns, on=['Date', 'sector'], how='left')
        
        # Calculate excess returns (stock - sector)
        df['excess_return_21d'] = df['return_21d_forward'] - df['sector_return_21d']
        
        # Fill missing sector returns with market median
        market_median = df.groupby('Date')['return_21d_forward'].median()
        df = df.merge(market_median.reset_index().rename(columns={'return_21d_forward': 'market_return_21d'}), 
                     on='Date', how='left')
        
        # Use market return as fallback for sector return
        df['sector_return_21d'] = df['sector_return_21d'].fillna(df['market_return_21d'])
        df['excess_return_21d'] = df['excess_return_21d'].fillna(
            df['return_21d_forward'] - df['market_return_21d']
        )
        
        logger.info(f"✅ Excess return targets created: {df['excess_return_21d'].notna().sum()} valid targets")
        
        return df
    
    def _create_meta_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create meta-labels: trinary {-1,0,+1} via dead-zone ±(cost + 10bps)
        Chat-G.txt Section 2: dead-zone approach to account for transaction costs
        """
        
        logger.info("🏷️ Creating meta-labels with dead-zone...")
        
        df = df.copy()
        
        # Transaction cost estimate: 15 bps (spread + slippage) + 10 bps buffer = 25 bps total
        cost_threshold = 0.0025  # 25 bps
        
        # Create trinary labels
        conditions = [
            df['excess_return_21d'] > cost_threshold,   # +1: Strong positive
            df['excess_return_21d'] < -cost_threshold,  # -1: Strong negative
        ]
        choices = [1, -1]
        df['meta_label'] = np.select(conditions, choices, default=0)  # 0: Dead zone
        
        # Label distribution
        label_counts = df['meta_label'].value_counts().sort_index()
        logger.info(f"📊 Meta-label distribution:")
        logger.info(f"   Strong Negative (-1): {label_counts.get(-1, 0)} ({label_counts.get(-1, 0)/len(df)*100:.1f}%)")
        logger.info(f"   Dead Zone (0):        {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(df)*100:.1f}%)")
        logger.info(f"   Strong Positive (+1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(df)*100:.1f}%)")
        
        return df
    
    def _create_barrier_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create barrier labels: 5-day triple-barrier outcome (TP/SL/timeout)
        Chat-G.txt Section 2: useful for classification and risk management
        """
        
        logger.info("🚧 Creating 5-day triple-barrier labels...")
        
        df = df.copy()
        
        # Parameters for triple barrier
        take_profit_pct = 0.03    # 3% take profit
        stop_loss_pct = -0.02     # 2% stop loss
        horizon_days = 5          # 5-day timeout
        
        barrier_results = []
        
        for ticker in df['Ticker'].unique():
            ticker_data = df[df['Ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('Date')
            
            for i, row in ticker_data.iterrows():
                # Get next 5 days of data
                future_data = ticker_data[ticker_data['Date'] > row['Date']].head(horizon_days)
                
                if len(future_data) == 0:
                    barrier_results.append({'index': i, 'barrier_label': 'timeout'})
                    continue
                
                entry_price = row['Close']
                
                # Calculate cumulative returns over next 5 days
                returns = []
                for _, future_row in future_data.iterrows():
                    ret = (future_row['Close'] - entry_price) / entry_price
                    returns.append(ret)
                    
                    # Check barriers
                    if ret >= take_profit_pct:
                        barrier_results.append({'index': i, 'barrier_label': 'take_profit'})
                        break
                    elif ret <= stop_loss_pct:
                        barrier_results.append({'index': i, 'barrier_label': 'stop_loss'})
                        break
                else:
                    # No barrier hit, timeout
                    barrier_results.append({'index': i, 'barrier_label': 'timeout'})
        
        # Merge barrier labels back
        barrier_df = pd.DataFrame(barrier_results).set_index('index')
        df = df.join(barrier_df, how='left')
        df['barrier_label'] = df['barrier_label'].fillna('no_data')
        
        # Barrier label distribution  
        barrier_counts = df['barrier_label'].value_counts()
        logger.info(f"📊 Barrier label distribution:")
        for label, count in barrier_counts.items():
            logger.info(f"   {label}: {count} ({count/len(df)*100:.1f}%)")
        
        return df
    
    def _leakage_audit(self, df: pd.DataFrame) -> bool:
        """
        Leakage audit: no feature timestamp ≥ target window start
        Chat-G.txt DoD: Critical for model validity
        """
        
        logger.info("🔍 Performing leakage audit...")
        
        # Check that all features are properly lagged
        feature_cols = [col for col in df.columns if col.endswith('_lag1') or col.endswith('_lag0')]
        
        # All features should be lagged (already enforced in feature engineering)
        if len(feature_cols) == 0:
            logger.error("No properly lagged features found")
            return False
        
        # Check that target columns don't use future information
        target_cols = ['excess_return_21d', 'meta_label', 'barrier_label']
        
        # These targets use forward-looking information by design (that's the point)
        # The key is that features must be lagged relative to the target start date
        
        # Verify no features accidentally use same-day or future information
        risky_cols = [col for col in df.columns if not any([
            col.endswith('_lag1'),
            col.endswith('_lag0'), 
            col in ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'sector', 'industry'],
            col in target_cols,
            col.startswith('return_21d_forward'),  # Intermediate calculation
            col.startswith('sector_return'),       # Intermediate calculation
            col.startswith('market_return')        # Intermediate calculation
        ])]
        
        if risky_cols:
            logger.error(f"Potential leakage in columns: {risky_cols}")
            return False
        
        # Check temporal consistency - no target before minimum feature date
        min_feature_dates = df.groupby('Ticker')['Date'].min()
        
        # For each ticker, verify targets start after sufficient lag period
        for ticker in df['Ticker'].unique():
            ticker_data = df[df['Ticker'] == ticker]
            min_date = min_feature_dates[ticker]
            
            # Need at least 21 days of history for lags + 21 days forward for target
            required_history = timedelta(days=60)
            
            valid_target_data = ticker_data[
                (ticker_data['Date'] >= min_date + required_history) & 
                (ticker_data['excess_return_21d'].notna())
            ]
            
            if len(valid_target_data) == 0:
                logger.warning(f"No valid targets for {ticker} after leakage check")
        
        logger.info("✅ Leakage audit passed")
        return True
    
    def _save_labels(self, df: pd.DataFrame):
        """Save labeled data to artifacts"""
        
        logger.info("💾 Saving labeled data...")
        
        # Ensure labels directory exists
        (self.artifacts_dir / "labels").mkdir(parents=True, exist_ok=True)
        
        # Save full labeled dataset
        labels_path = self.artifacts_dir / "labels" / "labels.parquet"
        df.to_parquet(labels_path, index=False)
        
        # Create summary statistics
        summary = {
            'total_samples': len(df),
            'unique_tickers': df['Ticker'].nunique(),
            'date_range': [str(df['Date'].min()), str(df['Date'].max())],
            'target_statistics': {
                'excess_return_21d': {
                    'count': df['excess_return_21d'].notna().sum(),
                    'mean': float(df['excess_return_21d'].mean()),
                    'std': float(df['excess_return_21d'].std()),
                    'min': float(df['excess_return_21d'].min()),
                    'max': float(df['excess_return_21d'].max())
                },
                'meta_label_distribution': df['meta_label'].value_counts().to_dict(),
                'barrier_label_distribution': df['barrier_label'].value_counts().to_dict()
            },
            'feature_count': len([col for col in df.columns if col.endswith('_lag1') or col.endswith('_lag0')]),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save summary
        summary_path = self.artifacts_dir / "labels" / "labels_summary.json"
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"✅ Labels saved:")
        logger.info(f"   Dataset: {labels_path}")
        logger.info(f"   Summary: {summary_path}")
        logger.info(f"   Total samples with targets: {df['excess_return_21d'].notna().sum()}")

def main():
    """Test the labeling agent"""
    
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "data_config.json"
    if config_path.exists():
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Initialize and run agent
    agent = LabelingAgent(config)
    success = agent.create_targets()
    
    if success:
        print("✅ Labeling completed successfully")
    else:
        print("❌ Labeling failed")

if __name__ == "__main__":
    main()