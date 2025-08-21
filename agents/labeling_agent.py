#!/usr/bin/env python3
"""
LABELING & TARGET AGENT - Chat-G.txt Section 2
Mission: Produce robust targets for ranking & classification
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
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
        logger.info("   Primary: Next-day residual return (stock - QQQ & sector effects)")
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

            # Create primary targets - next-day residual returns
            df = self._create_residual_return_targets(df)

            # Create meta-labels - trinary classification
            df = self._create_meta_labels(df)

            # Create barrier labels - triple-barrier outcomes
            df = self._create_barrier_labels(df)

            # Lag all feature columns by at least one day to avoid leakage
            df = self._lag_features(df)
            
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
    
    def _create_residual_return_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create next-day residual return targets via market/sector regression."""

        logger.info("📈 Creating next-day residual return targets...")

        df = df.copy()

        # Compute next-day forward return for each ticker
        df['next_close'] = df.groupby('Ticker')['Close'].shift(-1)
        df['return_1d_forward'] = (df['next_close'] - df['Close']) / df['Close']

        # Load QQQ returns for market proxy
        qqq_path = self.base_dir / "data" / "QQQ.csv"
        qqq = pd.read_csv(qqq_path, parse_dates=['Date'])
        qqq = qqq.sort_values('Date')
        qqq['qqq_next_close'] = qqq['Close'].shift(-1)
        qqq['qqq_return_1d'] = (qqq['qqq_next_close'] - qqq['Close']) / qqq['Close']
        qqq = qqq[['Date', 'qqq_return_1d']]

        # Merge market returns
        df = df.merge(qqq, on='Date', how='left')

        # Cross-sectional regression by date against QQQ and sector dummies
        df['residual_return_1d'] = np.nan
        for date, group in df.groupby('Date'):
            y = group['return_1d_forward'].values

            if np.isnan(y).all():
                continue

            X = np.ones((len(group), 1))  # intercept
            market = group['qqq_return_1d'].values.reshape(-1, 1)
            X = np.concatenate([X, market], axis=1)

            if 'sector' in group.columns:
                sector_dummies = pd.get_dummies(group['sector'])
                X = np.concatenate([X, sector_dummies.values], axis=1)

            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            fitted = X @ beta
            resid = y - fitted
            df.loc[group.index, 'residual_return_1d'] = resid

        # Clean up intermediate columns
        df = df.drop(columns=['next_close', 'return_1d_forward', 'qqq_return_1d'], errors='ignore')

        valid = df['residual_return_1d'].notna().sum()
        logger.info(f"✅ Residual return targets created: {valid} valid targets")

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
            df['residual_return_1d'] > cost_threshold,   # +1: Strong positive
            df['residual_return_1d'] < -cost_threshold,  # -1: Strong negative
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

    def _lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lag all non-target feature columns by one day to prevent leakage."""

        df = df.copy()

        exclude_cols = {
            'Date', 'Ticker', 'Close', 'sector',
            'residual_return_1d', 'meta_label', 'barrier_label'
        }
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        df[feature_cols] = df.groupby('Ticker')[feature_cols].shift(1)
        return df
    
    def _leakage_audit(self, df: pd.DataFrame) -> bool:
        """
        Leakage audit: no feature timestamp ≥ target window start
        Chat-G.txt DoD: Critical for model validity
        """
        
        logger.info("🔍 Performing leakage audit...")
        
        # Define columns that should not be treated as features
        target_cols = ['residual_return_1d', 'meta_label', 'barrier_label']
        exclude_cols = {
            'Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'sector', 'industry'
        }

        feature_cols = [col for col in df.columns if col not in target_cols and col not in exclude_cols]
        if not feature_cols:
            logger.error("No feature columns found for leakage audit")
            return False

        # First observation per ticker should be NaN after lagging
        for col in feature_cols:
            if df.groupby('Ticker')[col].first().notna().any():
                logger.error(f"Potential leakage detected in column: {col}")
                return False

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
                'residual_return_1d': {
                    'count': df['residual_return_1d'].notna().sum(),
                    'mean': float(df['residual_return_1d'].mean()),
                    'std': float(df['residual_return_1d'].std()),
                    'min': float(df['residual_return_1d'].min()),
                    'max': float(df['residual_return_1d'].max())
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
        logger.info(f"   Total samples with targets: {df['residual_return_1d'].notna().sum()}")

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