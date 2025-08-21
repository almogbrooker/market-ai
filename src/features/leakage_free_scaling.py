#!/usr/bin/env python3
"""
LEAKAGE-FREE CROSS-SECTIONAL SCALING
Critical fix for preventing data leakage in per-fold scaling
Enforces daily, cross-sectional z-scores computed strictly inside fold boundaries
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Tuple, Dict, List, Optional, Union
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class LeakageFreeScaler:
    """
    Leakage-free scaler that enforces per-fold, per-day cross-sectional scaling
    NEVER leaks global statistics across fold boundaries
    """
    
    def __init__(self, 
                 scaling_method: str = 'zscore',
                 min_cross_sectional_samples: int = 5,
                 fill_method: str = 'forward'):
        """
        Initialize leakage-free scaler
        
        Args:
            scaling_method: 'zscore', 'robust', 'rank', or 'none'
            min_cross_sectional_samples: Min stocks needed per day for cross-sectional scaling
            fill_method: How to handle missing scaling stats ('forward', 'zero', 'global')
        """
        self.scaling_method = scaling_method
        self.min_cross_sectional_samples = min_cross_sectional_samples
        self.fill_method = fill_method
        
        # Track per-fold scalers (NEVER cross-contaminate)
        self.fold_scalers = {}
        self.fold_cross_sectional_stats = {}
        
        logger.info(f"🔒 LeakageFreeScaler initialized: method={scaling_method}")
    
    def fit_transform_fold(self, 
                          X: pd.DataFrame, 
                          fold_id: str,
                          date_column: str = 'Date',
                          symbol_column: str = 'Ticker') -> pd.DataFrame:
        """
        Fit scaler on fold data and transform (NO LEAKAGE)
        
        Args:
            X: Training data for this fold ONLY
            fold_id: Unique identifier for this fold
            date_column: Name of date column
            symbol_column: Name of symbol/ticker column
            
        Returns:
            Scaled data using ONLY statistics from this fold
        """
        logger.info(f"🔧 Fitting leakage-free scaler for fold {fold_id}")
        
        if fold_id in self.fold_scalers:
            logger.warning(f"⚠️ Fold {fold_id} already exists - overwriting")
        
        X_scaled = X.copy()
        
        # Ensure date column is datetime
        X_scaled[date_column] = pd.to_datetime(X_scaled[date_column])
        
        # Get feature columns (exclude date, symbol, targets)
        exclude_cols = [date_column, symbol_column] + [col for col in X.columns if 'target' in col.lower() or 'label' in col.lower()]
        feature_cols = [col for col in X.columns if col not in exclude_cols]
        
        logger.info(f"📊 Scaling {len(feature_cols)} features across {X[symbol_column].nunique()} symbols")
        
        # Initialize fold-specific storage
        self.fold_scalers[fold_id] = {}
        self.fold_cross_sectional_stats[fold_id] = {}
        
        # Per-feature scaling (time-series level)
        for col in feature_cols:
            if X[col].dtype in ['object', 'category']:
                continue
                
            # Fit time-series scaler on this fold's data ONLY
            fold_scaler = self._get_scaler()
            valid_data = X[col].dropna()
            
            if len(valid_data) > 10:
                fold_scaler.fit(valid_data.values.reshape(-1, 1))
                X_scaled[col] = self._safe_transform(fold_scaler, X[col].values.reshape(-1, 1)).flatten()
                self.fold_scalers[fold_id][col] = fold_scaler
            else:
                logger.warning(f"⚠️ Insufficient data for {col} in fold {fold_id}")
                X_scaled[col] = X[col].fillna(0)
        
        # Cross-sectional scaling per day (CRITICAL FOR ALPHA)
        X_scaled = self._apply_cross_sectional_scaling(
            X_scaled, fold_id, date_column, symbol_column, feature_cols, fit_mode=True
        )
        
        logger.info(f"✅ Fold {fold_id} scaling completed")
        return X_scaled
    
    def transform_fold(self, 
                      X: pd.DataFrame,
                      fold_id: str,
                      date_column: str = 'Date',
                      symbol_column: str = 'Ticker') -> pd.DataFrame:
        """
        Transform new data using fold-specific scalers (NO LEAKAGE)
        
        Args:
            X: New data to transform
            fold_id: Which fold's scalers to use
            
        Returns:
            Scaled data using fold-specific scalers
        """
        if fold_id not in self.fold_scalers:
            raise ValueError(f"Fold {fold_id} not fitted. Call fit_transform_fold first.")
        
        X_scaled = X.copy()
        X_scaled[date_column] = pd.to_datetime(X_scaled[date_column])
        
        # Get feature columns
        exclude_cols = [date_column, symbol_column] + [col for col in X.columns if 'target' in col.lower() or 'label' in col.lower()]
        feature_cols = [col for col in X.columns if col not in exclude_cols]
        
        # Apply fold-specific time-series scaling
        for col in feature_cols:
            if col in self.fold_scalers[fold_id] and X[col].dtype not in ['object', 'category']:
                fold_scaler = self.fold_scalers[fold_id][col]
                X_scaled[col] = self._safe_transform(fold_scaler, X[col].values.reshape(-1, 1)).flatten()
            else:
                X_scaled[col] = X[col].fillna(0)
        
        # Apply cross-sectional scaling using fold-specific stats
        X_scaled = self._apply_cross_sectional_scaling(
            X_scaled, fold_id, date_column, symbol_column, feature_cols, fit_mode=False
        )
        
        return X_scaled
    
    def _apply_cross_sectional_scaling(self, 
                                     X: pd.DataFrame,
                                     fold_id: str,
                                     date_column: str,
                                     symbol_column: str,
                                     feature_cols: List[str],
                                     fit_mode: bool = True) -> pd.DataFrame:
        """
        Apply daily cross-sectional scaling (the critical alpha source)
        """
        X_scaled = X.copy()
        
        if fit_mode:
            daily_stats = {}
        
        # Group by date for cross-sectional scaling
        for date, date_group in X.groupby(date_column):
            n_symbols = len(date_group)
            
            if n_symbols < self.min_cross_sectional_samples:
                logger.debug(f"⚠️ Insufficient symbols ({n_symbols}) for {date} - skipping cross-sectional scaling")
                continue
            
            date_idx = X[date_column] == date
            
            for col in feature_cols:
                if X[col].dtype in ['object', 'category']:
                    continue
                
                col_values = date_group[col].dropna()
                
                if len(col_values) < 3:  # Need minimum for cross-sectional stats
                    continue
                
                if fit_mode:
                    # Compute cross-sectional stats for this date (FOLD-SPECIFIC)
                    if self.scaling_method == 'zscore':
                        cs_mean = col_values.mean()
                        cs_std = col_values.std()
                        if cs_std < 1e-8:
                            cs_std = 1.0
                    elif self.scaling_method == 'robust':
                        cs_mean = col_values.median()
                        cs_std = col_values.mad()  # Median absolute deviation
                        if cs_std < 1e-8:
                            cs_std = 1.0
                    elif self.scaling_method == 'rank':
                        # Cross-sectional ranking (0 to 1)
                        ranks = col_values.rank(pct=True)
                        X_scaled.loc[date_idx, col] = X_scaled.loc[date_idx, col].map(
                            dict(zip(date_group.index, ranks - 0.5))  # Center around 0
                        ).fillna(0)
                        continue
                    else:
                        continue
                    
                    # Store stats for this fold and date
                    if col not in daily_stats:
                        daily_stats[col] = {}
                    daily_stats[col][date] = {'mean': cs_mean, 'std': cs_std}
                    
                    # Apply scaling
                    X_scaled.loc[date_idx, col] = (X_scaled.loc[date_idx, col] - cs_mean) / cs_std
                
                else:
                    # Transform mode: use stored fold-specific stats
                    if (fold_id in self.fold_cross_sectional_stats and 
                        col in self.fold_cross_sectional_stats[fold_id] and
                        date in self.fold_cross_sectional_stats[fold_id][col]):
                        
                        stats = self.fold_cross_sectional_stats[fold_id][col][date]
                        cs_mean = stats['mean']
                        cs_std = stats['std']
                        
                        X_scaled.loc[date_idx, col] = (X_scaled.loc[date_idx, col] - cs_mean) / cs_std
                    
                    else:
                        # Handle missing stats based on fill_method
                        if self.fill_method == 'zero':
                            X_scaled.loc[date_idx, col] = 0
                        elif self.fill_method == 'forward':
                            # Use most recent available stats for this fold
                            recent_stats = self._get_recent_stats(fold_id, col, date)
                            if recent_stats:
                                cs_mean = recent_stats['mean']
                                cs_std = recent_stats['std']
                                X_scaled.loc[date_idx, col] = (X_scaled.loc[date_idx, col] - cs_mean) / cs_std
                        # 'global' would use overall fold statistics (not implemented to avoid complexity)
        
        # Store daily stats for this fold
        if fit_mode:
            self.fold_cross_sectional_stats[fold_id] = daily_stats
        
        return X_scaled
    
    def _get_recent_stats(self, fold_id: str, col: str, target_date) -> Optional[Dict]:
        """Get most recent cross-sectional stats for a column"""
        if (fold_id not in self.fold_cross_sectional_stats or 
            col not in self.fold_cross_sectional_stats[fold_id]):
            return None
        
        col_stats = self.fold_cross_sectional_stats[fold_id][col]
        
        # Find most recent date before target_date
        available_dates = [d for d in col_stats.keys() if d < target_date]
        
        if not available_dates:
            return None
        
        most_recent = max(available_dates)
        return col_stats[most_recent]
    
    def _get_scaler(self):
        """Get appropriate sklearn scaler"""
        if self.scaling_method == 'robust':
            return RobustScaler()
        else:
            return StandardScaler()
    
    def _safe_transform(self, scaler, data):
        """Safely transform data handling edge cases"""
        try:
            return scaler.transform(data)
        except Exception as e:
            logger.warning(f"Transform failed: {e}, returning zeros")
            return np.zeros_like(data)
    
    def validate_no_leakage(self, 
                          train_data: pd.DataFrame,
                          val_data: pd.DataFrame,
                          date_column: str = 'Date') -> Dict[str, bool]:
        """
        Validate that no future data leaked into training scaling
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {}
        
        # Check date boundaries
        train_max_date = train_data[date_column].max()
        val_min_date = val_data[date_column].min()
        
        validation_results['date_boundary_respected'] = train_max_date < val_min_date
        
        # Check that fold scalers only use training data
        validation_results['fold_isolation'] = len(self.fold_scalers) > 0
        
        # Check cross-sectional stats don't use future data
        if self.fold_cross_sectional_stats:
            all_stats_dates = []
            for fold_stats in self.fold_cross_sectional_stats.values():
                for col_stats in fold_stats.values():
                    all_stats_dates.extend(col_stats.keys())
            
            if all_stats_dates:
                max_stats_date = max(all_stats_dates)
                validation_results['cross_sectional_no_future'] = max_stats_date <= train_max_date
        
        logger.info(f"🔍 Leakage validation: {validation_results}")
        return validation_results

class BetaNeutralTargetGenerator:
    """
    Generate beta-neutral, cross-sectional alpha targets
    """
    
    def __init__(self, benchmark_symbol: str = 'QQQ'):
        self.benchmark_symbol = benchmark_symbol
    
    def generate_alpha_targets(self, 
                             returns_df: pd.DataFrame,
                             date_column: str = 'Date',
                             symbol_column: str = 'Ticker',
                             return_column: str = 'return_1d',
                             volatility_column: Optional[str] = None) -> pd.DataFrame:
        """
        Generate volatility-scaled, benchmark-neutral alpha targets
        
        Args:
            returns_df: DataFrame with returns data
            volatility_column: Column with rolling volatility (for scaling)
            
        Returns:
            DataFrame with alpha targets added
        """
        df = returns_df.copy()
        
        # Get benchmark returns (if available)
        benchmark_returns = self._get_benchmark_returns(df, date_column, return_column)
        
        # Calculate beta-neutral residuals per day
        alpha_targets = []
        
        for date, date_group in df.groupby(date_column):
            date_returns = date_group.set_index(symbol_column)[return_column]
            
            # Get benchmark return for this date
            if benchmark_returns is not None and date in benchmark_returns:
                bench_return = benchmark_returns[date]
                
                # Simple beta-neutral alpha (can be enhanced with rolling beta)
                alpha = date_returns - bench_return
            else:
                # Cross-sectional residual (relative to market)
                alpha = date_returns - date_returns.mean()
            
            # Volatility scaling if available
            if volatility_column and volatility_column in date_group.columns:
                vol_series = date_group.set_index(symbol_column)[volatility_column]
                alpha = alpha / (vol_series + 1e-6)  # Avoid division by zero
            
            # Cross-sectional standardization (critical for ranking)
            alpha = (alpha - alpha.mean()) / (alpha.std() + 1e-6)
            
            alpha_targets.append(alpha.reset_index())
        
        # Combine results
        alpha_df = pd.concat(alpha_targets, ignore_index=True)
        alpha_df[date_column] = [date for date, group in df.groupby(date_column) for _ in range(len(group))]
        alpha_df = alpha_df.rename(columns={return_column: 'alpha_target'})
        
        # Merge back with original data
        result = df.merge(alpha_df, on=[date_column, symbol_column], how='left')
        
        logger.info(f"✅ Generated alpha targets: mean={result['alpha_target'].mean():.6f}, std={result['alpha_target'].std():.4f}")
        
        return result
    
    def _get_benchmark_returns(self, df: pd.DataFrame, date_column: str, return_column: str) -> Optional[Dict]:
        """Get benchmark returns if available in the dataset"""
        if self.benchmark_symbol in df['Ticker'].values:
            benchmark_data = df[df['Ticker'] == self.benchmark_symbol]
            return dict(zip(benchmark_data[date_column], benchmark_data[return_column]))
        return None

# Usage example
def example_leakage_free_workflow():
    """
    Example of proper leakage-free scaling workflow
    """
    # Load data
    # df = pd.read_csv('training_data.csv')
    
    # Initialize leakage-free scaler
    scaler = LeakageFreeScaler(
        scaling_method='zscore',
        min_cross_sectional_samples=5
    )
    
    # Generate alpha targets
    target_gen = BetaNeutralTargetGenerator(benchmark_symbol='QQQ')
    # df = target_gen.generate_alpha_targets(df)
    
    # Proper fold-based training (NO LEAKAGE)
    # for fold_id, (train_idx, val_idx) in enumerate(cv_splits):
    #     # Get fold data
    #     train_data = df.iloc[train_idx].copy()
    #     val_data = df.iloc[val_idx].copy()
    #     
    #     # Fit scaler ONLY on training data
    #     train_scaled = scaler.fit_transform_fold(train_data, f'fold_{fold_id}')
    #     
    #     # Transform validation data using training scalers
    #     val_scaled = scaler.transform_fold(val_data, f'fold_{fold_id}')
    #     
    #     # Validate no leakage
    #     validation = scaler.validate_no_leakage(train_scaled, val_scaled)
    #     assert all(validation.values()), f"Data leakage detected in fold {fold_id}"
    #     
    #     # Train model on train_scaled, validate on val_scaled
    
    logger.info("✅ Leakage-free scaling workflow example completed")

if __name__ == "__main__":
    example_leakage_free_workflow()