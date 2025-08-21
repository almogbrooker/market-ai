#!/usr/bin/env python3
"""
Enhanced data loader for cross-sectional Alpha prediction (stock_return - QQQ_return)
"""

import pandas as pd
import numpy as np
import torch
from typing import Tuple, Optional, Dict, List
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class AlphaDataLoader:
    """Data loader for cross-sectional Alpha prediction relative to QQQ"""
    
    def __init__(self, 
                 sequence_length: int = 30,
                 prediction_horizon: int = 1,
                 neutral_zone_bps: float = 5.0,
                 label_mode: str = 'alpha'):
        """
        Args:
            sequence_length: Input sequence length for models
            prediction_horizon: Days ahead to predict (1, 5, or 20)
            neutral_zone_bps: Neutral zone in basis points (±5-10bps)
            label_mode: 'alpha' (regression), 'rank' (ranking), 'cls' (classification)
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.neutral_zone = neutral_zone_bps / 10_000.0  # Convert bps to decimal
        self.label_mode = label_mode
        
        self.feature_scaler = StandardScaler()
        self.features = None
        self.targets = None
        self.tickers = None
        self.dates = None
        
    def load_data(self, 
                  training_data_path: str = 'data/training_data_2020_2024_complete.csv',
                  qqq_data_path: str = 'data/QQQ.csv') -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and process data for Alpha prediction"""
        
        logger.info(f"Loading data for {self.label_mode} prediction...")

        # Validate file paths
        training_path = Path(training_data_path)
        qqq_path = Path(qqq_data_path)
        if not training_path.is_file():
            raise FileNotFoundError(f"Training data file not found: {training_data_path}")
        if not qqq_path.is_file():
            raise FileNotFoundError(f"QQQ data file not found: {qqq_data_path}")

        # Load stock data
        try:
            df = pd.read_csv(training_path)
        except pd.errors.ParserError:
            logger.exception(f"Failed to parse training data file: {training_data_path}")
            raise
        
        # Standardize column names
        if 'date' in df.columns and 'Date' not in df.columns:
            df = df.rename(columns={'date': 'Date'})
        if 'ticker' in df.columns and 'Ticker' not in df.columns:
            df = df.rename(columns={'ticker': 'Ticker'})
        if 'close' in df.columns and 'Close' not in df.columns:
            df = df.rename(columns={'close': 'Close'})
        
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Load QQQ benchmark data
        try:
            qqq_df = pd.read_csv(qqq_path)
        except pd.errors.ParserError:
            logger.exception(f"Failed to parse QQQ data file: {qqq_data_path}")
            raise
        qqq_df['Date'] = pd.to_datetime(qqq_df['Date'])
        qqq_returns = qqq_df.set_index('Date')['Return_1']
        
        logger.info(f"Stock data: {len(df)} rows, QQQ data: {len(qqq_df)} rows")
        
        # Calculate forward returns for different horizons
        df = self._calculate_forward_returns(df)
        
        # Merge with QQQ returns to calculate Alpha
        df = self._calculate_alpha_targets(df, qqq_returns)
        
        # Prepare features and targets
        X, y, metadata = self._prepare_sequences(df)
        
        logger.info(f"Final dataset: {X.shape[0]} samples, {X.shape[1]} timesteps, {X.shape[2]} features")
        
        return X, y, metadata
    
    def _calculate_forward_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate forward returns for multiple horizons"""
        
        # Sort by ticker and date
        df = df.sort_values(['Ticker', 'Date'])
        
        # 🔒 FIXED: Calculate forward returns with proper lag (no same-day signaling)
        # Use shift(-horizon-1) to ensure signal computed BEFORE tradeable period
        for horizon in [1, 5, 20]:
            future_col = f'Future_Return_{horizon}D'
            # CRITICAL FIX: Add 1-day buffer to prevent same-close signaling
            df[future_col] = df.groupby('Ticker')['Close'].pct_change(periods=horizon).shift(-(horizon + 1))
        
        return df
    
    def _calculate_alpha_targets(self, df: pd.DataFrame, qqq_returns: pd.Series) -> pd.DataFrame:
        """Calculate Alpha targets (stock_return - QQQ_return)"""
        
        # Merge QQQ returns
        df = df.merge(qqq_returns.to_frame('QQQ_Return').reset_index(), 
                      on='Date', how='left')
        
        # 🔒 FIXED: Calculate forward QQQ returns with proper lag
        qqq_df_temp = pd.DataFrame({'Date': qqq_returns.index, 'QQQ_Return': qqq_returns.values})
        for horizon in [1, 5, 20]:
            future_qqq_col = f'QQQ_Future_Return_{horizon}D'
            # CRITICAL FIX: Add 1-day buffer for QQQ returns too
            qqq_df_temp[future_qqq_col] = qqq_df_temp['QQQ_Return'].shift(-(horizon + 1))
        
        # Merge future QQQ returns
        df = df.merge(qqq_df_temp[['Date'] + [f'QQQ_Future_Return_{h}D' for h in [1, 5, 20]]], 
                      on='Date', how='left')
        
        # Calculate Alpha (excess return over QQQ) for each horizon
        for horizon in [1, 5, 20]:
            stock_return_col = f'Future_Return_{horizon}D'
            qqq_return_col = f'QQQ_Future_Return_{horizon}D'
            alpha_col = f'Alpha_{horizon}D'
            
            df[alpha_col] = df[stock_return_col] - df[qqq_return_col]
        
        # Main target is based on prediction horizon
        target_col = f'Alpha_{self.prediction_horizon}D'
        df['Target_Alpha'] = df[target_col]
        
        logger.info(f"Alpha statistics for {self.prediction_horizon}D horizon:")
        logger.info(f"Mean: {df['Target_Alpha'].mean():.6f}")
        logger.info(f"Std: {df['Target_Alpha'].std():.6f}")
        logger.info(f"Min: {df['Target_Alpha'].min():.6f}")
        logger.info(f"Max: {df['Target_Alpha'].max():.6f}")
        
        return df
    
    def _prepare_sequences(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Prepare sequences for training"""
        
        # Define feature columns (exclude target and metadata)
        exclude_cols = ['Date', 'Ticker', 'Target_Alpha', 'date'] + \
                      [f'Future_Return_{h}D' for h in [1, 5, 20]] + \
                      [f'Alpha_{h}D' for h in [1, 5, 20]] + \
                      [f'QQQ_Future_Return_{h}D' for h in [1, 5, 20]] + \
                      ['QQQ_Return']
        
        # Also exclude text columns that can't be converted to float
        text_keywords = ['text', 'summary', 'reasoning', 'factors', 'events', 'tickers', 
                        'sources', 'language', 'mentioned', 'extracted', 'key_factors']
        
        text_columns = []
        for col in df.columns:
            if col in ['Date', 'Ticker']:
                continue
            
            # Check if column contains text data
            if (df[col].dtype == 'object' or 
                any(keyword in col.lower() for keyword in text_keywords) or
                col.endswith('_language') or col.endswith('_sources')):
                text_columns.append(col)
                
        exclude_cols.extend(text_columns)
        
        if text_columns:
            logger.info(f"Excluding {len(text_columns)} text columns: {text_columns[:5]}...")
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # 🔒 CRITICAL: Validate no future-looking features leaked through
        forbidden_patterns = ['future', 'lead', 'target', 'next_', '_0d', 'same_day']
        leaked_features = []
        for feature_col in feature_cols:
            for pattern in forbidden_patterns:
                if pattern in feature_col.lower():
                    leaked_features.append(feature_col)
        
        if leaked_features:
            raise ValueError(f"CRITICAL: Future-looking features detected: {leaked_features}")
        
        logger.info(f"✅ Using {len(feature_cols)} validated features: {feature_cols[:10]}...")
        logger.info(f"🔒 Temporal validation passed - no future-looking features detected")
        
        sequences = []
        targets = []
        metadata = []
        
        # Group by ticker and create sequences
        for ticker in df['Ticker'].unique():
            ticker_data = df[df['Ticker'] == ticker].sort_values('Date')
            
            # Skip if not enough data
            if len(ticker_data) < self.sequence_length + self.prediction_horizon:
                continue
            
            # Create sequences for this ticker
            for i in range(len(ticker_data) - self.sequence_length - self.prediction_horizon + 1):
                # Features: sequence_length timesteps
                seq_data = ticker_data.iloc[i:i+self.sequence_length][feature_cols]
                
                # Target: alpha at prediction horizon
                target_idx = i + self.sequence_length + self.prediction_horizon - 1
                target_alpha = ticker_data.iloc[target_idx]['Target_Alpha']
                
                # Skip if target is NaN
                if pd.isna(target_alpha):
                    continue
                
                sequences.append(seq_data.values)
                targets.append(target_alpha)
                
                # Metadata
                metadata.append({
                    'ticker': ticker,
                    'date': ticker_data.iloc[target_idx]['Date'],
                    'sequence_start_date': ticker_data.iloc[i]['Date'],
                    'raw_return': ticker_data.iloc[target_idx][f'Future_Return_{self.prediction_horizon}D'],
                    'qqq_return': ticker_data.iloc[target_idx][f'QQQ_Future_Return_{self.prediction_horizon}D']
                })
        
        # Convert to arrays
        X = np.array(sequences, dtype=np.float32)
        y = np.array(targets, dtype=np.float32)
        
        # Handle NaN values in features
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize features
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_normalized = self.feature_scaler.fit_transform(X_reshaped)
        X = X_normalized.reshape(original_shape)
        
        # Apply label transformations based on mode
        if self.label_mode == 'alpha':
            # Regression: use raw alpha values
            pass
        elif self.label_mode == 'cls':
            # Classification: convert to buy/hold/sell signals
            y = self._convert_to_classification(y)
        elif self.label_mode == 'rank':
            # Ranking: will be handled during training with cross-sectional loss
            pass
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        logger.info(f"Created {len(sequences)} sequences")
        logger.info(f"Features shape: {X_tensor.shape}")
        logger.info(f"Targets shape: {y_tensor.shape}")
        
        return X_tensor, y_tensor, {
            'metadata': metadata,
            'feature_columns': feature_cols,
            'scaler': self.feature_scaler
        }
    
    def _convert_to_classification(self, alpha_values: np.ndarray) -> np.ndarray:
        """Convert alpha values to classification labels with neutral zone"""
        
        labels = np.zeros_like(alpha_values)
        
        # Buy signal: alpha > neutral_zone
        labels[alpha_values > self.neutral_zone] = 2
        
        # Sell signal: alpha < -neutral_zone  
        labels[alpha_values < -self.neutral_zone] = 0
        
        # Hold signal: within neutral zone
        labels[np.abs(alpha_values) <= self.neutral_zone] = 1
        
        logger.info(f"Classification distribution:")
        logger.info(f"Sell (0): {np.sum(labels == 0)} ({np.mean(labels == 0)*100:.1f}%)")
        logger.info(f"Hold (1): {np.sum(labels == 1)} ({np.mean(labels == 1)*100:.1f}%)")
        logger.info(f"Buy (2): {np.sum(labels == 2)} ({np.mean(labels == 2)*100:.1f}%)")
        
        return labels

class RankingLoss(torch.nn.Module):
    """Cross-sectional ranking loss for relative performance prediction"""
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                dates: torch.Tensor) -> torch.Tensor:
        """
        Compute ranking loss across stocks for each date
        
        Args:
            predictions: [N, 1] model predictions
            targets: [N, 1] true alpha values  
            dates: [N] date indices for grouping
        """
        unique_dates = torch.unique(dates)
        total_loss = 0
        valid_dates = 0
        
        for date in unique_dates:
            date_mask = dates == date
            date_preds = predictions[date_mask]
            date_targets = targets[date_mask]
            
            # Need at least 2 stocks for ranking
            if len(date_preds) < 2:
                continue
            
            # Convert to probabilities using softmax
            pred_probs = torch.softmax(date_preds / self.temperature, dim=0)
            
            # Convert targets to ranking probabilities (higher alpha = higher probability)
            target_ranks = torch.argsort(torch.argsort(date_targets, descending=True))
            target_probs = torch.softmax(-target_ranks.float() / self.temperature, dim=0)
            
            # KL divergence loss
            loss = torch.nn.functional.kl_div(
                torch.log(pred_probs + 1e-8), 
                target_probs, 
                reduction='sum'
            )
            
            total_loss += loss
            valid_dates += 1
        
        return total_loss / max(valid_dates, 1)

if __name__ == "__main__":
    # Test the data loader
    logging.basicConfig(level=logging.INFO)
    
    # Test Alpha mode
    loader = AlphaDataLoader(
        sequence_length=30,
        prediction_horizon=1,
        neutral_zone_bps=5.0,
        label_mode='alpha'
    )
    
    X, y, metadata = loader.load_data()
    print(f"Alpha regression: X={X.shape}, y={y.shape}")
    
    # Test Classification mode
    loader_cls = AlphaDataLoader(
        sequence_length=30,
        prediction_horizon=1,
        neutral_zone_bps=5.0,
        label_mode='cls'
    )
    
    X_cls, y_cls, metadata_cls = loader_cls.load_data()
    print(f"Classification: X={X_cls.shape}, y={y_cls.shape}")
    print(f"Label distribution: {np.bincount(y_cls.long().numpy())}")