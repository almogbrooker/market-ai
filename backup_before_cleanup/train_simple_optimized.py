#!/usr/bin/env python3
"""
SIMPLE OPTIMIZED MODEL TRAINING
Memory-efficient training with beta-neutral targets for maximum alpha
Focuses on practical performance over complex architectures
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from features.cross_sectional_ranking import create_beta_neutral_targets, CrossSectionalRanker
import yfinance as yf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleOptimizedLSTM(nn.Module):
    """Simplified LSTM optimized for financial time series"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention layer for better sequence modeling
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers with residual connection
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take the last time step
        last_hidden = attn_out[:, -1, :]
        
        # Prediction head
        out = self.relu(self.fc1(last_hidden))
        out = self.dropout(out)
        prediction = self.fc2(out)
        
        return prediction.squeeze()

class SimpleModelTrainer:
    """Simple, memory-efficient model trainer"""
    
    def __init__(self, model_save_dir: str = "artifacts/models/simple_optimized"):
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cpu')  # Use CPU for stability
        self.ranker = CrossSectionalRanker()
        
        logger.info(f"🚀 Simple Optimized Trainer initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Model save directory: {self.model_save_dir}")
    
    def create_simple_dataset(self, symbols: List[str], 
                            lookback_days: int = 365,
                            sequence_length: int = 20) -> Tuple:
        """Create simplified dataset with essential features"""
        logger.info("🔧 Creating simple optimized dataset...")
        
        # Generate beta-neutral targets
        beta_neutral_df = create_beta_neutral_targets(
            symbols=symbols,
            lookback_days=lookback_days,
            forward_days=1
        )
        
        if len(beta_neutral_df) == 0:
            raise ValueError("Failed to generate beta-neutral targets")
        
        # Download essential feature data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 50)
        
        feature_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval='1d')
                
                if len(df) < 50:
                    continue
                
                # Calculate essential features only
                df = self._calculate_essential_features(df)
                feature_data[symbol] = df
                
            except Exception as e:
                logger.error(f"Error downloading {symbol}: {e}")
                continue
        
        # Create sequences
        X_sequences = []
        y_targets = []
        
        # Essential feature columns (keep it simple)
        feature_columns = [
            'return_1d', 'return_5d', 'return_20d', 
            'price_vs_sma_10', 'price_vs_sma_20', 
            'rsi', 'volatility_20d', 'volume_ratio',
            'momentum_5d', 'bb_position'
        ]
        
        # Create training sequences
        for _, target_row in beta_neutral_df.iterrows():
            symbol = target_row['symbol']
            date = target_row['date']
            beta_neutral_target = target_row['beta_neutral_return']
            
            if symbol not in feature_data:
                continue
            
            df = feature_data[symbol]
            
            # Find date index
            date_mask = df.index.date == (date.date() if hasattr(date, 'date') else date)
            if not date_mask.any():
                continue
            
            date_idx = np.where(date_mask)[0][0]
            
            if date_idx < sequence_length:
                continue
            
            # Extract feature sequence
            start_idx = date_idx - sequence_length
            end_idx = date_idx
            
            feature_sequence = df.iloc[start_idx:end_idx][feature_columns].values
            
            # Handle missing data
            if np.isnan(feature_sequence).any():
                feature_sequence = pd.DataFrame(feature_sequence).fillna(method='ffill').fillna(0).values
            
            if feature_sequence.shape[0] == sequence_length:
                X_sequences.append(feature_sequence)
                y_targets.append(beta_neutral_target)
        
        X = np.array(X_sequences)
        y = np.array(y_targets)
        
        logger.info(f"✅ Simple dataset: X.shape={X.shape}, y.shape={y.shape}")
        return X, y, feature_columns
    
    def _calculate_essential_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate essential technical features only"""
        df = df.copy()
        
        # Returns
        df['return_1d'] = df['Close'].pct_change()
        df['return_5d'] = df['Close'].pct_change(5)
        df['return_20d'] = df['Close'].pct_change(20)
        
        # Moving averages
        df['sma_10'] = df['Close'].rolling(10).mean()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['price_vs_sma_10'] = df['Close'] / df['sma_10'] - 1
        df['price_vs_sma_20'] = df['Close'] / df['sma_20'] - 1
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volatility
        df['volatility_20d'] = df['return_1d'].rolling(20).std()
        
        # Volume
        df['volume_sma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        # Momentum
        df['momentum_5d'] = df['Close'] / df['Close'].shift(5) - 1
        
        # Bollinger position
        bb_middle = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        bb_upper = bb_middle + (2 * bb_std)
        bb_lower = bb_middle - (2 * bb_std)
        df['bb_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Clean data
        df = df.replace([np.inf, -np.inf], np.nan)
        return df
    
    def train_lstm_model(self, X: np.ndarray, y: np.ndarray, 
                        feature_names: List[str]) -> Dict:
        """Train simple LSTM model"""
        logger.info("🎯 Training Simple Optimized LSTM...")
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=3)
        fold_results = []
        oof_predictions = np.zeros(len(y))
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"  📁 Fold {fold + 1}/3")
            
            # Purged split
            purge_days = 5
            if len(train_idx) > purge_days:
                train_idx = train_idx[:-purge_days]
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
            X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
            
            # Create model
            model = SimpleOptimizedLSTM(
                input_size=X.shape[2],
                hidden_size=64,
                num_layers=2,
                dropout=0.3
            )
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train_scaled)
            y_train_tensor = torch.FloatTensor(y_train)
            X_val_tensor = torch.FloatTensor(X_val_scaled)
            y_val_tensor = torch.FloatTensor(y_val)
            
            # Training loop
            best_val_loss = float('inf')
            patience = 0
            max_patience = 15
            
            for epoch in range(100):
                model.train()
                
                # Forward pass
                train_pred = model(X_train_tensor)
                train_loss = criterion(train_pred, y_train_tensor)
                
                # Backward pass
                optimizer.zero_grad()
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val_tensor)
                    val_loss = criterion(val_pred, y_val_tensor)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = model.state_dict().copy()
                    patience = 0
                else:
                    patience += 1
                
                if patience >= max_patience:
                    break
            
            # Load best model and get OOF predictions
            model.load_state_dict(best_state)
            model.eval()
            
            with torch.no_grad():
                val_pred = model(X_val_tensor)
                oof_predictions[val_idx] = val_pred.numpy()
            
            # Calculate IC
            val_ic = np.corrcoef(y_val, oof_predictions[val_idx])[0, 1]
            fold_results.append({
                'fold': fold,
                'val_ic': val_ic,
                'val_loss': float(best_val_loss)
            })
            
            # Save model
            model_path = self.model_save_dir / f"simple_lstm_fold_{fold}.pt"
            scaler_path = self.model_save_dir / f"simple_lstm_fold_{fold}_scaler.pkl"
            
            torch.save(model.state_dict(), model_path)
            with open(scaler_path, 'wb') as f:
                joblib.dump(scaler, f)
            
            logger.info(f"    ✅ Fold {fold + 1}: IC={val_ic:.4f}, loss={best_val_loss:.6f}")
        
        overall_ic = np.corrcoef(y, oof_predictions)[0, 1]
        
        return {
            'model_name': 'simple_lstm',
            'overall_ic': overall_ic,
            'fold_results': fold_results,
            'oof_predictions': oof_predictions
        }
    
    def train_lightgbm_model(self, X: np.ndarray, y: np.ndarray, 
                           feature_names: List[str]) -> Dict:
        """Train LightGBM meta-model on flattened features"""
        logger.info("🎯 Training LightGBM Meta-Model...")
        
        # Flatten sequences to create tabular features
        # Take last 5 time steps and flatten
        X_flat = X[:, -5:, :].reshape(X.shape[0], -1)
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=3)
        fold_results = []
        oof_predictions = np.zeros(len(y))
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_flat)):
            # Purged split
            purge_days = 5
            if len(train_idx) > purge_days:
                train_idx = train_idx[:-purge_days]
            
            X_train, X_val = X_flat[train_idx], X_flat[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # LightGBM training
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            }
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=500,
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            # OOF predictions
            val_pred = model.predict(X_val)
            oof_predictions[val_idx] = val_pred
            
            # Calculate IC
            val_ic = np.corrcoef(y_val, val_pred)[0, 1]
            fold_results.append({
                'fold': fold,
                'val_ic': val_ic
            })
            
            # Save model
            model_path = self.model_save_dir / f"lightgbm_fold_{fold}.txt"
            model.save_model(str(model_path))
            
            logger.info(f"    ✅ Fold {fold + 1}: IC={val_ic:.4f}")
        
        overall_ic = np.corrcoef(y, oof_predictions)[0, 1]
        
        return {
            'model_name': 'lightgbm',
            'overall_ic': overall_ic,
            'fold_results': fold_results,
            'oof_predictions': oof_predictions
        }

def main():
    """Train simple optimized models"""
    
    # Focused stock universe
    symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'QCOM', 'AVGO',
        'ORCL', 'CRM', 'ADBE', 'INTC', 'TXN', 'PYPL', 'NFLX', 'CMCSA', 'PEP', 'COST'
    ]
    
    trainer = SimpleModelTrainer()
    
    try:
        # Create dataset
        X, y, feature_names = trainer.create_simple_dataset(symbols[:15])
        
        if len(X) == 0:
            raise ValueError("No training data created")
        
        # Train LSTM
        lstm_results = trainer.train_lstm_model(X, y, feature_names)
        
        # Train LightGBM
        lgb_results = trainer.train_lightgbm_model(X, y, feature_names)
        
        print("\n" + "="*60)
        print("🎯 SIMPLE OPTIMIZED MODEL RESULTS")
        print("="*60)
        print(f"🧠 LSTM IC: {lstm_results['overall_ic']:.4f}")
        print(f"🚀 LightGBM IC: {lgb_results['overall_ic']:.4f}")
        
        best_ic = max(lstm_results['overall_ic'], lgb_results['overall_ic'])
        print(f"\n🏆 BEST MODEL IC: {best_ic:.4f}")
        
        if best_ic > 0.01:
            print("✅ EXCELLENT: IC > 0.01 - Production ready!")
        elif best_ic > 0.005:
            print("✅ GOOD: IC > 0.005 - Strong alpha signal")
        else:
            print("⚠️ WEAK: IC < 0.005 - Need more features or data")
        
        print(f"\n📁 Models saved to: {trainer.model_save_dir}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()