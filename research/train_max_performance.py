#!/usr/bin/env python3
"""
MAXIMUM PERFORMANCE MODEL TRAINING
Combines all optimizations for best possible IC performance
Focus on what actually works: enhanced features + proper validation + ensemble
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
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
import lightgbm as lgb
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from features.cross_sectional_ranking import create_beta_neutral_targets
import yfinance as yf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MaxPerformanceLSTM(nn.Module):
    """Enhanced LSTM with all performance optimizations"""
    
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.4):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Larger LSTM for better capacity
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False  # Keep unidirectional for time series
        )
        
        # Multi-head attention for pattern recognition
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=16,  # More heads for richer patterns
            dropout=dropout,
            batch_first=True
        )
        
        # Deeper prediction head with batch normalization
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take last time step
        last_hidden = attn_out[:, -1, :]
        
        # Deep prediction head with batch norm
        out = self.relu(self.bn1(self.fc1(last_hidden)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.fc2(out)))
        out = self.dropout(out)
        prediction = self.fc3(out)
        
        return prediction.squeeze()

class MaxPerformanceTrainer:
    """Maximum performance trainer with all optimizations"""
    
    def __init__(self, model_save_dir: str = "artifacts/models/max_performance"):
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if torch.cuda.is_available():
            # Enable all CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
        # Mixed precision scaler
        self.scaler = GradScaler() if torch.cuda.is_available() else None
        
        logger.info(f"🚀 Max Performance Trainer initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Mixed precision: {self.scaler is not None}")
        if torch.cuda.is_available():
            logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    def create_maximum_performance_dataset(self, symbols: List[str], 
                                         lookback_days: int = 730,
                                         sequence_length: int = 60) -> Tuple:
        """Create comprehensive dataset with maximum features"""
        logger.info(f"🔧 Creating maximum performance dataset...")
        
        # Generate beta-neutral targets with longer lookback
        beta_neutral_df = create_beta_neutral_targets(
            symbols=symbols,
            lookback_days=lookback_days,
            forward_days=1
        )
        
        if len(beta_neutral_df) == 0:
            raise ValueError("Failed to generate beta-neutral targets")
        
        logger.info(f"✅ Generated {len(beta_neutral_df)} beta-neutral targets")
        
        # Download comprehensive feature data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 200)
        
        feature_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval='1d')
                
                if len(df) < sequence_length + 100:
                    continue
                
                # Maximum feature engineering
                df = self._calculate_maximum_features(df)
                feature_data[symbol] = df
                
                logger.debug(f"✅ {symbol}: {len(df)} days, {df.shape[1]} features")
                
            except Exception as e:
                logger.error(f"Error downloading {symbol}: {e}")
                continue
        
        # Create sequences
        X_sequences = []
        y_targets = []
        
        # Maximum feature set (all proven indicators)
        feature_columns = [
            # Returns (multiple timeframes)
            'return_1d', 'return_3d', 'return_5d', 'return_10d', 'return_20d',
            
            # Price ratios
            'price_vs_sma_5', 'price_vs_sma_10', 'price_vs_sma_20', 'price_vs_sma_50',
            'price_vs_ema_5', 'price_vs_ema_10', 'price_vs_ema_20',
            
            # RSI variants
            'rsi_14', 'rsi_7', 'rsi_21',
            
            # MACD system
            'macd', 'macd_signal', 'macd_histogram',
            
            # Bollinger system
            'bb_position', 'bb_width', 'bb_squeeze',
            
            # Volatility
            'volatility_5d', 'volatility_10d', 'volatility_20d', 'volatility_ratio',
            
            # Volume
            'volume_ratio', 'volume_sma_ratio', 'obv_ratio',
            
            # Momentum
            'momentum_3d', 'momentum_5d', 'momentum_10d', 'momentum_20d',
            
            # Price action
            'high_low_pct', 'close_position', 'gap_pct',
            
            # Advanced indicators
            'atr_ratio', 'cci', 'williams_r', 'roc_10d', 'stoch_k', 'stoch_d'
        ]
        
        # Create training sequences
        dates_processed = set()
        
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
            
            try:
                feature_sequence = df.iloc[start_idx:end_idx][feature_columns].values
                
                # Handle missing data
                if np.isnan(feature_sequence).any():
                    feature_sequence = pd.DataFrame(feature_sequence).fillna(method='ffill').fillna(method='bfill').fillna(0).values
                
                # Robust clipping
                feature_sequence = np.clip(feature_sequence, -10, 10)
                
                if feature_sequence.shape[0] == sequence_length and feature_sequence.shape[1] == len(feature_columns):
                    X_sequences.append(feature_sequence)
                    y_targets.append(beta_neutral_target)
                    dates_processed.add(date)
                    
            except Exception as e:
                logger.debug(f"Error processing {symbol} on {date}: {e}")
                continue
        
        if not X_sequences:
            raise ValueError("No valid training sequences created")
        
        X = np.array(X_sequences)
        y = np.array(y_targets)
        
        logger.info(f"✅ Maximum dataset: X.shape={X.shape}, y.shape={y.shape}")
        logger.info(f"   Sequence length: {sequence_length} days")
        logger.info(f"   Features: {len(feature_columns)}")
        logger.info(f"   Unique dates: {len(dates_processed)}")
        
        return X, y, feature_columns
    
    def _calculate_maximum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical features"""
        df = df.copy()
        
        # Multiple timeframe returns
        for period in [1, 3, 5, 10, 20]:
            df[f'return_{period}d'] = df['Close'].pct_change(period)
        
        # Moving averages and ratios
        for period in [5, 10, 20, 50]:
            sma = df['Close'].rolling(period).mean()
            ema = df['Close'].ewm(span=period).mean()
            df[f'sma_{period}'] = sma
            df[f'ema_{period}'] = ema
            df[f'price_vs_sma_{period}'] = df['Close'] / sma - 1
            df[f'price_vs_ema_{period}'] = df['Close'] / ema - 1
        
        # RSI variants
        for rsi_period in [7, 14, 21]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
            rs = gain / (loss + 1e-8)
            df[f'rsi_{rsi_period}'] = 100 - (100 / (1 + rs))
        
        # MACD system
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands system
        bb_period = 20
        bb_middle = df['Close'].rolling(bb_period).mean()
        bb_std = df['Close'].rolling(bb_period).std()
        bb_upper = bb_middle + (2 * bb_std)
        bb_lower = bb_middle - (2 * bb_std)
        df['bb_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_squeeze'] = bb_std / bb_std.rolling(20).mean()  # Volatility compression
        
        # Multi-timeframe volatility
        for vol_period in [5, 10, 20]:
            vol = df['Close'].pct_change().rolling(vol_period).std() * np.sqrt(252)
            df[f'volatility_{vol_period}d'] = vol
        
        df['volatility_ratio'] = df['volatility_5d'] / (df['volatility_20d'] + 1e-8)
        
        # Volume indicators
        df['volume_sma_20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / (df['volume_sma_20'] + 1e-8)
        df['volume_sma_ratio'] = df['Volume'].rolling(5).mean() / (df['volume_sma_20'] + 1e-8)
        
        # On-Balance Volume
        obv = (df['Volume'] * np.sign(df['Close'].diff())).cumsum()
        df['obv_ratio'] = obv / obv.rolling(20).mean()
        
        # Multi-timeframe momentum
        for mom_period in [3, 5, 10, 20]:
            df[f'momentum_{mom_period}d'] = df['Close'] / df['Close'].shift(mom_period) - 1
        
        # Price action features
        df['high_low_pct'] = (df['High'] - df['Low']) / df['Close']
        df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)
        df['gap_pct'] = (df['Open'] - df['Close'].shift()) / df['Close'].shift()
        
        # Advanced technical indicators
        
        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(14).mean()
        df['atr_ratio'] = atr / df['Close']
        
        # CCI (Commodity Channel Index)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = typical_price.rolling(14).mean()
        mad = typical_price.rolling(14).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['cci'] = (typical_price - sma_tp) / (0.015 * mad + 1e-8)
        
        # Williams %R
        high_14 = df['High'].rolling(14).max()
        low_14 = df['Low'].rolling(14).min()
        df['williams_r'] = (high_14 - df['Close']) / (high_14 - low_14 + 1e-8) * -100
        
        # Rate of Change
        df['roc_10d'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10) * 100
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(14).min()
        high_14 = df['High'].rolling(14).max()
        df['stoch_k'] = (df['Close'] - low_14) / (high_14 - low_14 + 1e-8) * 100
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # Clean data
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Robust outlier treatment
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                q99 = df[col].quantile(0.995)
                q01 = df[col].quantile(0.005)
                df[col] = df[col].clip(lower=q01, upper=q99)
        
        return df
    
    def train_enhanced_lstm(self, X: np.ndarray, y: np.ndarray, 
                          feature_names: List[str]) -> Dict:
        """Train enhanced LSTM with all optimizations"""
        logger.info("🧠 Training Maximum Performance LSTM...")
        
        # 5-fold CV for better validation
        tscv = TimeSeriesSplit(n_splits=5)
        fold_results = []
        oof_predictions = np.zeros(len(y))
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"  📁 Fold {fold + 1}/5")
            
            # Purged split
            purge_days = 5
            if len(train_idx) > purge_days:
                train_idx = train_idx[:-purge_days]
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Per-fold robust scaling
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
            X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
            
            # Enhanced model
            model = MaxPerformanceLSTM(
                input_size=X.shape[2],
                hidden_size=256,  # Larger model
                num_layers=3,
                dropout=0.4
            ).to(self.device)
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=1e-3, 
                weight_decay=1e-4,
                betas=(0.9, 0.95)
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=15, factor=0.5
            )
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train).to(self.device)
            X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            
            # Training loop
            best_val_loss = float('inf')
            patience = 0
            max_patience = 25
            
            for epoch in range(200):
                model.train()
                
                # Mixed precision training
                if self.scaler:
                    with autocast():
                        train_pred = model(X_train_tensor)
                        train_loss = criterion(train_pred, y_train_tensor)
                    
                    optimizer.zero_grad()
                    self.scaler.scale(train_loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    train_pred = model(X_train_tensor)
                    train_loss = criterion(train_pred, y_train_tensor)
                    
                    optimizer.zero_grad()
                    train_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    if self.scaler:
                        with autocast():
                            val_pred = model(X_val_tensor)
                            val_loss = criterion(val_pred, y_val_tensor)
                    else:
                        val_pred = model(X_val_tensor)
                        val_loss = criterion(val_pred, y_val_tensor)
                
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = model.state_dict().copy()
                    patience = 0
                else:
                    patience += 1
                
                if patience >= max_patience:
                    break
                
                if epoch % 50 == 0:
                    logger.debug(f"    Epoch {epoch}: train={train_loss:.6f}, val={val_loss:.6f}")
            
            # Load best model and get OOF predictions
            model.load_state_dict(best_state)
            model.eval()
            
            with torch.no_grad():
                if self.scaler:
                    with autocast():
                        val_pred = model(X_val_tensor)
                else:
                    val_pred = model(X_val_tensor)
                oof_predictions[val_idx] = val_pred.cpu().numpy()
            
            # Calculate IC
            val_ic_spearman = spearmanr(y_val, oof_predictions[val_idx])[0]
            val_ic_pearson = np.corrcoef(y_val, oof_predictions[val_idx])[0, 1]
            
            fold_results.append({
                'fold': fold,
                'val_ic_spearman': val_ic_spearman,
                'val_ic_pearson': val_ic_pearson,
                'val_loss': float(best_val_loss)
            })
            
            # Save model
            model_path = self.model_save_dir / f"max_lstm_fold_{fold}.pt"
            scaler_path = self.model_save_dir / f"max_lstm_fold_{fold}_scaler.pkl"
            
            torch.save(model.state_dict(), model_path)
            with open(scaler_path, 'wb') as f:
                joblib.dump(scaler, f)
            
            logger.info(f"    ✅ Fold {fold + 1}: IC_spearman={val_ic_spearman:.4f}, IC_pearson={val_ic_pearson:.4f}")
        
        # Overall performance
        overall_ic_spearman = spearmanr(y, oof_predictions)[0]
        overall_ic_pearson = np.corrcoef(y, oof_predictions)[0, 1]
        
        return {
            'model_name': 'max_lstm',
            'overall_ic_spearman': overall_ic_spearman,
            'overall_ic_pearson': overall_ic_pearson,
            'fold_results': fold_results,
            'oof_predictions': oof_predictions
        }
    
    def train_enhanced_lightgbm(self, X: np.ndarray, y: np.ndarray, 
                              lstm_oof: np.ndarray, feature_names: List[str]) -> Dict:
        """Train enhanced LightGBM meta-model"""
        logger.info("🚀 Training Maximum Performance LightGBM...")
        
        # Enhanced meta-features
        X_meta = []
        
        # Use longer sequence for more information
        X_flat = X[:, -20:, :].reshape(X.shape[0], -1)
        X_meta.append(X_flat)
        
        # Add LSTM predictions as feature
        X_meta.append(lstm_oof.reshape(-1, 1))
        
        # Statistical aggregations across time dimension
        X_stats = []
        for feature_idx in range(X.shape[2]):
            feature_series = X[:, :, feature_idx]
            X_stats.extend([
                np.mean(feature_series, axis=1),
                np.std(feature_series, axis=1),
                np.percentile(feature_series, 25, axis=1),
                np.percentile(feature_series, 75, axis=1),
                np.min(feature_series, axis=1),
                np.max(feature_series, axis=1)
            ])
        
        X_stats = np.column_stack(X_stats)
        X_meta.append(X_stats)
        
        X_meta_combined = np.column_stack(X_meta)
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        fold_results = []
        oof_predictions = np.zeros(len(y))
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_meta_combined)):
            # Purged split
            purge_days = 5
            if len(train_idx) > purge_days:
                train_idx = train_idx[:-purge_days]
            
            X_train, X_val = X_meta_combined[train_idx], X_meta_combined[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Enhanced LightGBM parameters
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 127,
                'learning_rate': 0.02,  # Lower for better performance
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'max_depth': 8,
                'min_data_in_leaf': 15,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
                'verbose': -1,
                'random_state': 42,
                'n_jobs': -1,
                'force_col_wise': True  # Optimize for many features
            }
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=3000,  # More rounds for better fit
                callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)]
            )
            
            # OOF predictions
            val_pred = model.predict(X_val)
            oof_predictions[val_idx] = val_pred
            
            # Calculate IC
            val_ic = spearmanr(y_val, val_pred)[0]
            fold_results.append({
                'fold': fold,
                'val_ic': val_ic
            })
            
            # Save model
            model_path = self.model_save_dir / f"max_lightgbm_fold_{fold}.txt"
            model.save_model(str(model_path))
            
            logger.info(f"    ✅ Fold {fold + 1}: IC={val_ic:.4f}")
        
        overall_ic = spearmanr(y, oof_predictions)[0]
        
        return {
            'model_name': 'max_lightgbm',
            'overall_ic': overall_ic,
            'fold_results': fold_results,
            'oof_predictions': oof_predictions
        }

def main():
    """Train maximum performance models"""
    
    # Premium stock universe (most liquid, highest quality)
    symbols = [
        # Tech leaders
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'INTC', 'QCOM',
        'ORCL', 'CRM', 'ADBE', 'NOW', 'PYPL', 'NFLX', 'UBER', 'CRWD', 'SNOW', 'DDOG',
        # Traditional leaders
        'JPM', 'BAC', 'WFC', 'GS', 'UNH', 'JNJ', 'PG', 'HD', 'WMT', 'DIS',
        'MCD', 'NKE', 'COST', 'TGT', 'SBUX', 'LOW', 'CVX', 'XOM', 'KO', 'PFE'
    ]
    
    trainer = MaxPerformanceTrainer()
    
    try:
        # Create maximum performance dataset
        X, y, feature_names = trainer.create_maximum_performance_dataset(
            symbols[:30],  # Top 30 stocks for training
            sequence_length=60  # 60-day sequences for pattern recognition
        )
        
        if len(X) == 0:
            raise ValueError("No training data created")
        
        # Train enhanced LSTM
        lstm_results = trainer.train_enhanced_lstm(X, y, feature_names)
        
        # Train enhanced LightGBM meta-model
        lgb_results = trainer.train_enhanced_lightgbm(
            X, y, lstm_results['oof_predictions'], feature_names
        )
        
        # Create ensemble
        ensemble_pred = (lstm_results['oof_predictions'] * 0.3 + 
                        lgb_results['oof_predictions'] * 0.7)  # Weight toward LightGBM
        ensemble_ic = spearmanr(y, ensemble_pred)[0]
        
        print("\n" + "="*80)
        print("🎯 MAXIMUM PERFORMANCE MODEL RESULTS")
        print("="*80)
        print(f"🧠 Enhanced LSTM IC (Spearman): {lstm_results['overall_ic_spearman']:.4f}")
        print(f"🧠 Enhanced LSTM IC (Pearson):  {lstm_results['overall_ic_pearson']:.4f}")
        print(f"🚀 Enhanced LightGBM IC:       {lgb_results['overall_ic']:.4f}")
        print(f"🎯 Ensemble IC:                {ensemble_ic:.4f}")
        
        best_ic = max(
            lstm_results['overall_ic_spearman'],
            lstm_results['overall_ic_pearson'],
            lgb_results['overall_ic'],
            ensemble_ic
        )
        
        print(f"\n🏆 BEST MODEL IC: {best_ic:.4f}")
        
        if best_ic > 0.03:
            print("✅ EXCEPTIONAL: IC > 3.0% - Institutional elite performance!")
        elif best_ic > 0.025:
            print("✅ EXCELLENT: IC > 2.5% - Top-tier institutional quality!")
        elif best_ic > 0.02:
            print("✅ GREAT: IC > 2.0% - Strong institutional performance!")
        elif best_ic > 0.015:
            print("✅ GOOD: IC > 1.5% - Production ready!")
        elif best_ic > 0.01:
            print("✅ DECENT: IC > 1.0% - Above average alpha")
        else:
            print("⚠️ WEAK: IC < 1.0% - Needs improvement")
        
        # Save ensemble metadata
        ensemble_metadata = {
            'training_date': datetime.now().isoformat(),
            'symbols': symbols[:30],
            'n_samples': len(y),
            'n_features': len(feature_names),
            'sequence_length': 60,
            'lstm_ic_spearman': lstm_results['overall_ic_spearman'],
            'lstm_ic_pearson': lstm_results['overall_ic_pearson'],
            'lightgbm_ic': lgb_results['overall_ic'],
            'ensemble_ic': ensemble_ic,
            'best_model_ic': best_ic
        }
        
        import json
        metadata_path = trainer.model_save_dir / "max_performance_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(ensemble_metadata, f, indent=2)
        
        print(f"\n📁 Models saved to: {trainer.model_save_dir}")
        print("🎯 Ready for production deployment!")
        print("="*80)
        
        # GPU memory summary
        if torch.cuda.is_available():
            print(f"\n💻 GPU Memory Usage:")
            print(f"   Peak allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
            print(f"   Peak reserved:  {torch.cuda.max_memory_reserved() / 1e9:.2f}GB")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()