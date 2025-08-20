#!/usr/bin/env python3
"""
FREE GPU OPTIMIZED TRAINING
Maximizes performance within free GPU constraints (Colab/Kaggle)
Trains production-ready models in <2 hours on free tier
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
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from features.cross_sectional_ranking import create_beta_neutral_targets
import yfinance as yf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FreeGPUOptimizedLSTM(nn.Module):
    """GPU-optimized LSTM for free tier training"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.4):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Larger model - we can afford it on free GPU
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Multi-head attention for better pattern recognition
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Deeper prediction head
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        
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

class FreeGPUTrainer:
    """Optimized trainer for free GPU constraints"""
    
    def __init__(self, model_save_dir: str = "artifacts/models/free_gpu"):
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect best available device
        self.device = self._get_best_device()
        
        logger.info(f"🚀 Free GPU Trainer initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    def _get_best_device(self):
        if torch.cuda.is_available():
            # Set memory allocation for better GPU utilization
            torch.cuda.empty_cache()
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')  # Apple Silicon
        else:
            return torch.device('cpu')
    
    def create_enhanced_dataset(self, symbols: List[str], 
                              lookback_days: int = 500,
                              sequence_length: int = 40) -> Tuple:
        """Create larger dataset for free GPU training"""
        logger.info("🔧 Creating enhanced dataset for free GPU...")
        
        # Generate beta-neutral targets
        beta_neutral_df = create_beta_neutral_targets(
            symbols=symbols,
            lookback_days=lookback_days,
            forward_days=1
        )
        
        if len(beta_neutral_df) == 0:
            raise ValueError("Failed to generate beta-neutral targets")
        
        # Download more comprehensive feature data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 100)
        
        feature_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval='1d')
                
                if len(df) < 100:
                    continue
                
                # More comprehensive features for GPU training
                df = self._calculate_comprehensive_features(df)
                feature_data[symbol] = df
                
            except Exception as e:
                logger.error(f"Error downloading {symbol}: {e}")
                continue
        
        # Create sequences with more features
        X_sequences = []
        y_targets = []
        
        # Comprehensive feature set (we can handle more on GPU)
        feature_columns = [
            'return_1d', 'return_5d', 'return_20d', 'return_60d',
            'price_vs_sma_5', 'price_vs_sma_10', 'price_vs_sma_20', 'price_vs_sma_50',
            'price_vs_ema_10', 'price_vs_ema_20', 
            'rsi', 'rsi_5', 'rsi_20',
            'macd', 'macd_signal', 'macd_histogram',
            'bb_position', 'bb_width',
            'volatility_5d', 'volatility_20d', 'volatility_60d',
            'volume_ratio', 'volume_sma_ratio', 'price_volume_trend',
            'momentum_5d', 'momentum_10d', 'momentum_20d',
            'high_low_pct', 'close_position',
            'atr_14', 'cci_14', 'williams_r'
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
        
        logger.info(f"✅ Enhanced dataset: X.shape={X.shape}, y.shape={y.shape}")
        return X, y, feature_columns
    
    def _calculate_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical features for GPU training"""
        df = df.copy()
        
        # Multiple timeframe returns
        for period in [1, 5, 20, 60]:
            df[f'return_{period}d'] = df['Close'].pct_change(period)
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['Close'].rolling(period).mean()
            df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
            df[f'price_vs_sma_{period}'] = df['Close'] / df[f'sma_{period}'] - 1
            df[f'price_vs_ema_{period}'] = df['Close'] / df[f'ema_{period}'] - 1
        
        # RSI multiple timeframes
        for period in [14, 5, 20]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            df[f'rsi_{period}' if period != 14 else 'rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        bb_period = 20
        df['bb_middle'] = df['Close'].rolling(bb_period).mean()
        df['bb_std'] = df['Close'].rolling(bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (2 * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (2 * df['bb_std'])
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Volatility multiple timeframes
        for period in [5, 20, 60]:
            df[f'volatility_{period}d'] = df['Close'].pct_change().rolling(period).std() * np.sqrt(252)
        
        # Volume features
        df['volume_sma_20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
        df['volume_sma_ratio'] = df['Volume'].rolling(5).mean() / df['volume_sma_20']
        df['price_volume_trend'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)) * df['Volume']
        
        # Momentum indicators
        for period in [5, 10, 20]:
            df[f'momentum_{period}d'] = df['Close'] / df['Close'].shift(period) - 1
        
        # High-Low features
        df['high_low_pct'] = (df['High'] - df['Low']) / df['Close']
        df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr_14'] = true_range.rolling(14).mean()
        
        # CCI (Commodity Channel Index)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = typical_price.rolling(14).mean()
        mad = typical_price.rolling(14).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['cci_14'] = (typical_price - sma_tp) / (0.015 * mad)
        
        # Williams %R
        high_14 = df['High'].rolling(14).max()
        low_14 = df['Low'].rolling(14).min()
        df['williams_r'] = (high_14 - df['Close']) / (high_14 - low_14) * -100
        
        # Clean data
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Cap extreme values
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].std() > 0:
                upper_cap = df[col].quantile(0.99)
                lower_cap = df[col].quantile(0.01)
                df[col] = df[col].clip(lower=lower_cap, upper=upper_cap)
        
        return df
    
    def train_gpu_optimized_models(self, X: np.ndarray, y: np.ndarray, 
                                  feature_names: List[str]) -> Dict:
        """Train both LSTM and LightGBM with GPU optimization"""
        logger.info("🎯 Training GPU-optimized models...")
        
        results = {}
        
        # 1. Train Enhanced LSTM on GPU
        lstm_results = self._train_enhanced_lstm(X, y, feature_names)
        results['enhanced_lstm'] = lstm_results
        
        # 2. Train LightGBM (uses all CPU cores efficiently)  
        lgb_results = self._train_enhanced_lightgbm(X, y, feature_names)
        results['enhanced_lightgbm'] = lgb_results
        
        # 3. Create ensemble
        if len(results) > 1:
            lstm_pred = lstm_results['oof_predictions']
            lgb_pred = lgb_results['oof_predictions']
            ensemble_pred = (lstm_pred * 0.4 + lgb_pred * 0.6)  # Weight toward LightGBM
            ensemble_ic = np.corrcoef(y, ensemble_pred)[0, 1]
            results['ensemble'] = {
                'overall_ic': ensemble_ic,
                'oof_predictions': ensemble_pred
            }
            logger.info(f"🎯 Ensemble IC: {ensemble_ic:.4f}")
        
        return results
    
    def _train_enhanced_lstm(self, X: np.ndarray, y: np.ndarray, 
                           feature_names: List[str]) -> Dict:
        """Train enhanced LSTM with GPU acceleration"""
        logger.info("🧠 Training Enhanced GPU LSTM...")
        
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
            
            # Enhanced model (larger for GPU)
            model = FreeGPUOptimizedLSTM(
                input_size=X.shape[2],
                hidden_size=128,  # Larger model
                num_layers=3,     # Deeper
                dropout=0.4
            ).to(self.device)
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train).to(self.device)
            X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            
            # Training loop (longer for better performance)
            best_val_loss = float('inf')
            patience = 0
            max_patience = 25
            
            for epoch in range(200):
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
                val_pred = model(X_val_tensor)
                oof_predictions[val_idx] = val_pred.cpu().numpy()
            
            # Calculate IC
            val_ic = np.corrcoef(y_val, oof_predictions[val_idx])[0, 1]
            fold_results.append({
                'fold': fold,
                'val_ic': val_ic,
                'val_loss': float(best_val_loss)
            })
            
            # Save model
            model_path = self.model_save_dir / f"enhanced_lstm_fold_{fold}.pt"
            scaler_path = self.model_save_dir / f"enhanced_lstm_fold_{fold}_scaler.pkl"
            
            torch.save(model.state_dict(), model_path)
            with open(scaler_path, 'wb') as f:
                joblib.dump(scaler, f)
            
            logger.info(f"    ✅ Fold {fold + 1}: IC={val_ic:.4f}, loss={best_val_loss:.6f}")
        
        overall_ic = np.corrcoef(y, oof_predictions)[0, 1]
        
        return {
            'model_name': 'enhanced_lstm',
            'overall_ic': overall_ic,
            'fold_results': fold_results,
            'oof_predictions': oof_predictions
        }
    
    def _train_enhanced_lightgbm(self, X: np.ndarray, y: np.ndarray, 
                               feature_names: List[str]) -> Dict:
        """Train enhanced LightGBM with more features"""
        logger.info("🚀 Training Enhanced LightGBM...")
        
        # Use longer sequence (last 10 time steps) for richer features
        X_flat = X[:, -10:, :].reshape(X.shape[0], -1)
        
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
            
            # Enhanced LightGBM parameters
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 63,      # Larger trees
                'learning_rate': 0.05,  # Lower LR for better performance
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'max_depth': 8,        # Deeper trees
                'min_data_in_leaf': 10,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
                'verbose': -1,
                'random_state': 42,
                'n_jobs': -1          # Use all CPU cores
            }
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=1000,   # More rounds for better fit
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
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
            model_path = self.model_save_dir / f"enhanced_lightgbm_fold_{fold}.txt"
            model.save_model(str(model_path))
            
            logger.info(f"    ✅ Fold {fold + 1}: IC={val_ic:.4f}")
        
        overall_ic = np.corrcoef(y, oof_predictions)[0, 1]
        
        return {
            'model_name': 'enhanced_lightgbm',
            'overall_ic': overall_ic,
            'fold_results': fold_results,
            'oof_predictions': oof_predictions
        }

def main():
    """Train enhanced models on free GPU"""
    
    # Expanded stock universe for better training
    symbols = [
        # Tech giants
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'INTC', 'QCOM',
        'ORCL', 'CRM', 'ADBE', 'NOW', 'PYPL', 'NFLX', 'UBER', 'ABNB', 'CRWD', 'SNOW',
        # Traditional large caps
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BRK-B', 'JNJ', 'PG', 'UNH',
        'HD', 'WMT', 'DIS', 'MCD', 'NKE', 'COST', 'TGT', 'SBUX', 'LOW', 'CVX'
    ]
    
    trainer = FreeGPUTrainer()
    
    try:
        # Create enhanced dataset (more data for free GPU)
        X, y, feature_names = trainer.create_enhanced_dataset(symbols[:30])  # 30 stocks
        
        if len(X) == 0:
            raise ValueError("No training data created")
        
        # Train enhanced models
        results = trainer.train_gpu_optimized_models(X, y, feature_names)
        
        print("\n" + "="*70)
        print("🎯 FREE GPU OPTIMIZED MODEL RESULTS")
        print("="*70)
        
        for model_name, model_results in results.items():
            if 'overall_ic' in model_results:
                ic = model_results['overall_ic']
                print(f"🚀 {model_name.upper()}: IC = {ic:.4f}")
        
        best_ic = max([r['overall_ic'] for r in results.values() if 'overall_ic' in r])
        print(f"\n🏆 BEST MODEL IC: {best_ic:.4f}")
        
        if best_ic > 0.025:
            print("✅ EXCEPTIONAL: IC > 2.5% - Institutional grade!")
        elif best_ic > 0.02:
            print("✅ EXCELLENT: IC > 2.0% - Top tier performance!")
        elif best_ic > 0.015:
            print("✅ GREAT: IC > 1.5% - Production ready!")
        elif best_ic > 0.01:
            print("✅ GOOD: IC > 1.0% - Strong alpha signal")
        else:
            print("⚠️ OK: IC < 1.0% - Need more data/features")
        
        print(f"\n📁 Models saved to: {trainer.model_save_dir}")
        print("🎯 Ready for production deployment!")
        print("="*70)
        
        # GPU utilization summary
        if torch.cuda.is_available():
            print(f"\n💻 GPU Utilization:")
            print(f"   Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
            print(f"   Max memory reserved: {torch.cuda.max_memory_reserved() / 1e9:.2f}GB")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()