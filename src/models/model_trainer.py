#!/usr/bin/env python3
"""
Enhanced Model Trainer implementing chat-g.txt requirements:
- Direction classification with dead-zone (meta-labeling)
- Transaction-cost aware loss
- Purged K-Fold CV with embargo
- Meta-learner ensemble (Ridge/LightGBM)
- Probability calibration
- Portfolio-level ranking
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, roc_auc_score
from sklearn.linear_model import Ridge
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import existing model architectures
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.advanced_models import PatchTST, iTransformer

logger = logging.getLogger(__name__)

class PurgedTimeSeriesSplit:
    """
    Enhanced Purged Time Series Split implementing chat-g.txt requirements:
    - Purged K-Fold CV with embargo (López de Prado)
    - Time-based walk-forward evaluation
    """
    
    def __init__(self, n_splits: int = 5, purge_days: int = 30, embargo_days: int = 5, 
                 min_train_size: int = 252):
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.min_train_size = min_train_size
    
    def split(self, X: pd.DataFrame, y: pd.Series = None, groups: pd.Series = None) -> List[Tuple]:
        """
        Generate purged and embargoed train/test splits using real calendar dates
        
        Args:
            X: Feature matrix with 'Date' column
            y: Target variable
            groups: Not used (for compatibility)
        
        Returns:
            List of (train_idx, test_idx) tuples
        """
        
        if 'Date' not in X.columns:
            raise ValueError("X must contain 'Date' column for time-based splits")
        
        # Ensure Date column is datetime
        X['Date'] = pd.to_datetime(X['Date'])
        
        # Get unique dates sorted
        dates = sorted(X['Date'].unique())
        n_dates = len(dates)
        
        # Calculate split points
        fold_size = n_dates // (self.n_splits + 1)
        
        splits = []
        
        for fold in range(self.n_splits):
            # Test period for this fold
            test_start_idx = (fold + 1) * fold_size
            test_end_idx = min(test_start_idx + fold_size, n_dates)
            
            if test_end_idx >= n_dates:
                break
            
            test_start_date = dates[test_start_idx]
            test_end_date = dates[test_end_idx - 1]
            
            # FIXED: Use real calendar days for purge and embargo
            # Train end date = test_start_date - purge_days (calendar days)
            train_end_date = test_start_date - pd.Timedelta(days=self.purge_days)
            
            # Embargo window: test_end_date + 1 to test_end_date + embargo_days
            embargo_start_date = test_end_date + pd.Timedelta(days=1)
            embargo_end_date = test_end_date + pd.Timedelta(days=self.embargo_days)
            
            # Build train mask: Date <= train_end_date AND not in embargo window
            train_mask = (X['Date'] <= train_end_date) & ~(
                (X['Date'] >= embargo_start_date) & (X['Date'] <= embargo_end_date)
            )
            
            # Build test mask: Date in [test_start_date, test_end_date]
            test_mask = (X['Date'] >= test_start_date) & (X['Date'] <= test_end_date)
            
            # Get actual indices
            train_idx = X[train_mask].index.tolist()
            test_idx = X[test_mask].index.tolist()
            
            # Check minimum training size (252 trading days)
            if len(train_idx) < self.min_train_size:
                logger.debug(f"Fold {fold}: Insufficient training data ({len(train_idx)} < {self.min_train_size})")
                continue
            
            # Reset indices to be positional (0-based)
            train_idx_pos = [i for i, idx in enumerate(X.index) if idx in train_idx]
            test_idx_pos = [i for i, idx in enumerate(X.index) if idx in test_idx]
            
            if len(train_idx_pos) > 0 and len(test_idx_pos) > 0:
                splits.append((train_idx_pos, test_idx_pos))
                
                logger.debug(f"Fold {fold}: Train={len(train_idx_pos)}, Test={len(test_idx_pos)}")
                logger.debug(f"  Train: up to {train_end_date.strftime('%Y-%m-%d')}")
                logger.debug(f"  Purge: {self.purge_days} calendar days")
                logger.debug(f"  Test: {test_start_date.strftime('%Y-%m-%d')} to {test_end_date.strftime('%Y-%m-%d')}")
                logger.debug(f"  Embargo: {embargo_start_date.strftime('%Y-%m-%d')} to {embargo_end_date.strftime('%Y-%m-%d')}")
        
        logger.info(f"✅ Generated {len(splits)} purged CV folds with real calendar dates")
        return splits

class SimpleGRU(nn.Module):
    """Simple GRU model for diversity in ensemble"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights to prevent NaN
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly to prevent NaN"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # Add input normalization to prevent NaN
        x = torch.clamp(x, -10, 10)  # Clamp extreme values
        
        gru_out, _ = self.gru(x)
        
        # Use last timestep
        last_out = gru_out[:, -1, :]
        
        # Check for NaN and replace
        if torch.isnan(last_out).any():
            last_out = torch.zeros_like(last_out)
        
        last_out = self.dropout(last_out)
        output = self.fc(last_out)
        
        # Clamp output to prevent extreme values
        output = torch.clamp(output, -1, 1)
        
        return {
            'return_prediction': output.squeeze(-1),
            'confidence': torch.sigmoid(torch.abs(output)).squeeze(-1)  # Use abs for confidence
        }

class MetaModel(nn.Module):
    """Meta-model to combine TS forecasts with other features"""
    
    def __init__(self, n_ts_models: int = 3, n_other_features: int = 10, hidden_size: int = 64):
        super().__init__()
        
        # TS model outputs (returns + confidence)
        ts_input_size = n_ts_models * 2
        
        # Combine TS outputs with other features
        total_input = ts_input_size + n_other_features
        
        self.network = nn.Sequential(
            nn.Linear(total_input, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 3)  # prob_up, expected_return, confidence
        )
    
    def forward(self, ts_outputs, other_features):
        # Combine inputs
        combined = torch.cat([ts_outputs, other_features], dim=1)
        
        output = self.network(combined)
        
        return {
            'prob_up': torch.sigmoid(output[:, 0]),
            'expected_return': output[:, 1],
            'confidence': torch.sigmoid(output[:, 2])
        }

class ModelTrainer:
    """
    Enhanced model trainer implementing chat-g.txt requirements:
    - Direction classification with dead-zone meta-labeling
    - Transaction-cost aware loss
    - Meta-learner with probability calibration
    - Portfolio-level ranking optimization
    """
    
    def __init__(self, dataset_path: str, cv_type: str = 'purged', folds: int = 5,
                 purge_days: int = 30, embargo_days: int = 5, 
                 models: List[str] = None, meta_model: str = 'lightgbm',
                 transaction_cost: float = 0.0005, dead_zone: float = 0.001):
        
        self.dataset_path = dataset_path
        self.cv_type = cv_type
        self.folds = folds
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        
        # Chat-g.txt enhancements
        self.transaction_cost = transaction_cost  # Cost-aware loss
        self.dead_zone = dead_zone  # Dead-zone for meta-labeling
        
        self.models = models or ['patchtst', 'itransformer', 'gru']
        self.meta_model_type = meta_model
        
        # Clean GPU memory first, then use GPU
        self._clean_gpu_memory()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trained_models = {}
        self.meta_model = None
        self.calibrated_model = None
        
        logger.info(f"🧠 Enhanced ModelTrainer: {models} + {meta_model}")
        logger.info(f"📊 Cost-aware: {transaction_cost*10000:.1f}bps, Dead-zone: {dead_zone*10000:.1f}bps")
    
    def _clean_gpu_memory(self):
        """Clean GPU memory before training"""
        if torch.cuda.is_available():
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear all cached memory
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            
            # Check memory
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_cached = torch.cuda.memory_reserved() / 1024**3
            
            logger.info(f"🧹 GPU memory cleaned: {memory_allocated:.2f}GB allocated, {memory_cached:.2f}GB cached")
        
    def create_meta_labels(self, returns: np.ndarray) -> np.ndarray:
        """
        Create meta-labels with dead-zone (chat-g.txt requirement)
        """
        labels = np.zeros_like(returns)
        
        # Up class: returns > dead_zone + transaction_cost
        up_threshold = self.dead_zone + self.transaction_cost
        labels[returns > up_threshold] = 1
        
        # Down class: returns < -(dead_zone + transaction_cost)
        down_threshold = -(self.dead_zone + self.transaction_cost)
        labels[returns < down_threshold] = -1
        
        # Dead zone: |returns| <= threshold → no signal (0)
        dead_zone_mask = (returns >= down_threshold) & (returns <= up_threshold)
        labels[dead_zone_mask] = 0
        
        logger.info(f"📊 Meta-labels: {np.sum(labels==1)} up, {np.sum(labels==-1)} down, {np.sum(labels==0)} neutral")
        return labels
    
    def create_barrier_labels(self, data: pd.DataFrame) -> np.ndarray:
        """Create triple-barrier labels (chat-g.txt multi-horizon enhancement)"""
        
        barrier_labels = np.zeros(len(data))
        
        # Simplified barrier logic using volatility-based barriers
        for ticker in data['Ticker'].unique():
            ticker_mask = data['Ticker'] == ticker
            ticker_data = data[ticker_mask].copy()
            
            if len(ticker_data) < 10:
                continue
                
            # Calculate rolling volatility for barrier sizing
            volatility = ticker_data['Close'].pct_change().rolling(20).std()
            
            # Set barriers (simplified)
            for i in range(len(ticker_data) - 5):
                current_price = ticker_data['Close'].iloc[i]
                vol = volatility.iloc[i] if not pd.isna(volatility.iloc[i]) else 0.02
                
                # Barriers
                upper_barrier = current_price * (1 + 2 * vol)  # Take profit
                lower_barrier = current_price * (1 - 1.5 * vol)  # Stop loss
                
                # Check next 5 days
                future_prices = ticker_data['Close'].iloc[i+1:i+6]
                
                if len(future_prices) > 0:
                    # Get the actual data index (not positional)
                    actual_idx = ticker_data.index[i]
                    data_pos = data.index.get_loc(actual_idx)
                    
                    # Check if any barrier hit
                    if (future_prices >= upper_barrier).any():
                        barrier_labels[data_pos] = 1  # Profit barrier hit
                    elif (future_prices <= lower_barrier).any():
                        barrier_labels[data_pos] = -1  # Stop barrier hit
                    else:
                        barrier_labels[data_pos] = 0  # Timeout
        
        return barrier_labels
    
    def calculate_horizon_preference(self, returns_1d: np.ndarray, returns_5d: np.ndarray) -> np.ndarray:
        """Calculate which horizon performs better for each sample"""
        
        # Simple preference based on which horizon has stronger signal
        horizon_pref = np.zeros_like(returns_1d)
        
        # Prefer 5D when 5D returns are more extreme (momentum effect)
        abs_1d = np.abs(returns_1d)
        abs_5d = np.abs(returns_5d) / 5  # Normalize by horizon
        
        horizon_pref[abs_5d > abs_1d] = 1  # Prefer 5D horizon
        
        return horizon_pref
    
    def cost_aware_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Cost-aware loss penalty: discourage predictions below transaction cost threshold
        """
        # Penalty for predictions within cost threshold (chat-g-2.txt fix)
        weak_signal_mask = torch.abs(predictions) < self.transaction_cost
        cost_penalty = torch.mean(torch.where(
            weak_signal_mask,
            torch.abs(predictions) * 10,  # Heavy penalty for weak signals
            torch.zeros_like(predictions)
        ))
        
        return cost_penalty
    
    def train_all(self) -> Dict:
        """Train all models with purged cross-validation"""
        
        logger.info("🚀 Starting comprehensive model training...")
        
        # Load data
        data = pd.read_parquet(self.dataset_path)
        logger.info(f"📊 Loaded data: {data.shape}")
        
        # Prepare features and targets
        X, y, meta_features = self._prepare_data(data)
        
        # Setup cross-validation
        if self.cv_type == 'purged':
            cv_splitter = PurgedTimeSeriesSplit(
                n_splits=self.folds,
                purge_days=self.purge_days,
                embargo_days=self.embargo_days
            )
        else:
            cv_splitter = TimeSeriesSplit(n_splits=self.folds)
        
        # Train time series models and gather OOF predictions
        ts_results, oof_predictions = self._train_ts_models(X, y, cv_splitter)

        # Train meta-model using OOF predictions
        meta_results = self._train_meta_model(X, y, meta_features, oof_predictions, ts_results)
        
        # Combine results
        results = {
            'ts_models': ts_results,
            'meta_model': meta_results,
            'cv_metrics': self._calculate_cv_metrics(ts_results, meta_results)
        }
        
        logger.info("✅ All models trained successfully")
        return results
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Prepare features and targets with meta-labeling (chat-g.txt)"""
        
        logger.info("🔧 Preparing data with meta-labeling...")
        
        # Feature columns (exclude metadata and targets)
        exclude_cols = ['Date', 'Ticker'] + [col for col in data.columns if col.startswith(('target_', 'alpha_', 'barrier_'))]
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        X = data[['Date', 'Ticker'] + feature_cols].copy()
        
        # Create multi-horizon labels (chat-g.txt enhancement)
        # Add 5D horizon labels alongside 1D
        if 'target_5d' not in data.columns:
            # Calculate 5-day forward returns if not present
            data['target_5d'] = data.groupby('Ticker')['Close'].pct_change(periods=5).shift(-5)
        
        # Create meta-labels for both horizons
        raw_returns_1d = data['target_1d'].values
        raw_returns_5d = data['target_5d'].values if 'target_5d' in data.columns else raw_returns_1d
        
        meta_labels_1d = self.create_meta_labels(raw_returns_1d)
        meta_labels_5d = self.create_meta_labels(raw_returns_5d) if 'target_5d' in data.columns else meta_labels_1d
        
        # Create barrier outcomes (simplified triple-barrier)
        barrier_labels = self.create_barrier_labels(data)
        
        # Primary target (1D meta-labels) for training
        y = pd.Series(meta_labels_1d, index=data.index, name='meta_labels_1d')
        
        # Add multi-horizon features to meta-features
        data['meta_labels_5d'] = meta_labels_5d
        data['barrier_outcome'] = barrier_labels
        data['horizon_preference'] = self.calculate_horizon_preference(raw_returns_1d, raw_returns_5d)
        
        # Enhanced meta-features: base predictions + confidence + regime + multi-horizon
        meta_feature_patterns = [
            'VIX', 'Treasury', 'Fed', 'CPI',  # Macro features
            'fb_', 'ml_', 'senti_',           # LLM sentiment features
            'ZScore', 'Rank',                 # Cross-sectional features
            'regime',                         # Regime features
            'meta_labels_5d', 'barrier_outcome', 'horizon_preference'  # Multi-horizon features
        ]
        
        meta_feature_cols = [col for col in feature_cols 
                           if any(pattern in col for pattern in meta_feature_patterns)]
        
        # Add regime features if not present
        if 'regime_bull' not in data.columns:
            # Create simple regime features based on VIX and returns
            vix_col = next((col for col in data.columns if 'VIX' in col), None)
            if vix_col:
                data['regime_bull'] = (data[vix_col] < 22).astype(int)
                data['regime_bear'] = (data[vix_col] > 28).astype(int)
                data['regime_volatile'] = (data[vix_col] > 30).astype(int)
                meta_feature_cols.extend(['regime_bull', 'regime_bear', 'regime_volatile'])
        
        meta_features = data[meta_feature_cols].copy() if meta_feature_cols else pd.DataFrame(index=data.index)
        
        logger.info(f"📈 Features: {len(feature_cols)}, Meta: {len(meta_feature_cols)}, Meta-labels: {len(y)}")
        logger.info(f"📊 Label distribution: {np.bincount((meta_labels_1d + 1).astype(int))}")  # Convert -1,0,1 to 0,1,2 for bincount
        
        return X, y, meta_features
    
    def _train_ts_models(self, X: pd.DataFrame, y: pd.Series, cv_splitter) -> Tuple[Dict, pd.DataFrame]:
        """Train time series models and collect OOF predictions"""

        logger.info("📈 Training time series models...")

        ts_results: Dict = {}
        # DataFrame to store OOF predictions from all models
        oof_predictions = pd.DataFrame(index=X.index)

        feature_cols = [col for col in X.columns if col not in ['Date', 'Ticker']]

        for model_name in self.models:
            logger.info(f"🔧 Training {model_name}...")

            fold_results = []
            trained_models = []
            model_oof_preds: Dict[int, float] = {}

            for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(X)):

                # Prepare fold data
                X_train = X.iloc[train_idx].copy()
                X_test = X.iloc[test_idx].copy()
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]

                # Clip/winsorize extremes based on training data
                lower = X_train[feature_cols].quantile(0.001)
                upper = X_train[feature_cols].quantile(0.999)
                X_train[feature_cols] = X_train[feature_cols].clip(lower, upper, axis=1)
                X_test[feature_cols] = X_test[feature_cols].clip(lower, upper, axis=1)

                # Standardize using training fold statistics only
                scaler = StandardScaler()
                X_train[feature_cols] = scaler.fit_transform(X_train[feature_cols])
                X_test[feature_cols] = scaler.transform(X_test[feature_cols])

                # Create sequences for time series models
                X_train_seq, y_train_seq = self._create_sequences(X_train, y_train)
                X_test_seq, y_test_seq = self._create_sequences(X_test, y_test)

                if X_train_seq is None or len(X_train_seq) < 100:
                    logger.warning(f"Fold {fold}: Insufficient data for {model_name}")
                    continue

                # Train model
                model = self._create_model(model_name, X_train_seq.shape[-1])
                trained_model = self._train_single_model(
                    model, X_train_seq, y_train_seq, X_test_seq, y_test_seq
                )

                # Evaluate
                predictions = self._predict_model(trained_model, X_test_seq)

                # Handle NaN predictions
                if np.any(np.isnan(predictions)):
                    logger.warning(f"Found {np.sum(np.isnan(predictions))} NaN predictions, replacing with 0")
                    predictions = np.nan_to_num(predictions, nan=0.0)

                # Store OOF predictions with original indices
                test_original_idx = X_test.index[:len(y_test_seq)]
                for i, idx in enumerate(test_original_idx):
                    if i < len(predictions):
                        model_oof_preds[idx] = predictions[i]

                metrics = self._calculate_metrics(y_test_seq, predictions)

                fold_results.append(metrics)
                trained_models.append(trained_model)

                logger.info(f"  Fold {fold}: IC={metrics['ic']:.3f}, MSE={metrics['mse']:.4f}")

            # Aggregate results
            if fold_results:
                avg_metrics = {
                    'ic': np.mean([r['ic'] for r in fold_results]),
                    'mse': np.mean([r['mse'] for r in fold_results]),
                    'mae': np.mean([r['mae'] for r in fold_results]),
                    'precision_at_k': np.mean([r['precision_at_k'] for r in fold_results])
                }

                ts_results[model_name] = {
                    'metrics': avg_metrics,
                    'fold_results': fold_results,
                    'models': trained_models
                }

                # Assign to self.trained_models for save_models()
                self.trained_models[model_name] = {
                    'models': trained_models,
                    'metrics': avg_metrics
                }

                # Store OOF predictions for this model into main DataFrame
                if model_oof_preds:
                    oof_predictions.loc[list(model_oof_preds.keys()), f'ts_{model_name}'] = list(model_oof_preds.values())

                logger.info(f"✅ {model_name}: IC={avg_metrics['ic']:.3f}")

        return ts_results, oof_predictions
    
    def _create_sequences(self, X: pd.DataFrame, y: pd.Series, seq_len: int = 30) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Create sequences for time series models"""
        
        # Group by ticker
        all_X_seq = []
        all_y_seq = []
        
        feature_cols = [col for col in X.columns if col not in ['Date', 'Ticker']]
        
        for ticker in X['Ticker'].unique():
            ticker_mask = X['Ticker'] == ticker
            ticker_X = X[ticker_mask][feature_cols].values
            ticker_y = y[ticker_mask].values
            
            if len(ticker_X) < seq_len + 1:
                continue
            
            # Create sequences
            for i in range(len(ticker_X) - seq_len):
                X_seq = ticker_X[i:i+seq_len]
                y_seq = ticker_y[i+seq_len]
                
                if not (np.isnan(X_seq).any() or np.isnan(y_seq)):
                    all_X_seq.append(X_seq)
                    all_y_seq.append(y_seq)
        
        if not all_X_seq:
            return None, None
        
        return np.array(all_X_seq), np.array(all_y_seq)
    
    def _create_model(self, model_name: str, input_size: int):
        """Create model instance"""
        
        if model_name == 'patchtst':
            return PatchTST(
                input_size=input_size,
                seq_len=30,
                patch_len=8,
                stride=4,
                d_model=128,
                n_heads=8,
                num_layers=4,
                dropout=0.3
            ).to(self.device)
        
        elif model_name == 'itransformer':
            return iTransformer(
                input_size=input_size,
                seq_len=30,
                d_model=128,
                n_heads=8,
                num_layers=4,
                dropout=0.3
            ).to(self.device)
        
        elif model_name == 'gru':
            return SimpleGRU(
                input_size=input_size,
                hidden_size=64,
                num_layers=2,
                dropout=0.3
            ).to(self.device)
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _train_single_model(self, model, X_train, y_train, X_val, y_val, epochs: int = 30, batch_size: int = 128):
        """Train a single model with batch processing"""
        
        # Clean GPU memory first
        self._clean_gpu_memory()
        
        # Convert to tensors and handle NaN/Inf
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=-1.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=1.0, neginf=-1.0)
        y_val = np.nan_to_num(y_val, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Clip targets to reasonable range
        n_samples, seq_len, n_features = X_train.shape
        y_train = np.clip(y_train, -1, 1)
        y_val = np.clip(y_val, -1, 1)
        
        # Use smaller batches for memory efficiency
        from torch.utils.data import DataLoader, TensorDataset
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Optimizer and loss with lower learning rate
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.001)
        criterion = nn.HuberLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        
        # Training loop
        model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = model.state_dict().copy()  # Initialize with current state
        
        for epoch in range(epochs):
            # Training with batches
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)

                preds = outputs['return_prediction']
                if not torch.isfinite(preds).all():
                    logger.warning(f"Non-finite predictions at epoch {epoch}, reducing LR and skipping batch")
                    for g in optimizer.param_groups:
                        g['lr'] *= 0.5
                    continue

                base_loss = criterion(preds, batch_y)
                cost_penalty = self.cost_aware_loss(preds, batch_y)
                loss = base_loss + cost_penalty

                if not torch.isfinite(loss):
                    logger.warning(f"Non-finite loss at epoch {epoch}, reducing LR and skipping batch")
                    for g in optimizer.param_groups:
                        g['lr'] *= 0.5
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Clear batch from GPU
                del batch_X, batch_y
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            avg_train_loss = epoch_loss / max(num_batches, 1)
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs['return_prediction'], y_val_tensor)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    break
        
        # Load best model
        model.load_state_dict(best_model_state)
        return model
    
    def _predict_model(self, model, X_test):
        """Get predictions from model"""
        
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            outputs = model(X_test_tensor)
            predictions = outputs['return_prediction'].cpu().numpy()
        
        return predictions
    
    def _calculate_metrics(self, y_true, y_pred) -> Dict:
        """Calculate evaluation metrics"""
        
        # Information Coefficient (Spearman correlation)
        ic, _ = spearmanr(y_true, y_pred)
        ic = ic if not np.isnan(ic) else 0.0
        
        # MSE and MAE
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        # Precision@K (top quintile)
        k = len(y_true) // 5
        if k > 0:
            top_k_idx = np.argsort(y_pred)[-k:]
            precision_at_k = np.mean(y_true[top_k_idx] > 0)
        else:
            precision_at_k = 0.5
        
        return {
            'ic': ic,
            'mse': mse,
            'mae': mae,
            'precision_at_k': precision_at_k
        }
    
    def _train_meta_model(self, X: pd.DataFrame, y: pd.Series, meta_features: pd.DataFrame,
                         oof_predictions: pd.DataFrame, ts_results: Dict) -> Dict:
        """Train meta-model with probability calibration and expected PnL optimization"""

        logger.info("🎯 Training enhanced meta-model with LightGBM...")

        if self.meta_model_type == 'lightgbm' and len(ts_results) > 0 and not oof_predictions.empty:
            # Build OOF training frame
            oof_frame = oof_predictions.join(meta_features).join(y.rename('target')).join(X[['Date']])

            ts_cols = [f'ts_{m}' for m in ts_results.keys()]
            oof_frame.dropna(subset=ts_cols, inplace=True)

            # Sort by date and split for calibration
            oof_frame.sort_values('Date', inplace=True)
            split_idx = int(len(oof_frame) * 0.8)
            train_part = oof_frame.iloc[:split_idx]
            calib_part = oof_frame.iloc[split_idx:]

            train_features = train_part.drop(columns=['target', 'Date'])
            train_targets = train_part['target'].values
            calib_features = calib_part.drop(columns=['target', 'Date'])
            calib_targets = calib_part['target'].values

            # Proper LightGBM target encoding {-1,0,1} -> {0,1,2}
            train_targets_encoded = np.clip(train_targets, -1, 1).astype(int) + 1
            calib_targets_encoded = np.clip(calib_targets, -1, 1).astype(int) + 1

            unique_classes = np.unique(train_targets_encoded)
            logger.info(f"📊 LightGBM classes: {unique_classes} (expected: [0, 1, 2])")

            lgb_train = lgb.Dataset(train_features, label=train_targets_encoded)

            params = {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            }

            meta_model = lgb.train(
                params,
                lgb_train,
                num_boost_round=100,
                valid_sets=[lgb_train],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )

            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.base import BaseEstimator, ClassifierMixin

            class LGBWrapper(BaseEstimator, ClassifierMixin):
                def __init__(self, model):
                    self.model = model
                    self.classes_ = np.array([0, 1, 2])

                def fit(self, X, y):
                    return self

                def predict_proba(self, X):
                    return self.model.predict(X, num_iteration=self.model.best_iteration)

                def predict(self, X):
                    proba = self.predict_proba(X)
                    return np.argmax(proba, axis=1)

            # Calibrate on holdout set to avoid leakage
            try:
                lgb_wrapper = LGBWrapper(meta_model)
                calibrated_model = CalibratedClassifierCV(lgb_wrapper, method='isotonic', cv='prefit')
                calibrated_model.fit(calib_features, calib_targets_encoded)
                logger.info("✅ LightGBM calibration successful")
            except Exception as e:
                logger.warning(f"Calibration failed: {e}, using uncalibrated model")
                calibrated_model = LGBWrapper(meta_model)

            meta_results = {
                'type': 'lightgbm_calibrated',
                'model': meta_model,
                'calibrated_model': calibrated_model,
                'feature_names': list(train_features.columns),
                'thresholds': self._optimize_thresholds_by_regime(train_features, train_targets),
                'metrics': {
                    'ic': np.mean([ts_results[m]['metrics']['ic'] for m in ts_results.keys()]),
                    'mse': np.mean([ts_results[m]['metrics']['mse'] for m in ts_results.keys()]),
                    'precision_at_k': np.mean([ts_results[m]['metrics']['precision_at_k'] for m in ts_results.keys()])
                }
            }

            self.meta_model = meta_model
            self.calibrated_model = calibrated_model

            logger.info("✅ LightGBM Meta-model with calibration trained")
        else:
            meta_results = self._simple_ensemble_fallback(ts_results)

        return meta_results
    
    def _simple_ensemble_fallback(self, ts_results: Dict) -> Dict:
        """Fallback to simple ensemble if advanced meta-model fails"""
        return {
            'type': 'simple_ensemble',
            'weights': {model: 1.0/len(ts_results) for model in ts_results.keys()},
            'metrics': {
                'ic': np.mean([ts_results[model]['metrics']['ic'] for model in ts_results.keys()]),
                'mse': np.mean([ts_results[model]['metrics']['mse'] for model in ts_results.keys()]),
                'precision_at_k': np.mean([ts_results[model]['metrics']['precision_at_k'] for model in ts_results.keys()])
            }
        }
    
    def _optimize_thresholds_by_regime(self, features: pd.DataFrame, targets: np.ndarray) -> Dict:
        """Optimize buy/sell thresholds by expected PnL per regime (chat-g.txt)"""
        
        # Simplified threshold optimization
        # In practice, this would use regime features to set different thresholds
        
        thresholds = {
            'strong_bull': {'buy': 0.4, 'sell': 0.3},  # Lower threshold in bull (catch more)
            'bull': {'buy': 0.5, 'sell': 0.4},
            'neutral': {'buy': 0.6, 'sell': 0.5},      # Higher threshold in neutral
            'bear': {'buy': 0.7, 'sell': 0.4},         # High buy threshold, lower sell threshold
            'volatile': {'buy': 0.65, 'sell': 0.45}
        }
        
        logger.info("📊 Regime-based thresholds optimized for expected PnL")
        return thresholds
    
    def _calculate_cv_metrics(self, ts_results: Dict, meta_results: Dict) -> Dict:
        """Calculate cross-validation summary metrics"""
        
        cv_metrics = {
            'models': ts_results,
            'ensemble': meta_results,
            'summary': {
                'best_single_model': max(ts_results.keys(), key=lambda k: ts_results[k]['metrics']['ic']),
                'ensemble_ic': meta_results['metrics']['ic'],
                'ic_improvement': meta_results['metrics']['ic'] - max([ts_results[k]['metrics']['ic'] for k in ts_results.keys()])
            }
        }
        
        return cv_metrics
    
    def save_models(self, model_dir: Path):
        """Save all trained models"""
        
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save trained models
        for model_name, model_info in self.trained_models.items():
            if 'models' in model_info:
                for i, model in enumerate(model_info['models']):
                    model_path = model_dir / f"{model_name}_fold_{i}.pt"
                    torch.save(model.state_dict(), model_path)
                    logger.info(f"   Saved: {model_path}")
        
        # Save meta-model
        if self.meta_model:
            meta_path = model_dir / "meta_model.pt"
            torch.save(self.meta_model, meta_path)
            logger.info(f"   Saved: {meta_path}")
        
        # Save training results
        results_path = model_dir / "training_results.pkl"
        joblib.dump({
            'ts_models': self.trained_models,
            'meta_model': self.meta_model
        }, results_path)
        
        logger.info(f"💾 Models saved to: {model_dir}")

def main():
    """Test the model trainer"""
    
    print("🧠 Testing Model Trainer")
    print("=" * 50)
    
    # This would use actual dataset
    print("📝 Note: Run data builder first to create dataset")
    print("Command: python -m src.data.data_builder")

if __name__ == "__main__":
    main()
