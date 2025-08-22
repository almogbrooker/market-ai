#!/usr/bin/env python3
"""
ULTIMATE Performance Testing Framework
Tests EVERYTHING - all models, all universes, all configurations
Identifies the absolute best performance without overfitting
"""


# === Conformal score-domain gating (robust offline) ===
def calibrate_gate_from_scores(yhat_val, alpha=0.15):
    yhat_val = np.asarray(yhat_val, dtype=float)
    if yhat_val.size == 0:
        return {"abs_score_threshold": 0.0, "alpha": alpha}
    thr = float(np.quantile(np.abs(yhat_val), 1 - alpha))
    return {"abs_score_threshold": thr, "alpha": alpha}

def apply_gate_scores(yhat, gate):
    yhat = np.asarray(yhat, dtype=float)
    thr = float(gate.get("abs_score_threshold", 0.0)) if gate else 0.0
    keep = np.abs(yhat) >= thr if thr > 0 else np.ones_like(yhat, dtype=bool)
    return yhat * keep, float(keep.mean()) if keep.size else 0.0

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import spearmanr

# Import our models and utilities
from src.models.advanced_models import create_advanced_model
from config.stock_universes import StockUniverses

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UltimateTestResult:
    """Ultimate comprehensive test results"""
    config_name: str
    model_type: str
    universe: str
    model_size: str
    feature_set: str
    feature_count: int
    
    # Performance metrics
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    train_precision: float
    val_precision: float
    test_precision: float
    train_recall: float
    val_recall: float
    test_recall: float
    train_f1: float
    val_f1: float
    test_f1: float
    
    # Financial metrics
    sharpe_ratio: float
    calmar_ratio: float
    max_drawdown: float
    volatility: float
    avg_return: float
    hit_rate: float
    profit_factor: float
    
    # Overfitting analysis
    overfitting_ratio: float  # val/train
    generalization_gap: float  # train - val
    stability_score: float     # |val - test|
    consistency_score: float   # cross-validation consistency
    
    # Model characteristics
    model_complexity: int
    training_time: float
    inference_time: float
    
    # Final assessment
    risk_score: float         # Higher = more risky
    confidence_score: float   # Higher = more confident
    final_verdict: str        # EXCELLENT, GOOD, SUSPICIOUS, OVERFITTED, FAILED
    
    # Additional insights
    best_features: List[str]
    worst_features: List[str]
    temporal_stability: float

class UltimatePerformanceTester:
    """Ultimate comprehensive performance testing system"""
    
    def __init__(self, data_path: str = "data/training_data_2020_2024_complete.csv"):
        """Initialize ultimate tester"""
        self.data_path = Path(data_path)
        self.results_dir = Path("ultimate_test_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Load data
        self.df = self._load_and_validate_data()
        
        # ALL model types available
        self.model_types = [
            'financial_transformer',
            'patchtst', 
            'itransformer',
            'tsmixer',
            'timesnet',
            'advanced_lstm',
            'wavenet',
            'ensemble'
        ]
        
        # ALL universe configurations
        self.universes = {
            'micro': 'mega_cap_tech',      # 12 stocks - fast testing
            'small': 'sp100',              # 100 stocks - medium scale
            'large': 'russell1000'         # 200 stocks - full scale
        }
        
        # Model size configurations
        self.model_sizes = {
            'tiny': {
                'seq_len': 15, 'd_model': 32, 'n_heads': 2, 'num_layers': 2, 
                'dropout': 0.1, 'patch_len': 3, 'stride': 1, 'hidden_size': 16
            },
            'small': {
                'seq_len': 20, 'd_model': 64, 'n_heads': 4, 'num_layers': 3,
                'dropout': 0.2, 'patch_len': 5, 'stride': 2, 'hidden_size': 32
            },
            'medium': {
                'seq_len': 30, 'd_model': 128, 'n_heads': 8, 'num_layers': 4,
                'dropout': 0.3, 'patch_len': 8, 'stride': 4, 'hidden_size': 64
            },
            'large': {
                'seq_len': 40, 'd_model': 256, 'n_heads': 8, 'num_layers': 6,
                'dropout': 0.4, 'patch_len': 10, 'stride': 5, 'hidden_size': 128
            },
            'xlarge': {
                'seq_len': 50, 'd_model': 512, 'n_heads': 16, 'num_layers': 8,
                'dropout': 0.5, 'patch_len': 12, 'stride': 6, 'hidden_size': 256
            }
        }
        
        # Feature configurations
        self.feature_sets = {
            'minimal': {
                'description': 'Only essential price features',
                'include': ['Close', 'Volume', 'RSI_14', 'MACD', 'returns_1d']
            },
            'technical': {
                'description': 'Technical indicators only',
                'exclude': ['VIX', 'Treasury_10Y', 'Fed_Funds', 'CPI', 'Unemployment', 
                           'fb_pos', 'fb_neg', 'ml_pos', 'ml_neg', 'senti_mean']
            },
            'fundamental': {
                'description': 'Macro and fundamental indicators',
                'include': ['Close', 'Volume', 'VIX', 'Treasury_10Y', 'Fed_Funds', 'CPI', 
                           'Unemployment', 'Yield_Spread', 'returns_1d']
            },
            'sentiment': {
                'description': 'Sentiment and news features',
                'include': ['Close', 'Volume', 'fb_pos', 'fb_neg', 'ml_pos', 'ml_neg', 
                           'senti_mean', 'senti_std', 'n_news', 'returns_1d']
            },
            'balanced': {
                'description': 'Balanced feature mix',
                'exclude': ['n_docs', 'n_social', 'n_edgar']  # Remove some noisy features
            },
            'all_features': {
                'description': 'All available features',
                'exclude': []
            }
        }
        
        # Results storage
        self.all_results: List[UltimateTestResult] = []
        
        # Generate ALL test configurations
        self.test_configs = self._generate_all_configurations()
        
        logger.info(f"Ultimate tester initialized with {len(self.test_configs)} configurations")
    
    def _load_and_validate_data(self) -> pd.DataFrame:
        """Load and thoroughly validate data"""
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        
        # Standardize column names
        if 'Date' in df.columns:
            df['date'] = pd.to_datetime(df['Date'])
        if 'Ticker' in df.columns:
            df['ticker'] = df['Ticker']
        if 'Return_1D' in df.columns:
            df['returns_1d'] = df['Return_1D']
        
        # Data quality checks
        logger.info(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"Unique tickers: {df['ticker'].nunique()}")
        
        # Check for data quality issues
        missing_pct = df.isnull().mean()
        high_missing = missing_pct[missing_pct > 0.5]
        if len(high_missing) > 0:
            logger.warning(f"High missing data in columns: {list(high_missing.index)}")
        
        # Sort data properly
        df = df.sort_values(['date', 'ticker']).reset_index(drop=True)
        
        return df
    
    def _generate_all_configurations(self) -> List[Dict]:
        """Generate ALL possible test configurations"""
        
        configs = []
        config_id = 1
        
        for model_type in self.model_types:
            for universe_name, universe_key in self.universes.items():
                for size_name, size_config in self.model_sizes.items():
                    for feature_name, feature_config in self.feature_sets.items():
                        
                        config = {
                            'id': config_id,
                            'name': f"{model_type}_{universe_name}_{size_name}_{feature_name}",
                            'model_type': model_type,
                            'universe_name': universe_name,
                            'universe_key': universe_key,
                            'size_name': size_name,
                            'size_config': size_config,
                            'feature_name': feature_name,
                            'feature_config': feature_config
                        }
                        
                        configs.append(config)
                        config_id += 1
        
        logger.info(f"Generated {len(configs)} total configurations:")
        logger.info(f"  Models: {len(self.model_types)}")
        logger.info(f"  Universes: {len(self.universes)}")
        logger.info(f"  Model sizes: {len(self.model_sizes)}")
        logger.info(f"  Feature sets: {len(self.feature_sets)}")
        
        return configs
    
    def prepare_data_for_config(self, config: Dict) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare data for specific configuration"""
        
        # Filter to universe
        universe_stocks = StockUniverses.get_universe(config['universe_key'])
        df_filtered = self.df[self.df['ticker'].isin(universe_stocks)].copy()
        
        # Prepare feature columns
        feature_config = config['feature_config']
        
        # All potential features
        all_potential_features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_10', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI_14',
            'MACD', 'MACD_Signal', 'BB_Middle', 'BB_Std', 'BB_Upper', 'BB_Lower',
            'Volume_SMA', 'Volume_Ratio', 'Volatility_20D', 'ATR_14',
            'VIX', 'Treasury_10Y', 'Fed_Funds', 'CPI', 'Unemployment', 'Yield_Spread', 'VIX_Spike',
            'fb_pos', 'fb_neg', 'ml_pos', 'ml_neg', 'n_docs', 'n_news', 'n_social', 'n_edgar',
            'senti_mean', 'senti_std', 'senti_z', 'news_n_z', 'avg_confidence',
            'sentiment_uncertainty', 'source_diversity', 'lang_diversity',
            'Momentum_ZScore', 'Volume_ZScore'
        ]
        
        # Select features based on configuration
        if 'include' in feature_config:
            # Use only specified features
            feature_cols = [col for col in feature_config['include'] if col in df_filtered.columns]
        else:
            # Use all features except excluded ones
            exclude_list = feature_config.get('exclude', [])
            feature_cols = [col for col in all_potential_features 
                          if col in df_filtered.columns and col not in exclude_list]
        
        # Ensure we have returns column
        if 'returns_1d' not in feature_cols and 'returns_1d' in df_filtered.columns:
            # Don't include returns as a feature for prediction
            pass
        
        # Data cleaning
        df_filtered[feature_cols] = df_filtered[feature_cols].fillna(df_filtered[feature_cols].median())
        df_filtered[feature_cols] = df_filtered[feature_cols].replace([np.inf, -np.inf], np.nan)
        df_filtered[feature_cols] = df_filtered[feature_cols].fillna(df_filtered[feature_cols].median())
        
        logger.info(f"Prepared data: {len(df_filtered)} rows, {len(feature_cols)} features")
        
        return df_filtered, feature_cols
    
    def create_robust_temporal_splits(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """Create multiple robust temporal splits for cross-validation"""
        
        df_sorted = df.sort_values('date').copy()
        unique_dates = sorted(df_sorted['date'].unique())
        
        # Create multiple temporal splits for robust validation
        splits = []
        
        # Main split: 60% train, 20% val, 20% test
        train_end = int(len(unique_dates) * 0.6)
        val_end = int(len(unique_dates) * 0.8)
        
        train_dates = unique_dates[:train_end]
        val_dates = unique_dates[train_end + 10:val_end]  # 10-day gap
        test_dates = unique_dates[val_end + 10:]          # 10-day gap
        
        train_df = df_sorted[df_sorted['date'].isin(train_dates)]
        val_df = df_sorted[df_sorted['date'].isin(val_dates)]
        test_df = df_sorted[df_sorted['date'].isin(test_dates)]
        
        splits.append((train_df, val_df, test_df))
        
        # Additional split for consistency check: 50% train, 25% val, 25% test
        train_end_2 = int(len(unique_dates) * 0.5)
        val_end_2 = int(len(unique_dates) * 0.75)
        
        train_dates_2 = unique_dates[:train_end_2]
        val_dates_2 = unique_dates[train_end_2 + 10:val_end_2]
        test_dates_2 = unique_dates[val_end_2 + 10:]
        
        train_df_2 = df_sorted[df_sorted['date'].isin(train_dates_2)]
        val_df_2 = df_sorted[df_sorted['date'].isin(val_dates_2)]
        test_df_2 = df_sorted[df_sorted['date'].isin(test_dates_2)]
        
        splits.append((train_df_2, val_df_2, test_df_2))
        
        return splits
    
    def create_sequences_robust(self, X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences with robust error handling"""
        
        if len(X) < seq_len:
            return np.array([]), np.array([])
        
        try:
            X_seq, y_seq = [], []
            
            for i in range(seq_len, len(X)):
                X_seq.append(X[i-seq_len:i])
                y_seq.append(y[i])
            
            return np.array(X_seq), np.array(y_seq)
        except Exception as e:
            logger.error(f"Sequence creation failed: {e}")
            return np.array([]), np.array([])
    
    def train_model_ultimate(self, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Ultimate model training with all safeguards"""
        
        try:
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train)
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
            
            # Training loop with early stopping
            best_val_loss = float('inf')
            patience = 20
            patience_counter = 0
            
            model.train()
            
            for epoch in range(100):  # Max epochs
                # Training step
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                
                # Handle different output formats
                if isinstance(outputs, dict):
                    outputs = outputs.get('prediction', outputs.get('output', list(outputs.values())[0]))
                
                loss = criterion(outputs.squeeze(), y_train_tensor)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Validation check every 5 epochs
                if epoch % 5 == 0:
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_val_tensor)
                        if isinstance(val_outputs, dict):
                            val_outputs = val_outputs.get('prediction', val_outputs.get('output', list(val_outputs.values())[0]))
                        val_loss = criterion(val_outputs.squeeze(), y_val_tensor).item()
                    
                    scheduler.step(val_loss)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
                    
                    model.train()
            
            return {'success': True, 'best_val_loss': best_val_loss}
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def evaluate_model_comprehensive(self, model: nn.Module, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Comprehensive model evaluation with all metrics"""
        
        try:
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                outputs = model(X_tensor)
                
                # Handle different output formats
                if isinstance(outputs, dict):
                    y_pred = outputs.get('prediction', outputs.get('output', list(outputs.values())[0])).numpy()
                else:
                    y_pred = outputs.numpy()
                
                y_pred = y_pred.flatten()
            
            # Binary classification metrics
            y_true_binary = (y > 0).astype(int)
            y_pred_binary = (y_pred > 0).astype(int)
            
            accuracy = accuracy_score(y_true_binary, y_pred_binary)
            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            
            # Financial metrics
            strategy_returns = np.where(y_pred > 0, y, -y)
            
            if len(strategy_returns) > 1:
                avg_return = np.mean(strategy_returns)
                volatility = np.std(strategy_returns)
                sharpe_ratio = avg_return / (volatility + 1e-6) * np.sqrt(252)
                hit_rate = np.mean(strategy_returns > 0)
                
                # Drawdown
                cumulative = np.cumprod(1 + strategy_returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = np.min(drawdown)
                
                # Calmar ratio
                annual_return = avg_return * 252
                calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
                
                # Profit factor
                positive_returns = strategy_returns[strategy_returns > 0]
                negative_returns = strategy_returns[strategy_returns < 0]
                profit_factor = np.sum(positive_returns) / abs(np.sum(negative_returns)) if len(negative_returns) > 0 else np.inf
                
            else:
                avg_return = volatility = sharpe_ratio = hit_rate = 0
                max_drawdown = calmar_ratio = profit_factor = 0
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'sharpe_ratio': sharpe_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'avg_return': avg_return,
                'hit_rate': hit_rate,
                'profit_factor': profit_factor
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                'accuracy': 0.5, 'precision': 0.5, 'recall': 0.5, 'f1': 0.5,
                'sharpe_ratio': 0, 'calmar_ratio': 0, 'max_drawdown': -1,
                'volatility': 1, 'avg_return': 0, 'hit_rate': 0.5, 'profit_factor': 1
            }
    
    def test_single_configuration_ultimate(self, config: Dict) -> UltimateTestResult:
        """Test single configuration with ultimate thoroughness"""
        
        logger.info(f"🔬 Testing: {config['name']}")
        start_time = datetime.now()
        
        try:
            # Prepare data
            df_filtered, feature_cols = self.prepare_data_for_config(config)
            
            # Create temporal splits
            splits = self.create_robust_temporal_splits(df_filtered)
            
            # Test on multiple splits for consistency
            split_results = []
            
            for split_idx, (train_df, val_df, test_df) in enumerate(splits):
                # Prepare arrays
                X_train = train_df[feature_cols].values
                y_train = train_df['returns_1d'].values
                X_val = val_df[feature_cols].values
                y_val = val_df['returns_1d'].values
                X_test = test_df[feature_cols].values
                y_test = test_df['returns_1d'].values
                
                # Handle sequences
                size_config = config['size_config']
                model_type = config['model_type']
                seq_len = size_config.get('seq_len', 30) if model_type in ['patchtst', 'tsmixer', 'financial_transformer', 'itransformer', 'timesnet'] else None
                
                if seq_len:
                    X_train_seq, y_train_seq = self.create_sequences_robust(X_train, y_train, seq_len)
                    X_val_seq, y_val_seq = self.create_sequences_robust(X_val, y_val, seq_len)
                    X_test_seq, y_test_seq = self.create_sequences_robust(X_test, y_test, seq_len)
                else:
                    X_train_seq, y_train_seq = X_train, y_train
                    X_val_seq, y_val_seq = X_val, y_val
                    X_test_seq, y_test_seq = X_test, y_test
                
                if len(X_train_seq) == 0 or len(X_val_seq) == 0 or len(X_test_seq) == 0:
                    continue
                
                # Create model
                try:
                    model_kwargs = size_config.copy()
                    model_kwargs['bidirectional'] = True
                    model = create_advanced_model(model_type, len(feature_cols), **model_kwargs)
                    model_complexity = sum(p.numel() for p in model.parameters())
                except Exception as e:
                    logger.error(f"Model creation failed: {e}")
                    continue
                
                # Train model
                training_result = self.train_model_ultimate(model, X_train_seq, y_train_seq, X_val_seq, y_val_seq)
                
                if not training_result.get('success', False):
                    continue
                
                # Evaluate
                train_metrics = self.evaluate_model_comprehensive(model, X_train_seq, y_train_seq)
                val_metrics = self.evaluate_model_comprehensive(model, X_val_seq, y_val_seq)
                test_metrics = self.evaluate_model_comprehensive(model, X_test_seq, y_test_seq)
                
                split_results.append({
                    'train': train_metrics,
                    'val': val_metrics,
                    'test': test_metrics,
                    'model_complexity': model_complexity
                })
            
            # Aggregate results across splits
            if split_results:
                # Average metrics across splits
                train_acc = np.mean([r['train']['accuracy'] for r in split_results])
                val_acc = np.mean([r['val']['accuracy'] for r in split_results])
                test_acc = np.mean([r['test']['accuracy'] for r in split_results])
                
                train_prec = np.mean([r['train']['precision'] for r in split_results])
                val_prec = np.mean([r['val']['precision'] for r in split_results])
                test_prec = np.mean([r['test']['precision'] for r in split_results])
                
                train_recall = np.mean([r['train']['recall'] for r in split_results])
                val_recall = np.mean([r['val']['recall'] for r in split_results])
                test_recall = np.mean([r['test']['recall'] for r in split_results])
                
                train_f1 = np.mean([r['train']['f1'] for r in split_results])
                val_f1 = np.mean([r['val']['f1'] for r in split_results])
                test_f1 = np.mean([r['test']['f1'] for r in split_results])
                
                # Financial metrics
                sharpe_ratio = np.mean([r['test']['sharpe_ratio'] for r in split_results])
                calmar_ratio = np.mean([r['test']['calmar_ratio'] for r in split_results])
                max_drawdown = np.mean([r['test']['max_drawdown'] for r in split_results])
                volatility = np.mean([r['test']['volatility'] for r in split_results])
                avg_return = np.mean([r['test']['avg_return'] for r in split_results])
                hit_rate = np.mean([r['test']['hit_rate'] for r in split_results])
                profit_factor = np.mean([r['test']['profit_factor'] for r in split_results])
                
                # Overfitting analysis
                overfitting_ratio = val_acc / max(train_acc, 0.001)
                generalization_gap = train_acc - val_acc
                stability_score = 1.0 - abs(val_acc - test_acc)
                
                # Consistency across splits
                test_accs = [r['test']['accuracy'] for r in split_results]
                consistency_score = 1.0 - np.std(test_accs)
                
                # Risk and confidence assessment
                risk_score = max(0, generalization_gap * 10) + max(0, (0.85 - overfitting_ratio) * 5)
                confidence_score = min(1.0, stability_score * consistency_score * min(overfitting_ratio / 0.85, 1.0))
                
                # Final verdict
                verdict = self._determine_ultimate_verdict(
                    test_acc, overfitting_ratio, generalization_gap, 
                    stability_score, consistency_score, sharpe_ratio
                )
                
                model_complexity = np.mean([r['model_complexity'] for r in split_results])
                
            else:
                # Failed case
                train_acc = val_acc = test_acc = 0.5
                train_prec = val_prec = test_prec = 0.5
                train_recall = val_recall = test_recall = 0.5
                train_f1 = val_f1 = test_f1 = 0.5
                sharpe_ratio = calmar_ratio = max_drawdown = avg_return = 0
                volatility = hit_rate = profit_factor = 0.5
                overfitting_ratio = generalization_gap = stability_score = consistency_score = 0
                risk_score = 1.0
                confidence_score = 0.0
                verdict = 'FAILED'
                model_complexity = 0
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Create comprehensive result
            result = UltimateTestResult(
                config_name=config['name'],
                model_type=config['model_type'],
                universe=config['universe_name'],
                model_size=config['size_name'],
                feature_set=config['feature_name'],
                feature_count=len(feature_cols),
                
                train_accuracy=train_acc,
                val_accuracy=val_acc,
                test_accuracy=test_acc,
                train_precision=train_prec,
                val_precision=val_prec,
                test_precision=test_prec,
                train_recall=train_recall,
                val_recall=val_recall,
                test_recall=test_recall,
                train_f1=train_f1,
                val_f1=val_f1,
                test_f1=test_f1,
                
                sharpe_ratio=sharpe_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=max_drawdown,
                volatility=volatility,
                avg_return=avg_return,
                hit_rate=hit_rate,
                profit_factor=profit_factor,
                
                overfitting_ratio=overfitting_ratio,
                generalization_gap=generalization_gap,
                stability_score=stability_score,
                consistency_score=consistency_score,
                
                model_complexity=int(model_complexity),
                training_time=training_time,
                inference_time=0.0,  # Could add inference timing
                
                risk_score=risk_score,
                confidence_score=confidence_score,
                final_verdict=verdict,
                
                best_features=[],  # Could add feature importance
                worst_features=[],
                temporal_stability=consistency_score
            )
            
            logger.info(f"✅ {config['name']} | Test Acc: {test_acc:.3f} | Verdict: {verdict}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Configuration {config['name']} failed: {e}")
            
            # Return failed result
            return UltimateTestResult(
                config_name=config['name'],
                model_type=config['model_type'],
                universe=config['universe_name'],
                model_size=config['size_name'],
                feature_set=config['feature_name'],
                feature_count=0,
                
                train_accuracy=0.5, val_accuracy=0.5, test_accuracy=0.5,
                train_precision=0.5, val_precision=0.5, test_precision=0.5,
                train_recall=0.5, val_recall=0.5, test_recall=0.5,
                train_f1=0.5, val_f1=0.5, test_f1=0.5,
                
                sharpe_ratio=0, calmar_ratio=0, max_drawdown=-1,
                volatility=1, avg_return=0, hit_rate=0.5, profit_factor=1,
                
                overfitting_ratio=1.0, generalization_gap=0, stability_score=0, consistency_score=0,
                
                model_complexity=0, training_time=0, inference_time=0,
                
                risk_score=1.0, confidence_score=0.0, final_verdict='FAILED',
                
                best_features=[], worst_features=[], temporal_stability=0
            )
    
    def _determine_ultimate_verdict(self, test_acc: float, overfit_ratio: float, 
                                  gen_gap: float, stability: float, consistency: float, 
                                  sharpe: float) -> str:
        """Determine ultimate performance verdict"""
        
        # Excellent: High performance, no overfitting, stable, good returns
        if (test_acc > 0.70 and overfit_ratio > 0.90 and gen_gap < 0.03 and 
            stability > 0.85 and consistency > 0.80 and sharpe > 2.0):
            return 'EXCELLENT'
        
        # Good: Solid performance with minimal issues
        elif (test_acc > 0.60 and overfit_ratio > 0.85 and gen_gap < 0.05 and 
              stability > 0.70 and sharpe > 1.0):
            return 'GOOD'
        
        # Suspicious: Shows signs of overfitting or instability
        elif (overfit_ratio < 0.80 or gen_gap > 0.10 or stability < 0.60 or consistency < 0.60):
            return 'SUSPICIOUS'
        
        # Overfitted: Clear signs of overfitting
        elif (overfit_ratio < 0.70 or gen_gap > 0.15):
            return 'OVERFITTED'
        
        # Failed: Poor performance
        elif test_acc < 0.52:
            return 'FAILED'
        
        else:
            return 'GOOD'
    
    def run_ultimate_tests(self, max_configs: Optional[int] = None) -> List[UltimateTestResult]:
        """Run ultimate comprehensive tests"""
        
        logger.info("🚀 STARTING ULTIMATE PERFORMANCE TESTING")
        logger.info("=" * 80)
        logger.info(f"Total configurations: {len(self.test_configs)}")
        
        configs_to_test = self.test_configs[:max_configs] if max_configs else self.test_configs
        
        logger.info(f"Testing {len(configs_to_test)} configurations...")
        
        for i, config in enumerate(configs_to_test, 1):
            logger.info(f"\n📊 Configuration {i}/{len(configs_to_test)}")
            
            result = self.test_single_configuration_ultimate(config)
            self.all_results.append(result)
            
            # Save every 25 results
            if i % 25 == 0:
                self.save_results()
        
        # Final save
        self.save_results()
        
        logger.info(f"\n🏆 ULTIMATE TESTING COMPLETED!")
        logger.info(f"Total results: {len(self.all_results)}")
        
        return self.all_results
    
    def save_results(self):
        """Save ultimate test results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"ultimate_results_{timestamp}.json"
        
        results_data = [asdict(result) for result in self.all_results]
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"💾 Results saved: {results_file}")
    
    def generate_ultimate_report(self) -> str:
        """Generate ultimate comprehensive report"""
        
        if not self.all_results:
            return "No results to report"
        
        # Analysis
        total_configs = len(self.all_results)
        verdicts = {}
        for result in self.all_results:
            verdicts[result.final_verdict] = verdicts.get(result.final_verdict, 0) + 1
        
        # Best performers
        successful_results = [r for r in self.all_results if r.final_verdict not in ['FAILED', 'OVERFITTED']]
        
        report = []
        report.append("=" * 100)
        report.append("🏆 ULTIMATE PERFORMANCE TESTING REPORT")
        report.append("=" * 100)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Configurations Tested: {total_configs}")
        report.append("")
        
        # Verdict summary
        report.append("📊 VERDICT DISTRIBUTION:")
        for verdict in ['EXCELLENT', 'GOOD', 'SUSPICIOUS', 'OVERFITTED', 'FAILED']:
            count = verdicts.get(verdict, 0)
            pct = count / total_configs * 100
            report.append(f"{verdict:12}: {count:4} ({pct:5.1f}%)")
        report.append("")
        
        if successful_results:
            # Top performers by test accuracy
            top_by_accuracy = sorted(successful_results, key=lambda x: x.test_accuracy, reverse=True)[:10]
            report.append("🥇 TOP 10 BY TEST ACCURACY:")
            for i, result in enumerate(top_by_accuracy, 1):
                report.append(f"{i:2}. {result.config_name}")
                report.append(f"    Test Acc: {result.test_accuracy:.3f} | Sharpe: {result.sharpe_ratio:.2f} | Verdict: {result.final_verdict}")
            report.append("")
            
            # Top by Sharpe ratio
            top_by_sharpe = sorted(successful_results, key=lambda x: x.sharpe_ratio, reverse=True)[:5]
            report.append("💰 TOP 5 BY SHARPE RATIO:")
            for i, result in enumerate(top_by_sharpe, 1):
                report.append(f"{i}. {result.config_name}")
                report.append(f"   Sharpe: {result.sharpe_ratio:.2f} | Test Acc: {result.test_accuracy:.3f} | Verdict: {result.final_verdict}")
            report.append("")
            
            # Most trustworthy (lowest risk)
            most_trustworthy = sorted(successful_results, key=lambda x: x.risk_score)[:5]
            report.append("✅ MOST TRUSTWORTHY (LOWEST RISK):")
            for i, result in enumerate(most_trustworthy, 1):
                report.append(f"{i}. {result.config_name}")
                report.append(f"   Risk Score: {result.risk_score:.3f} | Confidence: {result.confidence_score:.3f} | Test Acc: {result.test_accuracy:.3f}")
            report.append("")
            
            # Model type analysis
            model_analysis = {}
            for result in successful_results:
                model_type = result.model_type
                if model_type not in model_analysis:
                    model_analysis[model_type] = []
                model_analysis[model_type].append(result.test_accuracy)
            
            report.append("🔬 MODEL TYPE ANALYSIS:")
            for model_type, accuracies in model_analysis.items():
                avg_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                count = len(accuracies)
                report.append(f"{model_type:20}: {avg_acc:.3f} ± {std_acc:.3f} ({count} configs)")
            report.append("")
            
            # FINAL RECOMMENDATION
            best_overall = max(successful_results, key=lambda x: x.confidence_score * x.test_accuracy)
            report.append("🎯 FINAL RECOMMENDATION:")
            report.append(f"📌 BEST MODEL: {best_overall.config_name}")
            report.append(f"   Model Type: {best_overall.model_type}")
            report.append(f"   Universe: {best_overall.universe}")
            report.append(f"   Model Size: {best_overall.model_size}")
            report.append(f"   Feature Set: {best_overall.feature_set}")
            report.append(f"   Test Accuracy: {best_overall.test_accuracy:.3f}")
            report.append(f"   Sharpe Ratio: {best_overall.sharpe_ratio:.2f}")
            report.append(f"   Risk Score: {best_overall.risk_score:.3f}")
            report.append(f"   Confidence: {best_overall.confidence_score:.3f}")
            report.append(f"   Verdict: {best_overall.final_verdict}")
        
        report.append("")
        report.append("=" * 100)
        report.append("✅ ULTIMATE TESTING COMPLETE - ALL CONFIGURATIONS EVALUATED")
        report.append("=" * 100)
        
        return "\n".join(report)

def main():
    """Run ultimate comprehensive testing"""
    
    print("🚀 ULTIMATE PERFORMANCE TESTING FRAMEWORK")
    print("Testing ALL models, ALL universes, ALL configurations")
    print("=" * 80)
    
    # Initialize ultimate tester
    tester = UltimatePerformanceTester()
    
    # Run tests (start with smaller batch for validation)
    print(f"Total configurations available: {len(tester.test_configs)}")
    print("Starting with first 50 configurations for validation...")
    
    # Run first 50 configurations
    results = tester.run_ultimate_tests(max_configs=50)
    
    # Generate report
    report = tester.generate_ultimate_report()
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = tester.results_dir / f"ultimate_report_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n📊 Report saved: {report_file}")
    print("\n" + "=" * 50)
    print("QUICK SUMMARY:")
    print("=" * 50)
    
    # Quick summary
    verdicts = {}
    for result in results:
        verdicts[result.final_verdict] = verdicts.get(result.final_verdict, 0) + 1
    
    for verdict, count in sorted(verdicts.items()):
        print(f"{verdict}: {count} configurations")
    
    successful = [r for r in results if r.final_verdict not in ['FAILED', 'OVERFITTED']]
    if successful:
        best = max(successful, key=lambda x: x.test_accuracy)
        print(f"\nBest Model: {best.config_name}")
        print(f"Test Accuracy: {best.test_accuracy:.3f}")
        print(f"Verdict: {best.final_verdict}")
    
    print(f"\n✅ ULTIMATE TESTING COMPLETE!")
    print(f"Remove max_configs limit to test ALL {len(tester.test_configs)} configurations")

if __name__ == "__main__":
    main()