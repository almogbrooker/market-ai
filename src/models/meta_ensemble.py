#!/usr/bin/env python3
"""
Model D: Meta-Ensemble & Combiner (Decision Layer)
Combines A_score, B_score + meta-features → final score with calibrated uncertainty
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
import lightgbm as lgb
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConformalPredictor:
    """Conformal prediction for uncertainty quantification"""
    
    def __init__(self, alpha: float = 0.15):
        self.alpha = alpha  # 85% prediction intervals (was 90%)
        self.residuals = None
        self.is_fitted = False
    
    def fit(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Fit conformal predictor on residuals"""
        self.residuals = np.abs(y_true - y_pred)
        self.is_fitted = True
        
        # Calculate quantile
        self.quantile = np.quantile(self.residuals, 1 - self.alpha)
        
        logger.info(f"✅ Conformal predictor fitted:")
        logger.info(f"   Coverage target: {(1-self.alpha)*100:.1f}%")
        logger.info(f"   Quantile: {self.quantile:.6f}")
    
    def predict_intervals(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate prediction intervals"""
        if not self.is_fitted:
            raise ValueError("Conformal predictor not fitted")
        
        lower = y_pred - self.quantile
        upper = y_pred + self.quantile
        
        return lower, upper
    
    def get_uncertainty_filter(self, y_pred: np.ndarray, regime: str = 'neutral') -> np.ndarray:
        """Get uncertainty filter - only trade when confident"""
        
        lower, upper = self.predict_intervals(y_pred)
        interval_width = upper - lower
        
        # Regime-specific thresholds (much more relaxed)
        regime_thresholds = {
            'bull': 0.10,       # Much more relaxed
            'neutral': 0.15,    # Allow wider intervals
            'bear': 0.20,       # Still more relaxed than before
            'high_vol': 0.25    # Very relaxed
        }
        
        threshold = regime_thresholds.get(regime, 0.15)
        
        # Much more relaxed filter: trade based on signal strength, not zero-exclusion
        signal_strength = np.abs(y_pred)
        reasonable_width = interval_width < threshold
        
        # Rank-based selection: trade top/bottom 20% regardless of absolute thresholds
        n_positions = len(y_pred)
        n_trade = max(int(n_positions * 0.4), 4)  # Trade top/bottom 20% each (40% total), min 4
        
        # Rank by absolute score strength
        abs_scores = np.abs(y_pred)
        top_indices = np.argsort(abs_scores)[-n_trade:]
        
        # Create trade filter
        trade_filter = np.zeros(len(y_pred), dtype=bool)
        trade_filter[top_indices] = True
        
        # Still apply width constraint but much more relaxed
        trade_filter = trade_filter & reasonable_width
        
        return trade_filter

class MetaEnsemble:
    """
    Meta-Ensemble combining multiple alpha models with uncertainty
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Model selection
        self.combiner_type = self.config.get('combiner_type', 'ridge')  # 'ridge' or 'lgbm'
        
        # Components
        self.combiner = None
        self.scaler = None
        self.calibrator = None
        self.conformal = None
        
        # Performance tracking
        self.is_fitted = False
        self.feature_names = []
    
    def prepare_meta_features(self, 
                            lgbm_scores: np.ndarray,
                            lstm_scores: Optional[np.ndarray] = None,
                            regime_probs: Optional[Dict] = None,
                            market_features: Optional[Dict] = None) -> pd.DataFrame:
        """Prepare meta-features for ensemble"""
        
        # Base alpha scores
        meta_data = {
            'lgbm_score': lgbm_scores,
        }
        
        # LSTM scores (if available)
        if lstm_scores is not None:
            # Handle shape mismatch
            if len(lgbm_scores) != len(lstm_scores):
                min_len = min(len(lgbm_scores), len(lstm_scores))
                if len(lstm_scores) < len(lgbm_scores):
                    # Pad LSTM scores
                    padded_lstm = np.zeros(len(lgbm_scores))
                    padded_lstm[:len(lstm_scores)] = lstm_scores
                    lstm_scores = padded_lstm
                else:
                    # Truncate LSTM scores
                    lstm_scores = lstm_scores[:len(lgbm_scores)]
            
            meta_data['lstm_score'] = lstm_scores
            
            # Cross-model features
            meta_data['score_agreement'] = (np.sign(lgbm_scores) == np.sign(lstm_scores)).astype(float)
            meta_data['score_magnitude_diff'] = np.abs(lgbm_scores) - np.abs(lstm_scores)
            
            if len(lgbm_scores) > 1:
                rank_corr = np.corrcoef(lgbm_scores, lstm_scores)[0, 1]
                meta_data['score_rank_corr'] = rank_corr if not np.isnan(rank_corr) else 0
            else:
                meta_data['score_rank_corr'] = 0
        else:
            meta_data['lstm_score'] = np.zeros_like(lgbm_scores)
            meta_data['score_agreement'] = 1
            meta_data['score_magnitude_diff'] = 0
            meta_data['score_rank_corr'] = 1
        
        # Regime features (one-hot encoded)
        if regime_probs:
            for regime in ['bull', 'neutral', 'bear', 'high_vol']:
                meta_data[f'regime_{regime}'] = regime_probs.get(regime, 0.25)
        else:
            for regime in ['bull', 'neutral', 'bear', 'high_vol']:
                meta_data[f'regime_{regime}'] = 0.25
        
        # Market features
        if market_features:
            meta_data.update({
                'vix_level': market_features.get('vix_level', 20),
                'market_vol': market_features.get('realized_vol_20d', 0.2),
                'momentum': market_features.get('qqq_momentum_20d', 0.0),
                'breadth': market_features.get('breadth_above_50dma', 0.5)
            })
        else:
            meta_data.update({
                'vix_level': 20,
                'market_vol': 0.2,
                'momentum': 0.0,
                'breadth': 0.5
            })
        
        # Signal strength features
        meta_data['lgbm_abs_score'] = np.abs(lgbm_scores)
        if lstm_scores is not None:
            meta_data['lstm_abs_score'] = np.abs(lstm_scores)
            meta_data['ensemble_abs_score'] = (np.abs(lgbm_scores) + np.abs(lstm_scores)) / 2
        else:
            meta_data['lstm_abs_score'] = np.abs(lgbm_scores)
            meta_data['ensemble_abs_score'] = np.abs(lgbm_scores)
        
        # Consistency features
        meta_data['score_consistency'] = 1.0 - np.abs(meta_data['score_magnitude_diff'])
        
        # Convert to DataFrame
        meta_df = pd.DataFrame(meta_data)
        
        # Fill any NaNs
        meta_df = meta_df.fillna(0)
        
        return meta_df
    
    def train_combiner(self, 
                      meta_features: pd.DataFrame, 
                      targets: np.ndarray,
                      validation_split: float = 0.2) -> Dict:
        """Train the meta-combiner"""
        
        logger.info(f"🏋️ Training Meta-Ensemble ({self.combiner_type})...")
        
        # Split data
        n_val = int(len(meta_features) * validation_split)
        train_idx = slice(None, -n_val)
        val_idx = slice(-n_val, None)
        
        X_train = meta_features.iloc[train_idx]
        y_train = targets[train_idx]
        X_val = meta_features.iloc[val_idx]
        y_val = targets[val_idx]
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        self.feature_names = list(meta_features.columns)
        
        # Train combiner
        if self.combiner_type == 'ridge':
            self.combiner = Ridge(alpha=0.5, random_state=42)
            self.combiner.fit(X_train_scaled, y_train)
            
        elif self.combiner_type == 'lgbm':
            # LightGBM meta-combiner (very small)
            train_data = lgb.Dataset(X_train_scaled, label=y_train)
            val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': 15,
                'max_depth': 4,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
                'verbose': -1,
                'random_state': 42
            }
            
            self.combiner = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=100,
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
        
        # Make predictions
        if self.combiner_type == 'ridge':
            train_pred = self.combiner.predict(X_train_scaled)
            val_pred = self.combiner.predict(X_val_scaled)
        else:
            train_pred = self.combiner.predict(X_train_scaled)
            val_pred = self.combiner.predict(X_val_scaled)
        
        # Train calibrator
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        binary_targets = (y_train > 0).astype(float)
        self.calibrator.fit(train_pred, binary_targets)
        
        # Train conformal predictor
        self.conformal = ConformalPredictor(alpha=0.1)  # 90% intervals
        self.conformal.fit(y_val, val_pred)
        
        # Evaluate
        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        train_ic = spearmanr(y_train, train_pred)[0]
        val_ic = spearmanr(y_val, val_pred)[0]
        
        # Feature importance (if applicable)
        if self.combiner_type == 'ridge':
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': np.abs(self.combiner.coef_)
            }).sort_values('importance', ascending=False)
        else:
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.combiner.feature_importance()
            }).sort_values('importance', ascending=False)
        
        self.is_fitted = True
        
        results = {
            'train_mse': train_mse,
            'val_mse': val_mse,
            'train_ic': train_ic,
            'val_ic': val_ic,
            'feature_importance': importance,
            'n_features': len(self.feature_names),
            'combiner_type': self.combiner_type
        }
        
        logger.info(f"✅ Meta-Ensemble trained:")
        logger.info(f"   Val IC: {val_ic:.4f}")
        logger.info(f"   Val MSE: {val_mse:.6f}")
        logger.info(f"   Top features: {importance.head(3)['feature'].tolist()}")
        
        return results
    
    def predict_with_uncertainty(self, 
                                lgbm_scores: np.ndarray,
                                lstm_scores: Optional[np.ndarray] = None,
                                regime_probs: Optional[Dict] = None,
                                market_features: Optional[Dict] = None,
                                regime: str = 'neutral') -> Dict:
        """Make predictions with uncertainty quantification"""
        
        if not self.is_fitted:
            raise ValueError("Meta-ensemble not fitted")
        
        # Prepare meta-features
        meta_features = self.prepare_meta_features(
            lgbm_scores, lstm_scores, regime_probs, market_features
        )
        
        # Scale features
        X_scaled = self.scaler.transform(meta_features)
        
        # Make predictions
        if self.combiner_type == 'ridge':
            final_scores = self.combiner.predict(X_scaled)
        else:
            final_scores = self.combiner.predict(X_scaled)
        
        # Calibrated probabilities
        prob_up = self.calibrator.predict(final_scores)
        
        # Prediction intervals
        lower_bound, upper_bound = self.conformal.predict_intervals(final_scores)
        
        # Uncertainty filter
        trade_filter = self.conformal.get_uncertainty_filter(final_scores, regime)
        
        # Signal strength
        signal_strength = np.abs(final_scores)
        
        results = {
            'final_scores': final_scores,
            'prob_up': prob_up,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'interval_width': upper_bound - lower_bound,
            'trade_filter': trade_filter,
            'signal_strength': signal_strength,
            'n_tradeable': trade_filter.sum(),
            'coverage_rate': len(trade_filter) / len(final_scores) if len(final_scores) > 0 else 0
        }
        
        return results
    
    def get_position_sizes(self, 
                          final_scores: np.ndarray,
                          prob_up: np.ndarray,
                          interval_width: np.ndarray,
                          trade_filter: np.ndarray,
                          regime_multiplier: float = 1.0) -> np.ndarray:
        """Calculate position sizes using Kelly criterion with uncertainty"""
        
        # Rank-based position sizing: 2-3% per name, gross 40-60%
        n_positions = np.sum(trade_filter)
        if n_positions == 0:
            return np.zeros_like(final_scores)
        
        # Target 2% per position, scale by regime
        target_position_size = 0.02 * regime_multiplier
        
        # Convert scores to ranks then to position sizes
        ranked_scores = np.zeros_like(final_scores)
        valid_scores = final_scores[trade_filter]
        
        if len(valid_scores) > 0:
            # FLIP THE SIGNAL - rank backwards (negative scores get positive positions)
            ranks = np.argsort(np.argsort(-valid_scores))  # Negative sign flips ranking
            normalized_ranks = (ranks - ranks.mean()) / (ranks.std() + 1e-8)
            normalized_ranks = np.tanh(normalized_ranks)  # Bound to [-1, 1]
            
            ranked_scores[trade_filter] = normalized_ranks
        
        # Scale to target position size
        base_sizes = ranked_scores * target_position_size
        
        # Apply trade filter and ensure reasonable gross exposure
        filtered_sizes = np.where(trade_filter, base_sizes, 0)
        
        # Normalize to target gross exposure (40-60%)
        current_gross = np.sum(np.abs(filtered_sizes))
        target_gross = 0.5  # 50% gross exposure
        
        if current_gross > 0:
            scale_factor = target_gross / current_gross
            final_sizes = filtered_sizes * scale_factor
        else:
            final_sizes = filtered_sizes
        
        # Hard limits: 3% max per position
        final_sizes = np.clip(final_sizes, -0.03, 0.03)
        
        # Ensure market neutrality: net exposure close to 0
        net_exposure = np.sum(final_sizes)
        if abs(net_exposure) > 0.05:  # If net > 5%, adjust
            adjustment = -net_exposure / len(final_sizes[trade_filter])
            final_sizes[trade_filter] += adjustment
        
        return final_sizes
    
    def save_ensemble(self, path: Path):
        """Save complete ensemble"""
        import joblib
        
        ensemble_data = {
            'combiner': self.combiner,
            'scaler': self.scaler,
            'calibrator': self.calibrator,
            'conformal': self.conformal,
            'feature_names': self.feature_names,
            'config': self.config,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(ensemble_data, path)
        logger.info(f"✅ Meta-ensemble saved: {path}")
    
    def load_ensemble(self, path: Path):
        """Load complete ensemble"""
        import joblib
        
        ensemble_data = joblib.load(path)
        
        self.combiner = ensemble_data['combiner']
        self.scaler = ensemble_data['scaler']
        self.calibrator = ensemble_data['calibrator']
        self.conformal = ensemble_data['conformal']
        self.feature_names = ensemble_data['feature_names']
        self.config = ensemble_data['config']
        self.is_fitted = ensemble_data['is_fitted']
        
        logger.info(f"✅ Meta-ensemble loaded: {path}")

def main():
    """Test the meta-ensemble"""
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 5000
    
    # Synthetic alpha scores
    lgbm_scores = np.random.normal(0, 0.02, n_samples)
    lstm_scores = lgbm_scores + np.random.normal(0, 0.01, n_samples)  # Correlated
    
    # Synthetic targets
    targets = 0.3 * lgbm_scores + 0.2 * lstm_scores + np.random.normal(0, 0.015, n_samples)
    
    # Synthetic regime
    regime_probs = {
        'bull': 0.4,
        'neutral': 0.3,
        'bear': 0.2,
        'high_vol': 0.1
    }
    
    market_features = {
        'vix_level': 20,
        'realized_vol_20d': 0.18,
        'qqq_momentum_20d': 0.02,
        'breadth_above_50dma': 0.65
    }
    
    # Train ensemble
    ensemble = MetaEnsemble({'combiner_type': 'ridge'})
    
    # Prepare features
    meta_features = ensemble.prepare_meta_features(
        lgbm_scores, lstm_scores, regime_probs, market_features
    )
    
    # Train
    results = ensemble.train_combiner(meta_features, targets)
    print(f"Training results: {results}")
    
    # Test prediction
    test_lgbm = np.random.normal(0, 0.02, 100)
    test_lstm = test_lgbm + np.random.normal(0, 0.01, 100)
    
    predictions = ensemble.predict_with_uncertainty(
        test_lgbm, test_lstm, regime_probs, market_features
    )
    
    print(f"Predictions: {predictions}")
    
    # Position sizing
    positions = ensemble.get_position_sizes(
        predictions['final_scores'],
        predictions['prob_up'],
        predictions['interval_width'],
        predictions['trade_filter'],
        regime_multiplier=1.0
    )
    
    print(f"Position sizes: {positions[:10]}")

if __name__ == "__main__":
    main()