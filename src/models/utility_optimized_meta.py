#!/usr/bin/env python3
"""
UTILITY-OPTIMIZED META-MODEL
Optimizes for expected utility rather than accuracy
Focus: prob(return > costs + slippage) × E[excess return | trade]
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_score
from scipy import optimize
from typing import Dict, List, Tuple, Optional, Union
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class UtilityOptimizedMetaModel(BaseEstimator, RegressorMixin):
    """
    Meta-model that optimizes expected utility for trading
    Target: prob(return > costs) × expected_excess_return
    """
    
    def __init__(self,
                 transaction_cost: float = 0.001,
                 slippage_cost: float = 0.0005,
                 borrow_cost: float = 0.002,
                 confidence_threshold: float = 0.6,
                 utility_function: str = 'kelly',
                 regime_aware: bool = True):
        """
        Initialize utility-optimized meta-model
        
        Args:
            transaction_cost: Trading cost (bps)
            slippage_cost: Slippage cost (bps)  
            borrow_cost: Borrow cost for shorts (bps)
            confidence_threshold: Minimum confidence to trade
            utility_function: 'kelly', 'sharpe', or 'calmar'
            regime_aware: Use regime-specific costs/thresholds
        """
        self.transaction_cost = transaction_cost
        self.slippage_cost = slippage_cost
        self.borrow_cost = borrow_cost
        self.confidence_threshold = confidence_threshold
        self.utility_function = utility_function
        self.regime_aware = regime_aware
        
        # Models
        self.probability_model = None  # P(return > costs)
        self.magnitude_model = None    # E[return | trade]
        self.utility_model = None      # Combined utility
        self.calibrator = None         # Probability calibration
        
        # Regime-specific parameters
        self.regime_costs = {}
        self.regime_thresholds = {}
        
        logger.info(f"🎯 UtilityOptimizedMetaModel initialized")
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            base_predictions: Optional[np.ndarray] = None,
            regimes: Optional[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None) -> 'UtilityOptimizedMetaModel':
        """
        Fit meta-model to optimize expected utility
        
        Args:
            X: Base model predictions and features
            y: True returns/alpha
            base_predictions: Predictions from base models
            regimes: Market regime indicators
            sample_weight: Sample weights
        """
        logger.info(f"🔧 Fitting utility-optimized meta-model on {len(X)} samples")
        
        # Prepare features
        if base_predictions is not None:
            X_combined = np.hstack([X, base_predictions])
        else:
            X_combined = X
        
        # Calculate trading costs
        total_costs = self._calculate_total_costs(y, regimes)
        
        # Create utility targets
        utility_targets = self._calculate_utility_targets(y, total_costs, regimes)
        
        # Fit probability model: P(return > costs)
        self._fit_probability_model(X_combined, y, total_costs, sample_weight)
        
        # Fit magnitude model: E[return | return > costs]
        self._fit_magnitude_model(X_combined, y, total_costs, sample_weight)
        
        # Fit combined utility model
        self._fit_utility_model(X_combined, utility_targets, sample_weight)
        
        # Fit probability calibration
        self._fit_calibration(X_combined, y, total_costs)
        
        # Learn regime-specific parameters
        if self.regime_aware and regimes is not None:
            self._fit_regime_parameters(X_combined, y, regimes, total_costs)
        
        logger.info("✅ Meta-model fitting completed")
        return self
    
    def predict(self, X: np.ndarray, 
                base_predictions: Optional[np.ndarray] = None,
                regimes: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict expected utility for trading decisions
        
        Returns:
            Expected utility scores (higher = better trade)
        """
        # Prepare features
        if base_predictions is not None:
            X_combined = np.hstack([X, base_predictions])
        else:
            X_combined = X
        
        # Get regime-specific costs
        if self.regime_aware and regimes is not None:
            costs = self._get_regime_costs(regimes)
        else:
            costs = np.full(len(X), self.transaction_cost + self.slippage_cost)
        
        # Get probability predictions
        probabilities = self._predict_probability(X_combined, costs)
        
        # Get magnitude predictions  
        magnitudes = self._predict_magnitude(X_combined)
        
        # Calculate expected utility
        utility_scores = self._calculate_expected_utility(probabilities, magnitudes, costs)
        
        return utility_scores
    
    def predict_proba(self, X: np.ndarray,
                     base_predictions: Optional[np.ndarray] = None,
                     regimes: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict probability of profitable trade
        
        Returns:
            Calibrated probabilities
        """
        if base_predictions is not None:
            X_combined = np.hstack([X, base_predictions])
        else:
            X_combined = X
        
        if self.regime_aware and regimes is not None:
            costs = self._get_regime_costs(regimes)
        else:
            costs = np.full(len(X), self.transaction_cost + self.slippage_cost)
        
        return self._predict_probability(X_combined, costs)
    
    def get_trading_signals(self, X: np.ndarray,
                          base_predictions: Optional[np.ndarray] = None,
                          regimes: Optional[np.ndarray] = None,
                          confidence_threshold: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        Get trading signals with utility-based filtering
        
        Returns:
            Dictionary with signals, probabilities, and expected returns
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        
        # Get predictions
        utility_scores = self.predict(X, base_predictions, regimes)
        probabilities = self.predict_proba(X, base_predictions, regimes)
        magnitudes = self._predict_magnitude(
            np.hstack([X, base_predictions]) if base_predictions is not None else X
        )
        
        # Apply confidence threshold
        trade_mask = probabilities >= confidence_threshold
        
        # Generate signals
        signals = np.zeros(len(X))
        signals[trade_mask] = np.sign(magnitudes[trade_mask]) * utility_scores[trade_mask]
        
        return {
            'signals': signals,
            'probabilities': probabilities,
            'expected_returns': magnitudes,
            'utility_scores': utility_scores,
            'trade_mask': trade_mask
        }
    
    def _fit_probability_model(self, X: np.ndarray, y: np.ndarray, 
                             costs: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        """Fit model to predict P(return > costs)"""
        
        # Binary target: profitable after costs
        profitable = (np.abs(y) > costs).astype(int)
        
        # Fit LightGBM classifier
        train_data = lgb.Dataset(
            X, label=profitable, weight=sample_weight,
            feature_name=[f'feature_{i}' for i in range(X.shape[1])]
        )
        
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        self.probability_model = lgb.train(
            params, train_data,
            num_boost_round=200,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        logger.info(f"📊 Probability model fitted - positive rate: {profitable.mean():.3f}")
    
    def _fit_magnitude_model(self, X: np.ndarray, y: np.ndarray,
                           costs: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        """Fit model to predict E[return | return > costs]"""
        
        # Filter to profitable trades only
        profitable_mask = np.abs(y) > costs
        
        if profitable_mask.sum() < 50:
            logger.warning("⚠️ Too few profitable samples for magnitude model")
            # Fallback: use all samples
            X_filtered = X
            y_filtered = y
            weights_filtered = sample_weight
        else:
            X_filtered = X[profitable_mask]
            y_filtered = y[profitable_mask]
            weights_filtered = sample_weight[profitable_mask] if sample_weight is not None else None
        
        # Fit LightGBM regressor
        train_data = lgb.Dataset(
            X_filtered, label=y_filtered, weight=weights_filtered,
            feature_name=[f'feature_{i}' for i in range(X.shape[1])]
        )
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt', 
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        self.magnitude_model = lgb.train(
            params, train_data,
            num_boost_round=200,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        logger.info(f"📊 Magnitude model fitted on {len(y_filtered)} profitable samples")
    
    def _fit_utility_model(self, X: np.ndarray, utility_targets: np.ndarray,
                         sample_weight: Optional[np.ndarray] = None):
        """Fit model to directly predict utility"""
        
        train_data = lgb.Dataset(
            X, label=utility_targets, weight=sample_weight,
            feature_name=[f'feature_{i}' for i in range(X.shape[1])]
        )
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 63,  # Larger for complex utility function
            'learning_rate': 0.03,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        self.utility_model = lgb.train(
            params, train_data,
            num_boost_round=300,
            valid_sets=[train_data], 
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        logger.info(f"📊 Utility model fitted - mean utility: {utility_targets.mean():.6f}")
    
    def _fit_calibration(self, X: np.ndarray, y: np.ndarray, costs: np.ndarray):
        """Fit probability calibration"""
        
        # Get raw probabilities
        raw_probabilities = self.probability_model.predict(X)
        
        # True binary outcomes
        true_outcomes = (np.abs(y) > costs).astype(int)
        
        # Fit isotonic calibration
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(raw_probabilities, true_outcomes)
        
        logger.info("📊 Probability calibration fitted")
    
    def _fit_regime_parameters(self, X: np.ndarray, y: np.ndarray, 
                             regimes: np.ndarray, costs: np.ndarray):
        """Learn regime-specific costs and thresholds"""
        
        unique_regimes = np.unique(regimes)
        
        for regime in unique_regimes:
            regime_mask = regimes == regime
            regime_returns = y[regime_mask]
            regime_costs = costs[regime_mask]
            
            if len(regime_returns) < 20:
                continue
            
            # Learn regime-specific cost adjustments
            realized_volatility = np.std(regime_returns)
            base_vol = np.std(y)
            vol_adjustment = realized_volatility / (base_vol + 1e-6)
            
            # Adjust costs based on volatility
            adjusted_cost = self.transaction_cost * vol_adjustment
            
            self.regime_costs[regime] = adjusted_cost
            
            # Learn optimal confidence threshold for this regime
            X_regime = X[regime_mask]
            prob_preds = self.probability_model.predict(X_regime)
            
            # Find threshold that maximizes utility
            thresholds = np.percentile(prob_preds, [10, 20, 30, 40, 50, 60, 70, 80, 90])
            best_threshold = self.confidence_threshold
            best_utility = -np.inf
            
            for threshold in thresholds:
                trade_mask = prob_preds >= threshold
                if trade_mask.sum() < 5:
                    continue
                
                # Calculate utility for this threshold
                utility = self._calculate_regime_utility(
                    regime_returns[trade_mask], adjusted_cost
                )
                
                if utility > best_utility:
                    best_utility = utility
                    best_threshold = threshold
            
            self.regime_thresholds[regime] = best_threshold
            
            logger.info(f"📊 Regime {regime}: cost={adjusted_cost:.4f}, threshold={best_threshold:.3f}")
    
    def _calculate_total_costs(self, y: np.ndarray, regimes: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate total trading costs including regime adjustments"""
        
        base_cost = self.transaction_cost + self.slippage_cost
        
        # Add borrow costs for shorts
        short_mask = y < 0
        costs = np.full(len(y), base_cost)
        costs[short_mask] += self.borrow_cost
        
        # Regime adjustments
        if self.regime_aware and regimes is not None:
            for regime in np.unique(regimes):
                if regime in self.regime_costs:
                    regime_mask = regimes == regime
                    costs[regime_mask] = self.regime_costs[regime]
        
        return costs
    
    def _calculate_utility_targets(self, y: np.ndarray, costs: np.ndarray,
                                 regimes: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate utility targets for training"""
        
        if self.utility_function == 'kelly':
            # Kelly criterion: f* = (bp - q) / b
            # where b = odds, p = win prob, q = lose prob
            win_prob = (np.abs(y) > costs).astype(float)
            avg_win = np.where(np.abs(y) > costs, np.abs(y) - costs, 0)
            avg_loss = np.where(np.abs(y) <= costs, costs, 0)
            
            # Smooth estimates to avoid division by zero
            avg_win = np.clip(avg_win, 1e-6, None)
            avg_loss = np.clip(avg_loss, 1e-6, None)
            
            kelly_fraction = (avg_win * win_prob - avg_loss * (1 - win_prob)) / avg_win
            utility_targets = np.clip(kelly_fraction, -1, 1)
            
        elif self.utility_function == 'sharpe':
            # Sharpe-like utility: (return - cost) / volatility
            excess_returns = np.abs(y) - costs
            utility_targets = excess_returns / (np.std(y) + 1e-6)
            
        else:  # 'calmar' or default
            # Calmar-like: return / max drawdown approximation
            excess_returns = np.abs(y) - costs
            downside_risk = np.where(excess_returns < 0, -excess_returns, 0)
            utility_targets = excess_returns / (np.mean(downside_risk) + 1e-6)
        
        return utility_targets
    
    def _predict_probability(self, X: np.ndarray, costs: np.ndarray) -> np.ndarray:
        """Predict calibrated probabilities"""
        raw_probs = self.probability_model.predict(X)
        
        if self.calibrator is not None:
            calibrated_probs = self.calibrator.transform(raw_probs)
        else:
            calibrated_probs = raw_probs
        
        return np.clip(calibrated_probs, 0.01, 0.99)
    
    def _predict_magnitude(self, X: np.ndarray) -> np.ndarray:
        """Predict expected return magnitudes"""
        return self.magnitude_model.predict(X)
    
    def _calculate_expected_utility(self, probabilities: np.ndarray,
                                  magnitudes: np.ndarray, costs: np.ndarray) -> np.ndarray:
        """Calculate expected utility scores"""
        
        # Expected return = P(profit) × E[return | profit] - P(loss) × cost
        expected_returns = probabilities * np.abs(magnitudes) - (1 - probabilities) * costs
        
        # Utility adjustment based on function type
        if self.utility_function == 'kelly':
            # Kelly optimal betting fraction
            utility_scores = expected_returns / (np.abs(magnitudes) + 1e-6)
        else:
            utility_scores = expected_returns
        
        return utility_scores
    
    def _get_regime_costs(self, regimes: np.ndarray) -> np.ndarray:
        """Get regime-specific costs"""
        costs = np.full(len(regimes), self.transaction_cost + self.slippage_cost)
        
        for regime in np.unique(regimes):
            if regime in self.regime_costs:
                regime_mask = regimes == regime
                costs[regime_mask] = self.regime_costs[regime]
        
        return costs
    
    def _calculate_regime_utility(self, returns: np.ndarray, cost: float) -> float:
        """Calculate utility for a regime"""
        if len(returns) < 5:
            return -np.inf
        
        excess_returns = np.abs(returns) - cost
        win_rate = np.mean(excess_returns > 0)
        avg_win = np.mean(excess_returns[excess_returns > 0]) if np.any(excess_returns > 0) else 0
        avg_loss = np.mean(-excess_returns[excess_returns <= 0]) if np.any(excess_returns <= 0) else cost
        
        if avg_loss == 0:
            return avg_win * win_rate
        
        # Kelly-like utility
        utility = win_rate * avg_win / avg_loss - (1 - win_rate)
        return utility
    
    def feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from all models"""
        importance = {}
        
        if self.probability_model is not None:
            importance['probability'] = self.probability_model.feature_importance(importance_type='gain')
        
        if self.magnitude_model is not None:
            importance['magnitude'] = self.magnitude_model.feature_importance(importance_type='gain')
            
        if self.utility_model is not None:
            importance['utility'] = self.utility_model.feature_importance(importance_type='gain')
        
        return importance

# Usage Example
def example_utility_optimization():
    """Example of utility-optimized meta-model usage"""
    
    # Simulated data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    base_preds = np.random.randn(n_samples, 3)  # 3 base models
    y = np.random.randn(n_samples) * 0.02  # 2% volatility returns
    regimes = np.random.choice(['bull', 'bear', 'neutral'], n_samples)
    
    # Initialize and fit model
    meta_model = UtilityOptimizedMetaModel(
        transaction_cost=0.001,
        slippage_cost=0.0005,
        confidence_threshold=0.6,
        utility_function='kelly'
    )
    
    meta_model.fit(X, y, base_preds, regimes)
    
    # Get trading signals
    signals = meta_model.get_trading_signals(X, base_preds, regimes)
    
    logger.info(f"Trading signals generated:")
    logger.info(f"  Signals: {len(signals['signals'])} total")
    logger.info(f"  Tradeable: {signals['trade_mask'].sum()} ({signals['trade_mask'].mean():.1%})")
    logger.info(f"  Avg probability: {signals['probabilities'].mean():.3f}")
    logger.info(f"  Avg utility: {signals['utility_scores'].mean():.6f}")

if __name__ == "__main__":
    example_utility_optimization()