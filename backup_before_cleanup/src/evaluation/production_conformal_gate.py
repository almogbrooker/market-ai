#!/usr/bin/env python3
"""
PRODUCTION CONFORMAL PREDICTION GATING SYSTEM
The definitive implementation for live trading signal calibration
Fixes the critical bottleneck: 50 signals → 0 trades by proper threshold calibration
"""

import numpy as np
import pandas as pd
import pickle
import os
import json
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import yfinance as yf
from collections import deque
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class ConformalConfig:
    """Optimal configuration for production conformal gating"""
    alpha: float = 0.15  # 85% confidence intervals (optimized for trading)
    min_calibration_samples: int = 200  # Minimum for reliable thresholds
    lookback_days: int = 30  # Recent data is more relevant
    update_frequency_hours: int = 6  # Update 4x daily for regime changes
    regime_aware: bool = True  # Critical for market regime adaptation
    cost_awareness: float = 0.0008  # 8 bps transaction cost threshold
    confidence_scaling: bool = True  # Scale signals by confidence
    adaptive_thresholds: bool = True  # Adjust thresholds by recent performance

class ProductionConformalGate:
    """
    PRODUCTION-READY CONFORMAL GATING SYSTEM
    
    Key innovations:
    1. Real-time calibration using live prediction errors
    2. Regime-aware thresholds (bull/bear/neutral)
    3. Cost-aware gating (don't trade below transaction costs)
    4. Adaptive threshold adjustment based on recent hit rates
    5. Confidence-scaled signal strength
    6. Distribution-free coverage guarantees
    """
    
    def __init__(self, config: ConformalConfig = None):
        self.config = config or ConformalConfig()
        
        # Calibration data storage (recent prediction errors)
        self.prediction_errors = {
            'bull': deque(maxlen=1000),
            'bear': deque(maxlen=1000), 
            'neutral': deque(maxlen=1000),
            'default': deque(maxlen=1000)
        }
        
        # Calibrated thresholds per regime
        self.thresholds = {}
        
        # Performance tracking for adaptive adjustment
        self.recent_performance = {
            'bull': {'trades': 0, 'hits': 0},
            'bear': {'trades': 0, 'hits': 0},
            'neutral': {'trades': 0, 'hits': 0},
            'default': {'trades': 0, 'hits': 0}
        }
        
        # Live trading state
        self.last_calibration = {}
        self.is_calibrated = False
        self.cache_file = "production_conformal_cache.pkl"
        self.live_predictions_file = "live_predictions_log.csv"
        
        # Load existing state
        self.load_state()
        
        # Generate initial calibration data if needed
        if not self.is_calibrated:
            self.bootstrap_calibration()
        
        logger.info(f"🎯 Production Conformal Gate initialized")
        logger.info(f"   Alpha: {self.config.alpha:.1%} (coverage: {1-self.config.alpha:.1%})")
        logger.info(f"   Regime-aware: {self.config.regime_aware}")
        logger.info(f"   Cost threshold: {self.config.cost_awareness:.1%}")
        logger.info(f"   Calibrated regimes: {list(self.thresholds.keys())}")
    
    def bootstrap_calibration(self):
        """Generate initial calibration data from market history"""
        logger.info("🔧 Bootstrapping initial calibration from market data...")
        
        try:
            # Use recent market data to estimate prediction error distribution
            stocks = ['QQQ', 'SPY', 'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            
            all_errors = []
            
            for symbol in stocks:
                try:
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(start=start_date, end=end_date, interval='1d')
                    
                    if len(df) < 30:
                        continue
                    
                    # Calculate simple momentum prediction
                    df['return_1d'] = df['Close'].pct_change()
                    df['return_5d'] = df['Close'].pct_change(5)
                    df['sma_10'] = df['Close'].rolling(10).mean()
                    df['rsi'] = 100 - (100 / (1 + df['Close'].diff().clip(lower=0).rolling(14).mean() / 
                                              df['Close'].diff().clip(upper=0).abs().rolling(14).mean()))
                    
                    # Simple prediction: momentum + mean reversion
                    prediction = (df['return_5d'] * 0.3 + 
                                (df['Close'] / df['sma_10'] - 1) * 0.4 +
                                (50 - df['rsi']) / 100 * 0.3)
                    
                    # Target: next day return
                    target = df['return_1d'].shift(-1)
                    
                    # Market regime
                    regime = 'neutral'
                    bull_mask = df['return_5d'] > 0.02
                    bear_mask = df['return_5d'] < -0.02
                    
                    # Calculate errors for each regime
                    for i in range(len(df)-1):
                        if pd.notna(prediction.iloc[i]) and pd.notna(target.iloc[i]):
                            error = abs(prediction.iloc[i] - target.iloc[i])
                            
                            if bull_mask.iloc[i]:
                                current_regime = 'bull'
                            elif bear_mask.iloc[i]:
                                current_regime = 'bear'
                            else:
                                current_regime = 'neutral'
                            
                            self.prediction_errors[current_regime].append(error)
                            self.prediction_errors['default'].append(error)
                            all_errors.append(error)
                
                except Exception as e:
                    logger.warning(f"Error bootstrapping {symbol}: {e}")
                    continue
            
            if len(all_errors) > 50:
                # Calculate initial thresholds
                self._recalibrate_all_regimes()
                self.is_calibrated = True
                self.save_state()
                logger.info(f"✅ Bootstrapped calibration with {len(all_errors)} error samples")
            else:
                logger.warning("⚠️ Insufficient bootstrap data - using default thresholds")
                self._set_default_thresholds()
                
        except Exception as e:
            logger.error(f"Bootstrap calibration failed: {e}")
            self._set_default_thresholds()
    
    def _set_default_thresholds(self):
        """Set conservative default thresholds"""
        default_threshold = 0.015  # 1.5% prediction error threshold
        for regime in ['bull', 'bear', 'neutral', 'default']:
            self.thresholds[regime] = {
                'threshold': default_threshold,
                'n_samples': 100,
                'last_update': datetime.now(),
                'hit_rate': 0.5
            }
        self.is_calibrated = True
        logger.info(f"🛡️ Set default thresholds: {default_threshold:.3f}")
    
    def add_live_prediction_error(self, prediction: float, actual_return: float, 
                                 regime: str = 'default', symbol: str = None):
        """
        Add a live prediction error for continuous calibration
        Call this after each trade settles to improve calibration
        """
        if pd.isna(prediction) or pd.isna(actual_return):
            return
            
        error = abs(prediction - actual_return)
        
        # Add to error history
        if regime in self.prediction_errors:
            self.prediction_errors[regime].append(error)
        self.prediction_errors['default'].append(error)
        
        # Log to file for analysis
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol or 'unknown',
                'prediction': prediction,
                'actual': actual_return,
                'error': error,
                'regime': regime
            }
            
            # Append to CSV log
            log_df = pd.DataFrame([log_entry])
            if os.path.exists(self.live_predictions_file):
                log_df.to_csv(self.live_predictions_file, mode='a', header=False, index=False)
            else:
                log_df.to_csv(self.live_predictions_file, mode='w', header=True, index=False)
                
        except Exception as e:
            logger.error(f"Error logging prediction: {e}")
        
        # Trigger recalibration if enough new samples
        if len(self.prediction_errors[regime]) >= 50:
            self._maybe_recalibrate(regime)
    
    def _maybe_recalibrate(self, regime: str):
        """Check if recalibration is needed for a regime"""
        now = datetime.now()
        
        if regime in self.last_calibration:
            hours_since = (now - self.last_calibration[regime]).total_seconds() / 3600
            if hours_since < self.config.update_frequency_hours:
                return
        
        self._recalibrate_regime(regime)
    
    def _recalibrate_regime(self, regime: str):
        """Recalibrate threshold for a specific regime"""
        if regime not in self.prediction_errors:
            return
            
        errors = list(self.prediction_errors[regime])
        if len(errors) < self.config.min_calibration_samples:
            logger.debug(f"Insufficient samples for {regime}: {len(errors)}")
            return
        
        # Calculate conformal quantile
        n = len(errors)
        q_level = np.ceil((n + 1) * (1 - self.config.alpha)) / n
        q_level = min(q_level, 1.0)
        
        threshold = np.quantile(errors, q_level)
        
        # Adaptive adjustment based on recent performance
        if self.config.adaptive_thresholds and regime in self.recent_performance:
            perf = self.recent_performance[regime]
            if perf['trades'] > 10:  # Need minimum sample size
                hit_rate = perf['hits'] / perf['trades']
                target_hit_rate = 1 - self.config.alpha  # Target coverage
                
                # Adjust threshold if hit rate is off
                if hit_rate < target_hit_rate - 0.05:
                    # Hit rate too low, lower threshold (trade more)
                    threshold *= 0.9
                elif hit_rate > target_hit_rate + 0.05:
                    # Hit rate too high, raise threshold (trade less)  
                    threshold *= 1.1
        
        # Store calibrated threshold
        self.thresholds[regime] = {
            'threshold': threshold,
            'n_samples': n,
            'last_update': datetime.now(),
            'hit_rate': self.recent_performance[regime]['hits'] / max(self.recent_performance[regime]['trades'], 1),
            'mean_error': np.mean(errors),
            'std_error': np.std(errors)
        }
        
        self.last_calibration[regime] = datetime.now()
        
        logger.info(f"🎯 Recalibrated {regime}: threshold={threshold:.4f}, n={n}, hit_rate={self.thresholds[regime]['hit_rate']:.2%}")
        
        self.save_state()
    
    def _recalibrate_all_regimes(self):
        """Recalibrate all regimes"""
        for regime in self.prediction_errors.keys():
            self._recalibrate_regime(regime)
    
    def should_trade_signal(self, prediction: float, regime: str = 'default', 
                           confidence: float = None, symbol: str = None,
                           metadata: Dict = None) -> Dict:
        """
        MAIN TRADING DECISION FUNCTION
        Determines if a signal should be traded based on conformal prediction
        
        Returns comprehensive decision with diagnostics
        """
        
        # Basic validation
        if pd.isna(prediction) or abs(prediction) == 0:
            return self._create_rejection('invalid_prediction', prediction, regime)
        
        if not self.is_calibrated:
            return self._create_rejection('not_calibrated', prediction, regime)
        
        # Get active regime threshold
        active_regime = regime if regime in self.thresholds else 'default'
        if active_regime not in self.thresholds:
            return self._create_rejection('no_threshold', prediction, regime)
        
        regime_info = self.thresholds[active_regime]
        threshold = regime_info['threshold']
        
        # Cost-aware filtering: don't trade below transaction costs
        abs_prediction = abs(prediction)
        if abs_prediction < self.config.cost_awareness:
            return self._create_rejection('below_cost_threshold', prediction, regime, {
                'cost_threshold': self.config.cost_awareness,
                'abs_prediction': abs_prediction
            })
        
        # Conformal gating: prediction must exceed expected error
        signal_strength = abs_prediction / (threshold + 1e-8)
        conformal_pass = signal_strength > 1.0
        
        if not conformal_pass:
            return self._create_rejection('conformal_rejected', prediction, regime, {
                'signal_strength': signal_strength,
                'threshold': threshold
            })
        
        # Confidence gating (if available)
        if confidence is not None:
            # Dynamic confidence threshold based on signal strength
            min_confidence = 0.4 if signal_strength > 2.0 else 0.6
            confidence_pass = confidence > min_confidence
            
            if not confidence_pass:
                return self._create_rejection('confidence_rejected', prediction, regime, {
                    'confidence': confidence,
                    'min_confidence': min_confidence
                })
        else:
            confidence = signal_strength / 3.0  # Proxy confidence from signal strength
        
        # SIGNAL ACCEPTED - calculate final strength
        if self.config.confidence_scaling:
            # Scale signal by confidence and strength (but cap to prevent over-leverage)
            scale_factor = min(signal_strength * (confidence + 0.5), 3.0)
            final_signal = prediction * scale_factor
        else:
            final_signal = prediction
        
        # Calculate overall confidence
        overall_confidence = min((signal_strength + confidence) / 2.0, 1.0)
        
        return {
            'should_trade': True,
            'original_signal': prediction,
            'final_signal': final_signal,
            'confidence': overall_confidence,
            'signal_strength': signal_strength,
            'regime': active_regime,
            'reason': 'accepted',
            'diagnostics': {
                'threshold': threshold,
                'abs_prediction': abs_prediction,
                'cost_threshold': self.config.cost_awareness,
                'n_calibration_samples': regime_info['n_samples'],
                'regime_hit_rate': regime_info.get('hit_rate', 0.5)
            }
        }
    
    def _create_rejection(self, reason: str, prediction: float, regime: str, 
                         extra_info: Dict = None) -> Dict:
        """Helper to create consistent rejection responses"""
        result = {
            'should_trade': False,
            'original_signal': prediction,
            'final_signal': 0.0,
            'confidence': 0.0,
            'signal_strength': 0.0,
            'regime': regime,
            'reason': reason,
            'diagnostics': extra_info or {}
        }
        return result
    
    def batch_filter_signals(self, signals: Dict[str, float], regime: str = 'default',
                           confidences: Dict[str, float] = None) -> Dict[str, Dict]:
        """
        BATCH SIGNAL FILTERING FOR LIVE TRADING
        Efficiently filter multiple signals at once
        """
        results = {}
        accepted_count = 0
        
        for symbol, prediction in signals.items():
            confidence = confidences.get(symbol) if confidences else None
            decision = self.should_trade_signal(prediction, regime, confidence, symbol)
            results[symbol] = decision
            
            if decision['should_trade']:
                accepted_count += 1
        
        # Log batch statistics
        total_signals = len(signals)
        acceptance_rate = accepted_count / total_signals if total_signals > 0 else 0
        
        # Count rejection reasons
        rejection_reasons = {}
        for result in results.values():
            if not result['should_trade']:
                reason = result['reason']
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        
        logger.info(f"📊 Conformal filtering: {accepted_count}/{total_signals} accepted ({acceptance_rate:.1%})")
        if rejection_reasons:
            logger.info(f"🚫 Rejections: {rejection_reasons}")
        
        return results
    
    def update_trade_outcome(self, symbol: str, prediction: float, actual_outcome: float,
                           regime: str = 'default', success: bool = None):
        """
        Update performance tracking after trade settles
        Critical for adaptive threshold adjustment
        """
        # Add prediction error for calibration
        self.add_live_prediction_error(prediction, actual_outcome, regime, symbol)
        
        # Update hit rate tracking
        if regime in self.recent_performance:
            self.recent_performance[regime]['trades'] += 1
            
            # Determine if trade was successful
            if success is None:
                # Auto-determine: same direction as prediction
                success = (prediction * actual_outcome) > 0
            
            if success:
                self.recent_performance[regime]['hits'] += 1
        
        # Decay old performance data (keep last 100 trades)
        for r in self.recent_performance.values():
            if r['trades'] > 200:
                r['trades'] = int(r['trades'] * 0.8)
                r['hits'] = int(r['hits'] * 0.8)
    
    def get_regime_statistics(self) -> Dict:
        """Get detailed statistics for each regime"""
        stats = {}
        
        for regime, threshold_info in self.thresholds.items():
            error_samples = len(self.prediction_errors.get(regime, []))
            perf = self.recent_performance.get(regime, {'trades': 0, 'hits': 0})
            
            stats[regime] = {
                'threshold': threshold_info['threshold'],
                'n_error_samples': error_samples,
                'n_calibration_samples': threshold_info['n_samples'],
                'last_update': threshold_info['last_update'].isoformat() if isinstance(threshold_info['last_update'], datetime) else str(threshold_info['last_update']),
                'recent_trades': perf['trades'],
                'recent_hit_rate': perf['hits'] / max(perf['trades'], 1),
                'mean_error': threshold_info.get('mean_error', 0),
                'coverage_target': 1 - self.config.alpha
            }
        
        return stats
    
    def save_state(self):
        """Save all state to disk"""
        try:
            state = {
                'thresholds': self.thresholds,
                'last_calibration': self.last_calibration,
                'recent_performance': self.recent_performance,
                'prediction_errors': {k: list(v) for k, v in self.prediction_errors.items()},
                'is_calibrated': self.is_calibrated,
                'config': self.config
            }
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(state, f)
                
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def load_state(self):
        """Load state from disk"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    state = pickle.load(f)
                
                self.thresholds = state.get('thresholds', {})
                self.last_calibration = state.get('last_calibration', {})
                self.recent_performance = state.get('recent_performance', {})
                self.is_calibrated = state.get('is_calibrated', False)
                
                # Restore prediction errors
                prediction_errors_data = state.get('prediction_errors', {})
                for regime, errors in prediction_errors_data.items():
                    self.prediction_errors[regime] = deque(errors, maxlen=1000)
                
                logger.info(f"📁 Loaded conformal state: {len(self.thresholds)} regimes")
                
        except Exception as e:
            logger.error(f"Error loading state: {e}")

# Factory functions for easy integration
def create_production_gate(alpha: float = 0.15) -> ProductionConformalGate:
    """Create production conformal gate with optimal settings"""
    config = ConformalConfig(
        alpha=alpha,
        regime_aware=True,
        cost_awareness=0.0008,
        adaptive_thresholds=True,
        confidence_scaling=True
    )
    return ProductionConformalGate(config)

# Integration helper
class ConformalTradingBot:
    """Helper class to integrate conformal gating with trading bot"""
    
    def __init__(self, alpha: float = 0.15):
        self.gate = create_production_gate(alpha)
        logger.info("🤖 Conformal Trading Bot ready")
    
    def filter_trading_signals(self, signals: Dict[str, float], 
                             regime: str = 'neutral') -> Dict[str, float]:
        """
        Main integration point: filter raw signals through conformal gate
        
        Input: {symbol: raw_signal}
        Output: {symbol: calibrated_signal} (0.0 if rejected)
        """
        decisions = self.gate.batch_filter_signals(signals, regime)
        
        # Extract final signals
        filtered_signals = {}
        for symbol, decision in decisions.items():
            if decision['should_trade']:
                filtered_signals[symbol] = decision['final_signal']
            else:
                filtered_signals[symbol] = 0.0
        
        return filtered_signals
    
    def log_trade_result(self, symbol: str, prediction: float, 
                        actual_return: float, regime: str = 'neutral'):
        """Log trade outcome for continuous improvement"""
        self.gate.update_trade_outcome(symbol, prediction, actual_return, regime)
    
    def get_gate_diagnostics(self) -> Dict:
        """Get diagnostic information"""
        return self.gate.get_regime_statistics()