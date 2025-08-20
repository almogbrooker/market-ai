#!/usr/bin/env python3
"""
Production Conformal Prediction Gating System
Calibrates trading thresholds using OOF predictions to reduce false positives
Based on split-conformal prediction for distribution-free coverage guarantees
"""

import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class ConformalConfig:
    """Configuration for conformal prediction gating"""
    alpha: float = 0.20  # Miscoverage level (20% = 80% confidence intervals)
    min_calibration_samples: int = 500  # Minimum samples for reliable calibration
    lookback_days: int = 60  # Days of historical data for calibration
    update_frequency_hours: int = 24  # How often to recalibrate
    regime_aware: bool = True  # Separate thresholds per market regime
    cost_awareness: float = 0.0005  # Transaction cost threshold (5 bps)

class ProductionConformalGate:
    """
    Production-ready conformal prediction gating for live trading
    Calibrates on OOF predictions to provide distribution-free trade filtering
    """
    
    def __init__(self, config: ConformalConfig = None):
        self.config = config or ConformalConfig()
        self.calibration_data = {}  # Store per-regime calibration data
        self.quantiles = {}  # Store per-regime quantiles
        self.last_calibration = {}  # Track when each regime was last calibrated
        self.is_calibrated = False
        self.cache_file = "conformal_gate_cache.pkl"
        
        # Load existing calibration if available
        self.load_calibration()
        
        logger.info(f"Conformal gate initialized: alpha={self.config.alpha:.2%}, "
                   f"regime_aware={self.config.regime_aware}")
    
    def add_calibration_data(self, predictions: np.ndarray, targets: np.ndarray, 
                           timestamps: np.ndarray, regime: str = 'default',
                           metadata: Dict = None):
        """
        Add historical OOF predictions for calibration
        
        Args:
            predictions: Model predictions (raw scores, not probabilities)
            targets: True binary labels or returns
            timestamps: Timestamps for each prediction
            regime: Market regime identifier ('bull', 'bear', 'neutral', etc.)
            metadata: Additional info (volatility, confidence, etc.)
        """
        if regime not in self.calibration_data:
            self.calibration_data[regime] = {
                'predictions': [],
                'targets': [], 
                'timestamps': [],
                'metadata': []
            }
        
        # Add new data
        self.calibration_data[regime]['predictions'].extend(predictions.flatten())
        self.calibration_data[regime]['targets'].extend(targets.flatten())
        self.calibration_data[regime]['timestamps'].extend(timestamps.flatten())
        
        if metadata:
            self.calibration_data[regime]['metadata'].extend([metadata] * len(predictions))
        
        logger.debug(f"Added {len(predictions)} calibration points for regime {regime}")
    
    def load_oof_predictions(self, oof_file: str, target_col: str = 'target',
                            prediction_col: str = 'prediction', 
                            regime_col: str = 'regime',
                            timestamp_col: str = 'date'):
        """
        Load out-of-fold predictions from saved file
        Expected format: CSV/parquet with columns [date, prediction, target, regime, ...]
        """
        try:
            if oof_file.endswith('.parquet'):
                df = pd.read_parquet(oof_file)
            else:
                df = pd.read_csv(oof_file)
            
            logger.info(f"Loading OOF data from {oof_file}: {len(df)} samples")
            
            # Convert timestamps
            if timestamp_col in df.columns:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                timestamps = df[timestamp_col].values
            else:
                timestamps = np.arange(len(df))
            
            # Group by regime if regime-aware
            if self.config.regime_aware and regime_col in df.columns:
                for regime in df[regime_col].unique():
                    regime_data = df[df[regime_col] == regime]
                    
                    if len(regime_data) >= self.config.min_calibration_samples:
                        self.add_calibration_data(
                            predictions=regime_data[prediction_col].values,
                            targets=regime_data[target_col].values,
                            timestamps=regime_data[timestamp_col].values if timestamp_col in df.columns else np.arange(len(regime_data)),
                            regime=str(regime)
                        )
                        logger.info(f"Loaded {len(regime_data)} samples for regime {regime}")
                    else:
                        logger.warning(f"Insufficient data for regime {regime}: {len(regime_data)} < {self.config.min_calibration_samples}")
            else:
                # Single regime
                self.add_calibration_data(
                    predictions=df[prediction_col].values,
                    targets=df[target_col].values, 
                    timestamps=timestamps,
                    regime='default'
                )
                
        except Exception as e:
            logger.error(f"Error loading OOF predictions: {e}")
            raise
    
    def calibrate_regime(self, regime: str, force_update: bool = False) -> bool:
        """
        Calibrate conformal quantiles for a specific regime
        Returns True if successful, False if insufficient data
        """
        if regime not in self.calibration_data:
            logger.warning(f"No calibration data for regime {regime}")
            return False
        
        data = self.calibration_data[regime]
        
        # Check if we have enough recent data
        if len(data['predictions']) < self.config.min_calibration_samples:
            logger.warning(f"Insufficient calibration data for {regime}: {len(data['predictions'])} < {self.config.min_calibration_samples}")
            return False
        
        # Check if recalibration is needed
        now = datetime.now()
        if not force_update and regime in self.last_calibration:
            hours_since_calibration = (now - self.last_calibration[regime]).total_seconds() / 3600
            if hours_since_calibration < self.config.update_frequency_hours:
                logger.debug(f"Skipping {regime} recalibration: {hours_since_calibration:.1f}h < {self.config.update_frequency_hours}h")
                return True
        
        # Use only recent data for calibration
        predictions = np.array(data['predictions'])
        targets = np.array(data['targets'])
        timestamps = np.array(data['timestamps'])
        
        # Filter to recent lookback period
        if len(timestamps) > 0 and hasattr(timestamps[0], 'timestamp'):
            cutoff_time = now - timedelta(days=self.config.lookback_days)
            recent_mask = timestamps >= cutoff_time
            predictions = predictions[recent_mask]
            targets = targets[recent_mask]
            
            if len(predictions) < self.config.min_calibration_samples:
                logger.warning(f"Insufficient recent data for {regime}: {len(predictions)} samples")
                return False
        
        # Calculate nonconformity scores (absolute residuals)
        residuals = np.abs(predictions - targets)
        
        # Remove outliers (optional robust estimator)
        p99 = np.percentile(residuals, 99)
        residuals = np.clip(residuals, 0, p99)
        
        # Calculate conformal quantile
        n = len(residuals)
        # Adjusted quantile level for finite sample correction
        q_level = np.ceil((n + 1) * (1 - self.config.alpha)) / n
        q_level = min(q_level, 1.0)  # Ensure <= 1.0
        
        threshold = np.quantile(residuals, q_level)
        
        # Store calibrated threshold
        if regime not in self.quantiles:
            self.quantiles[regime] = {}
        
        self.quantiles[regime] = {
            'threshold': threshold,
            'n_samples': n,
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'calibration_time': now
        }
        
        self.last_calibration[regime] = now
        self.is_calibrated = True
        
        logger.info(f"Calibrated {regime}: threshold={threshold:.4f}, n_samples={n}, mean_residual={np.mean(residuals):.4f}")
        
        # Save calibration
        self.save_calibration()
        
        return True
    
    def calibrate_all_regimes(self, force_update: bool = False):
        """Calibrate all available regimes"""
        success_count = 0
        for regime in self.calibration_data.keys():
            if self.calibrate_regime(regime, force_update):
                success_count += 1
        
        if success_count > 0:
            self.is_calibrated = True
            logger.info(f"Successfully calibrated {success_count} regimes")
        else:
            logger.warning("No regimes could be calibrated")
    
    def should_trade(self, prediction: float, regime: str = 'default', 
                    confidence: float = None, metadata: Dict = None) -> Dict:
        """
        Determine if we should trade based on conformal prediction intervals
        
        Args:
            prediction: Raw model prediction (return forecast, signal strength, etc.)
            regime: Current market regime
            confidence: Optional model confidence score
            metadata: Additional context (volatility, etc.)
            
        Returns:
            Dict with trade decision, adjusted signal, and diagnostics
        """
        if not self.is_calibrated:
            logger.warning("Conformal gate not calibrated - defaulting to no trade")
            return {
                'should_trade': False,
                'adjusted_signal': 0.0,
                'confidence': 0.0,
                'reason': 'not_calibrated',
                'diagnostics': {}
            }
        
        # Fall back to default regime if specific regime not available
        active_regime = regime if regime in self.quantiles else 'default'
        if active_regime not in self.quantiles:
            logger.warning(f"No calibration for regime {regime} or default")
            return {
                'should_trade': False, 
                'adjusted_signal': 0.0,
                'confidence': 0.0,
                'reason': 'no_regime_calibration',
                'diagnostics': {'requested_regime': regime}
            }
        
        regime_info = self.quantiles[active_regime]
        threshold = regime_info['threshold']
        
        # Calculate prediction interval width as proxy for uncertainty
        # Wider intervals = less confident = should not trade
        abs_prediction = abs(prediction)
        
        # Cost-aware gating: only trade if expected edge > transaction costs
        if abs_prediction < self.config.cost_awareness:
            return {
                'should_trade': False,
                'adjusted_signal': 0.0, 
                'confidence': 0.0,
                'reason': 'below_cost_threshold',
                'diagnostics': {
                    'prediction': prediction,
                    'cost_threshold': self.config.cost_awareness,
                    'regime': active_regime
                }
            }
        
        # Conformal gating: trade if prediction magnitude exceeds calibrated threshold
        prediction_interval_half_width = threshold
        
        # Trade if prediction is significant relative to expected residual
        signal_to_noise = abs_prediction / (threshold + 1e-8)
        should_trade_conformal = signal_to_noise > 1.0
        
        # Combine with confidence if available
        if confidence is not None:
            # Higher confidence threshold when conformal is marginal
            conf_threshold = 0.5 if signal_to_noise > 1.5 else 0.7
            should_trade_confidence = confidence > conf_threshold
            final_should_trade = should_trade_conformal and should_trade_confidence
            reason = 'conformal_and_confidence'
        else:
            final_should_trade = should_trade_conformal
            reason = 'conformal_only'
        
        # Adjust signal strength based on confidence
        if final_should_trade:
            # Scale signal by how much it exceeds threshold
            scale_factor = min(signal_to_noise, 3.0)  # Cap at 3x
            adjusted_signal = prediction * scale_factor
        else:
            adjusted_signal = 0.0
        
        # Calculate overall confidence score
        overall_confidence = min(signal_to_noise / 2.0, 1.0)
        if confidence is not None:
            overall_confidence = (overall_confidence + confidence) / 2.0
        
        return {
            'should_trade': final_should_trade,
            'adjusted_signal': adjusted_signal,
            'confidence': overall_confidence,
            'reason': reason if final_should_trade else 'conformal_rejected',
            'diagnostics': {
                'prediction': prediction,
                'threshold': threshold,
                'signal_to_noise': signal_to_noise,
                'regime': active_regime,
                'n_calibration_samples': regime_info['n_samples']
            }
        }
    
    def batch_filter_signals(self, predictions: Dict[str, float], 
                           regime: str = 'default',
                           confidences: Dict[str, float] = None) -> Dict[str, Dict]:
        """
        Filter multiple signals at once for efficient batch processing
        
        Args:
            predictions: {symbol: prediction_value}
            regime: Current market regime
            confidences: {symbol: confidence_value}
            
        Returns:
            {symbol: trade_decision_dict}
        """
        results = {}
        
        for symbol, prediction in predictions.items():
            confidence = confidences.get(symbol) if confidences else None
            results[symbol] = self.should_trade(prediction, regime, confidence)
        
        # Log batch statistics
        total_signals = len(predictions)
        accepted_signals = sum(1 for r in results.values() if r['should_trade'])
        acceptance_rate = accepted_signals / total_signals if total_signals > 0 else 0
        
        logger.info(f"Batch filtering: {accepted_signals}/{total_signals} signals accepted ({acceptance_rate:.1%})")
        
        return results
    
    def save_calibration(self):
        """Save calibration state to disk"""
        try:
            calibration_state = {
                'quantiles': self.quantiles,
                'last_calibration': self.last_calibration,
                'config': self.config,
                'is_calibrated': self.is_calibrated
            }
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(calibration_state, f)
                
            logger.debug(f"Saved calibration state to {self.cache_file}")
            
        except Exception as e:
            logger.error(f"Error saving calibration: {e}")
    
    def load_calibration(self):
        """Load calibration state from disk"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    state = pickle.load(f)
                
                self.quantiles = state.get('quantiles', {})
                self.last_calibration = state.get('last_calibration', {})
                self.is_calibrated = state.get('is_calibrated', False)
                
                # Update config if needed but preserve user settings
                saved_config = state.get('config')
                if saved_config:
                    logger.info(f"Loaded calibration for {len(self.quantiles)} regimes")
                else:
                    logger.info("Loaded calibration state")
                    
        except Exception as e:
            logger.error(f"Error loading calibration: {e}")
            # Reset to empty state
            self.quantiles = {}
            self.last_calibration = {}
            self.is_calibrated = False
    
    def get_diagnostics(self) -> Dict:
        """Get diagnostic information about calibration state"""
        diagnostics = {
            'is_calibrated': self.is_calibrated,
            'n_regimes': len(self.quantiles),
            'regimes': list(self.quantiles.keys()),
            'config': {
                'alpha': self.config.alpha,
                'regime_aware': self.config.regime_aware,
                'cost_awareness': self.config.cost_awareness
            }
        }
        
        # Add per-regime details
        for regime, info in self.quantiles.items():
            diagnostics[f'regime_{regime}'] = {
                'threshold': info['threshold'],
                'n_samples': info['n_samples'],
                'mean_residual': info['mean_residual'],
                'last_calibration': info.get('calibration_time', 'unknown')
            }
        
        return diagnostics

# Convenience functions for integration
def create_conformal_gate(alpha: float = 0.20, regime_aware: bool = True) -> ProductionConformalGate:
    """Factory function to create a conformal gate with common settings"""
    config = ConformalConfig(alpha=alpha, regime_aware=regime_aware)
    return ProductionConformalGate(config)

def quick_calibrate_from_oof(oof_file: str, alpha: float = 0.20) -> ProductionConformalGate:
    """Quick setup: load OOF file and calibrate gate"""
    gate = create_conformal_gate(alpha=alpha)
    gate.load_oof_predictions(oof_file)
    gate.calibrate_all_regimes()
    return gate