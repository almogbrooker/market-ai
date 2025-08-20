#!/usr/bin/env python3
"""
Conformal Prediction for uncertainty-based trading decisions
Implements prediction intervals and "don't trade" logic based on uncertainty
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

class ConformalPredictor:
    """Conformal Prediction for financial forecasting with trading decisions"""
    
    def __init__(self, alpha: float = 0.1, tau: float = 0.05):
        """
        Args:
            alpha: Miscoverage level (0.1 for 90% prediction intervals)
            tau: Trading threshold - don't trade if interval width > tau
        """
        self.alpha = alpha
        self.tau = tau
        self.q_low = None
        self.q_high = None
        self.calibrated = False
        
    def calibrate(self, predictions: torch.Tensor, targets: torch.Tensor, 
                  volatilities: Optional[torch.Tensor] = None) -> None:
        """Calibrate conformal predictor on validation set"""
        
        if volatilities is None:
            # Use simple prediction errors for calibration
            residuals = torch.abs(predictions - targets).flatten()
        else:
            # Use normalized prediction errors (uncertainty-aware)
            residuals = torch.abs(predictions - targets).flatten() / (volatilities.flatten() + 1e-6)
        
        # Calculate empirical quantiles for prediction intervals
        n = len(residuals)
        q_level = np.ceil((n + 1) * (1 - self.alpha / 2)) / n
        
        self.q_low = torch.quantile(residuals, self.alpha / 2)
        self.q_high = torch.quantile(residuals, q_level)
        
        self.calibrated = True
        logger.info(f"Conformal calibration complete: q_low={self.q_low:.4f}, q_high={self.q_high:.4f}")
    
    def predict_with_intervals(self, predictions: torch.Tensor, 
                              volatilities: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Generate prediction intervals and trading decisions"""
        
        if not self.calibrated:
            raise ValueError("Must calibrate predictor first")
        
        if volatilities is None:
            volatilities = torch.ones_like(predictions)
        
        # Calculate prediction intervals
        lower_bound = predictions - self.q_high * volatilities
        upper_bound = predictions + self.q_high * volatilities
        
        # Calculate interval width (uncertainty measure)
        interval_width = upper_bound - lower_bound
        
        # Trading decision: trade only when confident (narrow intervals)
        should_trade = interval_width <= self.tau
        
        # Confidence score (inverse of interval width)
        confidence = 1.0 / (1.0 + interval_width)
        
        return {
            'predictions': predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'interval_width': interval_width,
            'should_trade': should_trade,
            'confidence': confidence,
            'trading_signal': predictions * should_trade.float()  # Zero out uncertain predictions
        }
    
    def evaluate_coverage(self, predictions: torch.Tensor, targets: torch.Tensor,
                         volatilities: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Evaluate prediction interval coverage on test set"""
        
        intervals = self.predict_with_intervals(predictions, volatilities)
        
        # Check if targets fall within prediction intervals
        in_interval = (targets >= intervals['lower_bound']) & (targets <= intervals['upper_bound'])
        coverage = in_interval.float().mean().item()
        
        # Average interval width
        avg_width = intervals['interval_width'].mean().item()
        
        # Trading frequency (how often we decide to trade)
        trade_freq = intervals['should_trade'].float().mean().item()
        
        return {
            'coverage': coverage,
            'average_interval_width': avg_width,
            'trading_frequency': trade_freq,
            'target_coverage': 1 - self.alpha
        }

class QuantileLoss(nn.Module):
    """Pinball/Quantile loss for training models to predict quantiles"""
    
    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [batch_size, num_quantiles] quantile predictions
            targets: [batch_size, 1] target values
        """
        losses = []
        
        for i, tau in enumerate(self.quantiles):
            pred_q = predictions[:, i:i+1]  # [batch_size, 1]
            
            # Pinball loss: max(tau * (y - q), (tau - 1) * (y - q))
            error = targets - pred_q
            loss = torch.max(tau * error, (tau - 1) * error)
            losses.append(loss.mean())
        
        return sum(losses) / len(losses)

class UncertaintyAwareModel(nn.Module):
    """Wrapper for models to output multiple quantiles for conformal prediction"""
    
    def __init__(self, base_model: nn.Module, quantiles: List[float] = [0.1, 0.5, 0.9]):
        super().__init__()
        self.base_model = base_model
        self.quantiles = quantiles
        
        # Get feature dimension from base model
        dummy_input = torch.randn(1, 30, 50)  # Assume typical input shape
        with torch.no_grad():
            base_output = base_model(dummy_input)
            if isinstance(base_output, dict):
                feature_dim = base_output['return_prediction'].shape[-1]
            else:
                feature_dim = base_output.shape[-1]
        
        # Additional quantile heads
        self.quantile_heads = nn.ModuleList([
            nn.Linear(feature_dim, len(quantiles)) for _ in range(1)
        ])
        
    def forward(self, x):
        # Get base model output
        base_output = self.base_model(x)
        
        if isinstance(base_output, dict):
            features = base_output['return_prediction']
            result = base_output.copy()
        else:
            features = base_output
            result = {'return_prediction': features}
        
        # Add quantile predictions
        quantile_preds = self.quantile_heads[0](features)
        result['quantile_predictions'] = quantile_preds
        result['quantiles'] = self.quantiles
        
        return result

def create_uncertainty_model(base_model: nn.Module, 
                           quantiles: List[float] = [0.1, 0.5, 0.9]) -> UncertaintyAwareModel:
    """Factory function to create uncertainty-aware models"""
    return UncertaintyAwareModel(base_model, quantiles)