#!/usr/bin/env python3
"""
Sharpe-aware training losses and DeepDow integration
Risk-adjusted loss functions for portfolio optimization
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class SharpeRatioLoss(nn.Module):
    """Differentiable Sharpe Ratio loss for end-to-end training"""
    
    def __init__(self, risk_free_rate: float = 0.02, annualization_factor: float = 252.0):
        super().__init__()
        self.risk_free_rate = risk_free_rate / annualization_factor  # Daily risk-free rate
        self.eps = 1e-8
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            predictions: [batch_size, num_assets] or [batch_size, 1] 
            targets: [batch_size, num_assets] or [batch_size, 1] actual returns
            weights: [batch_size, num_assets] portfolio weights (optional)
        """
        
        if weights is None:
            # Equal weights or single asset
            if predictions.shape[-1] == 1:
                portfolio_returns = predictions.squeeze(-1)
                actual_returns = targets.squeeze(-1)
            else:
                # Equal weight portfolio
                portfolio_returns = predictions.mean(dim=-1)
                actual_returns = targets.mean(dim=-1)
        else:
            # Weighted portfolio returns
            portfolio_returns = (predictions * weights).sum(dim=-1)
            actual_returns = (targets * weights).sum(dim=-1)
        
        # Calculate excess returns
        excess_returns = actual_returns - self.risk_free_rate
        
        # Sharpe ratio components
        mean_return = excess_returns.mean()
        std_return = excess_returns.std() + self.eps
        
        sharpe_ratio = mean_return / std_return
        
        # Return negative Sharpe (since we want to maximize Sharpe)
        return -sharpe_ratio

class SortinoRatioLoss(nn.Module):
    """Differentiable Sortino Ratio loss (downside deviation)"""
    
    def __init__(self, risk_free_rate: float = 0.02, annualization_factor: float = 252.0):
        super().__init__()
        self.risk_free_rate = risk_free_rate / annualization_factor
        self.eps = 1e-8
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        if weights is None:
            if predictions.shape[-1] == 1:
                portfolio_returns = predictions.squeeze(-1)
                actual_returns = targets.squeeze(-1)
            else:
                portfolio_returns = predictions.mean(dim=-1)
                actual_returns = targets.mean(dim=-1)
        else:
            portfolio_returns = (predictions * weights).sum(dim=-1)
            actual_returns = (targets * weights).sum(dim=-1)
        
        # Calculate excess returns
        excess_returns = actual_returns - self.risk_free_rate
        
        # Sortino ratio components
        mean_return = excess_returns.mean()
        
        # Downside deviation (only negative returns)
        downside_returns = torch.clamp(excess_returns, max=0.0)
        downside_deviation = torch.sqrt(torch.mean(downside_returns ** 2)) + self.eps
        
        sortino_ratio = mean_return / downside_deviation
        
        return -sortino_ratio

class MaxDrawdownLoss(nn.Module):
    """Differentiable Maximum Drawdown loss"""
    
    def __init__(self):
        super().__init__()
        self.eps = 1e-8
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        if weights is None:
            if predictions.shape[-1] == 1:
                actual_returns = targets.squeeze(-1)
            else:
                actual_returns = targets.mean(dim=-1)
        else:
            actual_returns = (targets * weights).sum(dim=-1)
        
        # Calculate cumulative returns
        cumulative_returns = torch.cumsum(actual_returns, dim=0)
        
        # Calculate running maximum
        running_max = torch.cummax(cumulative_returns, dim=0)[0]
        
        # Drawdown at each point
        drawdown = (cumulative_returns - running_max) / (running_max + self.eps)
        
        # Maximum drawdown
        max_drawdown = -torch.min(drawdown)  # Negative because drawdown is negative
        
        return max_drawdown

class CalmarRatioLoss(nn.Module):
    """Differentiable Calmar Ratio (return/max_drawdown)"""
    
    def __init__(self, risk_free_rate: float = 0.02, annualization_factor: float = 252.0):
        super().__init__()
        self.risk_free_rate = risk_free_rate / annualization_factor
        self.max_dd_loss = MaxDrawdownLoss()
        self.eps = 1e-8
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        if weights is None:
            if predictions.shape[-1] == 1:
                actual_returns = targets.squeeze(-1)
            else:
                actual_returns = targets.mean(dim=-1)
        else:
            actual_returns = (targets * weights).sum(dim=-1)
        
        # Mean excess return
        excess_returns = actual_returns - self.risk_free_rate
        mean_return = excess_returns.mean()
        
        # Maximum drawdown
        max_drawdown = self.max_dd_loss(predictions, targets, weights)
        
        # Calmar ratio
        calmar_ratio = mean_return / (max_drawdown + self.eps)
        
        return -calmar_ratio

class CompositeRiskLoss(nn.Module):
    """Composite loss combining multiple risk-adjusted metrics"""
    
    def __init__(self, sharpe_weight: float = 1.0, sortino_weight: float = 0.5, 
                 calmar_weight: float = 0.3, mse_weight: float = 0.1):
        super().__init__()
        
        self.sharpe_loss = SharpeRatioLoss()
        self.sortino_loss = SortinoRatioLoss()
        self.calmar_loss = CalmarRatioLoss()
        self.mse_loss = nn.MSELoss()
        
        self.sharpe_weight = sharpe_weight
        self.sortino_weight = sortino_weight
        self.calmar_weight = calmar_weight
        self.mse_weight = mse_weight
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        # Individual losses
        sharpe_loss = self.sharpe_loss(predictions, targets, weights)
        sortino_loss = self.sortino_loss(predictions, targets, weights)
        calmar_loss = self.calmar_loss(predictions, targets, weights)
        mse_loss = self.mse_loss(predictions, targets)
        
        # Composite loss
        total_loss = (self.sharpe_weight * sharpe_loss + 
                     self.sortino_weight * sortino_loss +
                     self.calmar_weight * calmar_loss +
                     self.mse_weight * mse_loss)
        
        return {
            'total_loss': total_loss,
            'sharpe_loss': sharpe_loss,
            'sortino_loss': sortino_loss,
            'calmar_loss': calmar_loss,
            'mse_loss': mse_loss
        }

class PortfolioOptimizationLoss(nn.Module):
    """DeepDow-inspired portfolio optimization loss"""
    
    def __init__(self, risk_aversion: float = 1.0, transaction_cost: float = 0.001):
        super().__init__()
        self.risk_aversion = risk_aversion
        self.transaction_cost = transaction_cost
        
    def forward(self, weights: torch.Tensor, expected_returns: torch.Tensor,
                covariance_matrix: torch.Tensor, 
                prev_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            weights: [batch_size, num_assets] portfolio weights
            expected_returns: [batch_size, num_assets] expected returns
            covariance_matrix: [batch_size, num_assets, num_assets] or [num_assets, num_assets]
            prev_weights: [batch_size, num_assets] previous weights for transaction costs
        """
        
        # Portfolio expected return
        portfolio_return = (weights * expected_returns).sum(dim=-1)
        
        # Portfolio variance
        if covariance_matrix.dim() == 2:
            # Shared covariance matrix
            portfolio_variance = torch.diag(weights @ covariance_matrix @ weights.T)
        else:
            # Batch-wise covariance matrices
            portfolio_variance = torch.bmm(
                weights.unsqueeze(1),
                torch.bmm(covariance_matrix, weights.unsqueeze(-1))
            ).squeeze()
        
        # Mean-variance objective
        objective = portfolio_return - 0.5 * self.risk_aversion * portfolio_variance
        
        # Transaction costs
        transaction_cost = 0.0
        if prev_weights is not None:
            turnover = torch.abs(weights - prev_weights).sum(dim=-1)
            transaction_cost = self.transaction_cost * turnover
        
        # Total utility (negative because we minimize)
        total_loss = -(objective - transaction_cost)
        
        return {
            'total_loss': total_loss.mean(),
            'portfolio_return': portfolio_return.mean(),
            'portfolio_variance': portfolio_variance.mean(),
            'transaction_cost': transaction_cost if isinstance(transaction_cost, torch.Tensor) else torch.tensor(transaction_cost)
        }

def create_risk_aware_loss(loss_type: str = "sharpe", **kwargs) -> nn.Module:
    """Factory function for risk-aware losses"""
    
    if loss_type == "sharpe":
        return SharpeRatioLoss(**kwargs)
    elif loss_type == "sortino":
        return SortinoRatioLoss(**kwargs)
    elif loss_type == "calmar":
        return CalmarRatioLoss(**kwargs)
    elif loss_type == "composite":
        return CompositeRiskLoss(**kwargs)
    elif loss_type == "portfolio":
        return PortfolioOptimizationLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")