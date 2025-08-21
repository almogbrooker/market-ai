#!/usr/bin/env python3
"""
RankIC Loss Functions for Direct IC Optimization
Optimizes models directly for Information Coefficient instead of MSE
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import spearmanr, pearsonr
from typing import Tuple, Optional

class RankICLoss(nn.Module):
    """
    Direct RankIC optimization loss
    Maximizes correlation between predictions and targets (Spearman/Pearson)
    """
    
    def __init__(self, 
                 correlation_type: str = 'spearman',
                 temperature: float = 1.0,
                 dead_zone_threshold: float = 0.001,
                 sample_weight: bool = True):
        super().__init__()
        self.correlation_type = correlation_type
        self.temperature = temperature
        self.dead_zone_threshold = dead_zone_threshold
        self.sample_weight = sample_weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute RankIC loss
        
        Args:
            predictions: Model predictions [batch_size, 1]
            targets: True returns/alpha [batch_size, 1]
            weights: Optional sample weights [batch_size, 1]
        
        Returns:
            Negative correlation (to minimize)
        """
        batch_size = predictions.size(0)
        
        if batch_size < 10:  # Need minimum samples for correlation
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        # Flatten tensors
        pred_flat = predictions.view(-1)
        target_flat = targets.view(-1)
        
        # Apply dead zone (don't optimize for tiny returns)
        if self.dead_zone_threshold > 0:
            mask = torch.abs(target_flat) > self.dead_zone_threshold
            if mask.sum() < 5:  # Need minimum samples
                return torch.tensor(0.0, device=predictions.device, requires_grad=True)
            pred_flat = pred_flat[mask]
            target_flat = target_flat[mask]
            if weights is not None:
                weights = weights.view(-1)[mask]
        
        # Compute differentiable correlation approximation
        if self.correlation_type == 'spearman':
            # Spearman correlation via differentiable ranking
            correlation = self._differentiable_spearman_correlation(pred_flat, target_flat, weights)
        else:
            # Pearson correlation
            correlation = self._differentiable_pearson_correlation(pred_flat, target_flat, weights)
        
        # Return negative correlation (minimize to maximize correlation)
        return -correlation
    
    def _differentiable_spearman_correlation(self, pred: torch.Tensor, target: torch.Tensor,
                                           weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Differentiable approximation of Spearman correlation using soft ranking
        """
        n = pred.size(0)
        
        # Soft ranking using temperature-scaled sigmoid
        pred_ranks = self._soft_rank(pred)
        target_ranks = self._soft_rank(target)
        
        # Compute correlation on soft ranks
        return self._pearson_correlation(pred_ranks, target_ranks, weights)
    
    def _differentiable_pearson_correlation(self, pred: torch.Tensor, target: torch.Tensor,
                                          weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Differentiable Pearson correlation
        """
        return self._pearson_correlation(pred, target, weights)
    
    def _pearson_correlation(self, x: torch.Tensor, y: torch.Tensor,
                           weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute weighted Pearson correlation
        """
        if weights is not None:
            # Weighted correlation
            w_sum = weights.sum()
            x_mean = (x * weights).sum() / w_sum
            y_mean = (y * weights).sum() / w_sum
            
            x_centered = (x - x_mean) * torch.sqrt(weights)
            y_centered = (y - y_mean) * torch.sqrt(weights)
        else:
            # Unweighted correlation
            x_mean = x.mean()
            y_mean = y.mean()
            
            x_centered = x - x_mean
            y_centered = y - y_mean
        
        # Compute correlation
        numerator = (x_centered * y_centered).sum()
        denominator = torch.sqrt((x_centered ** 2).sum() * (y_centered ** 2).sum())
        
        # Avoid division by zero
        correlation = numerator / (denominator + 1e-8)
        
        return correlation
    
    def _soft_rank(self, x: torch.Tensor) -> torch.Tensor:
        """
        Differentiable soft ranking using temperature-scaled comparisons
        """
        n = x.size(0)
        
        # Create pairwise comparison matrix
        x_expanded = x.unsqueeze(1)  # [n, 1]
        x_comparison = x.unsqueeze(0)  # [1, n]
        
        # Soft counting: how many elements are smaller
        comparisons = torch.sigmoid((x_expanded - x_comparison) / self.temperature)
        soft_ranks = comparisons.sum(dim=1)
        
        return soft_ranks

class PairwiseRankingLoss(nn.Module):
    """
    Pairwise ranking loss (ListNet/LambdaRank style)
    Optimizes relative ordering of predictions
    """
    
    def __init__(self, margin: float = 0.1, temperature: float = 1.0):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                sample_pairs: int = 1000) -> torch.Tensor:
        """
        Compute pairwise ranking loss
        
        Args:
            predictions: Model predictions [batch_size, 1]
            targets: True returns [batch_size, 1]
            sample_pairs: Number of pairs to sample (for efficiency)
        
        Returns:
            Pairwise ranking loss
        """
        batch_size = predictions.size(0)
        
        if batch_size < 2:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        pred_flat = predictions.view(-1)
        target_flat = targets.view(-1)
        
        # Sample pairs efficiently
        n_pairs = min(sample_pairs, batch_size * (batch_size - 1) // 2)
        
        if n_pairs < 10:
            # Use all pairs for small batches
            i_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, batch_size).flatten()
            j_indices = torch.arange(batch_size).unsqueeze(0).expand(batch_size, -1).flatten()
            mask = i_indices < j_indices
            i_indices = i_indices[mask]
            j_indices = j_indices[mask]
        else:
            # Sample random pairs for large batches
            i_indices = torch.randint(0, batch_size, (n_pairs,), device=predictions.device)
            j_indices = torch.randint(0, batch_size, (n_pairs,), device=predictions.device)
            
            # Ensure i != j
            mask = i_indices != j_indices
            i_indices = i_indices[mask]
            j_indices = j_indices[mask]
        
        if len(i_indices) == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        # Get pairs
        pred_i = pred_flat[i_indices]
        pred_j = pred_flat[j_indices]
        target_i = target_flat[i_indices]
        target_j = target_flat[j_indices]
        
        # True ordering: should pred_i > pred_j when target_i > target_j?
        target_diff = target_i - target_j  # Positive when i should rank higher than j
        pred_diff = pred_i - pred_j       # Model's predicted ordering
        
        # Hinge loss: encourage correct relative ordering
        loss = torch.clamp(self.margin - target_diff.sign() * pred_diff, min=0.0)
        
        return loss.mean()

class TopKPrecisionLoss(nn.Module):
    """
    Optimize for precision in top-K selections
    Useful for portfolio construction where we only trade top signals
    """
    
    def __init__(self, k_ratio: float = 0.2, temperature: float = 1.0):
        super().__init__()
        self.k_ratio = k_ratio  # Top 20% of predictions
        self.temperature = temperature
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Top-K precision loss
        
        Args:
            predictions: Model predictions [batch_size, 1]
            targets: True returns [batch_size, 1]
        
        Returns:
            Negative precision in top-K (to minimize)
        """
        batch_size = predictions.size(0)
        k = max(1, int(batch_size * self.k_ratio))
        
        pred_flat = predictions.view(-1)
        target_flat = targets.view(-1)
        
        # Get top-K predictions (soft selection)
        top_k_indices = torch.topk(pred_flat, k, largest=True)[1]
        top_k_targets = target_flat[top_k_indices]
        
        # Compute precision: fraction of top-K that are actually positive
        precision = (top_k_targets > 0).float().mean()
        
        return -precision  # Minimize negative precision

class CombinedICLoss(nn.Module):
    """
    Combined loss function optimizing multiple IC-related objectives
    """
    
    def __init__(self, 
                 rankic_weight: float = 1.0,
                 pairwise_weight: float = 0.5,
                 topk_weight: float = 0.3,
                 mse_weight: float = 0.1):
        super().__init__()
        self.rankic_loss = RankICLoss()
        self.pairwise_loss = PairwiseRankingLoss()
        self.topk_loss = TopKPrecisionLoss()
        self.mse_loss = nn.MSELoss()
        
        self.rankic_weight = rankic_weight
        self.pairwise_weight = pairwise_weight
        self.topk_weight = topk_weight
        self.mse_weight = mse_weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                weights: Optional[torch.Tensor] = None) -> dict:
        """
        Compute combined IC loss
        
        Returns:
            Dictionary with total loss and components
        """
        losses = {}
        
        # Individual losses
        losses['rankic'] = self.rankic_loss(predictions, targets, weights)
        losses['pairwise'] = self.pairwise_loss(predictions, targets)
        losses['topk'] = self.topk_loss(predictions, targets)
        losses['mse'] = self.mse_loss(predictions, targets)
        
        # Combined loss
        losses['total'] = (
            self.rankic_weight * losses['rankic'] +
            self.pairwise_weight * losses['pairwise'] +
            self.topk_weight * losses['topk'] +
            self.mse_weight * losses['mse']
        )
        
        return losses

def compute_ic_metrics(predictions: np.ndarray, targets: np.ndarray,
                      weights: Optional[np.ndarray] = None) -> dict:
    """
    Compute Information Coefficient metrics for evaluation
    
    Args:
        predictions: Model predictions
        targets: True returns/alpha
        weights: Optional sample weights
    
    Returns:
        Dictionary with IC metrics
    """
    if len(predictions) < 10:
        return {'ic_spearman': 0.0, 'ic_pearson': 0.0, 'ic_spearman_pval': 1.0, 'ic_pearson_pval': 1.0}
    
    # Remove NaN values
    mask = ~(np.isnan(predictions) | np.isnan(targets))
    pred_clean = predictions[mask]
    target_clean = targets[mask]
    
    if len(pred_clean) < 5:
        return {'ic_spearman': 0.0, 'ic_pearson': 0.0, 'ic_spearman_pval': 1.0, 'ic_pearson_pval': 1.0}
    
    # Compute correlations
    ic_spearman, spearman_pval = spearmanr(pred_clean, target_clean)
    ic_pearson, pearson_pval = pearsonr(pred_clean, target_clean)
    
    # Handle NaN results
    ic_spearman = ic_spearman if not np.isnan(ic_spearman) else 0.0
    ic_pearson = ic_pearson if not np.isnan(ic_pearson) else 0.0
    
    return {
        'ic_spearman': ic_spearman,
        'ic_pearson': ic_pearson,
        'ic_spearman_pval': spearman_pval if not np.isnan(spearman_pval) else 1.0,
        'ic_pearson_pval': pearson_pval if not np.isnan(pearson_pval) else 1.0,
        'n_samples': len(pred_clean)
    }

# Usage example for training loop integration
def example_training_with_rankic():
    """Example of how to integrate RankIC loss in training"""
    
    # Initialize loss
    criterion = CombinedICLoss(
        rankic_weight=1.0,     # Primary objective
        pairwise_weight=0.5,   # Ranking consistency  
        topk_weight=0.3,       # Portfolio precision
        mse_weight=0.1         # Magnitude accuracy
    )
    
    # In training loop:
    # predictions = model(features)  # [batch_size, 1]
    # targets = alpha_targets        # [batch_size, 1] - benchmark-neutral returns
    # weights = confidence_weights   # [batch_size, 1] - optional
    
    # loss_dict = criterion(predictions, targets, weights)
    # total_loss = loss_dict['total']
    # total_loss.backward()
    
    # For evaluation:
    # with torch.no_grad():
    #     pred_np = predictions.cpu().numpy()
    #     target_np = targets.cpu().numpy()
    #     ic_metrics = compute_ic_metrics(pred_np, target_np)
    #     print(f"IC Spearman: {ic_metrics['ic_spearman']:.4f}")
    
    pass

if __name__ == "__main__":
    # Test the loss functions
    torch.manual_seed(42)
    
    batch_size = 100
    predictions = torch.randn(batch_size, 1)
    targets = torch.randn(batch_size, 1)
    
    # Test RankIC loss
    rankic_loss = RankICLoss()
    loss_rankic = rankic_loss(predictions, targets)
    print(f"RankIC Loss: {loss_rankic.item():.4f}")
    
    # Test combined loss
    combined_loss = CombinedICLoss()
    loss_dict = combined_loss(predictions, targets)
    print(f"Combined Loss: {loss_dict['total'].item():.4f}")
    print(f"Components: {[(k, v.item()) for k, v in loss_dict.items() if k != 'total']}")
    
    # Test IC metrics
    pred_np = predictions.numpy().flatten()
    target_np = targets.numpy().flatten()
    ic_metrics = compute_ic_metrics(pred_np, target_np)
    print(f"IC Metrics: {ic_metrics}")