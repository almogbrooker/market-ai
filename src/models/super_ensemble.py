#!/usr/bin/env python3
"""
Advanced Super Ensemble System
Combines all 6 trained models with multiple ensemble techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import logging

logger = logging.getLogger(__name__)

from .advanced_models import create_advanced_model

@dataclass
class ModelInfo:
    name: str
    type: str
    params: Dict
    checkpoint_path: str
    val_accuracy: float
    val_loss: float
    weight: float = 1.0

class SuperEnsemble(nn.Module):
    """
    Advanced ensemble combining multiple techniques:
    1. Weighted Averaging (performance-based weights)
    2. Rank-based Voting 
    3. Meta-learner (neural network combining predictions)
    4. Confidence-weighted predictions
    5. Dynamic model selection
    """
    
    def __init__(self, input_size: int = 63, device: str = 'cuda'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.models = {}
        self.model_info = {}
        
        # Define model configurations (use absolute paths)
        base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                'experiments', 'financial_models_comparison', 'checkpoints')
        
        self.models_config = {
            'advanced_lstm_small': {
                'type': 'advanced_lstm',
                'params': {'num_layers': 4, 'hidden_size': 128, 'dropout': 0.2},
                'checkpoint': os.path.join(base_path, 'best_advanced_lstm_small.pth')
            },
            'advanced_lstm_large': {
                'type': 'advanced_lstm', 
                'params': {'num_layers': 6, 'hidden_size': 256, 'dropout': 0.3},
                'checkpoint': os.path.join(base_path, 'best_advanced_lstm_large.pth')
            },
            'financial_transformer_small': {
                'type': 'financial_transformer',
                'params': {'num_layers': 4, 'd_model': 256, 'n_heads': 8, 'dropout': 0.1},
                'checkpoint': os.path.join(base_path, 'best_financial_transformer_small.pth')
            },
            'financial_transformer_large': {
                'type': 'financial_transformer',
                'params': {'num_layers': 8, 'd_model': 512, 'n_heads': 16, 'dropout': 0.1},
                'checkpoint': os.path.join(base_path, 'best_financial_transformer_large.pth')
            },
            'ensemble_model': {
                'type': 'ensemble',
                'params': {'hidden_size': 256},
                'checkpoint': os.path.join(base_path, 'best_ensemble_model.pth')
            },
            'wavenet_model': {
                'type': 'wavenet',
                'params': {'num_blocks': 4, 'layers_per_block': 12, 'residual_channels': 128},
                'checkpoint': os.path.join(base_path, 'best_wavenet_model.pth')
            }
        }
        
        # Meta-learner for intelligent combination
        self.meta_learner = nn.Sequential(
            nn.Linear(12, 64),  # 6 predictions + 6 confidence scores
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        ).to(self.device)
        
        # Attention mechanism for dynamic weighting
        self.attention = nn.Sequential(
            nn.Linear(6, 32),  # 6 model predictions
            nn.GELU(),
            nn.Linear(32, 6),
            nn.Softmax(dim=-1)
        ).to(self.device)
        
        # Load all models
        self.load_models()
        
        # Calculate performance-based weights
        self.calculate_weights()
        
    def load_models(self):
        """Load all trained models"""
        logger.info("🤖 Loading Super Ensemble Models...")
        
        for model_name, config in self.models_config.items():
            try:
                # Create model
                model = create_advanced_model(
                    config['type'], 
                    self.input_size, 
                    **config['params']
                ).to(self.device)
                
                # Load checkpoint
                if os.path.exists(config['checkpoint']):
                    checkpoint = torch.load(config['checkpoint'], map_location=self.device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    
                    # Store model info
                    self.model_info[model_name] = ModelInfo(
                        name=model_name,
                        type=config['type'],
                        params=config['params'],
                        checkpoint_path=config['checkpoint'],
                        val_accuracy=checkpoint.get('val_acc', 0.5),
                        val_loss=checkpoint.get('val_loss', 1.0)
                    )
                    
                    model.eval()
                    self.models[model_name] = model
                    logger.info(f"  ✓ {model_name}: Acc={self.model_info[model_name].val_accuracy:.4f}")

                else:
                    logger.warning(f"  ❌ Checkpoint not found: {config['checkpoint']}")
                    
            except Exception as e:
                logger.error(f"  ❌ Failed to load {model_name}: {e}")

        logger.info(f"📊 Successfully loaded {len(self.models)}/6 models")
        
    def calculate_weights(self):
        """Calculate performance-based weights for each model"""
        if not self.model_info:
            return
            
        # Weight based on validation accuracy (higher accuracy = higher weight)
        accuracies = [info.val_accuracy for info in self.model_info.values()]
        losses = [info.val_loss for info in self.model_info.values()]
        
        # Normalize accuracies to weights with zero-sum check
        acc_array = np.array(accuracies, dtype=float)
        acc_sum = np.sum(acc_array)
        if acc_sum == 0:
            logger.warning("Accuracy sum is zero; using uniform weights.")
            acc_weights = np.ones_like(acc_array) / len(acc_array)
        else:
            acc_weights = acc_array / acc_sum
        
        # Inverse loss weights (lower loss = higher weight)
        loss_weights = 1.0 / np.array(losses)
        loss_weights = loss_weights / np.sum(loss_weights)
        
        # Combine accuracy and loss weights
        combined_weights = 0.7 * acc_weights + 0.3 * loss_weights
        
        # Update model info with weights
        for i, (name, info) in enumerate(self.model_info.items()):
            info.weight = combined_weights[i]
            
        logger.info("📈 Model Weights (Performance-based):")
        sorted_models = sorted(self.model_info.items(), key=lambda x: x[1].weight, reverse=True)
        for name, info in sorted_models:
            logger.info(f"  {name}: {info.weight:.4f} (Acc: {info.val_accuracy:.4f})")
    
    def get_single_prediction(self, model_name: str, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get prediction and confidence from a single model"""
        model = self.models[model_name]
        model_type = self.model_info[model_name].type
        
        with torch.no_grad():
            if model_type == 'financial_transformer':
                outputs = model(x)
                prediction = outputs['return_prediction']
                confidence = outputs.get('confidence_score', torch.ones_like(prediction) * 0.5)
            elif model_type == 'ensemble':
                outputs = model(x)
                prediction = outputs['meta_prediction']
                confidence = outputs.get('confidence', torch.ones_like(prediction) * 0.5)
            else:
                prediction = model(x)
                # Estimate confidence based on prediction magnitude
                confidence = torch.sigmoid(torch.abs(prediction))
                
        return prediction, confidence
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all ensemble methods
        """
        batch_size = x.size(0)
        x = x.to(self.device)
        
        # Collect predictions from all models
        predictions = []
        confidences = []
        model_names = []
        
        for model_name in self.models.keys():
            pred, conf = self.get_single_prediction(model_name, x)
            predictions.append(pred)
            confidences.append(conf)
            model_names.append(model_name)
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=-1)  # [batch, 1, n_models]
        confidences = torch.stack(confidences, dim=-1)  # [batch, 1, n_models]
        
        # Method 1: Weighted Average (performance-based)
        weights = torch.tensor([self.model_info[name].weight for name in model_names], 
                              device=self.device, dtype=torch.float32)
        weighted_pred = torch.sum(predictions * weights.unsqueeze(0).unsqueeze(0), dim=-1)
        
        # Method 2: Confidence-weighted Average
        conf_weights = F.softmax(confidences.squeeze(1), dim=-1)
        conf_weighted_pred = torch.sum(predictions * conf_weights.unsqueeze(1), dim=-1)
        
        # Method 3: Attention-based Dynamic Weighting
        pred_flat = predictions.squeeze(1)  # [batch, n_models]
        attn_weights = self.attention(pred_flat)
        attention_pred = torch.sum(predictions * attn_weights.unsqueeze(1), dim=-1)
        
        # Method 4: Rank-based Voting
        ranks = torch.argsort(torch.argsort(pred_flat, dim=-1, descending=True), dim=-1) + 1
        rank_weights = 1.0 / ranks.float()
        rank_weights = F.softmax(rank_weights, dim=-1)
        rank_pred = torch.sum(predictions * rank_weights.unsqueeze(1), dim=-1)
        
        # Method 5: Meta-learner
        meta_input = torch.cat([pred_flat, confidences.squeeze(1)], dim=-1)
        meta_pred = self.meta_learner(meta_input)
        
        # Method 6: Conservative Ensemble (only high-confidence predictions)
        high_conf_mask = confidences.squeeze(1) > 0.6
        if high_conf_mask.any():
            conservative_weights = high_conf_mask.float() * conf_weights
            conservative_weights = conservative_weights / (conservative_weights.sum(dim=-1, keepdim=True) + 1e-8)
            conservative_pred = torch.sum(predictions * conservative_weights.unsqueeze(1), dim=-1)
        else:
            conservative_pred = weighted_pred
        
        # Final Super Ensemble: Combine all methods intelligently
        # Higher weight to meta-learner and confidence-weighted methods
        super_ensemble = (
            0.3 * meta_pred +
            0.25 * conf_weighted_pred +
            0.2 * weighted_pred +
            0.15 * attention_pred +
            0.1 * conservative_pred
        )
        
        return {
            'super_ensemble': super_ensemble,
            'meta_learner': meta_pred,
            'weighted_average': weighted_pred,
            'confidence_weighted': conf_weighted_pred,
            'attention_weighted': attention_pred,
            'rank_based': rank_pred,
            'conservative': conservative_pred,
            'individual_predictions': predictions,
            'individual_confidences': confidences,
            'model_names': model_names
        }
    
    def get_ensemble_confidence(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate ensemble confidence based on prediction agreement"""
        individual_preds = outputs['individual_predictions']  # [batch, 1, n_models]
        
        # Calculate prediction variance (lower variance = higher confidence)
        pred_var = torch.var(individual_preds.squeeze(1), dim=-1, keepdim=True)
        confidence = torch.sigmoid(-pred_var * 10)  # High variance -> low confidence
        
        # Also consider individual model confidences
        individual_confs = outputs['individual_confidences'].mean(dim=-1)  # Average confidence
        
        # Combine variance-based and individual confidences
        ensemble_confidence = 0.6 * confidence + 0.4 * individual_confs
        
        return ensemble_confidence
    
    def predict_with_analysis(self, x: torch.Tensor) -> Dict:
        """
        Generate predictions with detailed analysis
        """
        outputs = self.forward(x)
        confidence = self.get_ensemble_confidence(outputs)
        
        # Directional predictions
        directions = {}
        for key in ['super_ensemble', 'meta_learner', 'weighted_average', 'confidence_weighted']:
            directions[key] = (outputs[key] > 0).float().mean().item()
        
        # Prediction statistics
        individual_preds = outputs['individual_predictions'].squeeze(1)  # [batch, n_models]
        pred_stats = {
            'mean': individual_preds.mean(dim=-1),
            'std': individual_preds.std(dim=-1),
            'min': individual_preds.min(dim=-1)[0],
            'max': individual_preds.max(dim=-1)[0],
            'agreement': (individual_preds > 0).float().mean(dim=-1)  # % bullish
        }
        
        return {
            'predictions': outputs,
            'confidence': confidence,
            'directions': directions,
            'statistics': pred_stats,
            'model_info': self.model_info
        }

def test_super_ensemble():
    """Test the super ensemble system"""
    logger.info("🚀 Testing Super Ensemble System")
    logger.info("=" * 50)
    
    # Create ensemble
    ensemble = SuperEnsemble()
    
    # Test data
    batch_size, seq_len = 32, 30
    test_data = torch.randn(batch_size, seq_len, 63)
    
    # Get predictions with analysis
    results = ensemble.predict_with_analysis(test_data)
    
    logger.info("📊 Super Ensemble Results:")
    logger.info("-" * 30)
    
    # Show ensemble predictions
    preds = results['predictions']
    conf = results['confidence']
    
    logger.info(f"Super Ensemble Prediction: {preds['super_ensemble'][:3].flatten()}")
    logger.info(f"Meta-learner Prediction:   {preds['meta_learner'][:3].flatten()}")
    logger.info(f"Weighted Average:          {preds['weighted_average'][:3].flatten()}")
    logger.info(f"Ensemble Confidence:       {conf[:3].flatten()}")
    
    # Show individual model predictions
    logger.info("🤖 Individual Model Predictions (first sample):")
    individual = preds['individual_predictions'][0, 0].cpu().numpy()
    for i, name in enumerate(preds['model_names']):
        logger.info(f"  {name}: {individual[i]:.6f}")
    
    # Show directional predictions
    logger.info("📈 Directional Analysis:")
    directions = results['directions']
    for method, bullish_pct in directions.items():
        logger.info(f"  {method}: {bullish_pct:.1%} bullish")
    
    # Show statistics
    stats = results['statistics']
    logger.info("📊 Prediction Statistics (first 3 samples):")
    logger.info(f"  Mean: {stats['mean'][:3]}")
    logger.info(f"  Std:  {stats['std'][:3]}")
    agreement_vals = stats['agreement'][:3].cpu().numpy()
    logger.info(f"  Agreement: {agreement_vals}")
    
    return ensemble, results

if __name__ == "__main__":
    test_super_ensemble()