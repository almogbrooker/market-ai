#!/usr/bin/env python3
"""
Advanced model architectures specifically designed for financial market prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Optional, Tuple

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for time series"""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.size()
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.w_o(context)
        return output, attn_weights

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(1)]

class TransformerBlock(nn.Module):
    """Enhanced Transformer block for financial time series"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights

class AdvancedLSTM(nn.Module):
    """Advanced LSTM with attention and residual connections"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 4,
                 dropout: float = 0.2, bidirectional: bool = True):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # LSTM layers with residual connections
        self.lstm_layers = nn.ModuleList()
        lstm_input_size = hidden_size
        for i in range(num_layers):
            self.lstm_layers.append(
                nn.LSTM(lstm_input_size, hidden_size, batch_first=True, 
                       dropout=dropout if i < num_layers-1 else 0,
                       bidirectional=bidirectional)
            )
            # Update input size for next layer (accounting for bidirectional)
            lstm_input_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Attention mechanism
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = MultiHeadAttention(lstm_output_size, n_heads=8, dropout=dropout)
        
        # Output layers
        self.layer_norm = nn.LayerNorm(lstm_output_size)
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, lstm_output_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 4, 1)
        )
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Input projection
        x = self.input_proj(x)
        
        # Multi-layer LSTM with residual connections
        for i, lstm_layer in enumerate(self.lstm_layers):
            residual = x
            lstm_out, _ = lstm_layer(x)
            
            # Residual connection (skip connection every 2 layers)
            if i > 0 and i % 2 == 0 and lstm_out.size(-1) == residual.size(-1):
                lstm_out = lstm_out + residual
            
            x = lstm_out
        
        # Self-attention
        attn_output, attn_weights = self.attention(x, x, x)
        x = self.layer_norm(x + attn_output)
        
        # Use attention-weighted combination instead of just last timestep
        attention_scores = torch.mean(attn_weights, dim=1)  # Average across heads
        attention_scores = torch.mean(attention_scores, dim=1)  # Average across queries
        
        # Weighted combination of all timesteps
        weighted_output = torch.sum(x * attention_scores.unsqueeze(-1), dim=1)
        
        # Final prediction
        output = self.output_proj(weighted_output)
        return output

class FinancialTransformer(nn.Module):
    """Advanced Transformer specifically designed for financial markets"""
    
    def __init__(self, input_size: int, d_model: int = 256, n_heads: int = 8, 
                 num_layers: int = 6, d_ff: int = 1024, dropout: float = 0.1,
                 max_len: int = 100):
        super().__init__()
        
        self.d_model = d_model
        
        # Input processing
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output processing
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Multi-task heads
        self.return_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self.volatility_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Softplus()  # Ensure positive volatility
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()  # Confidence score 0-1
        )
        
        # Feature importance attention
        self.feature_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        
    def forward(self, x, return_attention=False):
        batch_size, seq_len, _ = x.size()
        
        # Input projection and positional encoding
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        
        # Store attention weights if requested
        attention_weights = []
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x, attn_weights = transformer_block(x)
            if return_attention:
                attention_weights.append(attn_weights)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Feature importance attention
        attn_output, feature_weights = self.feature_attention(x, x, x)
        x = x + attn_output
        
        # Use the last timestep for prediction
        final_representation = x[:, -1, :]  # [batch_size, d_model]
        
        # Multi-task predictions
        return_pred = self.return_head(final_representation)
        volatility_pred = self.volatility_head(final_representation)
        confidence_pred = self.confidence_head(final_representation)
        
        outputs = {
            'return_prediction': return_pred,
            'volatility_prediction': volatility_pred,
            'confidence_score': confidence_pred
        }
        
        if return_attention:
            outputs['attention_weights'] = attention_weights
            outputs['feature_weights'] = feature_weights
        
        return outputs

class EnsembleModel(nn.Module):
    """Ensemble of different model architectures"""
    
    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()
        
        # Different model architectures
        self.lstm_model = AdvancedLSTM(input_size, hidden_size, num_layers=4)
        self.transformer_model = FinancialTransformer(input_size, d_model=256, num_layers=4)
        
        # Ensemble weights (learnable)
        self.ensemble_weights = nn.Parameter(torch.ones(2) / 2)
        
        # Meta-learner to combine predictions
        self.meta_learner = nn.Sequential(
            nn.Linear(4, 32),  # 2 returns + 2 confidences
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        # Get predictions from both models
        lstm_pred = self.lstm_model(x)
        transformer_outputs = self.transformer_model(x)
        transformer_pred = transformer_outputs['return_prediction']
        confidence = transformer_outputs['confidence_score']
        volatility = transformer_outputs['volatility_prediction']
        
        # Weighted ensemble
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_pred = weights[0] * lstm_pred + weights[1] * transformer_pred
        
        # Meta-learning combination
        meta_input = torch.cat([
            lstm_pred, transformer_pred, confidence, volatility
        ], dim=1)
        
        meta_pred = self.meta_learner(meta_input)
        
        return {
            'lstm_prediction': lstm_pred,
            'transformer_prediction': transformer_pred,
            'ensemble_prediction': ensemble_pred,
            'meta_prediction': meta_pred,
            'confidence': confidence,
            'volatility': volatility,
            'weights': weights
        }

class WaveNet(nn.Module):
    """WaveNet-inspired architecture for financial time series"""
    
    def __init__(self, input_size: int, residual_channels: int = 64, 
                 dilation_channels: int = 64, skip_channels: int = 64,
                 end_channels: int = 128, num_blocks: int = 3, layers_per_block: int = 10):
        super().__init__()
        
        self.input_size = input_size
        self.residual_channels = residual_channels
        
        # Input projection
        self.input_conv = nn.Conv1d(input_size, residual_channels, kernel_size=1)
        
        # Dilated convolution blocks
        self.dilated_blocks = nn.ModuleList()
        
        for block in range(num_blocks):
            for layer in range(layers_per_block):
                dilation = 2 ** layer
                
                # Dilated convolution with causal padding
                padding = dilation
                dilated_conv = nn.Conv1d(
                    residual_channels, dilation_channels, 
                    kernel_size=2, dilation=dilation, padding=padding
                )
                
                # Gated activation
                gate_conv = nn.Conv1d(
                    residual_channels, dilation_channels,
                    kernel_size=2, dilation=dilation, padding=padding
                )
                
                # Skip and residual connections
                skip_conv = nn.Conv1d(dilation_channels, skip_channels, kernel_size=1)
                residual_conv = nn.Conv1d(dilation_channels, residual_channels, kernel_size=1)
                
                self.dilated_blocks.append(nn.ModuleDict({
                    'dilated_conv': dilated_conv,
                    'gate_conv': gate_conv,
                    'skip_conv': skip_conv,
                    'residual_conv': residual_conv
                }))
        
        # Final layers
        self.end_conv1 = nn.Conv1d(skip_channels, end_channels, kernel_size=1)
        self.end_conv2 = nn.Conv1d(end_channels, 1, kernel_size=1)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, features]
        # Convert to [batch_size, features, seq_len] for conv1d
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_conv(x)
        skip_connections = None
        
        # Process through dilated blocks
        for block in self.dilated_blocks:
            residual = x
            
            # Dilated convolution
            filter_output = torch.tanh(block['dilated_conv'](x))
            gate_output = torch.sigmoid(block['gate_conv'](x))
            
            # Gated activation
            gated_output = filter_output * gate_output
            
            # Skip connection
            skip_output = block['skip_conv'](gated_output)
            if skip_connections is None:
                skip_connections = skip_output
            else:
                # Ensure same size for skip connections
                min_size = min(skip_connections.size(-1), skip_output.size(-1))
                skip_connections = skip_connections[:, :, :min_size] + skip_output[:, :, :min_size]
            
            # Residual connection with proper sizing
            residual_output = block['residual_conv'](gated_output)
            
            # Ensure residual and output have same size
            if residual.size() == residual_output.size():
                x = residual + residual_output
            else:
                # Crop or pad to match sizes
                min_size = min(residual.size(-1), residual_output.size(-1))
                x = residual[:, :, :min_size] + residual_output[:, :, :min_size]
        
        # Final processing
        x = F.relu(skip_connections)
        x = F.relu(self.end_conv1(x))
        x = self.end_conv2(x)
        
        # Return last timestep prediction
        return x[:, :, -1]  # [batch_size, 1]

def create_advanced_model(model_type: str, input_size: int, **kwargs) -> nn.Module:
    """Factory function to create advanced models"""
    
    if model_type == "advanced_lstm":
        return AdvancedLSTM(
            input_size=input_size,
            hidden_size=kwargs.get('hidden_size', 128),
            num_layers=kwargs.get('num_layers', 4),
            dropout=kwargs.get('dropout', 0.2),
            bidirectional=kwargs.get('bidirectional', True)
        )
    
    elif model_type == "financial_transformer":
        return FinancialTransformer(
            input_size=input_size,
            d_model=kwargs.get('d_model', 256),
            n_heads=kwargs.get('n_heads', 8),
            num_layers=kwargs.get('num_layers', 6),
            d_ff=kwargs.get('d_ff', 1024),
            dropout=kwargs.get('dropout', 0.1)
        )
    
    elif model_type == "ensemble":
        return EnsembleModel(
            input_size=input_size,
            hidden_size=kwargs.get('hidden_size', 128)
        )
    
    elif model_type == "wavenet":
        return WaveNet(
            input_size=input_size,
            residual_channels=kwargs.get('residual_channels', 64),
            num_blocks=kwargs.get('num_blocks', 3),
            layers_per_block=kwargs.get('layers_per_block', 10)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Advanced loss functions for financial markets
class FinancialLoss(nn.Module):
    """Multi-objective loss function for financial prediction"""
    
    def __init__(self, return_weight=1.0, directional_weight=0.5, volatility_weight=0.3):
        super().__init__()
        self.return_weight = return_weight
        self.directional_weight = directional_weight
        self.volatility_weight = volatility_weight
        
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, predictions, targets, volatility_pred=None):
        # Main return prediction loss
        return_loss = self.mse_loss(predictions, targets)
        
        # Directional accuracy loss
        pred_direction = torch.sign(predictions)
        target_direction = torch.sign(targets)
        directional_loss = self.bce_loss(
            pred_direction, (target_direction + 1) / 2
        )
        
        total_loss = (
            self.return_weight * return_loss +
            self.directional_weight * directional_loss
        )
        
        # Volatility loss if available
        if volatility_pred is not None:
            target_volatility = torch.abs(targets)
            volatility_loss = self.mse_loss(volatility_pred.squeeze(), target_volatility.squeeze())
            total_loss += self.volatility_weight * volatility_loss
        
        return total_loss

if __name__ == "__main__":
    # Test the models
    batch_size, seq_len, input_size = 32, 20, 23
    x = torch.randn(batch_size, seq_len, input_size)
    
    print("Testing Advanced Models:")
    print("=" * 50)
    
    # Test Advanced LSTM
    model1 = create_advanced_model("advanced_lstm", input_size, num_layers=6, hidden_size=256)
    out1 = model1(x)
    print(f"Advanced LSTM output shape: {out1.shape}")
    print(f"Advanced LSTM parameters: {sum(p.numel() for p in model1.parameters()):,}")
    
    # Test Financial Transformer
    model2 = create_advanced_model("financial_transformer", input_size, num_layers=8, d_model=512)
    out2 = model2(x)
    print(f"Financial Transformer outputs: {out2.keys()}")
    print(f"Financial Transformer parameters: {sum(p.numel() for p in model2.parameters()):,}")
    
    # Test Ensemble
    model3 = create_advanced_model("ensemble", input_size, hidden_size=256)
    out3 = model3(x)
    print(f"Ensemble outputs: {out3.keys()}")
    print(f"Ensemble parameters: {sum(p.numel() for p in model3.parameters()):,}")
    
    # Test WaveNet
    model4 = create_advanced_model("wavenet", input_size, num_blocks=4, layers_per_block=12)
    out4 = model4(x)
    print(f"WaveNet output shape: {out4.shape}")
    print(f"WaveNet parameters: {sum(p.numel() for p in model4.parameters()):,}")
    
    print("\nAll models tested successfully! 🚀")