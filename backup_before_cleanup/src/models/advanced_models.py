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
            # mask shape expected: [batch, heads, seq, seq] or [1, 1, seq, seq]; values 1=keep, 0=mask
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
        """Add positional encoding on the same device as x."""
        return x + self.pe[:x.size(1)].to(x.device)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample - for regularization"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class TransformerBlock(nn.Module):
    """Enhanced Transformer block for financial time series with DropPath"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, drop_path: float = 0.0):
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
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection + DropPath
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.drop_path(self.dropout(attn_output)))
        
        # Feed-forward with residual connection + DropPath
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.drop_path(self.dropout(ff_output)))
        
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
        
        # Shared feature extraction
        self.feature_proj = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, lstm_output_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Separate heads for mean and log-variance (uncertainty)
        self.mean_head = nn.Linear(lstm_output_size // 4, 1)
        self.log_var_head = nn.Linear(lstm_output_size // 4, 1)
        
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
        
        # Shared feature extraction
        features = self.feature_proj(weighted_output)
        
        # Mean and log-variance predictions
        mean_pred = self.mean_head(features)
        log_var_pred = self.log_var_head(features)
        
        # Return dictionary with uncertainty
        return {
            'return_prediction': mean_pred,
            'log_variance': log_var_pred,
            'volatility_prediction': torch.exp(0.5 * log_var_pred)  # Convert log-var to std
        }

class FinancialTransformer(nn.Module):
    """Advanced Transformer specifically designed for financial markets"""
    
    def __init__(self, input_size: int, d_model: int = 256, n_heads: int = 8, 
                num_layers: int = 6, d_ff: int = 1024, dropout: float = 0.1,
                max_len: int = 100, drop_path: float = 0.0):
        super().__init__()

        self.d_model = d_model

        # Input processing
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # Transformer layers with DropPath (linear scaling by depth)
        drop_path_rates = [drop_path * i / max(1, num_layers - 1) for i in range(num_layers)]
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, drop_path_rates[i])
            for i in range(num_layers)
        ])

        # Output processing
        self.layer_norm = nn.LayerNorm(d_model)

        # Heads
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
            nn.Softplus()
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        # Feature attention (time-major inputs, but we keep batch_first=True for [B,T,E])
        self.feature_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x, return_attention: bool = False, causal_mask: Optional[torch.Tensor] = None):
        """Forward with optional causal (look-ahead) mask over time axis."""
        # x: [B, T, F]
        batch_size, seq_len, _ = x.size()

        x = self.input_projection(x)
        x = self.positional_encoding(x)

        attention_weights = []

        # pass through transformer blocks with optional causal mask
        for transformer_block in self.transformer_blocks:
            x, attn_weights = transformer_block(x, mask=causal_mask)
            if return_attention:
                attention_weights.append(attn_weights)

        x = self.layer_norm(x)

        # feature attention (over embedding dimension; keep unmasked)
        attn_output, feature_weights = self.feature_attention(x, x, x)
        x = x + attn_output

        final_representation = x[:, -1, :]
        return_pred     = self.return_head(final_representation)
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
            'return_prediction': meta_pred,  # Main prediction for training
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
        
        # Final layers with uncertainty heads
        self.end_conv1 = nn.Conv1d(skip_channels, end_channels, kernel_size=1)
        self.mean_conv = nn.Conv1d(end_channels, 1, kernel_size=1)
        self.log_var_conv = nn.Conv1d(end_channels, 1, kernel_size=1)
        
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
        
        # Mean and log-variance predictions
        mean_pred = self.mean_conv(x)[:, :, -1]  # [batch_size, 1]
        log_var_pred = self.log_var_conv(x)[:, :, -1]  # [batch_size, 1]
        
        # Return dictionary with uncertainty
        return {
            'return_prediction': mean_pred,
            'log_variance': log_var_pred,
            'volatility_prediction': torch.exp(0.5 * log_var_pred)  # Convert log-var to std
        }

class TSMixer(nn.Module):
    """TSMixer - MLP-Mixer architecture for time series (2023)"""
    
    def __init__(self, input_size: int, seq_len: int = 30, hidden_dim: int = 256, 
                 num_blocks: int = 8, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_dim)
        
        # TSMixer blocks
        self.mixer_blocks = nn.ModuleList([
            TSMixerBlock(seq_len, hidden_dim, dropout) for _ in range(num_blocks)
        ])
        
        # Output heads
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Shared feature extraction
        self.feature_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Multiple heads for uncertainty
        self.mean_head = nn.Linear(hidden_dim // 4, 1)
        self.log_var_head = nn.Linear(hidden_dim // 4, 1)
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_proj(x)  # [batch, seq_len, hidden_dim]
        
        # Apply mixer blocks
        for block in self.mixer_blocks:
            x = block(x)
        
        x = self.layer_norm(x)
        
        # Global average pooling across time
        x = x.mean(dim=1)  # [batch, hidden_dim]
        
        # Feature extraction
        features = self.feature_proj(x)
        
        # Predictions with uncertainty
        mean_pred = self.mean_head(features)
        log_var_pred = self.log_var_head(features)
        
        return {
            'return_prediction': mean_pred,
            'log_variance': log_var_pred,
            'volatility_prediction': torch.exp(0.5 * log_var_pred)
        }

class TSMixerBlock(nn.Module):
    """Single TSMixer block with time and feature mixing"""
    
    def __init__(self, seq_len: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # Time mixing (across sequence dimension)
        self.time_mixing = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            Transpose(-1, -2),  # [batch, hidden_dim, seq_len]
            nn.Linear(seq_len, seq_len),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_len, seq_len),
            nn.Dropout(dropout),
            Transpose(-1, -2),  # [batch, seq_len, hidden_dim]
        )
        
        # Feature mixing (across feature dimension)  
        self.feature_mixing = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        # Time mixing with residual
        x = x + self.time_mixing(x)
        
        # Feature mixing with residual
        x = x + self.feature_mixing(x)
        
        return x

class Transpose(nn.Module):
    """Helper module for dimension swapping"""
    def __init__(self, dim1: int, dim2: int):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        
    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class TimesNet(nn.Module):
    """TimesNet - Multi-resolution time-spatial pattern extraction (2023)"""
    
    def __init__(self, input_size: int, seq_len: int = 30, d_model: int = 256, 
                 num_kernels: int = 6, top_k: int = 5, dropout: float = 0.1):
        super().__init__()
        
        self.seq_len = seq_len
        self.input_size = input_size
        self.d_model = d_model
        self.num_kernels = num_kernels
        self.top_k = top_k
        
        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Multi-scale convolution kernels for pattern extraction
        kernel_out_channels = d_model // num_kernels
        self.conv_kernels = nn.ModuleList([
            nn.Conv2d(1, kernel_out_channels, kernel_size=(i+1, 1), padding=(i//2, 0))
            for i in range(num_kernels)
        ])
        
        # Calculate actual concatenated channels
        self.concat_channels = kernel_out_channels * num_kernels
        
        # Inception-style blocks for multi-resolution processing
        self.inception_blocks = nn.ModuleList([
            TimesNetBlock(self.concat_channels, seq_len, dropout) for _ in range(3)
        ])
        
        # Adaptive pooling for temporal aggregation
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, d_model))
        
        # Output heads
        self.layer_norm = nn.LayerNorm(self.concat_channels)
        
        # Feature extraction
        self.feature_proj = nn.Sequential(
            nn.Linear(self.concat_channels, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Multiple heads for uncertainty
        self.mean_head = nn.Linear(d_model // 4, 1)
        self.log_var_head = nn.Linear(d_model // 4, 1)
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        batch_size, seq_len, input_size = x.shape
        
        # Input projection
        x = self.input_proj(x)  # [batch, seq_len, d_model]
        
        # Multi-scale feature extraction
        x_reshaped = x.unsqueeze(1)  # [batch, 1, seq_len, d_model]
        
        # Apply multi-scale convolutions
        multi_scale_features = []
        min_seq_len = None
        
        for conv in self.conv_kernels:
            conv_out = conv(x_reshaped)  # [batch, d_model//num_kernels, seq_len', d_model]
            conv_out = F.relu(conv_out)
            
            # Track minimum sequence length for alignment
            if min_seq_len is None:
                min_seq_len = conv_out.shape[2]
            else:
                min_seq_len = min(min_seq_len, conv_out.shape[2])
            
            multi_scale_features.append(conv_out)
        
        # Align all features to minimum sequence length
        aligned_features = []
        for feat in multi_scale_features:
            if feat.shape[2] > min_seq_len:
                # Trim to minimum length
                feat = feat[:, :, :min_seq_len, :]
            aligned_features.append(feat)
        
        # Concatenate multi-scale features
        x_multi = torch.cat(aligned_features, dim=1)  # [batch, d_model, seq_len', d_model]
        
        # Process through inception blocks
        for block in self.inception_blocks:
            x_multi = block(x_multi)
        
        # Adaptive pooling and reshape
        x_pooled = self.adaptive_pool(x_multi)  # [batch, concat_channels, 1, d_model]
        x_flat = x_pooled.squeeze(-2).mean(dim=-1)  # [batch, concat_channels]
        
        x_normed = self.layer_norm(x_flat)
        
        # Feature extraction
        features = self.feature_proj(x_normed)
        
        # Predictions with uncertainty
        mean_pred = self.mean_head(features)
        log_var_pred = self.log_var_head(features)
        
        return {
            'return_prediction': mean_pred,
            'log_variance': log_var_pred,
            'volatility_prediction': torch.exp(0.5 * log_var_pred)
        }

class TimesNetBlock(nn.Module):
    """TimesNet processing block with multi-resolution convolutions"""
    
    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.1):
        super().__init__()
        
        # Multi-branch convolutions for different temporal patterns
        self.conv1 = nn.Conv2d(d_model, d_model, kernel_size=(1, 1))
        self.conv3 = nn.Conv2d(d_model, d_model, kernel_size=(3, 1), padding=(1, 0))
        self.conv5 = nn.Conv2d(d_model, d_model, kernel_size=(5, 1), padding=(2, 0))
        
        # Pooling branch
        self.pool = nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.pool_conv = nn.Conv2d(d_model, d_model, kernel_size=(1, 1))
        
        # Normalization and activation
        self.norm = nn.BatchNorm2d(d_model * 4)
        self.dropout = nn.Dropout2d(dropout)
        
        # Output projection
        self.out_conv = nn.Conv2d(d_model * 4, d_model, kernel_size=(1, 1))
        
    def forward(self, x):
        # Multi-branch processing
        branch1 = F.relu(self.conv1(x))
        branch3 = F.relu(self.conv3(x))
        branch5 = F.relu(self.conv5(x))
        
        # Pooling branch
        pool_out = self.pool(x)
        branch_pool = F.relu(self.pool_conv(pool_out))
        
        # Concatenate all branches
        concat = torch.cat([branch1, branch3, branch5, branch_pool], dim=1)
        
        # Normalize and project
        normalized = self.norm(concat)
        dropped = self.dropout(normalized)
        output = self.out_conv(dropped)
        
        # Residual connection
        return F.relu(output + x)

class PatchTST(nn.Module):
    """PatchTST - Patching-based Transformer for time series (NeurIPS 2023)"""
    
    def __init__(self, input_size: int, seq_len: int = 30, patch_len: int = 8, 
                 stride: int = 4, d_model: int = 256, n_heads: int = 8, 
                 num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.input_size = input_size
        
        # Calculate number of patches
        self.num_patches = max(1, (seq_len - patch_len) // stride + 1)
        
        # Input projection for patches
        self.patch_embedding = nn.Linear(patch_len * input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Feature extraction
        self.feature_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Multiple heads for uncertainty
        self.mean_head = nn.Linear(d_model // 4, 1)
        self.log_var_head = nn.Linear(d_model // 4, 1)
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        batch_size, seq_len, input_size = x.shape
        
        # Create patches
        patches = []
        for i in range(0, seq_len - self.patch_len + 1, self.stride):
            patch = x[:, i:i+self.patch_len, :].reshape(batch_size, -1)
            patches.append(patch)
        
        if not patches:
            # Handle edge case where sequence is too short
            patch = x.reshape(batch_size, -1)
            patches = [patch]
            
        # Stack patches: [batch_size, num_patches, patch_len * input_size]
        x_patches = torch.stack(patches, dim=1)
        
        # Embed patches
        x_embedded = self.patch_embedding(x_patches)  # [batch, num_patches, d_model]
        
        # Add positional encoding
        x_embedded = x_embedded + self.pos_encoding[:, :x_embedded.size(1), :]
        
        # Apply transformer
        x_transformed = self.transformer(x_embedded)
        
        # Global pooling
        x_pooled = x_transformed.mean(dim=1)  # [batch, d_model]
        
        x_normed = self.layer_norm(x_pooled)
        
        # Feature extraction
        features = self.feature_proj(x_normed)
        
        # Predictions with uncertainty
        mean_pred = self.mean_head(features)
        log_var_pred = self.log_var_head(features)
        
        return {
            'return_prediction': mean_pred,
            'log_variance': log_var_pred,
            'volatility_prediction': torch.exp(0.5 * log_var_pred)
        }

class iTransformer(nn.Module):
    """iTransformer - Channel-wise first Transformer for multivariate time series (ICML 2024)"""
    
    def __init__(self, input_size: int, seq_len: int = 30, d_model: int = 256, 
                 n_heads: int = 8, num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        
        self.input_size = input_size
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Channel embedding (each variable gets its own embedding)
        self.channel_embedding = nn.Linear(seq_len, d_model)
        
        # Variable-wise positional encoding
        self.var_pos_encoding = nn.Parameter(torch.randn(1, input_size, d_model))
        
        # Transformer encoder (operates on variables, not time)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection back to time series
        self.time_projection = nn.Linear(d_model, seq_len)
        
        # Final processing
        self.layer_norm = nn.LayerNorm(input_size * seq_len)
        
        # Feature extraction
        self.feature_proj = nn.Sequential(
            nn.Linear(input_size * seq_len, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Multiple heads for uncertainty
        self.mean_head = nn.Linear(d_model // 4, 1)
        self.log_var_head = nn.Linear(d_model // 4, 1)
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        batch_size, seq_len, input_size = x.shape
        
        # Transpose to channel-first: [batch_size, input_size, seq_len]
        x = x.transpose(1, 2)
        
        # Channel embedding: each variable's time series -> embedding
        x_embedded = self.channel_embedding(x)  # [batch, input_size, d_model]
        
        # Add variable positional encoding
        x_embedded = x_embedded + self.var_pos_encoding
        
        # Apply transformer (operates across variables)
        x_transformed = self.transformer(x_embedded)  # [batch, input_size, d_model]
        
        # Project back to time dimension
        x_time = self.time_projection(x_transformed)  # [batch, input_size, seq_len]
        
        # Flatten for final processing
        x_flat = x_time.reshape(batch_size, -1)  # [batch, input_size * seq_len]
        
        x_normed = self.layer_norm(x_flat)
        
        # Feature extraction
        features = self.feature_proj(x_normed)
        
        # Predictions with uncertainty
        mean_pred = self.mean_head(features)
        log_var_pred = self.log_var_head(features)
        
        return {
            'return_prediction': mean_pred,
            'log_variance': log_var_pred,
            'volatility_prediction': torch.exp(0.5 * log_var_pred)
        }

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
            dropout=kwargs.get('dropout', 0.1),
            drop_path=kwargs.get('drop_path', 0.0)
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
    
    elif model_type == "tsmixer":
        return TSMixer(
            input_size=input_size,
            seq_len=kwargs.get('seq_len', 30),
            hidden_dim=kwargs.get('hidden_dim', 256),
            num_blocks=kwargs.get('num_blocks', 8),
            dropout=kwargs.get('dropout', 0.1)
        )
    
    elif model_type == "patchtst":
        return PatchTST(
            input_size=input_size,
            seq_len=kwargs.get('seq_len', 30),
            patch_len=kwargs.get('patch_len', 8),
            stride=kwargs.get('stride', 4),
            d_model=kwargs.get('d_model', 256),
            n_heads=kwargs.get('n_heads', 8),
            num_layers=kwargs.get('num_layers', 6),
            dropout=kwargs.get('dropout', 0.1)
        )
    
    elif model_type == "itransformer":
        return iTransformer(
            input_size=input_size,
            seq_len=kwargs.get('seq_len', 30),
            d_model=kwargs.get('d_model', 256),
            n_heads=kwargs.get('n_heads', 8),
            num_layers=kwargs.get('num_layers', 6),
            dropout=kwargs.get('dropout', 0.1)
        )
    
    elif model_type == "timesnet":
        return TimesNet(
            input_size=input_size,
            seq_len=kwargs.get('seq_len', 30),
            d_model=kwargs.get('d_model', 256),
            num_kernels=kwargs.get('num_kernels', 6),
            top_k=kwargs.get('top_k', 5),
            dropout=kwargs.get('dropout', 0.1)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Advanced loss functions for financial markets
class FinancialLoss(nn.Module):
    """Enhanced multi-objective loss: Margin + Focal + Huber for financial prediction"""
    
    def __init__(self, return_weight=1.0, directional_weight=0.5, volatility_weight=0.3, 
                 directional_margin_bps=10.0, focal_alpha=0.25, focal_gamma=2.0, huber_delta=0.01):
        super().__init__()
        self.return_weight = return_weight
        self.directional_weight = directional_weight
        self.volatility_weight = volatility_weight
        self.directional_margin = directional_margin_bps / 10_000.0  # Convert bps to decimal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.huber_delta = huber_delta
        
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')  # Need reduction='none' for focal
    
    def huber_loss(self, pred, target, delta=0.01):
        """Huber loss - robust to outliers"""
        abs_error = torch.abs(pred - target)
        quadratic = torch.clamp(abs_error, max=delta)
        linear = abs_error - quadratic
        return 0.5 * quadratic ** 2 + delta * linear
    
    def focal_loss(self, logits, targets, alpha=0.25, gamma=2.0):
        """Focal loss for hard examples emphasis"""
        bce_loss = self.bce_loss(logits, targets)
        p_t = torch.exp(-bce_loss)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** gamma
        return (focal_weight * bce_loss).mean()
    
    def margin_directional_loss(self, predictions, targets, margin):
        """Margin-based directional loss - only penalize if wrong AND beyond margin"""
        pred_sign = torch.sign(predictions.squeeze())
        target_sign = torch.sign(targets.squeeze())
        
        # Check if prediction is wrong direction AND magnitude exceeds margin
        wrong_direction = (pred_sign * target_sign) < 0
        exceeds_margin = torch.abs(targets.squeeze()) > margin
        
        # Only apply penalty for significant wrong predictions
        penalty_mask = wrong_direction & exceeds_margin
        
        # Convert to classification logits for focal loss
        pred_logits = predictions.squeeze() * 10  # Scale for better gradients
        target_binary = (targets.squeeze() > 0).float()
        
        return self.focal_loss(pred_logits, target_binary, self.focal_alpha, self.focal_gamma)
    
    def forward(self, predictions, targets, volatility_pred=None):
        """Enhanced loss: Huber + Margin + Focal + Volatility"""
        # 1. Huber loss for return prediction (robust to outliers)
        return_loss = self.huber_loss(predictions.squeeze(), targets.squeeze(), self.huber_delta).mean()
        
        # 2. Margin + Focal directional loss (focus on hard examples)
        directional_loss = self.margin_directional_loss(predictions, targets, self.directional_margin)
        
        total = self.return_weight * return_loss + self.directional_weight * directional_loss
        
        # 3. Volatility prediction loss (if available)
        if volatility_pred is not None:
            target_volatility = torch.abs(targets.squeeze())
            volatility_loss = self.huber_loss(volatility_pred.squeeze(), target_volatility, self.huber_delta).mean()
            total += self.volatility_weight * volatility_loss
        
        return total

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
