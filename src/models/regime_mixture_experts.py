#!/usr/bin/env python3
"""
REGIME MIXTURE OF EXPERTS
Train specialist models for different market regimes with gating network
Expected +0.002-0.004 IC improvement from regime specialization
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union
import logging
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class RegimeIdentifier:
    """
    Identifies market regimes for specialist model assignment
    """
    
    def __init__(self, 
                 method: str = 'vix_vol',
                 n_regimes: int = 3,
                 regime_names: Optional[List[str]] = None):
        """
        Initialize regime identifier
        
        Args:
            method: 'vix_vol', 'gmm', 'kmeans', or 'manual'
            n_regimes: Number of regimes to identify
            regime_names: Custom regime names
        """
        self.method = method
        self.n_regimes = n_regimes
        self.regime_names = regime_names or [f'regime_{i}' for i in range(n_regimes)]
        
        # Regime models
        self.regime_model = None
        self.scaler = StandardScaler()
        
        # Regime thresholds (for manual method)
        self.vix_thresholds = [15, 25, 40]  # Low, Medium, High, Extreme
        self.vol_thresholds = [0.8, 1.2, 2.0]  # Relative volatility thresholds
        
        logger.info(f"🎭 RegimeIdentifier initialized: method={method}, regimes={n_regimes}")
    
    def fit(self, features: np.ndarray, 
            market_data: Optional[pd.DataFrame] = None) -> 'RegimeIdentifier':
        """
        Fit regime identification model
        
        Args:
            features: Feature matrix for regime identification
            market_data: Optional market data (VIX, volatility, etc.)
        """
        if self.method == 'manual' and market_data is not None:
            # Manual regime assignment using VIX and volatility
            self._fit_manual_regimes(market_data)
        elif self.method == 'gmm':
            # Gaussian Mixture Model
            self._fit_gmm_regimes(features)
        elif self.method == 'kmeans':
            # K-Means clustering
            self._fit_kmeans_regimes(features)
        else:
            # Default: VIX and volatility based
            if market_data is not None:
                self._fit_vix_vol_regimes(market_data)
            else:
                # Fallback to GMM on features
                self._fit_gmm_regimes(features)
        
        logger.info("✅ Regime identification model fitted")
        return self
    
    def predict(self, features: np.ndarray,
                market_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Predict regime for new data
        
        Returns:
            Array of regime assignments
        """
        if self.method == 'manual' or self.method == 'vix_vol':
            if market_data is not None:
                return self._predict_manual_regimes(market_data)
            else:
                # Fallback
                return np.zeros(len(features), dtype=int)
        
        elif self.method == 'gmm' or self.method == 'kmeans':
            features_scaled = self.scaler.transform(features)
            return self.regime_model.predict(features_scaled)
        
        else:
            return np.zeros(len(features), dtype=int)
    
    def _fit_manual_regimes(self, market_data: pd.DataFrame):
        """Fit manual regime assignment using market indicators"""
        # This is rule-based, no training needed
        pass
    
    def _fit_vix_vol_regimes(self, market_data: pd.DataFrame):
        """Fit VIX and volatility based regimes"""
        # Extract regime features
        regime_features = self._extract_regime_features(market_data)
        
        # Use GMM on regime features
        self.scaler.fit(regime_features)
        features_scaled = self.scaler.transform(regime_features)
        
        self.regime_model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            random_state=42,
            max_iter=200
        )
        
        self.regime_model.fit(features_scaled)
    
    def _fit_gmm_regimes(self, features: np.ndarray):
        """Fit Gaussian Mixture Model for regime identification"""
        self.scaler.fit(features)
        features_scaled = self.scaler.transform(features)
        
        self.regime_model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            random_state=42,
            max_iter=200
        )
        
        self.regime_model.fit(features_scaled)
    
    def _fit_kmeans_regimes(self, features: np.ndarray):
        """Fit K-Means clustering for regime identification"""
        self.scaler.fit(features)
        features_scaled = self.scaler.transform(features)
        
        self.regime_model = KMeans(
            n_clusters=self.n_regimes,
            random_state=42,
            n_init=10
        )
        
        self.regime_model.fit(features_scaled)
    
    def _predict_manual_regimes(self, market_data: pd.DataFrame) -> np.ndarray:
        """Predict regimes using manual rules"""
        regimes = np.zeros(len(market_data), dtype=int)
        
        if 'VIX' in market_data.columns and 'realized_vol' in market_data.columns:
            vix = market_data['VIX'].values
            vol = market_data['realized_vol'].values
            
            # Combine VIX and realized volatility for regime assignment
            for i, (v, rv) in enumerate(zip(vix, vol)):
                if v < 15 and rv < 0.15:
                    regimes[i] = 0  # Low volatility
                elif v < 25 and rv < 0.25:
                    regimes[i] = 1  # Medium volatility
                else:
                    regimes[i] = 2  # High volatility
        
        elif 'VIX' in market_data.columns:
            vix = market_data['VIX'].values
            
            for i, v in enumerate(vix):
                if v < 15:
                    regimes[i] = 0
                elif v < 25:
                    regimes[i] = 1
                else:
                    regimes[i] = 2
        
        else:
            # Random assignment as fallback
            regimes = np.random.randint(0, self.n_regimes, len(market_data))
        
        return regimes
    
    def _extract_regime_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Extract features for regime identification"""
        features = []
        
        # VIX level and changes
        if 'VIX' in market_data.columns:
            vix = market_data['VIX'].ffill().fillna(20)
            features.append(vix.values)
            features.append(vix.pct_change().fillna(0).values)
        
        # Realized volatility
        if 'realized_vol' in market_data.columns:
            vol = market_data['realized_vol'].ffill().fillna(0.15)
            features.append(vol.values)
        
        # Market returns
        if 'market_return' in market_data.columns:
            ret = market_data['market_return'].fillna(0)
            features.append(ret.values)
            features.append(ret.rolling(5).mean().fillna(0).values)
        
        # Interest rates
        if 'interest_rate' in market_data.columns:
            ir = market_data['interest_rate'].ffill().fillna(0.02)
            features.append(ir.values)
        
        # If no features available, create dummy features
        if not features:
            features.append(np.random.randn(len(market_data)))
            features.append(np.random.randn(len(market_data)))
        
        # Combine features and clean infinite values
        feature_matrix = np.column_stack(features)
        
        # Replace infinite values with finite ones
        feature_matrix = np.where(np.isfinite(feature_matrix), feature_matrix, 0)
        
        # Replace extreme values
        feature_matrix = np.clip(feature_matrix, -1000, 1000)
        
        return feature_matrix
    
    def get_regime_names(self) -> List[str]:
        """Get regime names"""
        return self.regime_names

class RegimeExpert(nn.Module):
    """
    Individual expert model specialized for a specific regime
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 expert_type: str = 'lstm'):
        """
        Initialize regime expert
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden layer size
            num_layers: Number of layers
            dropout: Dropout rate
            expert_type: 'lstm', 'linear', or 'lightgbm'
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.expert_type = expert_type
        
        if expert_type == 'lstm':
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True, dropout=dropout
            )
            self.fc = nn.Linear(hidden_size, 1)
        
        elif expert_type == 'linear':
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1)
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through expert
        
        Args:
            x: Input features [batch_size, seq_len, input_size] for LSTM
               or [batch_size, input_size] for linear
        """
        if self.expert_type == 'lstm':
            if x.dim() == 2:
                x = x.unsqueeze(1)  # Add sequence dimension
            
            lstm_out, _ = self.lstm(x)
            output = self.fc(self.dropout(lstm_out[:, -1, :]))
        
        elif self.expert_type == 'linear':
            if x.dim() == 3:
                x = x.view(x.size(0), -1)  # Flatten
            
            output = self.network(x)
        
        return output

class GatingNetwork(nn.Module):
    """
    Gating network to combine expert outputs
    """
    
    def __init__(self, 
                 input_size: int,
                 n_experts: int,
                 hidden_size: int = 32):
        """
        Initialize gating network
        
        Args:
            input_size: Number of input features
            n_experts: Number of expert models
            hidden_size: Hidden layer size
        """
        super().__init__()
        
        self.gate = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, n_experts),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gating weights
        
        Args:
            x: Input features [batch_size, input_size]
            
        Returns:
            Gating weights [batch_size, n_experts]
        """
        if x.dim() == 3:
            x = x.view(x.size(0), -1)  # Flatten for gating
        
        return self.gate(x)

class RegimeMixtureOfExperts:
    """
    Mixture of Experts with regime-specific specialization
    """
    
    def __init__(self,
                 input_size: int,
                 n_regimes: int = 3,
                 expert_hidden_size: int = 64,
                 expert_type: str = 'lstm',
                 use_lightgbm_experts: bool = True):
        """
        Initialize Regime Mixture of Experts
        
        Args:
            input_size: Number of input features
            n_regimes: Number of market regimes
            expert_hidden_size: Hidden size for neural experts
            expert_type: Type of expert ('lstm' or 'linear')
            use_lightgbm_experts: Use LightGBM as experts
        """
        self.input_size = input_size
        self.n_regimes = n_regimes
        self.expert_type = expert_type
        self.use_lightgbm_experts = use_lightgbm_experts
        
        # Components
        self.regime_identifier = RegimeIdentifier(
            method='vix_vol',
            n_regimes=n_regimes,
            regime_names=[f'regime_{i}' for i in range(n_regimes)]
        )
        
        # Experts
        if use_lightgbm_experts:
            self.experts = {}  # Will be populated with LightGBM models
        else:
            self.experts = nn.ModuleDict({
                f'expert_{i}': RegimeExpert(
                    input_size=input_size,
                    hidden_size=expert_hidden_size,
                    expert_type=expert_type
                )
                for i in range(n_regimes)
            })
        
        # Gating network (for neural experts)
        if not use_lightgbm_experts:
            self.gating_network = GatingNetwork(input_size, n_regimes)
        else:
            self.gating_network = None
        
        # Training history
        self.regime_performance = {i: [] for i in range(n_regimes)}
        
        logger.info(f"🎭 RegimeMixtureOfExperts initialized: {n_regimes} regimes, {expert_type} experts")
    
    def fit(self, 
            X: np.ndarray,
            y: np.ndarray,
            market_data: Optional[pd.DataFrame] = None,
            sample_weight: Optional[np.ndarray] = None) -> 'RegimeMixtureOfExperts':
        """
        Fit mixture of experts model
        
        Args:
            X: Feature matrix [n_samples, n_features] or [n_samples, seq_len, n_features]
            y: Target values [n_samples]
            market_data: Market data for regime identification
            sample_weight: Sample weights
        """
        logger.info(f"🔧 Training Regime Mixture of Experts on {len(X)} samples")
        
        # Identify regimes
        if X.ndim == 3:
            regime_features = X[:, -1, :]  # Use last timestep for regime identification
        else:
            regime_features = X
        
        self.regime_identifier.fit(regime_features, market_data)
        regimes = self.regime_identifier.predict(regime_features, market_data)
        
        # Train experts per regime
        if self.use_lightgbm_experts:
            self._fit_lightgbm_experts(X, y, regimes, sample_weight)
        else:
            self._fit_neural_experts(X, y, regimes, sample_weight)
        
        logger.info("✅ Regime Mixture of Experts training completed")
        return self
    
    def predict(self,
                X: np.ndarray,
                market_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Predict using mixture of experts
        
        Args:
            X: Feature matrix
            market_data: Market data for regime identification
            
        Returns:
            Predictions
        """
        # Identify regimes
        if X.ndim == 3:
            regime_features = X[:, -1, :]
        else:
            regime_features = X
        
        regimes = self.regime_identifier.predict(regime_features, market_data)
        
        if self.use_lightgbm_experts:
            return self._predict_lightgbm(X, regimes)
        else:
            return self._predict_neural(X, regimes)
    
    def _fit_lightgbm_experts(self, 
                             X: np.ndarray, 
                             y: np.ndarray,
                             regimes: np.ndarray,
                             sample_weight: Optional[np.ndarray] = None):
        """Fit LightGBM experts for each regime"""
        
        # Flatten features if needed
        if X.ndim == 3:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X
        
        for regime_id in range(self.n_regimes):
            regime_mask = regimes == regime_id
            
            if regime_mask.sum() < 50:
                logger.warning(f"⚠️ Insufficient samples ({regime_mask.sum()}) for regime {regime_id}")
                # Use a simple mean predictor
                self.experts[regime_id] = np.mean(y)
                continue
            
            # Get regime-specific data
            X_regime = X_flat[regime_mask]
            y_regime = y[regime_mask]
            weights_regime = sample_weight[regime_mask] if sample_weight is not None else None
            
            # Train LightGBM
            train_data = lgb.Dataset(
                X_regime, label=y_regime, weight=weights_regime,
                feature_name=[f'feature_{i}' for i in range(X_flat.shape[1])]
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
                'random_state': 42 + regime_id
            }
            
            expert_model = lgb.train(
                params, train_data,
                num_boost_round=200,
                valid_sets=[train_data],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            self.experts[regime_id] = expert_model
            
            # Evaluate regime performance
            regime_pred = expert_model.predict(X_regime)
            regime_ic = np.corrcoef(regime_pred, y_regime)[0, 1]
            self.regime_performance[regime_id].append(regime_ic)
            
            logger.info(f"📊 Regime {regime_id}: {regime_mask.sum()} samples, IC={regime_ic:.4f}")
    
    def _fit_neural_experts(self,
                           X: np.ndarray,
                           y: np.ndarray,
                           regimes: np.ndarray,
                           sample_weight: Optional[np.ndarray] = None):
        """Fit neural network experts"""
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        regimes_tensor = torch.LongTensor(regimes)
        
        # Training parameters
        n_epochs = 100
        batch_size = 64
        learning_rate = 0.001
        
        # Optimizers for experts and gate
        expert_optimizers = {}
        for i in range(self.n_regimes):
            expert_optimizers[i] = torch.optim.Adam(
                self.experts[f'expert_{i}'].parameters(), lr=learning_rate
            )
        
        gate_optimizer = torch.optim.Adam(
            self.gating_network.parameters(), lr=learning_rate
        )
        
        criterion = nn.MSELoss()
        
        # Training loop
        for epoch in range(n_epochs):
            # Shuffle data
            perm = torch.randperm(len(X))
            X_shuffled = X_tensor[perm]
            y_shuffled = y_tensor[perm]
            regimes_shuffled = regimes_tensor[perm]
            
            epoch_loss = 0
            n_batches = 0
            
            for i in range(0, len(X), batch_size):
                batch_end = min(i + batch_size, len(X))
                X_batch = X_shuffled[i:batch_end]
                y_batch = y_shuffled[i:batch_end]
                regimes_batch = regimes_shuffled[i:batch_end]
                
                # Forward pass through experts
                expert_outputs = {}
                for regime_id in range(self.n_regimes):
                    expert_key = f'expert_{regime_id}'
                    expert_outputs[regime_id] = self.experts[expert_key](X_batch)
                
                # Gating weights
                gate_weights = self.gating_network(X_batch)
                
                # Combine expert outputs
                combined_output = torch.zeros_like(y_batch)
                for regime_id in range(self.n_regimes):
                    combined_output += gate_weights[:, regime_id:regime_id+1] * expert_outputs[regime_id]
                
                # Loss
                loss = criterion(combined_output, y_batch)
                
                # Backward pass
                for optimizer in expert_optimizers.values():
                    optimizer.zero_grad()
                gate_optimizer.zero_grad()
                
                loss.backward()
                
                for optimizer in expert_optimizers.values():
                    optimizer.step()
                gate_optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            if epoch % 20 == 0:
                avg_loss = epoch_loss / n_batches
                logger.debug(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
    
    def _predict_lightgbm(self, X: np.ndarray, regimes: np.ndarray) -> np.ndarray:
        """Predict using LightGBM experts"""
        
        # Flatten features if needed
        if X.ndim == 3:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X
        
        predictions = np.zeros(len(X))
        
        for regime_id in range(self.n_regimes):
            regime_mask = regimes == regime_id
            
            if regime_mask.sum() == 0:
                continue
            
            expert = self.experts[regime_id]
            
            if isinstance(expert, (int, float)):
                # Simple mean predictor fallback
                predictions[regime_mask] = expert
            else:
                # LightGBM prediction
                regime_pred = expert.predict(X_flat[regime_mask])
                predictions[regime_mask] = regime_pred
        
        return predictions
    
    def _predict_neural(self, X: np.ndarray, regimes: np.ndarray) -> np.ndarray:
        """Predict using neural experts"""
        
        X_tensor = torch.FloatTensor(X)
        
        with torch.no_grad():
            # Expert outputs
            expert_outputs = {}
            for regime_id in range(self.n_regimes):
                expert_key = f'expert_{regime_id}'
                expert_outputs[regime_id] = self.experts[expert_key](X_tensor)
            
            # Gating weights
            gate_weights = self.gating_network(X_tensor)
            
            # Combine outputs
            combined_output = torch.zeros(len(X), 1)
            for regime_id in range(self.n_regimes):
                combined_output += gate_weights[:, regime_id:regime_id+1] * expert_outputs[regime_id]
        
        return combined_output.numpy().flatten()
    
    def get_regime_performance(self) -> Dict[int, List[float]]:
        """Get performance metrics per regime"""
        return self.regime_performance
    
    def get_expert_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from LightGBM experts"""
        importance = {}
        
        if self.use_lightgbm_experts:
            for regime_id, expert in self.experts.items():
                if hasattr(expert, 'feature_importance'):
                    importance[f'regime_{regime_id}'] = expert.feature_importance(importance_type='gain')
        
        return importance
    
    def save_model(self, path: str):
        """Save the mixture of experts model"""
        model_data = {
            'regime_identifier': self.regime_identifier,
            'experts': self.experts,
            'regime_performance': self.regime_performance,
            'n_regimes': self.n_regimes,
            'input_size': self.input_size,
            'use_lightgbm_experts': self.use_lightgbm_experts
        }
        
        joblib.dump(model_data, path)
        logger.info(f"💾 Regime Mixture of Experts saved to {path}")
    
    @classmethod
    def load_model(cls, path: str) -> 'RegimeMixtureOfExperts':
        """Load a saved mixture of experts model"""
        model_data = joblib.load(path)
        
        # Recreate instance
        model = cls(
            input_size=model_data['input_size'],
            n_regimes=model_data['n_regimes'],
            use_lightgbm_experts=model_data['use_lightgbm_experts']
        )
        
        # Restore components
        model.regime_identifier = model_data['regime_identifier']
        model.experts = model_data['experts']
        model.regime_performance = model_data['regime_performance']
        
        logger.info(f"📂 Regime Mixture of Experts loaded from {path}")
        return model

# Usage example
def example_regime_mixture():
    """Example usage of Regime Mixture of Experts"""
    
    # Simulate data
    np.random.seed(42)
    n_samples = 1000
    seq_len = 30
    n_features = 20
    
    X = np.random.randn(n_samples, seq_len, n_features)
    y = np.random.randn(n_samples) * 0.02
    
    # Create market data for regime identification
    market_data = pd.DataFrame({
        'VIX': 20 + 10 * np.random.randn(n_samples),
        'realized_vol': 0.15 + 0.05 * np.random.randn(n_samples),
        'market_return': 0.001 + 0.01 * np.random.randn(n_samples)
    })
    
    # Initialize and train model
    model = RegimeMixtureOfExperts(
        input_size=n_features,
        n_regimes=3,
        use_lightgbm_experts=True
    )
    
    # Train
    model.fit(X, y, market_data)
    
    # Predict
    predictions = model.predict(X, market_data)
    
    # Evaluate
    ic = np.corrcoef(predictions, y)[0, 1]
    logger.info(f"🎯 Overall IC: {ic:.4f}")
    
    # Regime performance
    regime_perf = model.get_regime_performance()
    for regime_id, perfs in regime_perf.items():
        if perfs:
            avg_ic = np.mean(perfs)
            logger.info(f"📊 Regime {regime_id} IC: {avg_ic:.4f}")

if __name__ == "__main__":
    example_regime_mixture()