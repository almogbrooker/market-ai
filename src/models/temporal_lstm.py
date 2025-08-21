#!/usr/bin/env python3
"""
Model B: Compact LSTM (Temporal Specialist)
Captures local serial structure that trees can't
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import logging
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SequenceDataset(Dataset):
    """Dataset for sequence learning"""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class CompactLSTM(nn.Module):
    """
    Compact LSTM for temporal patterns
    Input: 30-day windows of core features
    Output: next-day cross-sectional return prediction
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, 
                 dropout: float = 0.2, seq_len: int = 30):
        super(CompactLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        
        # LSTM core
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        
        # Output heads
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Auxiliary binary head (up/down)
        self.binary_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM forward
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)
        
        # Take last timestep
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden)
        
        # Apply dropout
        last_hidden = self.dropout(last_hidden)
        
        # Predictions
        regression_out = self.regression_head(last_hidden).squeeze(-1)  # (batch,)
        binary_out = self.binary_head(last_hidden).squeeze(-1)  # (batch,)
        
        return {
            'regression': regression_out,
            'binary': binary_out
        }

class TemporalLSTMTrainer:
    """Trainer for compact LSTM model"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🔥 Using device: {self.device}")
        
        # Model parameters
        self.hidden_dim = config.get('hidden_dim', 64)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.2)
        self.seq_len = config.get('seq_len', 30)
        self.batch_size = config.get('batch_size', 256)
        self.learning_rate = config.get('learning_rate', 1e-3)
        self.max_epochs = config.get('max_epochs', 60)
        self.early_stop_patience = config.get('early_stop_patience', 8)
        
        self.model = None
        self.scaler = None
        
    def prepare_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare 30-day sequences for LSTM"""
        
        logger.info("🔧 Preparing sequences for LSTM...")
        
        # Core features (≤10 as specified)
        feature_cols = [
            'return_5d_lag1', 'return_20d_lag1', 'return_60d_lag1', 'return_12m_ex_1m_lag1',
            'vol_20d_lag1', 'vol_60d_lag1', 'volume_ratio_lag1', 'dollar_volume_ratio_lag1',
            'log_price_lag1', 'price_volume_trend_lag1'
        ]
        
        # Filter available features
        available_features = [col for col in feature_cols if col in data.columns]
        if not available_features:
            raise ValueError("No features found in data")
        
        logger.info(f"   Features: {len(available_features)} ({available_features[:3]}...)")
        
        # Sort by ticker and date
        data = data.sort_values(['Ticker', 'Date']).reset_index(drop=True)
        
        sequences = []
        targets = []
        valid_indices = []
        
        for ticker in data['Ticker'].unique():
            ticker_data = data[data['Ticker'] == ticker].reset_index(drop=True)
            
            if len(ticker_data) < self.seq_len + 1:
                continue
            
            # Create sequences
            for i in range(self.seq_len, len(ticker_data)):
                # Sequence: [i-seq_len:i] -> target: i
                seq_data = ticker_data.iloc[i-self.seq_len:i][available_features].values
                target = ticker_data.iloc[i]['next_return_1d']
                
                # Check for valid data
                if not (np.isnan(seq_data).any() or np.isnan(target)):
                    sequences.append(seq_data)
                    targets.append(target)
                    valid_indices.append(ticker_data.index[i])
        
        if not sequences:
            raise ValueError("No valid sequences created")
        
        sequences = np.array(sequences)  # (n_samples, seq_len, n_features)
        targets = np.array(targets)      # (n_samples,)
        valid_indices = np.array(valid_indices)
        
        logger.info(f"✅ Created {len(sequences):,} sequences")
        logger.info(f"   Shape: {sequences.shape}")
        logger.info(f"   Target range: [{targets.min():.4f}, {targets.max():.4f}]")
        
        return sequences, targets, valid_indices
    
    def train_model(self, train_data: pd.DataFrame, val_data: pd.DataFrame = None) -> Dict:
        """Train the compact LSTM model"""
        
        logger.info("🏋️ Training Compact LSTM...")
        
        # Prepare training sequences
        X_train, y_train, train_indices = self.prepare_sequences(train_data)
        
        # Prepare validation sequences
        if val_data is not None:
            X_val, y_val, val_indices = self.prepare_sequences(val_data)
        else:
            # Use 20% of training data for validation
            n_val = int(0.2 * len(X_train))
            X_val, y_val = X_train[-n_val:], y_train[-n_val:]
            X_train, y_train = X_train[:-n_val], y_train[:-n_val]
        
        # Standardize features (per sequence timestep)
        n_samples, seq_len, n_features = X_train.shape
        X_train_flat = X_train.reshape(-1, n_features)
        X_val_flat = X_val.reshape(-1, n_features)
        
        self.scaler = StandardScaler()
        X_train_flat = self.scaler.fit_transform(X_train_flat)
        X_val_flat = self.scaler.transform(X_val_flat)
        
        X_train = X_train_flat.reshape(n_samples, seq_len, n_features)
        X_val = X_val_flat.reshape(X_val.shape[0], seq_len, n_features)
        
        # Create datasets
        train_dataset = SequenceDataset(X_train, y_train)
        val_dataset = SequenceDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Initialize model
        self.model = CompactLSTM(
            input_dim=n_features,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            seq_len=self.seq_len
        ).to(self.device)
        
        # Optimizer and loss
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        regression_loss = nn.MSELoss()
        binary_loss = nn.BCELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.max_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(batch_x)
                
                # Combined loss
                reg_loss = regression_loss(outputs['regression'], batch_y)
                bin_targets = (batch_y > 0).float()
                bin_loss = binary_loss(outputs['binary'], bin_targets)
                
                total_loss = reg_loss + 0.1 * bin_loss  # Weight binary loss less
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += total_loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    
                    outputs = self.model(batch_x)
                    reg_loss = regression_loss(outputs['regression'], batch_y)
                    bin_targets = (batch_y > 0).float()
                    bin_loss = binary_loss(outputs['binary'], bin_targets)
                    
                    total_loss = reg_loss + 0.1 * bin_loss
                    val_loss += total_loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_lstm_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                logger.info(f"   Epoch {epoch:3d}: Train Loss {train_loss:.6f}, Val Loss {val_loss:.6f}")
            
            if patience_counter >= self.early_stop_patience:
                logger.info(f"   Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_lstm_model.pth'))
        
        results = {
            'best_val_loss': best_val_loss,
            'final_epoch': epoch,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        logger.info(f"✅ LSTM training completed")
        logger.info(f"   Best val loss: {best_val_loss:.6f}")
        logger.info(f"   Final epoch: {epoch}")
        
        return results
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions with trained LSTM"""
        
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained")
        
        # Prepare sequences
        X, _, indices = self.prepare_sequences(data)
        
        # Standardize
        n_samples, seq_len, n_features = X.shape
        X_flat = X.reshape(-1, n_features)
        X_flat = self.scaler.transform(X_flat)
        X = X_flat.reshape(n_samples, seq_len, n_features)
        
        # Predict
        self.model.eval()
        predictions = []
        
        dataset = SequenceDataset(X, np.zeros(len(X)))  # Dummy targets
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                predictions.extend(outputs['regression'].cpu().numpy())
        
        return np.array(predictions)
    
    def save_model(self, path: Path):
        """Save model and scaler"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'config': self.config
        }, path)
        logger.info(f"✅ Model saved: {path}")
    
    def load_model(self, path: Path):
        """Load model and scaler"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Recreate model
        self.config = checkpoint['config']
        # Model recreation logic here...
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        logger.info(f"✅ Model loaded: {path}")

def main():
    """Test the compact LSTM"""
    
    # Configuration
    config = {
        'hidden_dim': 64,
        'num_layers': 2,
        'dropout': 0.2,
        'seq_len': 30,
        'batch_size': 256,
        'learning_rate': 1e-3,
        'max_epochs': 60,
        'early_stop_patience': 8
    }
    
    # Test with synthetic data
    import pandas as pd
    np.random.seed(42)
    
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    data = []
    for ticker in tickers:
        for date in dates:
            row = {
                'Date': date,
                'Ticker': ticker,
                'next_return_1d': np.random.normal(0, 0.02),
                **{f'return_{d}d_lag1': np.random.normal(0, 0.01) for d in [5, 20, 60]},
                'return_12m_ex_1m_lag1': np.random.normal(0, 0.05),
                **{f'vol_{d}d_lag1': np.random.uniform(0.1, 0.5) for d in [20, 60]},
                'volume_ratio_lag1': np.random.uniform(0.5, 2.0),
                'dollar_volume_ratio_lag1': np.random.uniform(0.5, 2.0),
                'log_price_lag1': np.log(np.random.uniform(50, 300)),
                'price_volume_trend_lag1': np.random.normal(0, 0.01)
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    
    # Train model
    trainer = TemporalLSTMTrainer(config)
    
    # Split data
    split_date = '2023-01-01'
    train_data = df[df['Date'] < split_date]
    val_data = df[df['Date'] >= split_date]
    
    results = trainer.train_model(train_data, val_data)
    print(f"✅ Training completed: {results}")

if __name__ == "__main__":
    main()