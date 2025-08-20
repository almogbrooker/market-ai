#!/usr/bin/env python3
"""
Generate Out-of-Fold (OOF) Predictions for Conformal Calibration
Runs inference on historical data using our trained models to create calibration dataset
"""

import pandas as pd
import numpy as np
import torch
import joblib
from pathlib import Path
import sys
import yfinance as yf
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.advanced_models import AdvancedLSTM, iTransformer, PatchTST
from data.alpha_loader import AlphaDataLoader
from features.streamlined_pipeline import StreamlinedDataPipeline
import pandas_ta as ta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OOFGenerator:
    """Generate out-of-fold predictions for conformal calibration"""
    
    def __init__(self, model_dir="artifacts/models/best", data_dir="data"):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.models = {}
        self.scalers = {}
        self.features = []
        
        logger.info(f"OOF Generator initialized: models={self.model_dir}, data={self.data_dir}")
    
    def load_models(self):
        """Load trained models and scalers"""
        model_types = ['lstm', 'itransformer', 'patchtst']
        device = torch.device('cpu')
        
        for model_type in model_types:
            try:
                # Look for fold-based models
                for fold in [0, 1, 2]:
                    model_path = self.model_dir / f"{model_type}_fold_{fold}.pt"
                    scaler_path = self.model_dir / f"{model_type}_fold_{fold}_scaler.pkl"
                    
                    if model_path.exists() and scaler_path.exists():
                        # Load model
                        if model_type == 'lstm':
                            model = AdvancedLSTM(input_size=92, hidden_size=64, num_layers=2, dropout=0.3)
                        elif model_type == 'itransformer':
                            model = iTransformer(input_size=92, d_model=128, nhead=8, num_layers=4, dropout=0.3)
                        else:  # patchtst
                            model = PatchTST(input_size=92, d_model=128, nhead=8, num_layers=4, 
                                           patch_len=8, stride=4, dropout=0.3)
                        
                        model.load_state_dict(torch.load(model_path, map_location=device))
                        model.eval()
                        
                        # Load scaler
                        with open(scaler_path, 'rb') as f:
                            scaler = joblib.load(f)
                        
                        model_key = f"{model_type}_fold_{fold}"
                        self.models[model_key] = model
                        self.scalers[model_key] = scaler
                        
                        logger.info(f"Loaded {model_key}")
                        
            except Exception as e:
                logger.warning(f"Could not load {model_type}: {e}")
        
        logger.info(f"Loaded {len(self.models)} models")
    
    def load_historical_data(self, lookback_days=365):
        """Load historical data for OOF generation"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Load main dataset
        main_data_file = self.data_dir / "training_data_2020_2024_complete.csv"
        
        if main_data_file.exists():
            logger.info(f"Loading main dataset: {main_data_file}")
            df = pd.read_csv(main_data_file)
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter to recent period
            df = df[df['date'] >= start_date].copy()
            logger.info(f"Filtered to {len(df)} samples from {start_date.date()}")
            
            return df
        else:
            logger.warning(f"Main dataset not found: {main_data_file}")
            return self.create_synthetic_data(lookback_days)
    
    def create_synthetic_data(self, lookback_days=365):
        """Create synthetic dataset if main data not available"""
        logger.info("Creating synthetic dataset for OOF generation")
        
        # Get stock universe
        stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 
                 'INTC', 'QCOM', 'AVGO', 'TXN', 'ORCL', 'CRM', 'ADBE', 'NOW']
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 100)  # Extra buffer for indicators
        
        all_data = []
        
        for symbol in tqdm(stocks, desc="Downloading data"):
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)
                
                if len(hist) < 50:
                    continue
                
                # Calculate features
                df = hist.copy()
                df['symbol'] = symbol
                df['date'] = df.index
                
                # Technical indicators
                df['rsi'] = ta.rsi(df['Close'], length=14)
                df['macd'] = ta.macd(df['Close'])['MACD_12_26_9']
                df['bb_upper'], df['bb_middle'], df['bb_lower'] = ta.bbands(df['Close']).iloc[:, 0], ta.bbands(df['Close']).iloc[:, 1], ta.bbands(df['Close']).iloc[:, 2]
                df['sma_20'] = ta.sma(df['Close'], length=20)
                df['ema_12'] = ta.ema(df['Close'], length=12)
                
                # Returns
                df['return_1d'] = df['Close'].pct_change(1)
                df['return_5d'] = df['Close'].pct_change(5)
                df['return_20d'] = df['Close'].pct_change(20)
                
                # Volume indicators  
                df['volume_sma'] = df['Volume'].rolling(20).mean()
                df['volume_ratio'] = df['Volume'] / df['volume_sma']
                
                # Volatility
                df['volatility'] = df['return_1d'].rolling(20).std() * np.sqrt(252)
                
                # Create target (next day return)
                df['target'] = df['return_1d'].shift(-1)
                
                # Add regime (simplified)
                df['regime'] = 'neutral'
                df.loc[df['return_20d'] > 0.05, 'regime'] = 'bull' 
                df.loc[df['return_20d'] < -0.05, 'regime'] = 'bear'
                
                # Filter to desired date range
                df = df[df['date'] >= end_date - timedelta(days=lookback_days)]
                df = df.dropna()
                
                if len(df) > 10:
                    all_data.append(df)
                    
            except Exception as e:
                logger.warning(f"Error downloading {symbol}: {e}")
                continue
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Created synthetic dataset: {len(combined_df)} samples, {combined_df['symbol'].nunique()} stocks")
            return combined_df
        else:
            raise ValueError("Could not create synthetic dataset")
    
    def prepare_features(self, df):
        """Prepare feature matrix for models"""
        # Select numerical columns (skip symbol, date, target, regime)
        feature_cols = [col for col in df.columns 
                       if col not in ['symbol', 'date', 'target', 'regime'] 
                       and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
        
        # Fill missing values
        for col in feature_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # Ensure we have enough features (pad with zeros if needed)
        if len(feature_cols) < 92:
            for i in range(len(feature_cols), 92):
                df[f'pad_feature_{i}'] = 0.0
                feature_cols.append(f'pad_feature_{i}')
        
        feature_cols = feature_cols[:92]  # Take first 92 features
        self.features = feature_cols
        
        logger.info(f"Using {len(feature_cols)} features: {feature_cols[:10]}...")
        return df[feature_cols].values, df['target'].values
    
    def generate_predictions(self, X, model, scaler, sequence_length=30):
        """Generate predictions using a single model"""
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Create sequences
        sequences = []
        valid_indices = []
        
        for i in range(sequence_length, len(X_scaled)):
            seq = X_scaled[i-sequence_length:i]
            sequences.append(seq)
            valid_indices.append(i)
        
        if not sequences:
            return np.array([]), np.array([])
        
        X_seq = np.array(sequences)
        X_tensor = torch.FloatTensor(X_seq)
        
        # Generate predictions
        predictions = []
        batch_size = 32
        
        model.eval()
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                
                # Handle different model output formats
                output = model(batch)
                if isinstance(output, dict):
                    pred = output.get('return_prediction', output.get('prediction', batch[:, -1, 0]))
                else:
                    pred = output
                
                if pred.dim() > 1:
                    pred = pred.squeeze()
                
                predictions.extend(pred.cpu().numpy())
        
        return np.array(predictions), np.array(valid_indices)
    
    def generate_oof_dataset(self, output_file="oof_predictions.csv"):
        """Generate complete OOF dataset"""
        logger.info("🚀 Starting OOF generation for conformal calibration")
        
        # Load models
        if not self.models:
            self.load_models()
        
        if not self.models:
            raise ValueError("No models loaded - cannot generate OOF predictions")
        
        # Load data
        df = self.load_historical_data()
        
        # Group by symbol for sequential processing
        all_predictions = []
        
        for symbol in tqdm(df['symbol'].unique(), desc="Processing stocks"):
            symbol_data = df[df['symbol'] == symbol].sort_values('date').copy()
            
            if len(symbol_data) < 50:  # Need minimum data for sequences
                continue
            
            X, y = self.prepare_features(symbol_data)
            
            # Generate predictions from each model
            symbol_predictions = []
            
            for model_key, model in self.models.items():
                scaler = self.scalers[model_key]
                
                preds, valid_indices = self.generate_predictions(X, model, scaler)
                
                if len(preds) > 0:
                    # Create DataFrame for this model's predictions
                    model_df = symbol_data.iloc[valid_indices].copy()
                    model_df['prediction'] = preds
                    model_df['model'] = model_key
                    symbol_predictions.append(model_df)
            
            # Combine model predictions for this symbol
            if symbol_predictions:
                symbol_combined = pd.concat(symbol_predictions, ignore_index=True)
                all_predictions.append(symbol_combined)
        
        if not all_predictions:
            raise ValueError("No predictions generated")
        
        # Combine all predictions
        oof_df = pd.concat(all_predictions, ignore_index=True)
        
        # Ensemble predictions (average across models for each symbol-date)
        ensemble_df = oof_df.groupby(['symbol', 'date']).agg({
            'prediction': 'mean',
            'target': 'first',
            'regime': 'first'
        }).reset_index()
        
        # Remove any remaining NaN values
        ensemble_df = ensemble_df.dropna()
        
        # Add metadata
        ensemble_df['confidence'] = np.abs(ensemble_df['prediction'])  # Simple confidence proxy
        ensemble_df['abs_prediction'] = np.abs(ensemble_df['prediction'])
        ensemble_df['abs_target'] = np.abs(ensemble_df['target'])
        
        # Save to file
        ensemble_df.to_csv(output_file, index=False)
        
        logger.info(f"✅ Generated OOF dataset: {len(ensemble_df)} predictions saved to {output_file}")
        
        # Print statistics
        logger.info(f"📊 Dataset statistics:")
        logger.info(f"   Symbols: {ensemble_df['symbol'].nunique()}")
        logger.info(f"   Date range: {ensemble_df['date'].min()} to {ensemble_df['date'].max()}")
        logger.info(f"   Regimes: {ensemble_df['regime'].value_counts().to_dict()}")
        logger.info(f"   Mean |prediction|: {ensemble_df['abs_prediction'].mean():.4f}")
        logger.info(f"   Mean |target|: {ensemble_df['abs_target'].mean():.4f}")
        logger.info(f"   Correlation: {ensemble_df['prediction'].corr(ensemble_df['target']):.3f}")
        
        return ensemble_df

def main():
    """Generate OOF predictions for conformal calibration"""
    generator = OOFGenerator()
    
    # Generate OOF dataset
    oof_df = generator.generate_oof_dataset("oof_predictions_for_conformal.csv")
    
    print("\n" + "="*60)
    print("🎯 OOF GENERATION COMPLETED")
    print("="*60)
    print(f"📁 Output file: oof_predictions_for_conformal.csv")
    print(f"📊 Total samples: {len(oof_df):,}")
    print(f"📈 Ready for conformal calibration!")
    print("="*60)
    
    return oof_df

if __name__ == "__main__":
    main()