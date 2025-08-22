#!/usr/bin/env python3
"""
DAILY RETRAINING PIPELINE
Automatically retrain model with fresh data each day
"""

import json
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from src.utils.artifacts import save_model_artifacts

class DailyRetrainer:
    """Automated daily model retraining system"""
    
    def __init__(self, model_dir="PRODUCTION/models/best_institutional_model"):
        self.model_dir = Path(model_dir)
        self.logger = self.setup_logging()
        
        # Training parameters
        self.training_config = {
            "lookback_days": 365,      # 1 year of training data
            "min_samples": 1000,       # Minimum samples required
            "validation_split": 0.2,   # 20% validation
            "patience": 10,            # Early stopping patience
            "max_epochs": 50,          # Maximum training epochs
            "learning_rate": 0.001,    # Learning rate
            "random_seed": 42          # Reproducibility
        }
        
    def setup_logging(self):
        """Setup logging"""
        log_dir = Path("PRODUCTION/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'daily_retrain.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('DailyRetrainer')
    
    def fetch_fresh_data(self):
        """Fetch fresh training data"""
        self.logger.info("📊 Fetching fresh training data")
        
        try:
            # Load base dataset
            data_path = Path("data/training_data_enhanced_FIXED.csv")
            if not data_path.exists():
                self.logger.error(f"Training dataset not found: {data_path}")
                return None
            
            df = pd.read_csv(data_path)
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Use recent data only (last year)
            cutoff_date = datetime.now() - timedelta(days=self.training_config["lookback_days"])
            recent_df = df[df['Date'] >= cutoff_date].copy()
            
            self.logger.info(f"✅ Loaded {len(recent_df)} recent samples")
            self.logger.info(f"Date range: {recent_df['Date'].min()} to {recent_df['Date'].max()}")
            
            if len(recent_df) < self.training_config["min_samples"]:
                self.logger.warning(f"⚠️ Only {len(recent_df)} samples, minimum {self.training_config['min_samples']} required")
                return df.tail(self.training_config["min_samples"])  # Use last N samples
            
            return recent_df
            
        except Exception as e:
            self.logger.error(f"❌ Failed to fetch training data: {e}")
            return None
    
    def prepare_training_data(self, df):
        """Prepare data for training"""
        self.logger.info("🔧 Preparing training data")
        
        try:
            # Load current features
            with open(self.model_dir / "feature_list.json", 'r') as f:
                features = json.load(f)
            
            # Target column
            target_col = "Return_1D"
            
            # Filter available features
            available_features = [f for f in features if f in df.columns]
            missing_features = set(features) - set(available_features)
            
            if missing_features:
                self.logger.warning(f"⚠️ Missing features: {missing_features}")
            
            # Clean data
            clean_df = df.dropna(subset=available_features + [target_col]).copy()
            
            if len(clean_df) < self.training_config["min_samples"]:
                self.logger.error(f"❌ Insufficient clean samples: {len(clean_df)}")
                return None, None, None
            
            # Prepare X and y
            X = clean_df[available_features]
            y = clean_df[target_col]
            dates = clean_df['Date']
            
            self.logger.info(f"✅ Training data prepared: {len(X)} samples, {len(available_features)} features")
            
            return X, y, dates
            
        except Exception as e:
            self.logger.error(f"❌ Data preparation failed: {e}")
            return None, None, None
    
    def train_model(self, X, y):
        """Train the model with fresh data"""
        self.logger.info("🧠 Training model with fresh data")
        
        try:
            # Set random seeds for reproducibility
            torch.manual_seed(self.training_config["random_seed"])
            np.random.seed(self.training_config["random_seed"])

            # Load current model configuration
            with open(self.model_dir / "config.json", 'r') as f:
                config = json.load(f)
            
            # Create preprocessing pipeline
            preprocessing = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ])
            
            # Fit preprocessing
            X_processed = preprocessing.fit_transform(X)
            
            # Create model
            import sys
            sys.path.append('.')
            from src.models.advanced_models import FinancialTransformer
            
            model_config = config['size_config']
            model = FinancialTransformer(
                input_size=len(X.columns),
                d_model=model_config.get('d_model', 64),
                n_heads=model_config.get('n_heads', 4),
                num_layers=model_config.get('num_layers', 3),
                d_ff=1024,
                dropout=model_config.get('dropout', 0.2)
            )
            
            # Training setup
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.training_config["learning_rate"])
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X_processed)
            y_tensor = torch.FloatTensor(y.values).unsqueeze(-1)
            
            if len(X_tensor.shape) == 2:
                X_tensor = X_tensor.unsqueeze(1)  # Add sequence dimension
            
            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=3)
            best_val_loss = float('inf')
            patience_counter = 0
            
            # Training loop
            model.train()
            for epoch in range(self.training_config["max_epochs"]):
                epoch_losses = []
                val_losses = []
                
                for train_idx, val_idx in tscv.split(X_tensor):
                    # Training step
                    optimizer.zero_grad()
                    
                    train_X = X_tensor[train_idx]
                    train_y = y_tensor[train_idx]
                    
                    output = model(train_X)
                    if isinstance(output, dict):
                        predictions = output['return_prediction']
                    else:
                        predictions = output
                    
                    loss = criterion(predictions, train_y)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_losses.append(loss.item())
                    
                    # Validation step
                    with torch.no_grad():
                        val_X = X_tensor[val_idx]
                        val_y = y_tensor[val_idx]
                        
                        val_output = model(val_X)
                        if isinstance(val_output, dict):
                            val_predictions = val_output['return_prediction']
                        else:
                            val_predictions = val_output
                        
                        val_loss = criterion(val_predictions, val_y)
                        val_losses.append(val_loss.item())
                
                avg_train_loss = np.mean(epoch_losses)
                avg_val_loss = np.mean(val_losses)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model state
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
                
                if patience_counter >= self.training_config["patience"]:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Load best model state
            model.load_state_dict(best_model_state)
            model.eval()
            
            self.logger.info(f"✅ Model training completed. Best validation loss: {best_val_loss:.6f}")
            
            return model, preprocessing
            
        except Exception as e:
            self.logger.error(f"❌ Model training failed: {e}")
            return None, None
    
    def recalibrate_gate(self, model, preprocessing, X, y):
        """Recalibrate conformal gate with fresh data"""
        self.logger.info("🚪 Recalibrating conformal gate")
        
        try:
            # Generate predictions on training data
            X_processed = preprocessing.transform(X)
            X_tensor = torch.FloatTensor(X_processed)
            
            if len(X_tensor.shape) == 2:
                X_tensor = X_tensor.unsqueeze(1)
            
            with torch.no_grad():
                model.eval()
                output = model(X_tensor)
                if isinstance(output, dict):
                    predictions = output['return_prediction'].cpu().numpy().flatten()
                else:
                    predictions = output.cpu().numpy().flatten()
            
            # Calculate scores (absolute prediction values)
            scores = np.abs(predictions)
            
            # Target acceptance rate (20%)
            target_accept_rate = 0.20
            threshold = np.percentile(scores, target_accept_rate * 100)
            
            actual_accept_rate = np.mean(scores <= threshold)
            
            gate_config = {
                "method": "score_absolute",
                "abs_score_threshold": float(threshold),
                "target_accept_rate": target_accept_rate,
                "actual_accept_rate": float(actual_accept_rate)
            }
            
            self.logger.info(f"✅ Gate recalibrated: threshold={threshold:.6f}, accept_rate={actual_accept_rate:.1%}")
            
            return gate_config
            
        except Exception as e:
            self.logger.error(f"❌ Gate recalibration failed: {e}")
            return None
    
    def save_retrained_model(self, model, scaler, gate_config, feature_list, dates):
        """Save the retrained model"""
        self.logger.info("💾 Saving retrained model")

        training_meta = {
            "data_range": {
                "start": str(dates.min()),
                "end": str(dates.max()),
            },
            "random_seed": self.training_config["random_seed"],
            "training_params": self.training_config,
        }

        try:
            backup_dir = save_model_artifacts(
                self.model_dir,
                model,
                scaler,
                feature_list,
                gate_config,
                training_meta,
            )
            self.logger.info(f"✅ Model saved successfully. Backup: {backup_dir}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to save model: {e}")
            return False
    
    def run_daily_retrain(self):
        """Execute complete daily retraining pipeline"""
        self.logger.info("🔄 STARTING DAILY RETRAINING PIPELINE")
        
        try:
            # 1. Fetch fresh data
            fresh_data = self.fetch_fresh_data()
            if fresh_data is None:
                return False
            
            # 2. Prepare training data
            X, y, dates = self.prepare_training_data(fresh_data)
            if X is None:
                return False
            
            # 3. Train model
            model, preprocessing = self.train_model(X, y)
            if model is None:
                return False
            
            # 4. Recalibrate gate
            gate_config = self.recalibrate_gate(model, preprocessing, X, y)
            if gate_config is None:
                return False
            
            # 5. Save retrained model
            if not self.save_retrained_model(model, preprocessing, gate_config, X.columns.tolist(), dates):
                return False
            
            self.logger.info("✅ DAILY RETRAINING COMPLETED SUCCESSFULLY")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Daily retraining failed: {e}")
            return False

def main():
    """Main retraining function"""
    print("🔄 DAILY MODEL RETRAINING SYSTEM")
    print("=" * 60)
    
    # Initialize retrainer
    retrainer = DailyRetrainer()
    
    # Run retraining
    success = retrainer.run_daily_retrain()
    
    if success:
        print("✅ Daily retraining completed successfully")
    else:
        print("❌ Daily retraining failed")

if __name__ == "__main__":
    main()