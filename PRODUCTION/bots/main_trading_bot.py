#!/usr/bin/env python3
"""
PRODUCTION TRADING BOT
Main production-ready trading bot with all safeguards
"""

import json
import time
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
import logging
from datetime import datetime
import warnings
from monitoring import (
    LATENCY,
    GROSS_EXPOSURE,
    DAILY_LOSS,
    SIGNAL_ACCEPT_RATE,
    start_monitoring,
)

warnings.filterwarnings('ignore')

class ProductionTradingBot:
    """Production-ready trading bot with institutional safeguards"""
    
    def __init__(self, model_path="PRODUCTION/models/best_institutional_model"):
        self.model_path = Path(model_path)
        self.logger = self.setup_logging()
        self.model = None
        self.preprocessing = None
        self.features = None
        self.gate_config = None
        self.risk_limits = {
            "max_position_size": 0.03,  # 3% max per position
            "baseline_gross_exposure": 0.33,  # 33% baseline gross
            "max_gross_exposure": 0.60,  # 60% emergency max gross
            "daily_loss_limit": 0.02,   # 2% daily loss limit
            "min_confidence": 0.15      # Minimum gate acceptance
        }
        
    def setup_logging(self):
        """Setup production logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('PRODUCTION/logs/trading_bot.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('ProductionBot')
    
    def load_model(self):
        """Load production model"""
        try:
            self.logger.info(f"Loading model from: {self.model_path}")
            
            # Load components
            with open(self.model_path / "config.json", 'r') as f:
                config = json.load(f)
            
            with open(self.model_path / "features.json", 'r') as f:
                self.features = json.load(f)
            
            with open(self.model_path / "gate.json", 'r') as f:
                self.gate_config = json.load(f)
            
            self.preprocessing = joblib.load(self.model_path / "preprocessing.pkl")
            
            # Create model
            import sys
            sys.path.append('.')
            from src.models.advanced_models import FinancialTransformer
            
            model_config = config['size_config']
            self.model = FinancialTransformer(
                input_size=len(self.features),
                d_model=model_config.get('d_model', 64),
                n_heads=model_config.get('n_heads', 4),
                num_layers=model_config.get('num_layers', 3),
                d_ff=1024,
                dropout=model_config.get('dropout', 0.2)
            )
            
            # Load weights
            state_dict = torch.load(self.model_path / "model.pt", map_location='cpu')
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            self.logger.info("✅ Model loaded successfully")
            self.logger.info(f"Features: {len(self.features)}")
            self.logger.info(f"Gate config: {self.gate_config}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Model loading failed: {e}")
            return False
    
    def generate_signals(self, market_data):
        """Generate trading signals with all safeguards"""
        start_time = time.time()
        try:
            if self.model is None:
                self.logger.error("Model not loaded")
                return None
            
            # Validate input data
            if not self.validate_market_data(market_data):
                return None
            
            # Prepare features
            available_features = [f for f in self.features if f in market_data.columns]
            if len(available_features) != len(self.features):
                self.logger.warning(f"Missing features: {len(self.features) - len(available_features)}")
            
            # Clean data
            clean_data = market_data.dropna(subset=available_features)
            if len(clean_data) == 0:
                self.logger.warning("No valid data after cleaning")
                return None
            
            # Generate predictions
            X = clean_data[available_features]
            X_processed = self.preprocessing.transform(X)
            
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_processed)
                if len(X_tensor.shape) == 2:
                    X_tensor = X_tensor.unsqueeze(1)
                
                model_output = self.model(X_tensor)
                predictions = model_output['return_prediction'].cpu().numpy().flatten()
            
            # Apply conformal gate
            if self.gate_config.get("method") == "score_absolute":
                threshold = self.gate_config["abs_score_threshold"]
                gate_mask = np.abs(predictions) <= threshold
            else:
                # Fallback gate
                gate_mask = np.ones_like(predictions, dtype=bool)
            
            # Create signals dataframe
            signals_df = clean_data.copy()
            signals_df["prediction"] = predictions
            signals_df["confidence"] = gate_mask.astype(float)
            signals_df["signal_strength"] = np.abs(predictions)
            
            # Filter by gate
            signals_df = signals_df[gate_mask]

            accept_rate = gate_mask.mean()
            SIGNAL_ACCEPT_RATE.set(accept_rate)
            self.logger.info(
                f"Generated {len(signals_df)} signals (accept rate: {accept_rate:.1%})"
            )

            LATENCY.labels(operation="generate_signals").observe(time.time() - start_time)
            return signals_df

        except Exception as e:
            self.logger.error(f"❌ Signal generation failed: {e}")
            LATENCY.labels(operation="generate_signals").observe(time.time() - start_time)
            return None
    
    def validate_market_data(self, data):
        """Validate incoming market data"""
        required_cols = ['Date', 'ticker'] if 'ticker' in data.columns else ['Date']
        
        for col in required_cols:
            if col not in data.columns:
                self.logger.error(f"Missing required column: {col}")
                return False
        
        if len(data) == 0:
            self.logger.error("Empty market data")
            return False
        
        return True
    
    def apply_risk_management(self, signals_df, current_positions=None):
        """Apply production risk management"""
        if signals_df is None or len(signals_df) == 0:
            return None
        
        # Sort by signal strength
        signals_df = signals_df.sort_values("signal_strength", ascending=False)
        
        # Apply position sizing
        signals_df["position_size"] = self.risk_limits["max_position_size"]
        
        # Limit number of positions to baseline exposure (33%)
        baseline_positions = int(self.risk_limits["baseline_gross_exposure"] / self.risk_limits["max_position_size"])
        max_positions = int(self.risk_limits["max_gross_exposure"] / self.risk_limits["max_position_size"])
        
        # Use baseline unless emergency conditions (high confidence signals)
        target_positions = baseline_positions
        
        # Take top signals (use baseline positions)
        final_signals = signals_df.head(target_positions).copy()

        gross_exposure = len(final_signals) * self.risk_limits['max_position_size']
        GROSS_EXPOSURE.set(gross_exposure)
        self.logger.info(
            f"Risk management: {len(final_signals)} positions, "
            f"gross exposure: {gross_exposure:.1%}"
        )

        return final_signals
    
    def run_live_trading(self):
        """Main live trading loop (demo version)"""
        start_time = time.time()
        self.logger.info("🚀 Starting production trading bot")
        
        if not self.load_model():
            self.logger.error("Failed to load model, exiting")
            return
        
        # This would connect to your broker API
        # For demo, we'll simulate with recent data
        self.logger.info("📊 Demo mode: Loading recent market data")
        
        try:
            # Load recent data for demo
            data_path = Path("data/training_data_enhanced_FIXED.csv")
            if data_path.exists():
                df = pd.read_csv(data_path)
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Get most recent data
                recent_data = df[df['Date'] >= df['Date'].max() - pd.Timedelta(days=30)]
                
                self.logger.info(f"Demo data: {len(recent_data)} recent samples")
                
                # Generate signals
                signals = self.generate_signals(recent_data)
                
                if signals is not None:
                    # Apply risk management
                    final_signals = self.apply_risk_management(signals)
                    
                    if final_signals is not None and len(final_signals) > 0:
                        self.logger.info(f"✅ Generated {len(final_signals)} trading signals")
                        
                        # Log top signals
                        for idx, row in final_signals.head(5).iterrows():
                            self.logger.info(f"  Signal: {row.get('ticker', 'N/A')} "
                                           f"pred={row['prediction']:.4f} "
                                           f"conf={row['confidence']:.2f}")
                    else:
                        self.logger.warning("No signals passed risk management")
                else:
                    self.logger.warning("No signals generated")
            
        except Exception as e:
            self.logger.error(f"❌ Trading loop error: {e}")
        finally:
            LATENCY.labels(operation="run_live_trading").observe(time.time() - start_time)

def main():
    """Main bot execution"""
    start_monitoring()
    bot = ProductionTradingBot()
    bot.run_live_trading()

if __name__ == "__main__":
    main()
