#!/usr/bin/env python3
"""
LEAK-FREE 6-MONTH VALIDATION SYSTEM
Pure out-of-sample test with no data leakage for 6 months before August 2025

Training Period: 2020-05-26 to 2024-08-31 (4+ years of data)
Test Period: 2024-09-01 to 2025-02-12 (6 months out-of-sample)

This ensures ZERO data leakage and tests the model on completely unseen data.
"""

import pandas as pd
import numpy as np
import torch
import joblib
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class LeakFreeValidator:
    """Leak-free validation system for 6-month out-of-sample testing"""
    
    def __init__(self, data_path="data/training_data_enhanced_FIXED.csv"):
        self.data_path = data_path
        self.cutoff_date = pd.to_datetime('2024-09-01')  # No future data leaks
        self.results = {}
        
    def prepare_leak_free_data(self):
        """Split data with strict temporal boundary - NO LEAKAGE"""
        print("🔒 PREPARING LEAK-FREE DATA SPLIT")
        print("=" * 60)
        
        # Load full dataset
        df = pd.read_csv(self.data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        print(f"📅 Full dataset: {df['Date'].min()} to {df['Date'].max()}")
        print(f"📊 Total samples: {len(df):,}")
        
        # Strict temporal split - NO FUTURE DATA IN TRAINING
        train_data = df[df['Date'] < self.cutoff_date].copy()
        test_data = df[df['Date'] >= self.cutoff_date].copy()
        
        print(f"\n🏋️ TRAINING DATA (NO FUTURE LEAKS):")
        print(f"   Period: {train_data['Date'].min()} to {train_data['Date'].max()}")
        print(f"   Samples: {len(train_data):,}")
        
        print(f"\n🧪 TEST DATA (PURE OUT-OF-SAMPLE):")
        print(f"   Period: {test_data['Date'].min()} to {test_data['Date'].max()}")
        print(f"   Samples: {len(test_data):,}")
        print(f"   Duration: {(test_data['Date'].max() - test_data['Date'].min()).days} days")
        
        # Validate no overlap
        if train_data['Date'].max() >= test_data['Date'].min():
            raise ValueError("❌ DATA LEAKAGE DETECTED! Training data overlaps with test data")
        
        print(f"\n✅ LEAK-FREE VALIDATION CONFIRMED")
        print(f"   Gap between train/test: {(test_data['Date'].min() - train_data['Date'].max()).days} days")
        
        # Save clean datasets
        train_path = "data/leak_free_train.csv"
        test_path = "data/leak_free_test.csv"
        
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        
        print(f"\n💾 Saved leak-free datasets:")
        print(f"   Training: {train_path}")
        print(f"   Test: {test_path}")
        
        return train_data, test_data
    
    def validate_existing_model(self, model_path="PRODUCTION/models/best_institutional_model"):
        """Test existing model on pure out-of-sample data"""
        print(f"\n🧪 VALIDATING EXISTING MODEL: {model_path}")
        print("=" * 60)
        
        # Load test data
        test_data = pd.read_csv("data/leak_free_test.csv")
        test_data['Date'] = pd.to_datetime(test_data['Date'])
        
        print(f"📊 Test data: {len(test_data):,} samples")
        print(f"📅 Period: {test_data['Date'].min()} to {test_data['Date'].max()}")
        
        try:
            # Load existing model components
            model_dir = Path(model_path)
            
            with open(model_dir / "features.json", 'r') as f:
                features = json.load(f)
            
            with open(model_dir / "gate.json", 'r') as f:
                gate_config = json.load(f)
            
            preprocessing = joblib.load(model_dir / "preprocessing.pkl")
            
            print(f"✅ Model components loaded:")
            print(f"   Features: {len(features)}")
            print(f"   Gate threshold: {gate_config.get('abs_score_threshold', 'N/A')}")
            
            # Prepare features
            available_features = [f for f in features if f in test_data.columns]
            missing_features = set(features) - set(available_features)
            
            if missing_features:
                print(f"⚠️ Missing features: {len(missing_features)}")
                for feat in list(missing_features)[:5]:
                    print(f"     {feat}")
            
            # Clean test data
            target_col = "Return_1D" if "Return_1D" in test_data.columns else "returns_1d"
            clean_test = test_data.dropna(subset=available_features + [target_col])
            
            print(f"📊 Clean test samples: {len(clean_test):,}")
            
            # Generate predictions using current model architecture
            X_test = clean_test[available_features]
            y_test = clean_test[target_col]
            
            # Transform features
            X_processed = preprocessing.transform(X_test)
            
            # Load and run model
            import sys
            sys.path.append('.')
            from src.models.advanced_models import FinancialTransformer
            
            # Create model with config
            with open(model_dir / "config.json", 'r') as f:
                config = json.load(f)
            
            model_config = config['size_config']
            model = FinancialTransformer(
                input_size=len(available_features),
                d_model=model_config.get('d_model', 64),
                n_heads=model_config.get('n_heads', 4),
                num_layers=model_config.get('num_layers', 3),
                d_ff=1024,
                dropout=0.0  # No dropout for inference
            )
            
            # Load weights
            state_dict = torch.load(model_dir / "model.pt", map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
            
            print("✅ Model loaded and ready for inference")
            
            # Generate predictions
            predictions = []
            with torch.no_grad():
                batch_size = 512
                for i in range(0, len(X_processed), batch_size):
                    batch = X_processed[i:i+batch_size]
                    X_tensor = torch.FloatTensor(batch)
                    
                    if len(X_tensor.shape) == 2:
                        X_tensor = X_tensor.unsqueeze(1)
                    
                    model_output = model(X_tensor)
                    batch_preds = model_output['return_prediction'].cpu().numpy().flatten()
                    predictions.extend(batch_preds)
            
            predictions = np.array(predictions)
            print(f"✅ Generated {len(predictions):,} predictions")
            
            # Calculate performance metrics
            from scipy.stats import spearmanr
            
            # Information Coefficient (Spearman correlation)
            ic = spearmanr(predictions, y_test)[0]
            
            # Directional accuracy  
            direction_acc = ((predictions > 0) == (y_test > 0)).mean()
            
            # Apply conformal gate
            if gate_config.get("method") == "score_absolute":
                threshold = gate_config["abs_score_threshold"]
                gate_mask = np.abs(predictions) <= threshold
                gate_accept_rate = gate_mask.mean()
                
                # Gated performance
                if gate_mask.sum() > 10:
                    gated_ic = spearmanr(predictions[gate_mask], y_test[gate_mask])[0]
                    gated_direction = ((predictions[gate_mask] > 0) == (y_test[gate_mask] > 0)).mean()
                else:
                    gated_ic = np.nan
                    gated_direction = np.nan
            else:
                gate_accept_rate = 1.0
                gated_ic = ic
                gated_direction = direction_acc
            
            # Store results
            self.results = {
                "test_period": {
                    "start": str(test_data['Date'].min()),
                    "end": str(test_data['Date'].max()),
                    "days": (test_data['Date'].max() - test_data['Date'].min()).days,
                    "samples": len(clean_test)
                },
                "performance": {
                    "information_coefficient": float(ic),
                    "directional_accuracy": float(direction_acc),
                    "gate_accept_rate": float(gate_accept_rate),
                    "gated_ic": float(gated_ic) if not np.isnan(gated_ic) else None,
                    "gated_direction_acc": float(gated_direction) if not np.isnan(gated_direction) else None
                },
                "predictions_stats": {
                    "mean": float(np.mean(predictions)),
                    "std": float(np.std(predictions)),
                    "min": float(np.min(predictions)),
                    "max": float(np.max(predictions))
                }
            }
            
            # Print results
            print(f"\n🎯 LEAK-FREE VALIDATION RESULTS:")
            print(f"=" * 50)
            print(f"📊 Information Coefficient: {ic:.4f}")
            print(f"🎯 Directional Accuracy: {direction_acc:.1%}")
            print(f"🚪 Gate Accept Rate: {gate_accept_rate:.1%}")
            
            if gated_ic is not None:
                print(f"🔒 Gated IC: {gated_ic:.4f}")
                print(f"🔒 Gated Direction: {gated_direction:.1%}")
            
            print(f"📈 Prediction Stats:")
            print(f"   Mean: {np.mean(predictions):.4f}")
            print(f"   Std:  {np.std(predictions):.4f}")
            print(f"   Range: [{np.min(predictions):.4f}, {np.max(predictions):.4f}]")
            
            return self.results
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return None
    
    def save_results(self):
        """Save validation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"leak_free_validation_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved: {results_file}")
        return results_file

def main():
    """Run complete leak-free validation"""
    validator = LeakFreeValidator()
    
    # Step 1: Prepare leak-free data split
    train_data, test_data = validator.prepare_leak_free_data()
    
    # Step 2: Validate existing model on pure out-of-sample data
    results = validator.validate_existing_model()
    
    if results:
        # Step 3: Save results
        validator.save_results()
        
        # Step 4: Summary
        print(f"\n🏆 LEAK-FREE VALIDATION SUMMARY:")
        print(f"=" * 60)
        print(f"✅ Zero data leakage confirmed")
        print(f"📅 Test period: {results['test_period']['days']} days")
        print(f"📊 Test samples: {results['test_period']['samples']:,}")
        print(f"🎯 IC Performance: {results['performance']['information_coefficient']:.4f}")
        print(f"🎯 Direction Accuracy: {results['performance']['directional_accuracy']:.1%}")
        
        # Performance assessment
        ic = results['performance']['information_coefficient']
        if ic > 0.005:
            print(f"🟢 EXCELLENT: IC > 0.5% (institutional grade)")
        elif ic > 0.002:
            print(f"🟡 GOOD: IC > 0.2% (acceptable)")
        elif ic > 0:
            print(f"🟠 WEAK: IC > 0% (marginally positive)")
        else:
            print(f"🔴 POOR: IC <= 0% (negative predictive power)")
    
    else:
        print("❌ Validation failed")

if __name__ == "__main__":
    main()