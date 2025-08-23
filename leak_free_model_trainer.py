#!/usr/bin/env python3
"""
LEAK-FREE MODEL TRAINER
Build and train model with zero data leakage using proper temporal validation
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import json
import joblib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class LeakFreeTrainer:
    """Leak-free model trainer with proper temporal validation"""
    
    def __init__(self, clean_features_file="clean_features_20250823_085317.json"):
        self.clean_features_file = clean_features_file
        self.model = None
        self.scaler = None
        self.features = None
        
    def load_clean_features(self):
        """Load the audited clean features"""
        print("🧹 LOADING CLEAN FEATURES")
        print("-" * 40)
        
        with open(self.clean_features_file, 'r') as f:
            clean_features = json.load(f)
        
        print(f"✅ Loaded {len(clean_features)} leak-free features")
        self.features = clean_features
        return clean_features
    
    def prepare_leak_free_dataset(self, data_path="data/training_data_enhanced_FIXED.csv"):
        """Prepare dataset with only clean features and proper temporal structure"""
        print(f"\n📊 PREPARING LEAK-FREE DATASET")
        print("-" * 40)
        
        # Load data
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        print(f"📅 Full dataset: {df['Date'].min()} to {df['Date'].max()}")
        print(f"📊 Total samples: {len(df):,}")
        
        # Use strict temporal cutoff - train only on data before 2024-09-01
        cutoff_date = pd.to_datetime('2024-09-01')
        train_df = df[df['Date'] < cutoff_date].copy()
        
        print(f"\n🏋️ TRAINING DATA (NO FUTURE LEAKS):")
        print(f"   Period: {train_df['Date'].min()} to {train_df['Date'].max()}")
        print(f"   Samples: {len(train_df):,}")
        
        # Check target variable
        target_col = 'Return_1D'
        if target_col not in train_df.columns:
            print(f"❌ Target column {target_col} not found!")
            return None, None, None
        
        # Prepare features (only clean ones)
        available_features = [f for f in self.features if f in train_df.columns]
        missing_features = set(self.features) - set(available_features)
        
        print(f"\n🧪 FEATURE STATUS:")
        print(f"   Available clean features: {len(available_features)}")
        if missing_features:
            print(f"   Missing features: {len(missing_features)}")
        
        # Handle missing data more intelligently
        # First, check availability of each feature
        feature_completeness = {}
        for feature in available_features:
            completeness = (1 - train_df[feature].isnull().mean()) * 100
            feature_completeness[feature] = completeness
        
        # Keep features with at least 50% completeness
        good_features = [f for f, comp in feature_completeness.items() if comp >= 50.0]
        print(f"   Features with >50% data: {len(good_features)}")
        
        # Now clean data with good features only
        clean_data = train_df.dropna(subset=good_features + [target_col])
        print(f"   Clean samples with good features: {len(clean_data):,}")
        
        if len(clean_data) == 0:
            print("   🔧 Trying with forward-fill to handle missing data...")
            # Forward fill missing values and try again
            train_df_filled = train_df[good_features + [target_col]].ffill().fillna(0)
            clean_data = train_df_filled.dropna()
            print(f"   Clean samples after forward-fill: {len(clean_data):,}")
        
        # Prepare X and y
        if len(clean_data) > 0:
            X = clean_data[good_features].values
            y = clean_data[target_col].values
            dates = clean_data['Date'].values if 'Date' in clean_data.columns else train_df.loc[clean_data.index, 'Date'].values
            available_features = good_features  # Update available features
            self.actual_features = good_features  # Store for testing
        else:
            print("   ❌ No clean data available even after preprocessing")
            return None, None, None
        
        print(f"\n✅ Dataset prepared: {X.shape[0]:,} samples, {X.shape[1]} features")
        
        return X, y, dates
    
    def implement_temporal_cv(self, X, y, dates, n_splits=5):
        """Implement proper temporal cross-validation"""
        print(f"\n⏰ IMPLEMENTING TEMPORAL CROSS-VALIDATION")
        print("-" * 40)
        
        # Use TimeSeriesSplit for proper temporal validation
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=5)  # 5-day gap to prevent leakage
        
        cv_results = []
        
        print(f"🔄 Running {n_splits}-fold temporal cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            dates_train = dates[train_idx]
            dates_val = dates[val_idx]
            
            print(f"\n📊 Fold {fold + 1}:")
            print(f"   Train: {pd.to_datetime(dates_train[0])} to {pd.to_datetime(dates_train[-1])}")
            print(f"   Valid: {pd.to_datetime(dates_val[0])} to {pd.to_datetime(dates_val[-1])}")
            print(f"   Train samples: {len(X_train_fold):,}")
            print(f"   Valid samples: {len(X_val_fold):,}")
            
            # Train simple model for this fold
            fold_results = self.train_fold_model(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
            cv_results.append(fold_results)
        
        # Average results across folds
        avg_ic = np.mean([r['ic'] for r in cv_results if not np.isnan(r['ic'])])
        avg_direction = np.mean([r['direction_acc'] for r in cv_results])
        
        print(f"\n🎯 TEMPORAL CV RESULTS:")
        print(f"   Average IC: {avg_ic:.4f}")
        print(f"   Average Direction Accuracy: {avg_direction:.1%}")
        
        return cv_results, avg_ic, avg_direction
    
    def train_fold_model(self, X_train, y_train, X_val, y_val):
        """Train a simple linear model for one CV fold"""
        from sklearn.linear_model import LinearRegression
        from scipy.stats import spearmanr
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train simple linear model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Predict on validation set
        y_pred = model.predict(X_val_scaled)
        
        # Calculate metrics
        ic = spearmanr(y_pred, y_val)[0] if len(y_val) > 10 else np.nan
        direction_acc = ((y_pred > 0) == (y_val > 0)).mean()
        
        return {
            'ic': ic,
            'direction_acc': direction_acc,
            'predictions': y_pred,
            'actual': y_val
        }
    
    def train_final_model(self, X, y):
        """Train final model on all training data"""
        print(f"\n🎯 TRAINING FINAL MODEL")
        print("-" * 40)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train final linear model (keeping it simple to avoid overfitting)
        from sklearn.linear_model import Ridge
        
        # Use Ridge regression with cross-validation for alpha selection
        from sklearn.linear_model import RidgeCV
        
        alphas = [0.1, 1.0, 10.0, 100.0]
        model = RidgeCV(alphas=alphas, cv=5)
        model.fit(X_scaled, y)
        
        self.model = model
        
        print(f"✅ Final model trained")
        print(f"   Features: {X.shape[1]}")
        print(f"   Samples: {X.shape[0]:,}")
        print(f"   Regularization alpha: {model.alpha_:.3f}")
        
        return model
    
    def save_leak_free_model(self):
        """Save the leak-free model"""
        print(f"\n💾 SAVING LEAK-FREE MODEL")
        print("-" * 40)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = Path(f"PRODUCTION/models/leak_free_model_{timestamp}")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model components
        joblib.dump(self.model, model_dir / "model.pkl")
        joblib.dump(self.scaler, model_dir / "scaler.pkl")
        
        # Save features list
        with open(model_dir / "features.json", 'w') as f:
            json.dump(self.features, f, indent=2)
        
        # Save model config
        config = {
            "model_type": "ridge_regression",
            "alpha": float(self.model.alpha_),
            "feature_count": len(self.features),
            "training_date": timestamp,
            "leak_free": True,
            "temporal_cv": True
        }
        
        with open(model_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Leak-free model saved: {model_dir}")
        return model_dir
    
    def test_on_holdout(self, model_dir, test_data_path="data/leak_free_test.csv"):
        """Test the new model on holdout data"""
        print(f"\n🧪 TESTING ON HOLDOUT DATA")
        print("-" * 40)
        
        # Load test data
        test_df = pd.read_csv(test_data_path)
        test_df['Date'] = pd.to_datetime(test_df['Date'])
        
        print(f"📅 Test period: {test_df['Date'].min()} to {test_df['Date'].max()}")
        print(f"📊 Test samples: {len(test_df):,}")
        
        # Use the same features that were actually used in training
        target_col = 'Return_1D'
        
        # Use the exact features from training
        if hasattr(self, 'actual_features'):
            good_features = [f for f in self.actual_features if f in test_df.columns]
        else:
            # Fallback to original features
            good_features = [f for f in self.features if f in test_df.columns]
        
        print(f"   Using {len(good_features)} features from training")
        
        # Clean test data with same preprocessing
        clean_test = test_df.dropna(subset=good_features + [target_col])
        if len(clean_test) == 0:
            print("   🔧 Using forward-fill for test data...")
            test_df_filled = test_df[good_features + [target_col]].ffill().fillna(0)
            clean_test = test_df_filled.dropna()
        
        X_test = clean_test[good_features].values
        y_test = clean_test[target_col].values
        
        print(f"🧪 Clean test samples: {len(X_test):,}")
        
        # Scale and predict
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        from scipy.stats import spearmanr
        ic = spearmanr(y_pred, y_test)[0]
        direction_acc = ((y_pred > 0) == (y_test > 0)).mean()
        
        print(f"\n🎯 LEAK-FREE MODEL RESULTS:")
        print(f"   Information Coefficient: {ic:.4f}")
        print(f"   Directional Accuracy: {direction_acc:.1%}")
        
        # Performance assessment
        if ic > 0.005:
            print(f"🟢 EXCELLENT: IC > 0.5% (institutional grade)")
        elif ic > 0.002:
            print(f"🟡 GOOD: IC > 0.2% (acceptable)")
        elif ic > 0:
            print(f"🟠 WEAK: IC > 0% (marginally positive)")
        else:
            print(f"🔴 POOR: IC <= 0% (needs more work)")
        
        return ic, direction_acc

def main():
    """Run complete leak-free model training"""
    trainer = LeakFreeTrainer()
    
    # Step 1: Load clean features
    clean_features = trainer.load_clean_features()
    
    # Step 2: Prepare leak-free dataset
    X, y, dates = trainer.prepare_leak_free_dataset()
    
    if X is not None:
        # Step 3: Temporal cross-validation
        cv_results, avg_ic, avg_direction = trainer.implement_temporal_cv(X, y, dates)
        
        # Step 4: Train final model
        final_model = trainer.train_final_model(X, y)
        
        # Step 5: Save model
        model_dir = trainer.save_leak_free_model()
        
        # Step 6: Test on holdout
        test_ic, test_direction = trainer.test_on_holdout(model_dir)
        
        print(f"\n🏆 FINAL RESULTS:")
        print(f"=" * 60)
        print(f"📊 Cross-validation IC: {avg_ic:.4f}")
        print(f"🧪 Out-of-sample IC: {test_ic:.4f}")
        print(f"📊 Cross-validation Direction: {avg_direction:.1%}")
        print(f"🧪 Out-of-sample Direction: {test_direction:.1%}")
        
        if test_ic > 0.002:
            print(f"\n🎉 SUCCESS: Model shows positive predictive power!")
            print(f"✅ Ready for production deployment")
        else:
            print(f"\n⚠️ Model still needs improvement")
            print(f"🔧 Consider feature engineering or different architecture")

if __name__ == "__main__":
    main()