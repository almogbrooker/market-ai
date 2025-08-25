#!/usr/bin/env python3
"""
FIX MODEL ISSUES
================
Investigate and fix Lasso (0 features) and LightGBM (overfitting) issues
"""

import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import lightgbm as lgb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def investigate_data():
    """Investigate the processed data"""
    print("🔍 INVESTIGATING PROCESSED DATA")
    print("=" * 50)
    
    # Load processed data
    processed_dir = Path("../artifacts/processed")
    df = pd.read_parquet(processed_dir / "train_institutional.parquet")
    
    print(f"📊 Data shape: {df.shape}")
    print(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Check feature columns
    feature_cols = [col for col in df.columns if col.endswith('_t1')]
    target_col = 'target_forward'
    
    print(f"🎯 Features: {len(feature_cols)}")
    print(f"📈 Target: {target_col}")
    
    # Check data quality
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"\n📊 FEATURE ANALYSIS:")
    for col in feature_cols:
        values = X[col].dropna()
        print(f"   {col}:")
        print(f"     Range: [{values.min():.4f}, {values.max():.4f}]")
        print(f"     Std: {values.std():.4f}")
        print(f"     Missing: {X[col].isnull().sum()}")
    
    print(f"\n🎯 TARGET ANALYSIS:")
    print(f"   Range: [{y.min():.4f}, {y.max():.4f}]")
    print(f"   Mean: {y.mean():.6f}")
    print(f"   Std: {y.std():.4f}")
    print(f"   Missing: {y.isnull().sum()}")
    
    return df, feature_cols, target_col

def fix_lasso_model(X_train, X_val, y_train, y_val):
    """Fix Lasso with proper alpha range"""
    print("\n🔧 FIXING LASSO MODEL")
    print("-" * 30)
    
    # Scale features first
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Try much smaller alphas for Lasso
    alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    
    best_alpha = None
    best_ic = -np.inf
    best_n_features = 0
    
    print("   Testing alpha values:")
    
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=5000, random_state=42)
        lasso.fit(X_train_scaled, y_train)
        
        # Check selected features
        selected_features = np.sum(np.abs(lasso.coef_) > 1e-8)
        
        if selected_features > 0:
            val_pred = lasso.predict(X_val_scaled)
            ic, _ = spearmanr(y_val, val_pred)
            
            if not np.isnan(ic):
                print(f"     Alpha {alpha:.6f}: {selected_features} features, IC={ic:.4f}")
                
                if abs(ic) > abs(best_ic):
                    best_ic = ic
                    best_alpha = alpha
                    best_n_features = selected_features
            else:
                print(f"     Alpha {alpha:.6f}: {selected_features} features, IC=NaN")
        else:
            print(f"     Alpha {alpha:.6f}: 0 features selected")
    
    if best_alpha is not None:
        print(f"   ✅ Best Lasso: alpha={best_alpha:.6f}, features={best_n_features}, IC={best_ic:.4f}")
        
        # Train final model
        final_lasso = Lasso(alpha=best_alpha, max_iter=5000, random_state=42)
        final_lasso.fit(X_train_scaled, y_train)
        
        return {
            'model': final_lasso,
            'scaler': scaler,
            'alpha': best_alpha,
            'ic': best_ic,
            'n_features': best_n_features,
            'status': 'SUCCESS'
        }
    else:
        print(f"   ❌ Lasso failed - all alphas result in 0 features or NaN")
        return {'status': 'FAILED'}

def fix_lightgbm_model(X_train, X_val, y_train, y_val):
    """Fix LightGBM with more conservative parameters"""
    print("\n🔧 FIXING LIGHTGBM MODEL")
    print("-" * 30)
    
    # Much more conservative LightGBM parameters
    conservative_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 5,           # Very conservative
        'learning_rate': 0.01,     # Very slow
        'feature_fraction': 0.6,   # Use fewer features
        'bagging_fraction': 0.6,   # Use fewer samples
        'bagging_freq': 3,
        'min_data_in_leaf': 200,   # Much higher minimum
        'lambda_l1': 10.0,         # Strong L1 regularization
        'lambda_l2': 10.0,         # Strong L2 regularization
        'max_depth': 3,            # Limit depth
        'verbose': -1,
        'random_state': 42,
        'force_col_wise': True
    }
    
    print("   Using very conservative parameters:")
    for key, value in conservative_params.items():
        if key not in ['verbose', 'random_state', 'force_col_wise']:
            print(f"     {key}: {value}")
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train with very limited rounds
    model = lgb.train(
        conservative_params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=50,        # Very limited boosting
        callbacks=[
            lgb.early_stopping(10), 
            lgb.log_evaluation(0)
        ]
    )
    
    # Evaluate
    val_pred = model.predict(X_val)
    ic, _ = spearmanr(y_val, val_pred)
    mse = mean_squared_error(y_val, val_pred)
    
    print(f"   📊 Conservative LightGBM results:")
    print(f"     Validation IC: {ic:.4f}")
    print(f"     Validation MSE: {mse:.6f}")
    print(f"     Best iteration: {model.best_iteration}")
    
    # Check if within institutional bounds
    if abs(ic) <= 0.08:  # Within institutional limit
        print(f"   ✅ IC within institutional bounds")
        return {
            'model': model,
            'ic': ic,
            'mse': mse,
            'best_iteration': model.best_iteration,
            'status': 'SUCCESS'
        }
    else:
        print(f"   ⚠️ IC still above institutional limit: {abs(ic):.4f} > 0.08")
        return {
            'model': model,
            'ic': ic,
            'mse': mse,
            'best_iteration': model.best_iteration,
            'status': 'HIGH_IC'
        }

def retrain_models():
    """Retrain models with fixes"""
    print("🔧 RETRAINING MODELS WITH FIXES")
    print("=" * 50)
    
    # Load and investigate data
    df, feature_cols, target_col = investigate_data()
    
    # Prepare data
    X = df[feature_cols].fillna(0)  # Simple fillna for missing values
    y = df[target_col]
    
    # Split (same as original)
    val_split_idx = int(len(df) * 0.7)
    X_train = X.iloc[:val_split_idx]
    X_val = X.iloc[val_split_idx:]
    y_train = y.iloc[:val_split_idx]
    y_val = y.iloc[val_split_idx:]
    
    print(f"\n📊 Training split: {len(X_train)} samples")
    print(f"📊 Validation split: {len(X_val)} samples")
    
    # Fix Lasso
    lasso_results = fix_lasso_model(X_train, X_val, y_train, y_val)
    
    # Fix LightGBM
    lgb_results = fix_lightgbm_model(X_train, X_val, y_train, y_val)
    
    # Summary
    print(f"\n📊 FIXED MODELS SUMMARY")
    print("=" * 40)
    
    if lasso_results['status'] == 'SUCCESS':
        print(f"✅ Lasso: IC={lasso_results['ic']:.4f}, Features={lasso_results['n_features']}")
        
        # Check institutional compliance
        if 0.005 <= abs(lasso_results['ic']) <= 0.08:
            print(f"   🏛️ Lasso: INSTITUTIONAL APPROVED")
        else:
            print(f"   ⚠️ Lasso: IC outside institutional range")
    else:
        print(f"❌ Lasso: Failed to fix")
    
    if lgb_results['status'] in ['SUCCESS', 'HIGH_IC']:
        print(f"✅ LightGBM: IC={lgb_results['ic']:.4f}")
        
        if 0.005 <= abs(lgb_results['ic']) <= 0.08:
            print(f"   🏛️ LightGBM: INSTITUTIONAL APPROVED")
        elif abs(lgb_results['ic']) > 0.08:
            print(f"   ⚠️ LightGBM: IC still too high ({abs(lgb_results['ic']):.4f} > 0.08)")
        else:
            print(f"   ⚠️ LightGBM: IC too low ({abs(lgb_results['ic']):.4f} < 0.005)")
    
    # Compare with current Ridge
    print(f"\n🏆 MODEL COMPARISON:")
    print(f"   Ridge (current): IC=0.0713 ✅ APPROVED")
    
    if lasso_results['status'] == 'SUCCESS':
        lasso_status = "✅ APPROVED" if 0.005 <= abs(lasso_results['ic']) <= 0.08 else "❌ NOT APPROVED"
        print(f"   Lasso (fixed): IC={lasso_results['ic']:.4f} {lasso_status}")
    else:
        print(f"   Lasso: FAILED")
    
    if lgb_results['status'] in ['SUCCESS', 'HIGH_IC']:
        lgb_status = "✅ APPROVED" if 0.005 <= abs(lgb_results['ic']) <= 0.08 else "❌ NOT APPROVED"
        print(f"   LightGBM (fixed): IC={lgb_results['ic']:.4f} {lgb_status}")
    
    return {
        'lasso': lasso_results,
        'lightgbm': lgb_results
    }

def main():
    """Main execution"""
    results = retrain_models()
    
    print(f"\n🎯 CONCLUSION:")
    
    working_models = 1  # Ridge is working
    
    if results['lasso']['status'] == 'SUCCESS':
        if 0.005 <= abs(results['lasso']['ic']) <= 0.08:
            working_models += 1
            print(f"✅ Lasso model fixed and approved")
    
    if results['lightgbm']['status'] in ['SUCCESS', 'HIGH_IC']:
        if 0.005 <= abs(results['lightgbm']['ic']) <= 0.08:
            working_models += 1
            print(f"✅ LightGBM model fixed and approved")
        elif abs(results['lightgbm']['ic']) > 0.08:
            print(f"⚠️ LightGBM still overfitting despite conservative parameters")
    
    print(f"\n📊 Total institutional-approved models: {working_models}/3")
    
    if working_models >= 2:
        print(f"✅ MODEL ISSUES RESOLVED")
    else:
        print(f"⚠️ Some model issues remain")
    
    return results

if __name__ == "__main__":
    results = main()