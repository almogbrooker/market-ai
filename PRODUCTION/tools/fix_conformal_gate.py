#!/usr/bin/env python3
"""
Fix Conformal Gate Calibration
Recalibrate gates to achieve target 15-30% accept rate instead of 100%
"""

import json
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

def load_model_and_predict(model_dir, data_df):
    """Load model and generate predictions"""
    import sys
    sys.path.append('.')
    from src.models.advanced_models import FinancialTransformer
    
    # Load model components
    with open(model_dir / "config.json", 'r') as f:
        config = json.load(f)
    
    with open(model_dir / "features.json", 'r') as f:
        features = json.load(f)
    
    preprocessing = joblib.load(model_dir / "preprocessing.pkl")
    
    # Create model
    model_config = config['size_config']
    model = FinancialTransformer(
        input_size=len(features),
        d_model=model_config.get('d_model', 64),
        n_heads=model_config.get('n_heads', 4),
        num_layers=model_config.get('num_layers', 3),
        d_ff=1024,
        dropout=model_config.get('dropout', 0.2)
    )
    
    # Load weights
    state_dict = torch.load(model_dir / "model.pt", map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    # Prepare data
    available_features = [f for f in features if f in data_df.columns]
    target_col = "Return_1D" if "Return_1D" in data_df.columns else "returns_1d"
    
    eval_data = data_df.dropna(subset=available_features + [target_col]).copy()
    
    # Make predictions
    X = eval_data[available_features]
    X_processed = preprocessing.transform(X)
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_processed)
        if len(X_tensor.shape) == 2:
            X_tensor = X_tensor.unsqueeze(1)
        
        model_output = model(X_tensor)
        predictions = model_output['return_prediction'].cpu().numpy().flatten()
    
    eval_data["pred_raw"] = predictions
    return eval_data

def recalibrate_conformal_gate(eval_data, target_accept_rate=0.25):
    """Recalibrate conformal gate to achieve target accept rate"""
    
    predictions = eval_data["pred_raw"].values
    target_col = "Return_1D" if "Return_1D" in eval_data.columns else "returns_1d"
    actual_returns = eval_data[target_col].values
    
    print(f"🔧 Recalibrating conformal gate...")
    print(f"Target accept rate: {target_accept_rate:.1%}")
    print(f"Current predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    # Calculate residuals for conformal prediction
    residuals = actual_returns - predictions
    
    # Method 1: Quantile-based conformal intervals
    alpha = 1 - target_accept_rate
    lo_quantile = np.quantile(residuals, alpha/2)
    hi_quantile = np.quantile(residuals, 1 - alpha/2)
    
    method1_gate = {
        "method": "quantile_conformal",
        "lo": float(lo_quantile),
        "hi": float(hi_quantile),
        "alpha": alpha,
        "target_accept_rate": target_accept_rate
    }
    
    # Test method 1 accept rate
    method1_mask = (predictions >= lo_quantile) & (predictions <= hi_quantile)
    method1_accept = method1_mask.mean()
    
    print(f"Method 1 (Quantile Conformal):")
    print(f"  Bounds: [{lo_quantile:.4f}, {hi_quantile:.4f}]")
    print(f"  Accept rate: {method1_accept:.1%}")
    
    # Method 2: Score-based absolute threshold
    score_abs = np.abs(predictions)
    threshold_score = np.quantile(score_abs, target_accept_rate)
    
    method2_gate = {
        "method": "score_absolute",
        "abs_score_threshold": float(threshold_score),
        "target_accept_rate": target_accept_rate
    }
    
    # Test method 2 accept rate
    method2_mask = score_abs <= threshold_score
    method2_accept = method2_mask.mean()
    
    print(f"Method 2 (Score Absolute):")
    print(f"  Threshold: {threshold_score:.4f}")
    print(f"  Accept rate: {method2_accept:.1%}")
    
    # Method 3: Prediction confidence intervals
    pred_std = np.std(predictions)
    pred_mean = np.mean(predictions)
    
    # Find multiplier that gives target accept rate
    multipliers = np.linspace(0.1, 3.0, 100)
    best_multiplier = None
    best_accept_diff = float('inf')
    
    for mult in multipliers:
        lo_bound = pred_mean - mult * pred_std
        hi_bound = pred_mean + mult * pred_std
        
        mask = (predictions >= lo_bound) & (predictions <= hi_bound)
        accept_rate = mask.mean()
        
        diff = abs(accept_rate - target_accept_rate)
        if diff < best_accept_diff:
            best_accept_diff = diff
            best_multiplier = mult
    
    method3_lo = pred_mean - best_multiplier * pred_std
    method3_hi = pred_mean + best_multiplier * pred_std
    
    method3_gate = {
        "method": "prediction_confidence",
        "lo": float(method3_lo),
        "hi": float(method3_hi),
        "multiplier": float(best_multiplier),
        "target_accept_rate": target_accept_rate
    }
    
    method3_mask = (predictions >= method3_lo) & (predictions <= method3_hi)
    method3_accept = method3_mask.mean()
    
    print(f"Method 3 (Prediction Confidence):")
    print(f"  Bounds: [{method3_lo:.4f}, {method3_hi:.4f}]")
    print(f"  Multiplier: {best_multiplier:.2f}")
    print(f"  Accept rate: {method3_accept:.1%}")
    
    # Choose best method (closest to target)
    methods = [
        ("quantile_conformal", method1_gate, method1_accept),
        ("score_absolute", method2_gate, method2_accept),
        ("prediction_confidence", method3_gate, method3_accept)
    ]
    
    best_method = min(methods, key=lambda x: abs(x[2] - target_accept_rate))
    
    print(f"\n🎯 Best method: {best_method[0]}")
    print(f"   Accept rate: {best_method[2]:.1%} (target: {target_accept_rate:.1%})")
    
    return best_method[1], best_method[2]

def validate_gate_performance(eval_data, gate_config, target_accept_rate=0.25):
    """Validate gate performance on different time periods"""
    
    predictions = eval_data["pred_raw"].values
    target_col = "Return_1D" if "Return_1D" in eval_data.columns else "returns_1d"
    actual_returns = eval_data[target_col].values
    
    # Apply gate based on method
    if gate_config["method"] == "quantile_conformal":
        lo = gate_config["lo"]
        hi = gate_config["hi"]
        gate_mask = (predictions >= lo) & (predictions <= hi)
        
    elif gate_config["method"] == "score_absolute":
        threshold = gate_config["abs_score_threshold"]
        gate_mask = np.abs(predictions) <= threshold
        
    elif gate_config["method"] == "prediction_confidence":
        lo = gate_config["lo"]
        hi = gate_config["hi"]
        gate_mask = (predictions >= lo) & (predictions <= hi)
    
    # Overall performance
    overall_accept = gate_mask.mean()
    gated_predictions = predictions[gate_mask]
    gated_returns = actual_returns[gate_mask]
    
    if len(gated_predictions) > 10:
        gated_ic = spearmanr(gated_predictions, gated_returns).correlation
        if np.isnan(gated_ic):
            gated_ic = 0.0
    else:
        gated_ic = 0.0
    
    # Time-based validation
    date_col = "Date" if "Date" in eval_data.columns else eval_data.columns[0]
    
    if date_col in eval_data.columns:
        eval_data_clean = eval_data.copy()
        eval_data_clean["gate_mask"] = gate_mask
        
        # Split into periods
        dates = sorted(eval_data_clean[date_col].unique())
        mid_point = len(dates) // 2
        
        early_dates = dates[:mid_point]
        late_dates = dates[mid_point:]
        
        early_data = eval_data_clean[eval_data_clean[date_col].isin(early_dates)]
        late_data = eval_data_clean[eval_data_clean[date_col].isin(late_dates)]
        
        early_accept = early_data["gate_mask"].mean()
        late_accept = late_data["gate_mask"].mean()
        
        print(f"\n📊 Gate Performance Validation:")
        print(f"Overall accept rate: {overall_accept:.1%}")
        print(f"Gated IC: {gated_ic:.4f}")
        print(f"Early period accept: {early_accept:.1%}")
        print(f"Late period accept: {late_accept:.1%}")
        print(f"Accept rate stability: {abs(early_accept - late_accept):.1%}")
        
        # Validate stability
        stability_check = abs(early_accept - late_accept) < 0.15  # Within 15%
        target_check = abs(overall_accept - target_accept_rate) < 0.05  # Within 5%
        ic_check = gated_ic > 0  # Positive IC after gating
        
        validation_status = "PASS" if all([stability_check, target_check, ic_check]) else "FAIL"
        
        return {
            "status": validation_status,
            "overall_accept": overall_accept,
            "gated_ic": gated_ic,
            "early_accept": early_accept,
            "late_accept": late_accept,
            "stability": abs(early_accept - late_accept),
            "checks": {
                "stability": stability_check,
                "target": target_check,
                "ic": ic_check
            }
        }
    
    return {
        "status": "PARTIAL",
        "overall_accept": overall_accept,
        "gated_ic": gated_ic
    }

def main():
    """Main gate recalibration function"""
    print("🔧 CONFORMAL GATE RECALIBRATION")
    print("=" * 50)
    
    # Find institutional model
    models_root = Path("institutional_test_results/202508")
    model_dirs = []
    
    for d in models_root.iterdir():
        if d.is_dir() and (d / "model.pt").exists():
            score = 10 if "all_features" in d.name else 5
            model_dirs.append((score, d))
    
    if not model_dirs:
        raise FileNotFoundError("No institutional models found")
    
    model_dir = sorted(model_dirs, reverse=True)[0][1]
    print(f"📊 Using model: {model_dir.name}")
    
    # Load data
    data_path = Path("data/training_data_enhanced_FIXED.csv")
    data_df = pd.read_csv(data_path)
    print(f"📈 Using dataset: {data_path.name}")
    
    # Generate predictions
    print("🔮 Generating predictions...")
    eval_data = load_model_and_predict(model_dir, data_df)
    print(f"✅ Generated predictions for {len(eval_data)} samples")
    
    # Test different target accept rates
    target_rates = [0.15, 0.20, 0.25, 0.30]
    
    best_gate = None
    best_validation = None
    best_score = -1
    
    for target_rate in target_rates:
        print(f"\n🎯 Testing target accept rate: {target_rate:.1%}")
        print("-" * 40)
        
        # Recalibrate gate
        gate_config, actual_rate = recalibrate_conformal_gate(eval_data, target_rate)
        
        # Validate performance
        validation = validate_gate_performance(eval_data, gate_config, target_rate)
        
        # Score this configuration
        score = 0
        if validation["status"] == "PASS":
            score += 100
        
        # Bonus for good IC
        if validation.get("gated_ic", 0) > 0.01:
            score += 50
        
        # Penalty for being far from target
        if abs(actual_rate - target_rate) < 0.05:
            score += 25
        
        # Bonus for stability
        if validation.get("stability", 1) < 0.1:
            score += 25
        
        print(f"Configuration score: {score}")
        
        if score > best_score:
            best_score = score
            best_gate = gate_config
            best_validation = validation
            best_gate["actual_accept_rate"] = actual_rate
    
    # Save best gate configuration
    if best_gate:
        print(f"\n🏆 BEST GATE CONFIGURATION")
        print("=" * 40)
        print(f"Method: {best_gate['method']}")
        print(f"Target accept rate: {best_gate['target_accept_rate']:.1%}")
        print(f"Actual accept rate: {best_gate['actual_accept_rate']:.1%}")
        
        if best_gate["method"] == "quantile_conformal":
            print(f"Bounds: [{best_gate['lo']:.4f}, {best_gate['hi']:.4f}]")
        elif best_gate["method"] == "score_absolute":
            print(f"Threshold: {best_gate['abs_score_threshold']:.4f}")
        elif best_gate["method"] == "prediction_confidence":
            print(f"Bounds: [{best_gate['lo']:.4f}, {best_gate['hi']:.4f}]")
        
        print(f"Gated IC: {best_validation['gated_ic']:.4f}")
        print(f"Validation: {best_validation['status']}")
        
        # Save new gate to all institutional models
        for _, model_d in model_dirs:
            gate_file = model_d / "gate.json"
            
            # Backup original
            backup_file = model_d / "gate_original.json"
            if gate_file.exists() and not backup_file.exists():
                with open(gate_file, 'r') as f:
                    original_gate = json.load(f)
                with open(backup_file, 'w') as f:
                    json.dump(original_gate, f, indent=2)
                print(f"✅ Backed up original gate: {backup_file}")
            
            # Save new gate
            with open(gate_file, 'w') as f:
                json.dump(best_gate, f, indent=2)
            print(f"✅ Updated gate: {gate_file}")
        
        print(f"\n🎉 Conformal gate recalibration complete!")
        print(f"Accept rate reduced from 100% to {best_gate['actual_accept_rate']:.1%}")
        
        return best_gate, best_validation
    
    else:
        print("❌ Failed to find suitable gate configuration")
        return None, None

if __name__ == "__main__":
    main()