#!/usr/bin/env python3
"""
IC Reality Check Tool - Validates real Information Coefficient performance
Compatible with PyTorch models from institutional testing
"""

import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

def safe_spearman(x, y):
    """Safe Spearman correlation that handles edge cases"""
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size < 2 or y.size < 2:
        return np.nan
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan
    try:
        return float(spearmanr(x, y).correlation)
    except:
        return np.nan

def apply_conformal_gate(predictions, gate_config):
    """Apply conformal prediction gate to filter low-confidence predictions"""
    if not gate_config:
        return np.ones_like(predictions, dtype=bool)
    
    lo = gate_config.get('lo', -float('inf'))
    hi = gate_config.get('hi', float('inf'))
    
    # Keep predictions within confidence interval
    return (predictions >= lo) & (predictions <= hi)

def load_pytorch_model(model_dir):
    """Load PyTorch model from institutional testing results"""
    model_file = model_dir / "model.pt"
    features_file = model_dir / "features.json"
    preprocessing_file = model_dir / "preprocessing.pkl"
    gate_file = model_dir / "gate.json"
    config_file = model_dir / "config.json"
    
    if not all(f.exists() for f in [model_file, features_file, preprocessing_file, config_file]):
        raise FileNotFoundError(f"Missing required files in {model_dir}")
    
    # Load configuration to recreate model
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Load features
    with open(features_file, 'r') as f:
        features = json.load(f)
    
    # Import model creation function
    import sys
    sys.path.append('.')
    from src.models.advanced_models import create_advanced_model
    
    # Recreate model architecture
    model_config = config['size_config'].copy()
    model_type = config['model_type']
    
    # Create model with explicit parameters
    if model_type == "financial_transformer":
        from src.models.advanced_models import FinancialTransformer
        model = FinancialTransformer(
            input_size=len(features),
            d_model=model_config.get('d_model', 64),
            n_heads=model_config.get('n_heads', 4),
            num_layers=model_config.get('num_layers', 3),
            d_ff=1024,  # Fixed from saved model architecture
            dropout=model_config.get('dropout', 0.2),
            drop_path=model_config.get('drop_path', 0.0)
        )
    else:
        model_config['input_size'] = len(features)
        model = create_advanced_model(model_type, model_config)
    
    # Load saved state dict
    state_dict = torch.load(model_file, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load preprocessing pipeline
    preprocessing = joblib.load(preprocessing_file)
    
    # Load conformal gate
    gate_config = {}
    if gate_file.exists():
        with open(gate_file, 'r') as f:
            gate_config = json.load(f)
    
    return model, features, preprocessing, gate_config

def pick_latest_institutional_model(root_path="PRODUCTION/models"):
    """Pick the latest institutional testing model"""
    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(f"No institutional results found at {root_path}")
    
    # Find best model in PRODUCTION/models directory
    model_dirs = []
    for d in root.iterdir():
        if d.is_dir() and (d / "model.pt").exists():
            # Prefer best_institutional_model
            score = 0
            if "best_institutional" in d.name:
                score += 10
            elif "drift_corrected" in d.name:
                score += 5
            model_dirs.append((score, d))
    
    if not model_dirs:
        raise FileNotFoundError("No valid model directories found")
    
    best_model_dir = sorted(model_dirs, reverse=True)[0][1]
    return best_model_dir

def pick_dataset(repo_root: Path, prefer: str = None):
    """Pick the best available dataset for evaluation"""
    if prefer:
        p = Path(prefer)
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {prefer}")
        return p
    
    # Search for suitable datasets
    candidates = []
    for p in repo_root.rglob("*.csv"):
        score = 0
        name = p.name.lower()
        
        # Prefer certain naming patterns
        if any(k in name for k in ["training", "enhanced", "complete", "validation"]):
            score += 5
        if "data" in str(p.parent).lower():
            score += 3
        if p.stat().st_size > 1_000_000:  # Prefer larger files
            score += 2
        
        try:
            # Quick check for required columns
            df = pd.read_csv(p, nrows=100)
            cols = set(df.columns.str.lower())
            if "date" in cols and "ticker" in cols:
                score += 6
            if any(c in cols for c in ["returns_1d", "ret_1d", "target", "future_return"]):
                score += 6
            if len(df.columns) > 20:  # Prefer datasets with more features
                score += 3
        except:
            continue
        
        candidates.append((score, p))
    
    if not candidates:
        raise FileNotFoundError("No suitable CSV dataset found")
    
    return sorted(candidates, reverse=True)[0][1]

def evaluate_model_ic(model, features, preprocessing, data_df, target_col):
    """Evaluate model IC on given dataset"""
    # Prepare data
    valid_data = data_df.dropna(subset=features + [target_col]).copy()
    
    if len(valid_data) == 0:
        raise ValueError("No valid data after dropping NaN values")
    
    # Apply preprocessing
    X = valid_data[features]
    X_processed = preprocessing.transform(X)
    
    # Make predictions
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_processed)
        # Add sequence dimension if needed (batch_size, 1, features)
        if len(X_tensor.shape) == 2:
            X_tensor = X_tensor.unsqueeze(1)
        
        model_output = model(X_tensor)
        
        # Extract predictions based on model output type
        if isinstance(model_output, dict):
            # FinancialTransformer returns dict with 'return_prediction'
            predictions = model_output['return_prediction'].cpu().numpy().flatten()
        else:
            # Other models return tensor directly
            predictions = model_output.cpu().numpy().flatten()
    
    valid_data["pred_raw"] = predictions
    
    return valid_data

def main(models_root="institutional_test_results", dataset_csv=None, out_json="reports/ic_audit.json"):
    """Main IC reality check function"""
    repo = Path(".")
    
    try:
        # Load institutional model
        model_dir = pick_latest_institutional_model(models_root)
        print(f"📊 Loading model from: {model_dir}")
        
        model, features, preprocessing, gate_config = load_pytorch_model(model_dir)
        
        # Load dataset
        data_path = pick_dataset(repo, dataset_csv)
        print(f"📈 Loading dataset: {data_path}")
        
        df = pd.read_csv(data_path, low_memory=False)
        
        # Normalize column names
        col_mapping = {c.lower(): c for c in df.columns}
        
        def get_col(name):
            return col_mapping.get(name.lower(), name)
        
        # Validate required columns
        date_col = get_col("date")
        ticker_col = get_col("ticker")
        
        if date_col not in df.columns or ticker_col not in df.columns:
            raise RuntimeError(f"Dataset missing date/ticker columns: {data_path}")
        
        # Convert date column
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date
        
        # Find target column
        target_col = None
        for target_name in ["returns_1d", "ret_1d", "target", "future_return", "y"]:
            col_name = get_col(target_name)
            if col_name in df.columns:
                target_col = col_name
                break
        
        if target_col is None:
            # Last resort: find any column with 'return' in name
            return_cols = [c for c in df.columns if 'return' in c.lower()]
            if not return_cols:
                raise RuntimeError("No target/returns column found")
            target_col = return_cols[0]
        
        print(f"🎯 Using target column: {target_col}")
        print(f"🔧 Using {len(features)} features")
        
        # Filter features to those available in dataset
        available_features = [f for f in features if f in df.columns]
        if len(available_features) != len(features):
            print(f"⚠️ Warning: Only {len(available_features)}/{len(features)} features available")
        
        # Evaluate model
        eval_data = evaluate_model_ic(model, available_features, preprocessing, df, target_col)
        
        # Apply conformal gate
        if gate_config:
            gate_mask = apply_conformal_gate(eval_data["pred_raw"], gate_config)
            gate_accept_rate = float(gate_mask.mean())
            print(f"🚪 Conformal gate accept rate: {gate_accept_rate:.1%}")
        else:
            gate_mask = np.ones(len(eval_data), dtype=bool)
            gate_accept_rate = 1.0
        
        # Calculate daily ICs
        daily_ics = []
        per_day_results = []
        
        for date, group in eval_data.groupby(date_col):
            if len(group) < 2:
                continue
            
            # Apply gate filter
            group_filtered = group[gate_mask[group.index]] if gate_config else group
            
            if len(group_filtered) < 2:
                continue
            
            ic = safe_spearman(group_filtered["pred_raw"], group_filtered[target_col])
            daily_ics.append(ic)
            per_day_results.append({
                "date": str(date),
                "ic": ic,
                "n_total": len(group),
                "n_filtered": len(group_filtered)
            })
        
        # Calculate statistics
        ic_mean = float(np.nanmean(daily_ics)) if daily_ics else float('nan')
        ic_std = float(np.nanstd(daily_ics)) if daily_ics else float('nan')
        ic_days = int(np.sum(~np.isnan(daily_ics)))
        
        # Monthly aggregation
        eval_data["month"] = pd.to_datetime(eval_data[date_col]).dt.to_period("M").astype(str)
        monthly_ics = []
        
        for month, group in eval_data.groupby("month"):
            if len(group) < 10:  # Require minimum observations
                continue
            
            # Apply gate filter
            group_filtered = group[gate_mask[group.index]] if gate_config else group
            
            if len(group_filtered) < 10:
                continue
            
            monthly_ic = safe_spearman(group_filtered["pred_raw"], group_filtered[target_col])
            monthly_ics.append({"month": month, "ic": monthly_ic, "n": len(group_filtered)})
        
        # Prepare output
        results = {
            "model_dir": str(model_dir),
            "dataset": str(data_path),
            "rows_total": len(df),
            "rows_used": len(eval_data),
            "n_features": len(available_features),
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ic_days": ic_days,
            "ic_sharpe": float(ic_mean / ic_std) if ic_std > 0 else float('nan'),
            "gate_config": gate_config,
            "gate_accept_rate": gate_accept_rate,
            "per_day": per_day_results[:100],  # Limit for JSON size
            "per_month": monthly_ics,
            "evaluation_summary": {
                "model_type": "institutional_pytorch",
                "features_available": f"{len(available_features)}/{len(features)}",
                "date_range": f"{eval_data[date_col].min()} to {eval_data[date_col].max()}",
                "unique_tickers": len(eval_data[ticker_col].unique())
            }
        }
        
        # Save results
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print(f"\n🎯 IC REALITY CHECK RESULTS")
        print(f"=" * 50)
        print(f"Model: {model_dir.name}")
        print(f"Dataset: {data_path.name}")
        print(f"Rows used: {len(eval_data):,}")
        print(f"Features: {len(available_features)}")
        print(f"IC Mean: {ic_mean:.4f}")
        print(f"IC Std: {ic_std:.4f}")
        print(f"IC Sharpe: {ic_mean/ic_std:.2f}" if ic_std > 0 else "IC Sharpe: N/A")
        print(f"IC Days: {ic_days}")
        print(f"Gate Accept Rate: {gate_accept_rate:.1%}")
        
        if monthly_ics:
            print(f"\n📅 Monthly IC Performance:")
            for month_data in monthly_ics[-6:]:  # Last 6 months
                print(f"  {month_data['month']}: {month_data['ic']:.4f} (n={month_data['n']})")
        
        print(f"\n✅ Full results saved to: {out_json}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="IC Reality Check for Institutional Models")
    parser.add_argument("--models_root", default="PRODUCTION/models", 
                       help="Path to institutional test results")
    parser.add_argument("--dataset_csv", default=None, 
                       help="Optional explicit dataset path")
    parser.add_argument("--out_json", default="reports/ic_audit.json",
                       help="Output JSON report path")
    
    args = parser.parse_args()
    main(args.models_root, args.dataset_csv, args.out_json)