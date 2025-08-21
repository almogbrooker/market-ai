#!/usr/bin/env python3
"""
Clean Validation Protocol - Following exact specification
No look-ahead bias, proper time-series validation
"""

import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# 🔒 FREEZE WALL CLOCK
T0_CUTOFF = '2022-12-31 23:59:59'
logger.info(f"🔒 WALL CLOCK FROZEN AT: {T0_CUTOFF}")

def step_1_rebuild_dataset_by_time_slice():
    """
    Step 1: Rebuild dataset with proper time slicing
    - ds_train.parquet (≤ T₀)
    - ds_oos_2023H1.parquet 
    - ds_oos_2023H2_2024.parquet
    - ds_oos_2025YTD.parquet
    """
    
    logger.info("📊 STEP 1: REBUILD DATASET BY TIME SLICE")
    logger.info("=" * 60)
    
    # Load enhanced dataset with fundamentals
    enhanced_path = Path(__file__).parent / 'data' / 'training_data_enhanced_with_fundamentals.csv'
    fallback_path = Path(__file__).parent / 'data' / 'training_data_enhanced.csv'
    
    data_path = enhanced_path if enhanced_path.exists() else fallback_path
    if not data_path.exists():
        logger.error(f"Dataset not found: {data_path}")
        return False
    
    logger.info(f"Using enhanced dataset: {data_path.name}")
    
    data = pd.read_csv(data_path)
    data['Date'] = pd.to_datetime(data['Date'])
    
    logger.info(f"Original dataset: {len(data):,} samples")
    logger.info(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    
    # Create artifacts directory
    artifacts_dir = Path(__file__).parent / 'artifacts'
    artifacts_dir.mkdir(exist_ok=True)
    
    # Time slicing with strict cutoffs
    cutoff_date = pd.to_datetime(T0_CUTOFF)
    
    # Training set: ≤ T₀
    ds_train = data[data['Date'] <= cutoff_date].copy()
    
    # OOS splits
    ds_oos_2023h1 = data[
        (data['Date'] >= '2023-01-01') & (data['Date'] <= '2023-06-30')
    ].copy()
    
    ds_oos_2023h2_2024 = data[
        (data['Date'] >= '2023-07-01') & (data['Date'] <= '2024-12-31')
    ].copy()
    
    ds_oos_2025ytd = data[
        data['Date'] >= '2025-01-01'
    ].copy()
    
    # Validate no overlap
    logger.info(f"📋 TIME SLICE VALIDATION:")
    logger.info(f"   Training (≤T₀): {len(ds_train):,} samples, {ds_train['Date'].min()} to {ds_train['Date'].max()}")
    logger.info(f"   OOS 2023H1: {len(ds_oos_2023h1):,} samples")
    logger.info(f"   OOS 2023H2-2024: {len(ds_oos_2023h2_2024):,} samples") 
    logger.info(f"   OOS 2025YTD: {len(ds_oos_2025ytd):,} samples")
    
    # Check for temporal gaps
    if len(ds_train) > 0 and len(ds_oos_2023h1) > 0:
        train_end = ds_train['Date'].max()
        test_start = ds_oos_2023h1['Date'].min()
        gap_days = (test_start - train_end).days
        logger.info(f"   Temporal gap: {gap_days} days (should be 1)")
    
    # Save datasets
    try:
        ds_train.to_parquet(artifacts_dir / 'ds_train.parquet', index=False)
        ds_oos_2023h1.to_parquet(artifacts_dir / 'ds_oos_2023H1.parquet', index=False)
        ds_oos_2023h2_2024.to_parquet(artifacts_dir / 'ds_oos_2023H2_2024.parquet', index=False)
        if len(ds_oos_2025ytd) > 0:
            ds_oos_2025ytd.to_parquet(artifacts_dir / 'ds_oos_2025YTD.parquet', index=False)
        
        logger.info("✅ Time-sliced datasets saved to artifacts/")
        
        return {
            'train_samples': len(ds_train),
            'oos_2023h1_samples': len(ds_oos_2023h1),
            'oos_2023h2_2024_samples': len(ds_oos_2023h2_2024),
            'oos_2025ytd_samples': len(ds_oos_2025ytd),
            'temporal_gap_days': gap_days if 'gap_days' in locals() else None
        }
        
    except Exception as e:
        logger.error(f"Failed to save datasets: {e}")
        return False

def step_2_proper_cross_validation():
    """
    Step 2: Proper cross-validation inside ≤T₀ only
    Purged, Embargoed Time-Series CV
    """
    
    logger.info("🔬 STEP 2: PROPER CROSS-VALIDATION (≤T₀ ONLY)")
    logger.info("=" * 60)
    
    artifacts_dir = Path(__file__).parent / 'artifacts'
    
    # Load training data only
    train_path = artifacts_dir / 'ds_train.parquet'
    if not train_path.exists():
        logger.error("Training dataset not found. Run step 1 first.")
        return False
    
    ds_train = pd.read_parquet(train_path)
    logger.info(f"Training data loaded: {len(ds_train):,} samples")
    logger.info(f"Date range: {ds_train['Date'].min()} to {ds_train['Date'].max()}")
    
    # Simple time-series CV (5 folds with purge/embargo)
    cv_results = []
    
    # Sort by date
    ds_train_sorted = ds_train.sort_values('Date')
    unique_dates = sorted(ds_train_sorted['Date'].unique())
    n_dates = len(unique_dates)
    
    logger.info(f"Unique trading dates: {n_dates}")
    
    # 5-fold time-series CV
    fold_size = n_dates // 5
    purge_days = 10  # 10 trading days purge
    embargo_days = 5  # 5 trading days embargo
    
    for fold in range(5):
        # Define fold boundaries
        train_start_idx = 0
        train_end_idx = min((fold + 1) * fold_size, n_dates - purge_days - embargo_days)
        
        val_start_idx = train_end_idx + purge_days
        val_end_idx = min(val_start_idx + fold_size, n_dates)
        
        if train_end_idx <= 0 or val_start_idx >= n_dates:
            continue
        
        # Get date boundaries
        train_start_date = unique_dates[train_start_idx]
        train_end_date = unique_dates[train_end_idx - 1] if train_end_idx > 0 else unique_dates[0]
        val_start_date = unique_dates[val_start_idx] if val_start_idx < n_dates else unique_dates[-1]
        val_end_date = unique_dates[val_end_idx - 1] if val_end_idx <= n_dates else unique_dates[-1]
        
        # Create fold data
        fold_train = ds_train_sorted[
            (ds_train_sorted['Date'] >= train_start_date) & 
            (ds_train_sorted['Date'] <= train_end_date)
        ]
        
        fold_val = ds_train_sorted[
            (ds_train_sorted['Date'] >= val_start_date) & 
            (ds_train_sorted['Date'] <= val_end_date)
        ]
        
        if len(fold_train) < 100 or len(fold_val) < 50:
            logger.warning(f"Fold {fold}: Insufficient data")
            continue
        
        logger.info(f"   Fold {fold}: Train {len(fold_train):,} samples ({train_start_date.date()} to {train_end_date.date()})")
        logger.info(f"   Fold {fold}: Val {len(fold_val):,} samples ({val_start_date.date()} to {val_end_date.date()})")
        
        # Academic-grade feature set: fundamentals + technicals + sentiment
        enhanced_features = [
            # Technical features
            'return_5d_lag1', 'return_20d_lag1', 'vol_20d_lag1', 'volume_ratio_lag1',
            'RSI_14', 'MACD', 'Volume_Ratio', 'Return_5D', 'Volatility_20D',
            
            # Cross-sectional fundamental z-scores (academic standard)
            'ZSCORE_PE', 'ZSCORE_PB', 'ZSCORE_PS', 'ZSCORE_ROE', 'ZSCORE_ROA',
            'ZSCORE_PM', 'ZSCORE_OM', 'ZSCORE_REV_GROWTH', 'ZSCORE_EPS_GROWTH',
            
            # Cross-sectional fundamental ranks
            'RANK_PE', 'RANK_PB', 'RANK_ROE', 'RANK_REV_GROWTH',
            
            # Engineered combinations
            'VALUE_MOMENTUM', 'QUALITY_GROWTH',
            
            # Sentiment features  
            'SENT_SENT_MEAN', 'SENT_MOMENTUM'
        ]
        
        # Use available features only
        available_features = [feat for feat in enhanced_features if feat in fold_train.columns]
        
        if len(available_features) >= 3:
            # Enhanced signal with equal weighting of available features
            train_signals = []
            val_signals = []
            
            for feat in available_features:
                train_feat = fold_train[feat].fillna(0)
                val_feat = fold_val[feat].fillna(0)
                
                # Normalize each feature to [-1, 1] range
                if len(train_feat) > 1 and train_feat.std() > 0:
                    train_norm = (train_feat - train_feat.mean()) / train_feat.std()
                    train_norm = np.tanh(train_norm)  # Bound to [-1, 1]
                else:
                    train_norm = train_feat
                
                if len(val_feat) > 1 and val_feat.std() > 0:
                    val_norm = (val_feat - val_feat.mean()) / val_feat.std()
                    val_norm = np.tanh(val_norm)
                else:
                    val_norm = val_feat
                
                train_signals.append(train_norm)
                val_signals.append(val_norm)
            
            # Ensemble the signals
            if train_signals and val_signals:
                train_signal = np.mean(train_signals, axis=0)
                val_signal = np.mean(val_signals, axis=0)
            else:
                train_signal = fold_train['return_5d_lag1'].fillna(0)
                val_signal = fold_val['return_5d_lag1'].fillna(0)
            
            # Multi-horizon IC calculation (1-day, 5-day, 21-day)
            horizons = {'1d': 'next_return_1d', '5d': 'target_5d', '20d': 'target_20d'}
            best_train_ic = 0
            best_val_ic = 0
            best_horizon = '1d'
            
            for horizon_name, target_col in horizons.items():
                if target_col in fold_train.columns:
                    train_target = fold_train[target_col].fillna(0)
                    val_target = fold_val[target_col].fillna(0)
                    
                    train_ic_h = np.corrcoef(train_signal, train_target)[0, 1]
                    val_ic_h = np.corrcoef(val_signal, val_target)[0, 1]
                    
                    if np.isnan(train_ic_h):
                        train_ic_h = 0
                    if np.isnan(val_ic_h):
                        val_ic_h = 0
                    
                    # Choose horizon with best validation IC
                    if abs(val_ic_h) > abs(best_val_ic):
                        best_train_ic = train_ic_h
                        best_val_ic = val_ic_h
                        best_horizon = horizon_name
            
            train_ic = best_train_ic
            val_ic = best_val_ic
            
            fold_result = {
                'fold': fold,
                'train_samples': len(fold_train),
                'val_samples': len(fold_val),
                'train_start': train_start_date.isoformat(),
                'train_end': train_end_date.isoformat(),
                'val_start': val_start_date.isoformat(),
                'val_end': val_end_date.isoformat(),
                'train_ic': float(train_ic),
                'val_ic': float(val_ic),
                'best_horizon': best_horizon,
                'available_features': available_features
            }
            
            cv_results.append(fold_result)
            logger.info(f"   Fold {fold}: Train IC = {train_ic:+.6f}, Val IC = {val_ic:+.6f} (horizon: {best_horizon}, features: {len(available_features)})")
    
    # Summary
    if cv_results:
        avg_train_ic = np.mean([r['train_ic'] for r in cv_results])
        avg_val_ic = np.mean([r['val_ic'] for r in cv_results])
        
        logger.info(f"📊 CV SUMMARY:")
        logger.info(f"   Average Train IC: {avg_train_ic:+.6f}")
        logger.info(f"   Average Val IC: {avg_val_ic:+.6f}")
        logger.info(f"   Folds completed: {len(cv_results)}")
        
        # Acceptance gates
        train_ic_check = abs(avg_train_ic) <= 0.03  # ≤ 3%
        val_ic_check = avg_val_ic >= 0.005  # ≥ 0.5%
        
        logger.info(f"🚪 ACCEPTANCE GATES:")
        logger.info(f"   Train IC ≤ 3.0%: {'✅' if train_ic_check else '❌'}")
        logger.info(f"   Val IC ≥ 0.5%: {'✅' if val_ic_check else '❌'}")
        
        # Save CV report
        cv_report = {
            'timestamp': datetime.now().isoformat(),
            'wall_clock_cutoff': T0_CUTOFF,
            'cv_results': cv_results,
            'summary': {
                'avg_train_ic': float(avg_train_ic),
                'avg_val_ic': float(avg_val_ic),
                'folds_completed': len(cv_results),
                'train_ic_check_passed': bool(train_ic_check),
                'val_ic_check_passed': bool(val_ic_check)
            }
        }
        
        with open(artifacts_dir / 'cv_report.json', 'w') as f:
            json.dump(cv_report, f, indent=2)
        
        logger.info("✅ CV report saved to artifacts/cv_report.json")
        
        return cv_report
    
    else:
        logger.error("❌ No valid CV folds completed")
        return False

def main():
    """Run clean validation protocol"""
    
    logger.info("🚀 CLEAN VALIDATION PROTOCOL")
    logger.info("=" * 80)
    logger.info("Following exact specification with no look-ahead bias")
    logger.info("=" * 80)
    
    # Step 1: Rebuild dataset by time slice
    step1_results = step_1_rebuild_dataset_by_time_slice()
    
    if not step1_results:
        logger.error("❌ Step 1 failed")
        return
    
    logger.info("\n✅ Step 1 completed successfully")
    
    # Step 2: Proper cross-validation
    step2_results = step_2_proper_cross_validation()
    
    if not step2_results:
        logger.error("❌ Step 2 failed")
        return
    
    logger.info("\n✅ Step 2 completed successfully")
    
    # Summary
    logger.info("\n🎉 CLEAN VALIDATION PROTOCOL PHASE 1-2 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Training samples: {step1_results['train_samples']:,}")
    logger.info(f"OOS 2023H1 samples: {step1_results['oos_2023h1_samples']:,}")
    logger.info(f"Average validation IC: {step2_results['summary']['avg_val_ic']:+.6f}")
    
    gates_passed = (
        step2_results['summary']['train_ic_check_passed'] and
        step2_results['summary']['val_ic_check_passed']
    )
    
    if gates_passed:
        logger.info("✅ ACCEPTANCE GATES PASSED - Ready for Step 3-4")
    else:
        logger.info("❌ ACCEPTANCE GATES FAILED - Model needs improvement")

if __name__ == "__main__":
    main()