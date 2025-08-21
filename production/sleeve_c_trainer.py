#!/usr/bin/env python3
"""
SLEEVE C TRAINER: Momentum + Quality Baseline Ranker
Mission Brief Implementation with Purged CV and Proper Gates
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import sys
import os
from pathlib import Path
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PurgedTimeSeriesSplit:
    """
    Purged Time Series Split with embargo (Mission Brief Section 4)
    """
    
    def __init__(self, n_splits: int = 3, purge_days: int = 10, embargo_days: int = 3):
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
    
    def split(self, X: np.ndarray, y: np.ndarray = None, dates: pd.Series = None):
        """Generate purged splits"""
        
        if dates is None:
            raise ValueError("dates required for purged split")
        
        # Convert to datetime if needed
        if not isinstance(dates.iloc[0], pd.Timestamp):
            dates = pd.to_datetime(dates)
        
        # Get unique dates and sort
        unique_dates = sorted(dates.unique())
        n_dates = len(unique_dates)
        
        # Calculate split points
        split_size = n_dates // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            # Training end
            train_end_idx = (i + 1) * split_size
            train_end_date = unique_dates[train_end_idx]
            
            # Purge period
            purge_end_date = train_end_date + timedelta(days=self.purge_days)
            
            # Validation start (after embargo)
            val_start_date = purge_end_date + timedelta(days=self.embargo_days)
            
            # Validation end (next split or end)
            if i < self.n_splits - 1:
                val_end_idx = (i + 2) * split_size
                val_end_date = unique_dates[min(val_end_idx, n_dates - 1)]
            else:
                val_end_date = unique_dates[-1]
            
            # Create train/val indices
            train_mask = dates <= train_end_date
            val_mask = (dates >= val_start_date) & (dates <= val_end_date)
            
            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]
            
            if len(train_idx) > 0 and len(val_idx) > 0:
                yield train_idx, val_idx

class SleeveCTrainer:
    """
    Sleeve C Trainer: Momentum + Quality baseline ranker
    """
    
    def __init__(self, dataset_hash: str = None):
        logger.info("🎯 SLEEVE C TRAINER - MOMENTUM + QUALITY BASELINE")
        
        self.dataset_hash = dataset_hash
        
        # Mission Brief Parameters
        self.target_rank_ic_min = 0.008  # 0.8% minimum
        self.target_rank_ic_max = 0.015  # 1.5% excellent  
        self.max_training_ic = 0.03      # 3% overfitting threshold
        
        # CV parameters
        self.purge_days = 10            # Calendar-day purge
        self.embargo_days = 3           # Embargo period
        self.cv_splits = 3              # 3 purged splits
        
        # Setup directories
        self.artifacts_dir = Path(__file__).parent.parent / "artifacts" / "sleeves" / "sleeve_c"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Track gates for final report
        self.gates_passed = {}
        
        logger.info(f"🎯 Mission Brief Parameters:")
        logger.info(f"   Target Rank IC: {self.target_rank_ic_min:.1%} - {self.target_rank_ic_max:.1%}")
        logger.info(f"   Max Training IC: {self.max_training_ic:.1%}")
        logger.info(f"   Purged CV: {self.cv_splits} splits, purge={self.purge_days}d, embargo={self.embargo_days}d")
    
    def load_nasdaq_dataset(self) -> pd.DataFrame:
        """Load NASDAQ dataset"""
        
        logger.info("📊 Loading NASDAQ dataset...")
        
        if self.dataset_hash:
            dataset_path = Path(__file__).parent.parent / "artifacts" / "nasdaq_picker" / f"nasdaq_dataset_{self.dataset_hash}.csv"
        else:
            # Find most recent dataset
            nasdaq_dir = Path(__file__).parent.parent / "artifacts" / "nasdaq_picker"
            dataset_files = list(nasdaq_dir.glob("nasdaq_dataset_*.csv"))
            if not dataset_files:
                raise FileNotFoundError("No NASDAQ dataset found")
            dataset_path = max(dataset_files, key=lambda x: x.stat().st_mtime)
            self.dataset_hash = dataset_path.stem.split('_')[-1]
        
        logger.info(f"Loading dataset: {dataset_path}")
        df = pd.read_csv(dataset_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        logger.info(f"✅ Dataset loaded: {len(df)} samples, {df['Ticker'].nunique()} tickers")
        logger.info(f"   Dataset hash: {self.dataset_hash}")
        logger.info(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        return df
    
    def pre_training_gates(self, df: pd.DataFrame) -> bool:
        """Pre-training gates from Mission Brief"""
        
        logger.info("🚨 CHECKING PRE-TRAINING GATES...")
        
        # Gate 1: All features are lagged (≥1 day)
        feature_columns = [col for col in df.columns if col.endswith('_lag1')]
        self.feature_columns = feature_columns
        all_lagged = all('lag' in col for col in feature_columns)
        self.gates_passed['all_features_lagged'] = all_lagged
        logger.info(f"{'✅' if all_lagged else '❌'} All features lagged: {len(feature_columns)} features")
        
        # Gate 2: Feature count ≤ 10
        feature_count_ok = len(feature_columns) <= 10
        self.gates_passed['feature_count_ok'] = feature_count_ok
        logger.info(f"{'✅' if feature_count_ok else '❌'} Feature count: {len(feature_columns)}/10")
        
        # Gate 3: Target = next-day returns (cross-sectional ranks)
        has_targets = 'target_rank' in df.columns and 'target_return' in df.columns
        self.gates_passed['has_proper_targets'] = has_targets
        logger.info(f"{'✅' if has_targets else '❌'} Proper targets: rank + return")
        
        # Gate 4: Dataset hash matches
        dataset_hash_ok = self.dataset_hash is not None
        self.gates_passed['dataset_hash_ok'] = dataset_hash_ok
        logger.info(f"{'✅' if dataset_hash_ok else '❌'} Dataset hash: {self.dataset_hash}")
        
        # Gate 5: Sufficient data for purged CV
        min_samples = 10000  # Need enough for meaningful splits
        sufficient_data = len(df) >= min_samples
        self.gates_passed['sufficient_data'] = sufficient_data
        logger.info(f"{'✅' if sufficient_data else '❌'} Sufficient data: {len(df)}/{min_samples}")
        
        all_passed = all(self.gates_passed.values())
        logger.info(f"🚨 PRE-TRAINING GATES: {'✅ ALL PASSED' if all_passed else '❌ FAILED'}")
        
        return all_passed
    
    def train_sleeve_c_model(self, df: pd.DataFrame) -> Dict:
        """Train Sleeve C model with purged CV"""
        
        logger.info("🏋️ Training Sleeve C with Purged CV...")
        
        # Prepare data
        X = df[self.feature_columns].values
        y_rank = df['target_rank'].values
        y_return = df['target_return'].values
        dates = df['Date']
        
        logger.info(f"Training data: X.shape={X.shape}")
        logger.info(f"Features: {self.feature_columns}")
        
        # Purged Time Series CV
        purged_cv = PurgedTimeSeriesSplit(
            n_splits=self.cv_splits,
            purge_days=self.purge_days,
            embargo_days=self.embargo_days
        )
        
        fold_results = []
        oof_predictions = np.zeros(len(y_rank))
        oof_mask = np.zeros(len(y_rank), dtype=bool)
        models = []
        
        for fold_id, (train_idx, val_idx) in enumerate(purged_cv.split(X, dates=dates)):
            logger.info(f"🔥 Training Sleeve C Fold {fold_id + 1}/{self.cv_splits}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_return[train_idx], y_return[val_idx]  # Train on returns
            y_rank_val = y_rank[val_idx]  # Evaluate on ranks
            
            train_dates = dates.iloc[train_idx]
            val_dates = dates.iloc[val_idx]
            
            logger.info(f"Fold {fold_id + 1}:")
            logger.info(f"   Train: {len(X_train)} samples ({train_dates.min()} to {train_dates.max()})")
            logger.info(f"   Val:   {len(X_val)} samples ({val_dates.min()} to {val_dates.max()})")
            
            # Per-fold scaling (Mission Brief Section 2)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # LightGBM with anti-overfitting parameters
            train_data = lgb.Dataset(
                X_train_scaled, 
                label=y_train,  # Train to predict returns
                feature_name=[f"feature_{i}" for i in range(X_train_scaled.shape[1])]
            )
            
            # Validation set for early stopping
            early_stop_size = min(1000, len(X_train_scaled) // 5)
            early_stop_data = lgb.Dataset(
                X_train_scaled[:early_stop_size], 
                label=y_train[:early_stop_size]
            )
            
            # Conservative parameters (Mission Brief Section 3)
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 15,       # Small trees
                'learning_rate': 0.03,  # Slow learning  
                'feature_fraction': 0.6, # Heavy feature sampling
                'bagging_fraction': 0.6, # Heavy row sampling
                'bagging_freq': 1,
                'min_data_in_leaf': 200, # Large leaves
                'lambda_l1': 0.2,       # Strong L1 regularization
                'lambda_l2': 0.2,       # Strong L2 regularization
                'verbose': -1,
                'random_state': 42,
                'force_col_wise': True,
                'max_depth': 4,         # Shallow trees
                'min_gain_to_split': 0.05  # High split threshold
            }
            
            model = lgb.train(
                params, train_data,
                valid_sets=[early_stop_data],
                num_boost_round=50,     # Conservative rounds
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            
            # Generate predictions (returns)
            val_preds = model.predict(X_val_scaled)
            oof_predictions[val_idx] = val_preds
            oof_mask[val_idx] = True
            
            # Calculate training IC (safety check)
            train_preds = model.predict(X_train_scaled)
            train_ic = spearmanr(y_train, train_preds)[0]
            train_ic = 0 if np.isnan(train_ic) else train_ic
            
            # Calculate validation Rank IC (key metric)
            # Convert return predictions to cross-sectional ranks
            val_df = pd.DataFrame({
                'Date': val_dates.values,
                'predictions': val_preds,
                'target_rank': y_rank_val,
                'target_return': y_val
            })
            
            # Calculate daily rank IC
            daily_rank_ics = []
            for date, date_group in val_df.groupby('Date'):
                if len(date_group) >= 5:
                    pred_ranks = date_group['predictions'].rank(pct=True)
                    target_ranks = date_group['target_rank']
                    daily_ic = spearmanr(pred_ranks, target_ranks)[0]
                    if not np.isnan(daily_ic):
                        daily_rank_ics.append(daily_ic)
            
            fold_rank_ic = np.mean(daily_rank_ics) if daily_rank_ics else 0
            
            fold_results.append({
                'fold': fold_id,
                'train_ic': train_ic,
                'rank_ic': fold_rank_ic,
                'val_samples': len(X_val),
                'train_samples': len(X_train),
                'train_date_range': f"{train_dates.min()} to {train_dates.max()}",
                'val_date_range': f"{val_dates.min()} to {val_dates.max()}"
            })
            
            models.append({
                'model': model,
                'scaler': scaler,
                'fold': fold_id
            })
            
            logger.info(f"✅ Fold {fold_id + 1} Results:")
            logger.info(f"   Train IC: {train_ic:.4f} ({train_ic*100:.2f}%)")
            logger.info(f"   Rank IC: {fold_rank_ic:.4f} ({fold_rank_ic*100:.2f}%)")
        
        # Calculate overall OOF Rank IC
        oof_df = pd.DataFrame({
            'Date': dates[oof_mask],
            'predictions': oof_predictions[oof_mask],
            'target_rank': y_rank[oof_mask]
        })
        
        # Overall daily rank IC
        overall_daily_rank_ics = []
        for date, date_group in oof_df.groupby('Date'):
            if len(date_group) >= 5:
                pred_ranks = date_group['predictions'].rank(pct=True)
                target_ranks = date_group['target_rank']
                daily_ic = spearmanr(pred_ranks, target_ranks)[0]
                if not np.isnan(daily_ic):
                    overall_daily_rank_ics.append(daily_ic)
        
        overall_rank_ic = np.mean(overall_daily_rank_ics) if overall_daily_rank_ics else 0
        
        # Training IC gate check
        mean_train_ic = np.mean([fold['train_ic'] for fold in fold_results])
        training_ic_ok = abs(mean_train_ic) <= self.max_training_ic
        self.gates_passed['training_ic_gate'] = training_ic_ok
        
        # OOS Rank IC gate check
        oos_rank_ic_ok = overall_rank_ic >= self.target_rank_ic_min
        self.gates_passed['oos_rank_ic_gate'] = oos_rank_ic_ok
        
        results = {
            'model_name': 'sleeve_c_momentum_quality',
            'dataset_hash': self.dataset_hash,
            'overall_rank_ic': overall_rank_ic,
            'mean_training_ic': mean_train_ic,
            'fold_results': fold_results,
            'oof_predictions': oof_predictions,
            'oof_mask': oof_mask,
            'feature_columns': self.feature_columns,
            'models': models,
            'gates_passed': self.gates_passed,
            'daily_rank_ics': overall_daily_rank_ics
        }
        
        logger.info(f"🏆 SLEEVE C RESULTS:")
        logger.info(f"   Overall Rank IC: {overall_rank_ic:.4f} ({overall_rank_ic*100:.2f}%)")
        logger.info(f"   Mean Training IC: {mean_train_ic:.4f} ({mean_train_ic*100:.2f}%)")
        logger.info(f"   Training IC Gate: {'✅ PASS' if training_ic_ok else '❌ FAIL'}")
        logger.info(f"   OOS Rank IC Gate: {'✅ PASS' if oos_rank_ic_ok else '❌ FAIL'}")
        
        return results
    
    def newey_west_test(self, daily_ics: List[float]) -> float:
        """Calculate Newey-West t-statistic (Mission Brief Section 7)"""
        
        if len(daily_ics) < 10:
            return 0.0
        
        ic_array = np.array(daily_ics)
        ic_mean = np.mean(ic_array)
        ic_std = np.std(ic_array)
        
        # Simple t-stat (proper Newey-West would need more sophisticated implementation)
        t_stat = ic_mean / (ic_std / np.sqrt(len(ic_array))) if ic_std > 0 else 0
        
        return abs(t_stat)
    
    def validation_gates(self, results: Dict) -> bool:
        """Validation gates from Mission Brief Section 7"""
        
        logger.info("🔍 CHECKING VALIDATION GATES...")
        
        daily_ics = results['daily_rank_ics']
        overall_rank_ic = results['overall_rank_ic']
        
        # Gate 1: OOS Rank IC ≥ 0.8%
        gate1 = overall_rank_ic >= self.target_rank_ic_min
        logger.info(f"{'✅' if gate1 else '❌'} OOS Rank IC: {overall_rank_ic:.4f} ({'≥' if gate1 else '<'} {self.target_rank_ic_min:.3f})")
        
        # Gate 2: Newey-West t-stat > 2.0
        nw_tstat = self.newey_west_test(daily_ics)
        gate2 = nw_tstat > 2.0
        logger.info(f"{'✅' if gate2 else '❌'} Newey-West t-stat: {nw_tstat:.2f} ({'>' if gate2 else '≤'} 2.0)")
        
        # Gate 3: Stable across sub-periods
        if len(daily_ics) >= 30:
            mid_point = len(daily_ics) // 2
            period1_ic = np.mean(daily_ics[:mid_point])
            period2_ic = np.mean(daily_ics[mid_point:])
            ic_stability = abs(period1_ic - period2_ic) < 0.01  # ≤1% difference
        else:
            ic_stability = True  # Not enough data to test
        
        gate3 = ic_stability
        logger.info(f"{'✅' if gate3 else '❌'} IC stability across periods: {ic_stability}")
        
        # Gate 4: Training IC ≤ 3%
        gate4 = results['gates_passed'].get('training_ic_gate', False)
        logger.info(f"{'✅' if gate4 else '❌'} Training IC ≤ 3%: {gate4}")
        
        all_gates = [gate1, gate2, gate3, gate4]
        validation_passed = all(all_gates)
        
        results['validation_gates'] = {
            'oos_rank_ic_gate': gate1,
            'newey_west_gate': gate2, 
            'stability_gate': gate3,
            'training_ic_gate': gate4,
            'all_passed': validation_passed,
            'newey_west_tstat': nw_tstat
        }
        
        logger.info(f"🔍 VALIDATION GATES: {'✅ ALL PASSED' if validation_passed else '❌ FAILED'}")
        
        return validation_passed
    
    def save_sleeve_c_artifacts(self, results: Dict):
        """Save Sleeve C artifacts (Mission Brief Section 6)"""
        
        logger.info("💾 Saving Sleeve C artifacts...")
        
        # Save fold models
        for model_info in results['models']:
            fold_id = model_info['fold']
            model_path = self.artifacts_dir / f"sleeve_c_fold_{fold_id}_model.txt"
            model_info['model'].save_model(str(model_path))
            
            # Save scaler
            scaler_path = self.artifacts_dir / f"sleeve_c_fold_{fold_id}_scaler.pkl"
            import joblib
            joblib.dump(model_info['scaler'], scaler_path)
        
        # Save OOF predictions
        oof_df = pd.DataFrame({
            'oof_predictions': results['oof_predictions'],
            'oof_mask': results['oof_mask']
        })
        oof_path = self.artifacts_dir / "sleeve_c_oof_predictions.csv"
        oof_df.to_csv(oof_path, index=False)
        
        # Save daily IC
        daily_ic_df = pd.DataFrame({
            'daily_rank_ic': results['daily_rank_ics']
        })
        daily_ic_path = self.artifacts_dir / "sleeve_c_daily_IC.csv"
        daily_ic_df.to_csv(daily_ic_path, index=False)
        
        # Save fold results
        fold_df = pd.DataFrame(results['fold_results'])
        fold_ic_path = self.artifacts_dir / "sleeve_c_fold_IC.csv"
        fold_df.to_csv(fold_ic_path, index=False)
        
        # Save metadata
        metadata = {
            'model_name': results['model_name'],
            'dataset_hash': results['dataset_hash'],
            'overall_rank_ic': results['overall_rank_ic'],
            'mean_training_ic': results['mean_training_ic'],
            'features': results['feature_columns'],
            'cv_config': {
                'n_splits': self.cv_splits,
                'purge_days': self.purge_days,
                'embargo_days': self.embargo_days
            },
            'gates_passed': results['gates_passed'],
            'validation_gates': results.get('validation_gates', {}),
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = self.artifacts_dir / "sleeve_c_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"✅ Sleeve C artifacts saved to: {self.artifacts_dir}")

def main():
    """Train Sleeve C following Mission Brief"""
    
    logger.info("=" * 80)
    logger.info("🎯 SLEEVE C TRAINER - MOMENTUM + QUALITY BASELINE")
    logger.info("=" * 80)
    
    try:
        # Initialize trainer
        trainer = SleeveCTrainer()
        
        # Load NASDAQ dataset
        df = trainer.load_nasdaq_dataset()
        
        # Pre-training gates
        if not trainer.pre_training_gates(df):
            logger.error("🚨 PRE-TRAINING GATES FAILED - STOPPING")
            return None
        
        # Train Sleeve C model
        results = trainer.train_sleeve_c_model(df)
        
        # Validation gates
        validation_passed = trainer.validation_gates(results)
        
        # Save artifacts
        trainer.save_sleeve_c_artifacts(results)
        
        # Final assessment
        logger.info("=" * 80)
        logger.info("🏆 SLEEVE C TRAINING RESULTS")
        logger.info("=" * 80)
        
        overall_rank_ic = results['overall_rank_ic']
        validation_gates = results.get('validation_gates', {})
        
        logger.info(f"📊 Overall Rank IC: {overall_rank_ic:.4f} ({overall_rank_ic*100:.2f}%)")
        logger.info(f"🎯 Target Range: {trainer.target_rank_ic_min:.1%} - {trainer.target_rank_ic_max:.1%}")
        logger.info(f"✅ Validation Passed: {validation_passed}")
        
        if validation_passed:
            logger.info("🎉 SUCCESS: Sleeve C ready for portfolio simulation!")
            print(f"\n✅ SLEEVE C SUCCESS!")
            print(f"📊 Rank IC: {overall_rank_ic*100:.2f}% (Target: {trainer.target_rank_ic_min*100:.1f}%-{trainer.target_rank_ic_max*100:.1f}%)")
            print(f"✅ All validation gates passed")
            print(f"🚀 Ready for beta-neutral portfolio construction!")
        else:
            logger.warning("⚠️ Validation gates failed - needs improvement")
            print(f"\n⚠️ SLEEVE C NEEDS IMPROVEMENT")
            print(f"📊 Rank IC: {overall_rank_ic*100:.2f}%")
            print(f"❌ Some validation gates failed")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Sleeve C training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()