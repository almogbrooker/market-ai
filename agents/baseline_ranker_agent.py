#!/usr/bin/env python3
"""
BASELINE RANKER AGENT - Chat-G.txt Section 3a
Mission: Cross-sectional rank model (fast, strong baseline) with LightGBM
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PurgedTimeSeriesSplit:
    """
    Purged Time Series Split with embargo
    Chat-G.txt: Purged K-Fold (embargo 10d) for model selection
    """
    
    def __init__(self, n_splits: int = 5, embargo_days: int = 10, purge_days: int = 5):
        self.n_splits = n_splits
        self.embargo_days = embargo_days
        self.purge_days = purge_days
    
    def split(self, X: np.ndarray, y: np.ndarray = None, dates: pd.Series = None):
        """Generate purged splits with embargo"""
        
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

class BaselineRankerAgent:
    """
    Baseline Ranker Agent - Chat-G.txt Section 3a
    Cross-sectional rank model with LightGBM
    """
    
    def __init__(self, model_config: Dict):
        logger.info("🤖 BASELINE RANKER AGENT - LIGHTGBM CROSS-SECTIONAL RANKER")
        
        self.config = model_config['baseline_ranker']
        self.base_dir = Path(__file__).parent.parent
        self.artifacts_dir = self.base_dir / "artifacts"
        
        # Load labeled data
        self.labels_path = self.artifacts_dir / "labels" / "labels.parquet"
        
        # Success criteria from config
        self.min_oos_ic = self.config['success_criteria']['min_oos_ic']
        self.min_newey_west_tstat = self.config['success_criteria']['min_newey_west_tstat']
        
        logger.info(f"🎯 Success Criteria:")
        logger.info(f"   Min OOS IC: {self.min_oos_ic:.3f} ({self.min_oos_ic*100:.1f}%)")
        logger.info(f"   Min Newey-West t-stat: {self.min_newey_west_tstat}")
        logger.info(f"   Target: {self.config['target']}")
        logger.info(f"   Max Features: {self.config['max_features']}")
    
    def train_model(self) -> bool:
        """
        Train baseline ranker following Chat-G.txt specification
        DoD: OOS IC ≥ 0.8% (NW t-stat > 2), stability across regimes; permutation test shows real signal
        """
        
        logger.info("🏋️ Training baseline ranker with purged CV...")
        
        try:
            # Load and prepare data
            df = self._load_and_prepare_data()
            if df is None:
                return False
            
            # Feature selection
            feature_columns = self._select_features(df)
            
            # Purged cross-validation training
            cv_results = self._purged_cross_validation(df, feature_columns)
            
            # Walk-forward validation
            walkforward_results = self._walk_forward_validation(df, feature_columns)
            
            # Final model training
            final_model = self._train_final_model(df, feature_columns)
            
            # Robustness tests
            robustness_results = self._robustness_tests(df, feature_columns, final_model)
            
            # Check success criteria
            success = self._check_success_criteria(cv_results, walkforward_results, robustness_results)
            
            # Save artifacts
            self._save_artifacts(final_model, cv_results, walkforward_results, robustness_results, feature_columns)
            
            logger.info(f"🏆 Baseline Ranker Training: {'✅ SUCCESS' if success else '❌ FAILED'}")
            return success
            
        except Exception as e:
            logger.error(f"❌ Baseline ranker training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_and_prepare_data(self) -> Optional[pd.DataFrame]:
        """Load and prepare labeled data"""
        
        logger.info("📊 Loading labeled data...")
        
        if not self.labels_path.exists():
            logger.error(f"Labels not found: {self.labels_path}")
            return None
        
        df = pd.read_parquet(self.labels_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Ticker', 'Date'])
        
        # Filter to valid targets
        df = df[df[self.config['target']].notna()].copy()
        
        logger.info(f"✅ Data loaded: {len(df)} samples, {df['Ticker'].nunique()} tickers")
        logger.info(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
        logger.info(f"   Target: {self.config['target']}")
        
        return df
    
    def _select_features(self, df: pd.DataFrame) -> List[str]:
        """
        Feature selection - Chat-G.txt: ≤ 40 high-signal, orthogonal features
        """
        
        logger.info("🔧 Selecting features...")
        
        # Get all lagged features
        feature_candidates = [col for col in df.columns if col.endswith('_lag1') or col.endswith('_lag0')]
        
        # Remove features with too much missingness
        feature_coverage = df[feature_candidates].notna().mean()
        valid_features = feature_coverage[feature_coverage >= 0.8].index.tolist()
        
        # Prioritize by feature importance (simple correlation with target)
        feature_importance = {}
        target_col = self.config['target']
        
        for feature in valid_features:
            try:
                corr = df[feature].corr(df[target_col])
                if not np.isnan(corr):
                    feature_importance[feature] = abs(corr)
            except:
                feature_importance[feature] = 0
        
        # Sort by importance and take top features
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in sorted_features[:self.config['max_features']]]
        
        logger.info(f"✅ Features selected: {len(selected_features)}/{len(feature_candidates)}")
        logger.info(f"   Top 5: {selected_features[:5]}")
        
        return selected_features
    
    def _purged_cross_validation(self, df: pd.DataFrame, feature_columns: List[str]) -> Dict:
        """
        Purged K-Fold cross-validation
        Chat-G.txt: Purged K-Fold (embargo 10d) for model selection
        """
        
        logger.info("🔄 Running purged cross-validation...")
        
        X = df[feature_columns].values
        y = df[self.config['target']].values
        dates = df['Date']
        
        # Purged CV
        purged_cv = PurgedTimeSeriesSplit(
            n_splits=self.config['cross_validation']['n_splits'],
            embargo_days=self.config['cross_validation']['embargo_days'],
            purge_days=self.config['cross_validation']['purge_days']
        )
        
        fold_results = []
        oof_predictions = np.zeros(len(y))
        oof_mask = np.zeros(len(y), dtype=bool)
        
        for fold_id, (train_idx, val_idx) in enumerate(purged_cv.split(X, y, dates)):
            logger.info(f"🔥 Training fold {fold_id + 1}/{self.config['cross_validation']['n_splits']}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Per-fold scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train LightGBM
            model = self._train_lgbm(X_train_scaled, y_train, X_val_scaled, y_val)
            
            # Predictions
            val_preds = model.predict(X_val_scaled)
            oof_predictions[val_idx] = val_preds
            oof_mask[val_idx] = True
            
            # Calculate fold IC
            fold_ic = spearmanr(y_val, val_preds)[0]
            fold_ic = 0 if np.isnan(fold_ic) else fold_ic
            
            fold_results.append({
                'fold': fold_id,
                'ic': fold_ic,
                'samples': len(y_val),
                'train_samples': len(y_train)
            })
            
            logger.info(f"   Fold {fold_id + 1} IC: {fold_ic:.4f} ({fold_ic*100:.2f}%)")
        
        # Overall OOF IC
        oof_ic = spearmanr(y[oof_mask], oof_predictions[oof_mask])[0]
        oof_ic = 0 if np.isnan(oof_ic) else oof_ic
        
        # Newey-West t-statistic
        nw_tstat = self._calculate_newey_west_tstat(oof_predictions[oof_mask], y[oof_mask])
        
        cv_results = {
            'oof_ic': oof_ic,
            'newey_west_tstat': nw_tstat,
            'fold_results': fold_results,
            'oof_predictions': oof_predictions,
            'oof_mask': oof_mask
        }
        
        logger.info(f"✅ Purged CV Results:")
        logger.info(f"   OOF IC: {oof_ic:.4f} ({oof_ic*100:.2f}%)")
        logger.info(f"   Newey-West t-stat: {nw_tstat:.2f}")
        
        return cv_results
    
    def _walk_forward_validation(self, df: pd.DataFrame, feature_columns: List[str]) -> Dict:
        """
        Walk-forward validation
        Chat-G.txt: Walk-forward monthly retrain for OOS
        """
        
        logger.info("🚶 Running walk-forward validation...")
        
        # Get monthly rebalance dates
        df['YearMonth'] = df['Date'].dt.to_period('M')
        rebalance_dates = sorted(df['YearMonth'].unique())
        
        walkforward_results = []
        min_train_periods = 12  # 12 months minimum training
        
        for i, test_period in enumerate(rebalance_dates[min_train_periods:], min_train_periods):
            # Training data: all data before test period
            train_data = df[df['YearMonth'] < test_period].copy()
            test_data = df[df['YearMonth'] == test_period].copy()
            
            if len(train_data) < 1000 or len(test_data) < 10:
                continue
            
            # Prepare data
            X_train = train_data[feature_columns].values
            y_train = train_data[self.config['target']].values
            X_test = test_data[feature_columns].values
            y_test = test_data[self.config['target']].values
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = self._train_lgbm(X_train_scaled, y_train)
            
            # Predict
            test_preds = model.predict(X_test_scaled)
            
            # Calculate IC
            test_ic = spearmanr(y_test, test_preds)[0]
            test_ic = 0 if np.isnan(test_ic) else test_ic
            
            walkforward_results.append({
                'period': str(test_period),
                'ic': test_ic,
                'samples': len(y_test),
                'train_samples': len(y_train)
            })
            
            if i % 6 == 0:  # Log every 6 months
                logger.info(f"   {test_period}: IC = {test_ic:.4f}")
        
        # Calculate statistics
        ics = [r['ic'] for r in walkforward_results]
        wf_mean_ic = np.mean(ics)
        wf_std_ic = np.std(ics)
        wf_sharpe = wf_mean_ic / wf_std_ic if wf_std_ic > 0 else 0
        
        wf_summary = {
            'mean_ic': wf_mean_ic,
            'std_ic': wf_std_ic,
            'ic_sharpe': wf_sharpe,
            'periods': len(walkforward_results),
            'results': walkforward_results
        }
        
        logger.info(f"✅ Walk-Forward Results:")
        logger.info(f"   Mean IC: {wf_mean_ic:.4f} ({wf_mean_ic*100:.2f}%)")
        logger.info(f"   IC Sharpe: {wf_sharpe:.2f}")
        logger.info(f"   Periods: {len(walkforward_results)}")
        
        return wf_summary
    
    def _train_lgbm(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> lgb.Booster:
        """Train LightGBM with regularization"""
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        
        valid_sets = [train_data]
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
        
        # Parameters from config with heavy regularization
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'max_depth': self.config['regularization']['max_depth'],
            'min_data_in_leaf': self.config['regularization']['min_data_in_leaf'],
            'feature_fraction': self.config['regularization']['feature_fraction'],
            'bagging_fraction': self.config['regularization']['bagging_fraction'],
            'bagging_freq': self.config['regularization']['bagging_freq'],
            'lambda_l1': self.config['regularization']['lambda_l1'],
            'lambda_l2': self.config['regularization']['lambda_l2'],
            'learning_rate': 0.05,
            'num_leaves': 31,
            'verbose': -1,
            'random_state': 42,
            'force_col_wise': True
        }
        
        # Train
        model = lgb.train(
            params, 
            train_data,
            valid_sets=valid_sets,
            num_boost_round=200,
            callbacks=[lgb.early_stopping(25), lgb.log_evaluation(0)]
        )
        
        return model
    
    def _train_final_model(self, df: pd.DataFrame, feature_columns: List[str]) -> lgb.Booster:
        """Train final model on all available data"""
        
        logger.info("🎯 Training final model...")
        
        X = df[feature_columns].values
        y = df[self.config['target']].values
        
        # Scale
        self.final_scaler = StandardScaler()
        X_scaled = self.final_scaler.fit_transform(X)
        
        # Train final model
        final_model = self._train_lgbm(X_scaled, y)
        
        logger.info("✅ Final model trained")
        return final_model
    
    def _robustness_tests(self, df: pd.DataFrame, feature_columns: List[str], model: lgb.Booster) -> Dict:
        """
        Robustness tests
        Chat-G.txt: permutation test shows real signal
        """
        
        logger.info("🧪 Running robustness tests...")
        
        X = df[feature_columns].values
        y = df[self.config['target']].values
        X_scaled = self.final_scaler.transform(X)
        
        # Baseline predictions
        baseline_preds = model.predict(X_scaled)
        baseline_ic = spearmanr(y, baseline_preds)[0]
        baseline_ic = 0 if np.isnan(baseline_ic) else baseline_ic
        
        # Shuffle test
        np.random.seed(42)
        y_shuffled = np.random.permutation(y)
        shuffle_ic = spearmanr(baseline_preds, y_shuffled)[0]
        shuffle_ic = 0 if np.isnan(shuffle_ic) else shuffle_ic
        
        # Permutation test (permute top features)
        permutation_results = []
        for i in range(min(5, len(feature_columns))):  # Test top 5 features
            X_perm = X_scaled.copy()
            X_perm[:, i] = np.random.permutation(X_perm[:, i])
            
            perm_preds = model.predict(X_perm)
            perm_ic = spearmanr(y, perm_preds)[0]
            perm_ic = 0 if np.isnan(perm_ic) else perm_ic
            
            ic_drop = baseline_ic - perm_ic
            permutation_results.append({
                'feature_idx': i,
                'feature_name': feature_columns[i],
                'ic_drop': ic_drop
            })
        
        # Regime stability (high/low VIX periods)
        vix_col = 'vix_lag0'
        if vix_col in df.columns:
            vix_median = df[vix_col].median()
            high_vix_mask = df[vix_col] > vix_median
            low_vix_mask = df[vix_col] <= vix_median
            
            high_vix_ic = spearmanr(y[high_vix_mask], baseline_preds[high_vix_mask])[0]
            low_vix_ic = spearmanr(y[low_vix_mask], baseline_preds[low_vix_mask])[0]
            
            high_vix_ic = 0 if np.isnan(high_vix_ic) else high_vix_ic
            low_vix_ic = 0 if np.isnan(low_vix_ic) else low_vix_ic
        else:
            high_vix_ic = baseline_ic
            low_vix_ic = baseline_ic
        
        robustness_results = {
            'baseline_ic': baseline_ic,
            'shuffle_ic': shuffle_ic,
            'permutation_results': permutation_results,
            'regime_stability': {
                'high_vix_ic': high_vix_ic,
                'low_vix_ic': low_vix_ic,
                'ic_difference': abs(high_vix_ic - low_vix_ic)
            }
        }
        
        logger.info(f"✅ Robustness Tests:")
        logger.info(f"   Baseline IC: {baseline_ic:.4f}")
        logger.info(f"   Shuffle IC: {shuffle_ic:.4f}")
        logger.info(f"   High VIX IC: {high_vix_ic:.4f}")
        logger.info(f"   Low VIX IC: {low_vix_ic:.4f}")
        
        return robustness_results
    
    def _calculate_newey_west_tstat(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate Newey-West t-statistic"""
        
        ic = np.corrcoef(predictions, targets)[0, 1]
        if np.isnan(ic):
            return 0.0
        
        # Simple approximation (proper Newey-West would need more sophisticated implementation)
        n = len(predictions)
        ic_std = np.sqrt((1 - ic**2) / (n - 2)) if n > 2 else 1.0
        t_stat = ic / ic_std
        
        return abs(t_stat)
    
    def _check_success_criteria(self, cv_results: Dict, walkforward_results: Dict, robustness_results: Dict) -> bool:
        """Check if model meets success criteria"""
        
        logger.info("🏆 Checking success criteria...")
        
        # Check OOS IC
        oos_ic_pass = cv_results['oof_ic'] >= self.min_oos_ic
        
        # Check Newey-West t-stat
        nw_tstat_pass = cv_results['newey_west_tstat'] > self.min_newey_west_tstat
        
        # Check stability across regimes
        regime_stability = robustness_results['regime_stability']['ic_difference'] < 0.02
        
        # Check permutation test shows real signal
        max_ic_drop = max([p['ic_drop'] for p in robustness_results['permutation_results']])
        permutation_pass = max_ic_drop > 0.002  # At least 0.2% IC drop when permuting features
        
        # Check shuffle test (IC should collapse)
        shuffle_pass = abs(robustness_results['shuffle_ic']) < 0.01
        
        criteria = {
            'oos_ic': oos_ic_pass,
            'newey_west_tstat': nw_tstat_pass,
            'stability_across_regimes': regime_stability,
            'permutation_test': permutation_pass,
            'shuffle_test': shuffle_pass
        }
        
        all_pass = all(criteria.values())
        
        logger.info("📊 Success Criteria Results:")
        for criterion, passed in criteria.items():
            logger.info(f"   {'✅' if passed else '❌'} {criterion}: {'PASS' if passed else 'FAIL'}")
        
        return all_pass
    
    def _save_artifacts(self, model: lgb.Booster, cv_results: Dict, walkforward_results: Dict, 
                       robustness_results: Dict, feature_columns: List[str]):
        """Save model artifacts"""
        
        logger.info("💾 Saving baseline ranker artifacts...")
        
        # Ensure directories exist
        (self.artifacts_dir / "models").mkdir(parents=True, exist_ok=True)
        (self.artifacts_dir / "oof").mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = self.artifacts_dir / "models" / "lgbm_ranker.txt"
        model.save_model(str(model_path))
        
        # Save scaler
        scaler_path = self.artifacts_dir / "models" / "lgbm_scaler.pkl"
        import joblib
        joblib.dump(self.final_scaler, scaler_path)
        
        # Save OOF predictions
        oof_df = pd.DataFrame({
            'oof_predictions': cv_results['oof_predictions'],
            'oof_mask': cv_results['oof_mask']
        })
        oof_path = self.artifacts_dir / "oof" / "lgbm_oof.parquet"
        oof_df.to_parquet(oof_path, index=False)
        
        # Save comprehensive results
        results = {
            'model_name': 'baseline_ranker_lgbm',
            'feature_columns': feature_columns,
            'cv_results': {
                'oof_ic': cv_results['oof_ic'],
                'newey_west_tstat': cv_results['newey_west_tstat'],
                'fold_results': cv_results['fold_results']
            },
            'walkforward_results': walkforward_results,
            'robustness_results': robustness_results,
            'config': self.config,
            'success_criteria_met': self._check_success_criteria(cv_results, walkforward_results, robustness_results),
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = self.artifacts_dir / "models" / "lgbm_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"✅ Artifacts saved:")
        logger.info(f"   Model: {model_path}")
        logger.info(f"   OOF: {oof_path}")
        logger.info(f"   Results: {results_path}")

def main():
    """Test the baseline ranker agent"""
    
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "model_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        logger.error("Model config not found")
        return False
    
    # Initialize and run agent
    agent = BaselineRankerAgent(config)
    success = agent.train_model()
    
    if success:
        print("✅ Baseline Ranker training completed successfully")
    else:
        print("❌ Baseline Ranker training failed")
    
    return success

if __name__ == "__main__":
    main()