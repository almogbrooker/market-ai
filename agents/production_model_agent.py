#!/usr/bin/env python3
"""
PRODUCTION MODEL AGENT - LambdaRank Implementation
Research-backed cross-sectional ranking model with proper validation
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
import joblib
from scipy.stats import spearmanr
import warnings

# Only suppress non-critical warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionModelAgent:
    """
    Production Model Agent - LambdaRank with Walk-Forward Validation
    Implements research-backed cross-sectional ranking
    """
    
    def __init__(self, config_path: str):
        logger.info("🤖 PRODUCTION MODEL AGENT - LAMBDARANK RANKER")
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.base_dir = Path(__file__).parent.parent
        self.artifacts_dir = self.base_dir / "artifacts"
        
        # Model configuration
        self.model_config = self.config['model_config']['stage_1']
        self.validation_config = self.config['training_validation']
        
        logger.info("🎯 Model Configuration:")
        logger.info(f"   Model: {self.model_config['model_type']}")
        logger.info(f"   Objective: {self.model_config['objective']}")
        logger.info(f"   Per-Date Groups: {self.model_config['per_date_groups']}")
        logger.info(f"   Max Depth: {self.model_config['regularization']['max_depth']}")
        
    def train_production_model(self, data_path: Optional[str] = None) -> Dict[str, any]:
        """
        Train production LambdaRank model with walk-forward validation
        """
        
        logger.info("🏭 Training production model...")
        
        try:
            # Load production data
            data = self._load_production_data(data_path)
            if data is None:
                return {'success': False, 'reason': 'No data available'}
            
            # Prepare features and targets
            features, targets, groups, scaler = self._prepare_model_data(data)
            self.scaler = scaler
            
            # Walk-forward validation
            oof_results = self._walk_forward_validation(features, targets, groups, data)
            
            # Final model training on all data
            final_model = self._train_final_model(features, targets, groups)
            
            # Comprehensive evaluation
            evaluation_results = self._evaluate_model_performance(oof_results, data)
            
            # Robustness tests
            robustness_results = self._run_robustness_tests(features, targets, groups, final_model)
            
            # Check production gates
            gates_passed = self._check_production_gates(evaluation_results, robustness_results)
            
            # Save model artifacts
            self._save_model_artifacts(final_model, oof_results, evaluation_results, robustness_results)
            
            result = {
                'success': True,
                'gates_passed': gates_passed,
                'oof_results': oof_results,
                'evaluation': evaluation_results,
                'robustness': robustness_results,
                'model_saved': True
            }
            
            logger.info(f"🏆 Model Training: {'✅ SUCCESS' if gates_passed else '❌ FAILED GATES'}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'reason': f'Error: {e}'}
    
    def _load_production_data(self, data_path: Optional[str]) -> Optional[pd.DataFrame]:
        """Load production dataset"""
        
        logger.info("📂 Loading production data...")
        
        if data_path is None:
            # Find most recent production dataset
            production_dir = self.artifacts_dir / "production"
            if not production_dir.exists():
                logger.error("No production data directory found")
                return None
            
            data_files = list(production_dir.glob("universe_*.parquet"))
            if not data_files:
                logger.error("No production datasets found")
                return None
            
            data_files.sort()
            data_path = data_files[-1]
        
        try:
            data = pd.read_parquet(data_path)
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.sort_values(['Date', 'Ticker'])
            
            # Filter to valid samples
            data = data[data['residual_return_target'].notna()].copy()
            
            logger.info(f"✅ Data loaded: {len(data)} samples, {data['Ticker'].nunique()} stocks")
            logger.info(f"   Date range: {data['Date'].min()} to {data['Date'].max()}")
            logger.info(f"   Target: residual_return_target")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return None
    
    def _prepare_model_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
        """Prepare features, targets, groups, and scaler for LambdaRank"""
        
        logger.info("🔧 Preparing model data...")
        
        # Get z-scored features (research-backed signals)
        feature_cols = [col for col in data.columns if col.endswith('_zscore')]
        
        if not feature_cols:
            logger.warning("No z-scored features found, using raw features")
            feature_cols = [
                'momentum_12_1m', 'momentum_3m', 'momentum_20d',
                'reversal_5d', 'overnight_gap',
                'volatility_20d', 'dollar_volume_20d', 'idiosyncratic_vol'
            ]
            feature_cols = [col for col in feature_cols if col in data.columns]
        
        # Fill missing values
        for col in feature_cols:
            data[col] = data[col].fillna(0)  # Cross-sectional z-scores, 0 = neutral

        # Scale features
        scaler = StandardScaler()
        data[feature_cols] = scaler.fit_transform(data[feature_cols])

        # Prepare arrays
        features = data[feature_cols].values
        targets = data['residual_return_target'].values
        
        # Create group array for LambdaRank (one group per date)
        date_map = {date: i for i, date in enumerate(data['Date'].unique())}
        groups = data['Date'].map(date_map).values
        
        logger.info(f"✅ Model data prepared:")
        logger.info(f"   Features: {len(feature_cols)} ({feature_cols[:3]}...)")
        logger.info(f"   Samples: {len(features)}")
        logger.info(f"   Groups (dates): {len(np.unique(groups))}")

        return features, targets, groups, scaler
    
    def _walk_forward_validation(self, features: np.ndarray, targets: np.ndarray, 
                                groups: np.ndarray, data: pd.DataFrame) -> Dict[str, any]:
        """Walk-forward validation with purge and embargo"""
        
        logger.info("🚶 Running walk-forward validation...")
        
        # Get unique dates sorted
        unique_dates = sorted(data['Date'].unique())
        
        # Configuration
        train_months = self.validation_config['walk_forward']['train_months']
        test_months = self.validation_config['walk_forward']['test_months'] 
        purge_days = self.validation_config['walk_forward']['purge_days']
        embargo_days = self.validation_config['walk_forward']['embargo_days']
        
        fold_results = []
        oof_predictions = np.zeros(len(targets))
        oof_mask = np.zeros(len(targets), dtype=bool)
        
        # Start walk-forward after minimum training period
        start_idx = train_months  # Start after train_months of data
        
        for i in range(start_idx, len(unique_dates), test_months):
            # Training period
            train_end_date = unique_dates[i]
            train_start_date = unique_dates[max(0, i - train_months)]
            
            # Purge period
            purge_end_date = train_end_date + timedelta(days=purge_days)
            
            # Test period (after embargo)
            test_start_date = purge_end_date + timedelta(days=embargo_days)
            test_end_idx = min(i + test_months, len(unique_dates))
            test_end_date = unique_dates[test_end_idx - 1] if test_end_idx > i else test_start_date
            
            # Create masks
            train_mask = (data['Date'] >= train_start_date) & (data['Date'] <= train_end_date)
            test_mask = (data['Date'] >= test_start_date) & (data['Date'] <= test_end_date)
            
            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]
            
            if len(train_indices) < 1000 or len(test_indices) < 50:
                continue
            
            # Extract fold data
            X_train, y_train, g_train = features[train_indices], targets[train_indices], groups[train_indices]
            X_test, y_test, g_test = features[test_indices], targets[test_indices], groups[test_indices]
            
            # Train model
            model = self._train_lambdarank_model(X_train, y_train, g_train)
            
            # Predict
            test_predictions = model.predict(X_test)
            oof_predictions[test_indices] = test_predictions
            oof_mask[test_indices] = True
            
            # Calculate fold metrics
            fold_ic = spearmanr(y_test, test_predictions)[0]
            fold_ic = 0 if np.isnan(fold_ic) else fold_ic
            
            # Cross-sectional IC (per date within fold)
            test_data = data.iloc[test_indices].copy()
            test_data['predictions'] = test_predictions
            daily_ics = []
            
            for date in test_data['Date'].unique():
                date_mask = test_data['Date'] == date
                date_targets = test_data.loc[date_mask, 'residual_return_target']
                date_preds = test_data.loc[date_mask, 'predictions']
                
                if len(date_targets) >= 10:  # Minimum stocks per date
                    daily_ic = spearmanr(date_targets, date_preds)[0]
                    if not np.isnan(daily_ic):
                        daily_ics.append(daily_ic)
            
            avg_daily_ic = np.mean(daily_ics) if daily_ics else 0
            
            fold_results.append({
                'fold': len(fold_results),
                'train_start': str(train_start_date),
                'train_end': str(train_end_date),
                'test_start': str(test_start_date),
                'test_end': str(test_end_date),
                'train_samples': len(train_indices),
                'test_samples': len(test_indices),
                'fold_ic': fold_ic,
                'avg_daily_ic': avg_daily_ic,
                'daily_ic_count': len(daily_ics)
            })
            
            logger.info(f"   Fold {len(fold_results)}: IC={fold_ic:.4f}, Daily IC={avg_daily_ic:.4f}")
        
        # Overall OOF metrics
        oof_ic = spearmanr(targets[oof_mask], oof_predictions[oof_mask])[0]
        oof_ic = 0 if np.isnan(oof_ic) else oof_ic
        
        # Newey-West t-statistic (simplified)
        n_oof = oof_mask.sum()
        ic_std = np.sqrt((1 - oof_ic**2) / (n_oof - 2)) if n_oof > 2 else 1.0
        newey_west_tstat = abs(oof_ic / ic_std) if ic_std > 0 else 0
        
        oof_results = {
            'fold_results': fold_results,
            'oof_ic': oof_ic,
            'newey_west_tstat': newey_west_tstat,
            'oof_predictions': oof_predictions,
            'oof_mask': oof_mask,
            'total_folds': len(fold_results),
            'avg_fold_ic': np.mean([f['fold_ic'] for f in fold_results]),
            'avg_daily_ic': np.mean([f['avg_daily_ic'] for f in fold_results])
        }
        
        logger.info(f"✅ Walk-Forward Results:")
        logger.info(f"   Folds: {len(fold_results)}")
        logger.info(f"   OOF IC: {oof_ic:.4f} ({oof_ic*10000:.1f} bps)")
        logger.info(f"   Newey-West t-stat: {newey_west_tstat:.2f}")
        logger.info(f"   Avg Daily IC: {oof_results['avg_daily_ic']:.4f}")
        
        return oof_results
    
    def _train_lambdarank_model(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> lgb.Booster:
        """Train LightGBM LambdaRank model"""
        
        # Convert groups to group sizes for LightGBM
        unique_groups, group_counts = np.unique(groups, return_counts=True)
        group_sizes = group_counts.tolist()
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X, label=y, group=group_sizes)
        
        # LambdaRank parameters (research-backed)
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [5, 10],
            'max_depth': self.model_config['regularization']['max_depth'],
            'num_leaves': self.model_config['regularization']['num_leaves'],
            'min_data_in_leaf': self.model_config['regularization']['min_data_in_leaf'],
            'feature_fraction': self.model_config['regularization']['feature_fraction'],
            'bagging_fraction': self.model_config['regularization']['bagging_fraction'],
            'lambda_l1': self.model_config['regularization']['lambda_l1'],
            'lambda_l2': self.model_config['regularization']['lambda_l2'],
            'learning_rate': 0.05,
            'verbose': -1,
            'random_state': 42,
            'force_col_wise': True
        }
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            callbacks=[lgb.early_stopping(25), lgb.log_evaluation(0)]
        )
        
        return model
    
    def _train_final_model(self, features: np.ndarray, targets: np.ndarray, groups: np.ndarray) -> lgb.Booster:
        """Train final model on all available data"""
        
        logger.info("🎯 Training final model...")
        
        final_model = self._train_lambdarank_model(features, targets, groups)
        
        logger.info("✅ Final model trained")
        return final_model
    
    def _evaluate_model_performance(self, oof_results: Dict, data: pd.DataFrame) -> Dict[str, any]:
        """Comprehensive model evaluation"""
        
        logger.info("📊 Evaluating model performance...")
        
        # Basic metrics
        oof_ic = oof_results['oof_ic']
        newey_west_tstat = oof_results['newey_west_tstat']
        avg_daily_ic = oof_results['avg_daily_ic']
        
        # IC stability
        fold_ics = [f['fold_ic'] for f in oof_results['fold_results']]
        ic_stability = np.std(fold_ics) if fold_ics else 0
        
        # Hit rates in top/bottom deciles
        oof_mask = oof_results['oof_mask']
        oof_predictions = oof_results['oof_predictions']
        
        if oof_mask.sum() > 0:
            oof_data = data[oof_mask].copy()
            oof_data['predictions'] = oof_predictions[oof_mask]
            
            # Calculate decile hit rates
            hit_rates = self._calculate_decile_hit_rates(oof_data)
        else:
            hit_rates = {}
        
        # Turnover analysis
        turnover_stats = self._analyze_turnover(oof_data) if oof_mask.sum() > 0 else {}
        
        evaluation = {
            'oof_ic': oof_ic,
            'oof_ic_bps': oof_ic * 10000,
            'newey_west_tstat': newey_west_tstat,
            'avg_daily_ic': avg_daily_ic,
            'ic_stability': ic_stability,
            'hit_rates': hit_rates,
            'turnover_stats': turnover_stats,
            'total_oof_samples': oof_mask.sum(),
            'evaluation_period': {
                'start': str(data[oof_mask]['Date'].min()) if oof_mask.sum() > 0 else None,
                'end': str(data[oof_mask]['Date'].max()) if oof_mask.sum() > 0 else None,
                'days': len(data[oof_mask]['Date'].unique()) if oof_mask.sum() > 0 else 0
            }
        }
        
        logger.info(f"📊 Performance Evaluation:")
        logger.info(f"   OOF IC: {oof_ic:.4f} ({oof_ic*10000:.1f} bps)")
        logger.info(f"   Daily IC: {avg_daily_ic:.4f}")
        logger.info(f"   IC Stability (std): {ic_stability:.4f}")
        logger.info(f"   Newey-West t-stat: {newey_west_tstat:.2f}")
        
        return evaluation
    
    def _calculate_decile_hit_rates(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate hit rates in top/bottom deciles"""
        
        hit_rates = {}
        
        for date in data['Date'].unique():
            date_data = data[data['Date'] == date].copy()
            
            if len(date_data) < 20:  # Need minimum stocks
                continue
            
            # Rank by predictions
            date_data['pred_rank'] = date_data['predictions'].rank(pct=True)
            
            # Define deciles
            top_decile = date_data['pred_rank'] >= 0.9
            bottom_decile = date_data['pred_rank'] <= 0.1
            
            # Calculate hit rates (positive returns in top decile, negative in bottom)
            if top_decile.sum() > 0:
                top_hit_rate = (date_data.loc[top_decile, 'residual_return_target'] > 0).mean()
                hit_rates[f'{date}_top_decile'] = top_hit_rate
            
            if bottom_decile.sum() > 0:
                bottom_hit_rate = (date_data.loc[bottom_decile, 'residual_return_target'] < 0).mean()
                hit_rates[f'{date}_bottom_decile'] = bottom_hit_rate
        
        # Aggregate hit rates
        top_decile_rates = [v for k, v in hit_rates.items() if 'top_decile' in k]
        bottom_decile_rates = [v for k, v in hit_rates.items() if 'bottom_decile' in k]
        
        return {
            'avg_top_decile_hit_rate': np.mean(top_decile_rates) if top_decile_rates else 0,
            'avg_bottom_decile_hit_rate': np.mean(bottom_decile_rates) if bottom_decile_rates else 0,
            'top_decile_consistency': np.std(top_decile_rates) if top_decile_rates else 0,
            'bottom_decile_consistency': np.std(bottom_decile_rates) if bottom_decile_rates else 0
        }
    
    def _analyze_turnover(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze prediction turnover"""
        
        if len(data['Date'].unique()) < 2:
            return {}
        
        dates = sorted(data['Date'].unique())
        turnovers = []
        
        for i in range(1, len(dates)):
            prev_date = dates[i-1]
            curr_date = dates[i]
            
            prev_data = data[data['Date'] == prev_date].copy()
            curr_data = data[data['Date'] == curr_date].copy()
            
            # Get common stocks
            common_stocks = set(prev_data['Ticker']) & set(curr_data['Ticker'])
            
            if len(common_stocks) < 10:
                continue
            
            # Calculate rank correlation (stability)
            prev_ranks = prev_data.set_index('Ticker')['predictions'].rank()
            curr_ranks = curr_data.set_index('Ticker')['predictions'].rank()
            
            common_prev = prev_ranks.loc[list(common_stocks)]
            common_curr = curr_ranks.loc[list(common_stocks)]
            
            rank_corr = common_prev.corr(common_curr)
            turnover = 1 - rank_corr if not np.isnan(rank_corr) else 1
            turnovers.append(turnover)
        
        return {
            'avg_daily_turnover': np.mean(turnovers) if turnovers else 0,
            'turnover_stability': np.std(turnovers) if turnovers else 0,
            'min_turnover': np.min(turnovers) if turnovers else 0,
            'max_turnover': np.max(turnovers) if turnovers else 0
        }
    
    def _run_robustness_tests(self, features: np.ndarray, targets: np.ndarray, 
                            groups: np.ndarray, model: lgb.Booster) -> Dict[str, any]:
        """Run robustness tests (shuffle, permutation)"""
        
        logger.info("🧪 Running robustness tests...")
        
        # Baseline predictions
        baseline_predictions = model.predict(features)
        baseline_ic = spearmanr(targets, baseline_predictions)[0]
        baseline_ic = 0 if np.isnan(baseline_ic) else baseline_ic
        
        # Shuffle test - IC should collapse
        np.random.seed(42)
        shuffled_targets = np.random.permutation(targets)
        shuffle_ic = spearmanr(baseline_predictions, shuffled_targets)[0]
        shuffle_ic = 0 if np.isnan(shuffle_ic) else shuffle_ic
        
        # Feature permutation test
        n_features = features.shape[1]
        permutation_results = []
        
        for i in range(min(5, n_features)):  # Test first 5 features
            permuted_features = features.copy()
            permuted_features[:, i] = np.random.permutation(permuted_features[:, i])
            
            # Retrain model with permuted feature
            permuted_model = self._train_lambdarank_model(permuted_features, targets, groups)
            permuted_predictions = permuted_model.predict(permuted_features)
            permuted_ic = spearmanr(targets, permuted_predictions)[0]
            permuted_ic = 0 if np.isnan(permuted_ic) else permuted_ic
            
            ic_drop = baseline_ic - permuted_ic
            
            permutation_results.append({
                'feature_idx': i,
                'ic_drop': ic_drop,
                'permuted_ic': permuted_ic
            })
        
        robustness = {
            'baseline_ic': baseline_ic,
            'shuffle_ic': shuffle_ic,
            'permutation_results': permutation_results,
            'shuffle_test_pass': abs(shuffle_ic) < 0.01,  # IC should collapse
            'permutation_test_pass': any(p['ic_drop'] > 0.002 for p in permutation_results)  # Features should matter
        }
        
        logger.info(f"🧪 Robustness Results:")
        logger.info(f"   Baseline IC: {baseline_ic:.4f}")
        logger.info(f"   Shuffle IC: {shuffle_ic:.4f} ({'✅' if robustness['shuffle_test_pass'] else '❌'})")
        logger.info(f"   Max Feature Drop: {max([p['ic_drop'] for p in permutation_results]):.4f}")
        
        return robustness
    
    def _check_production_gates(self, evaluation: Dict, robustness: Dict) -> bool:
        """Check if model passes production gates"""
        
        logger.info("🚪 Checking production gates...")
        
        gates = self.validation_config['gates']
        
        # Gate checks
        ic_gate = evaluation['oof_ic'] >= gates['min_oos_ic']
        tstat_gate = evaluation['newey_west_tstat'] >= gates['min_newey_west_tstat']
        shuffle_gate = robustness['shuffle_test_pass']
        permutation_gate = robustness['permutation_test_pass']
        
        gates_status = {
            'oos_ic_gate': ic_gate,
            'newey_west_gate': tstat_gate,
            'shuffle_gate': shuffle_gate,
            'permutation_gate': permutation_gate
        }
        
        all_pass = all(gates_status.values())
        
        logger.info("🚪 Production Gates:")
        for gate, passed in gates_status.items():
            logger.info(f"   {'✅' if passed else '❌'} {gate}: {'PASS' if passed else 'FAIL'}")
        
        logger.info(f"🏆 Overall: {'✅ ALL GATES PASSED' if all_pass else '❌ GATES FAILED'}")
        
        return all_pass
    
    def _save_model_artifacts(self, model: lgb.Booster, oof_results: Dict, 
                            evaluation: Dict, robustness: Dict):
        """Save all model artifacts"""
        
        logger.info("💾 Saving model artifacts...")
        
        # Ensure directories exist
        (self.artifacts_dir / "production_models").mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save model
        model_path = self.artifacts_dir / "production_models" / f"lambdarank_{timestamp}.txt"
        model.save_model(str(model_path))

        # Save scaler
        scaler_path = None
        if hasattr(self, 'scaler') and self.scaler is not None:
            scaler_path = self.artifacts_dir / "production_models" / f"scaler_{timestamp}.pkl"
            joblib.dump(self.scaler, scaler_path)

        # Save OOF predictions
        oof_path = self.artifacts_dir / "production_models" / f"oof_predictions_{timestamp}.npz"
        np.savez(oof_path,
                predictions=oof_results['oof_predictions'],
                mask=oof_results['oof_mask'])
        
        # Save comprehensive results
        results = {
            'model_type': 'production_lambdarank',
            'timestamp': timestamp,
            'model_config': self.model_config,
            'validation_config': self.validation_config,
            'oof_results': {k: v for k, v in oof_results.items() if k not in ['oof_predictions', 'oof_mask']},
            'evaluation': evaluation,
            'robustness': robustness,
            'production_ready': self._check_production_gates(evaluation, robustness)
        }
        
        results_path = self.artifacts_dir / "production_models" / f"results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"✅ Model artifacts saved:")
        logger.info(f"   Model: {model_path}")
        if scaler_path is not None:
            logger.info(f"   Scaler: {scaler_path}")
        logger.info(f"   OOF: {oof_path}")
        logger.info(f"   Results: {results_path}")

def main():
    """Test production model agent"""
    
    config_path = Path(__file__).parent.parent / "config" / "production_config.json"
    
    if not config_path.exists():
        print("❌ Production config not found")
        return False
    
    agent = ProductionModelAgent(str(config_path))
    result = agent.train_production_model()
    
    if result['success']:
        gates_passed = result['gates_passed']
        oof_ic = result['evaluation']['oof_ic']
        print(f"✅ Model training completed")
        print(f"🎯 OOF IC: {oof_ic:.4f} ({oof_ic*10000:.1f} bps)")
        print(f"🚪 Production Gates: {'✅ PASSED' if gates_passed else '❌ FAILED'}")
        
        if gates_passed:
            print("🚀 Model ready for production deployment")
        else:
            print("⚠️ Model needs improvement before deployment")
    else:
        print("❌ Model training failed")
        print(f"Reason: {result.get('reason', 'Unknown error')}")
    
    return result['success']

if __name__ == "__main__":
    main()