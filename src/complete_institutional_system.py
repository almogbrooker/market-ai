#!/usr/bin/env python3
"""
COMPLETE INSTITUTIONAL SYSTEM
==============================
Integrated system with comprehensive data validation for each model
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, kstest
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

class CompleteInstitutionalSystem:
    """Complete institutional system with integrated validation"""
    
    def __init__(self):
        print("🏛️ COMPLETE INSTITUTIONAL SYSTEM")
        print("=" * 70)
        
        # Directory structure
        self.base_dir = Path("../artifacts")
        self.models_dir = self.base_dir / "models"
        self.processed_dir = self.base_dir / "processed"
        self.validation_dir = self.base_dir / "validation"
        
        # Institutional thresholds
        self.thresholds = {
            'min_ic': 0.005,
            'max_ic': 0.080,
            'min_samples': 1000,
            'max_missing_pct': 5.0,
            'psi_threshold': 0.25,
            'universe_size': 24,
            'min_feature_variance': 1e-8
        }
        
        # Results tracking
        self.validation_results = {
            'data_checks': {},
            'model_results': {},
            'institutional_status': {}
        }
    
    def comprehensive_data_validation(self) -> dict:
        """Comprehensive data validation for all models"""
        print("\n📊 COMPREHENSIVE DATA VALIDATION")
        print("=" * 50)
        
        # Load raw data
        raw_data = pd.read_parquet(self.base_dir / "ds_train.parquet")
        raw_data['Date'] = pd.to_datetime(raw_data['Date'])
        
        # Load processed data
        processed_data = pd.read_parquet(self.processed_dir / "train_institutional.parquet")
        processed_data['Date'] = pd.to_datetime(processed_data['Date'])
        
        validation_report = {}
        
        # 1. Raw Data Validation
        print("\n   🔍 RAW DATA VALIDATION:")
        raw_validation = self._validate_raw_data(raw_data)
        validation_report['raw_data'] = raw_validation
        
        # 2. Processed Data Validation  
        print("\n   🔧 PROCESSED DATA VALIDATION:")
        processed_validation = self._validate_processed_data(processed_data)
        validation_report['processed_data'] = processed_validation
        
        # 3. Feature Quality Assessment
        print("\n   📈 FEATURE QUALITY ASSESSMENT:")
        feature_cols = [col for col in processed_data.columns if col.endswith('_t1')]
        feature_validation = self._validate_features_for_models(processed_data, feature_cols)
        validation_report['feature_quality'] = feature_validation
        
        # 4. Target Distribution Analysis
        print("\n   🎯 TARGET DISTRIBUTION ANALYSIS:")
        target_validation = self._validate_target_distribution(processed_data)
        validation_report['target_analysis'] = target_validation
        
        # 5. Temporal Integrity Check
        print("\n   ⏰ TEMPORAL INTEGRITY CHECK:")
        temporal_validation = self._validate_temporal_integrity(processed_data)
        validation_report['temporal_integrity'] = temporal_validation
        
        # Overall data status
        all_checks = [
            raw_validation['status'] == 'PASS',
            processed_validation['status'] == 'PASS', 
            feature_validation['status'] == 'PASS',
            target_validation['status'] == 'PASS',
            temporal_validation['status'] == 'PASS'
        ]
        
        validation_report['overall_status'] = 'PASS' if all(all_checks) else 'WARNING'
        validation_report['ready_for_modeling'] = sum(all_checks) >= 4  # Allow one warning
        
        print(f"\n   📋 DATA VALIDATION SUMMARY:")
        print(f"      Raw data: {'✅' if raw_validation['status'] == 'PASS' else '⚠️'}")
        print(f"      Processed data: {'✅' if processed_validation['status'] == 'PASS' else '⚠️'}")
        print(f"      Feature quality: {'✅' if feature_validation['status'] == 'PASS' else '⚠️'}")
        print(f"      Target analysis: {'✅' if target_validation['status'] == 'PASS' else '⚠️'}")
        print(f"      Temporal integrity: {'✅' if temporal_validation['status'] == 'PASS' else '⚠️'}")
        print(f"      Overall: {'✅ READY FOR MODELING' if validation_report['ready_for_modeling'] else '❌ NOT READY'}")
        
        self.validation_results['data_checks'] = validation_report
        return validation_report
    
    def _validate_raw_data(self, df: pd.DataFrame) -> dict:
        """Validate raw data quality"""
        issues = []
        
        # Universe size check
        actual_tickers = df['Ticker'].nunique()
        if actual_tickers != self.thresholds['universe_size']:
            issues.append(f"Universe size: {actual_tickers} (expected {self.thresholds['universe_size']})")
        
        # Date range check
        date_span = (df['Date'].max() - df['Date'].min()).days
        if date_span < 500:
            issues.append(f"Limited date range: {date_span} days")
        
        # Target column validation
        if 'target_1d' not in df.columns:
            issues.append("Target column missing")
        else:
            target = df['target_1d'].dropna()
            if not (0.01 <= target.std() <= 0.05):
                issues.append(f"Unusual target volatility: {target.std():.4f}")
        
        print(f"      Tickers: {actual_tickers}, Date span: {date_span} days")
        if issues:
            print(f"      Issues: {len(issues)}")
            for issue in issues:
                print(f"        • {issue}")
        
        return {
            'status': 'PASS' if len(issues) == 0 else 'WARNING',
            'issues': issues,
            'ticker_count': actual_tickers,
            'date_span_days': date_span
        }
    
    def _validate_processed_data(self, df: pd.DataFrame) -> dict:
        """Validate processed data quality"""
        issues = []
        
        # Data retention check
        raw_data_count = pd.read_parquet(self.base_dir / "ds_train.parquet").shape[0]
        retention_rate = len(df) / raw_data_count * 100
        
        if retention_rate < 50:
            issues.append(f"Low data retention: {retention_rate:.1f}%")
        
        # Missing values check
        missing_counts = df.isnull().sum()
        high_missing = missing_counts[missing_counts > len(df) * 0.05]
        
        if len(high_missing) > 0:
            issues.append(f"{len(high_missing)} columns with >5% missing")
        
        # Duplicate check
        if 'Date' in df.columns and 'Ticker' in df.columns:
            duplicates = df.duplicated(subset=['Date', 'Ticker']).sum()
            if duplicates > 0:
                issues.append(f"{duplicates} duplicate rows")
        
        print(f"      Retention: {retention_rate:.1f}%, Missing cols: {len(high_missing)}")
        
        return {
            'status': 'PASS' if len(issues) == 0 else 'WARNING',
            'issues': issues,
            'retention_rate': retention_rate,
            'high_missing_cols': len(high_missing)
        }
    
    def _validate_features_for_models(self, df: pd.DataFrame, feature_cols: list) -> dict:
        """Validate features specifically for each model type"""
        print(f"      Analyzing {len(feature_cols)} features for model suitability:")
        
        X = df[feature_cols]
        
        # Feature quality analysis
        feature_quality = {}
        problematic_features = []
        
        for col in feature_cols:
            values = X[col].dropna()
            
            # Basic quality checks
            quality = {
                'variance': values.var(),
                'missing_pct': X[col].isnull().sum() / len(X) * 100,
                'unique_values': values.nunique(),
                'has_outliers': False,
                'distribution_issues': False
            }
            
            # Variance check
            if quality['variance'] < self.thresholds['min_feature_variance']:
                quality['issues'] = quality.get('issues', []) + ['low_variance']
                problematic_features.append(col)
            
            # Missing data check
            if quality['missing_pct'] > self.thresholds['max_missing_pct']:
                quality['issues'] = quality.get('issues', []) + ['high_missing']
                problematic_features.append(col)
            
            # Outlier detection
            if len(values) > 100:
                q1, q3 = values.quantile([0.25, 0.75])
                iqr = q3 - q1
                if iqr > 0:
                    outlier_mask = (values < q1 - 5*iqr) | (values > q3 + 5*iqr)
                    outlier_pct = outlier_mask.sum() / len(values) * 100
                    
                    if outlier_pct > 5:
                        quality['has_outliers'] = True
                        quality['outlier_pct'] = outlier_pct
            
            feature_quality[col] = quality
        
        # Model-specific suitability
        ridge_suitable = len([f for f, q in feature_quality.items() if q['variance'] > 1e-8])
        lasso_suitable = len([f for f, q in feature_quality.items() 
                            if q['variance'] > 1e-6 and not q.get('has_outliers', False)])
        lgb_suitable = len([f for f, q in feature_quality.items() 
                           if q['missing_pct'] < 10 and q['unique_values'] > 10])
        
        print(f"        Ridge suitable: {ridge_suitable}/{len(feature_cols)} features")
        print(f"        Lasso suitable: {lasso_suitable}/{len(feature_cols)} features")  
        print(f"        LightGBM suitable: {lgb_suitable}/{len(feature_cols)} features")
        
        return {
            'status': 'PASS' if len(problematic_features) < len(feature_cols) * 0.3 else 'WARNING',
            'feature_quality': feature_quality,
            'problematic_features': problematic_features,
            'model_suitability': {
                'ridge': ridge_suitable,
                'lasso': lasso_suitable, 
                'lightgbm': lgb_suitable
            }
        }
    
    def _validate_target_distribution(self, df: pd.DataFrame) -> dict:
        """Validate target variable distribution"""
        target = df['target_forward'].dropna()
        
        # Distribution statistics
        stats_dict = {
            'mean': target.mean(),
            'std': target.std(),
            'skewness': target.skew(),
            'kurtosis': target.kurtosis(),
            'min': target.min(),
            'max': target.max()
        }
        
        # Distribution tests
        issues = []
        
        # Mean should be near zero for equity returns
        if abs(stats_dict['mean']) > 0.01:
            issues.append(f"High mean return: {stats_dict['mean']:.4f}")
        
        # Standard deviation should be reasonable for daily returns
        if not (0.01 <= stats_dict['std'] <= 0.05):
            issues.append(f"Unusual volatility: {stats_dict['std']:.4f}")
        
        # Extreme skewness/kurtosis
        if abs(stats_dict['skewness']) > 5:
            issues.append(f"High skewness: {stats_dict['skewness']:.2f}")
        
        if abs(stats_dict['kurtosis']) > 10:
            issues.append(f"High kurtosis: {stats_dict['kurtosis']:.2f}")
        
        print(f"      Mean: {stats_dict['mean']:.6f}, Std: {stats_dict['std']:.4f}")
        print(f"      Skewness: {stats_dict['skewness']:.2f}, Kurtosis: {stats_dict['kurtosis']:.2f}")
        
        return {
            'status': 'PASS' if len(issues) == 0 else 'WARNING',
            'issues': issues,
            'statistics': stats_dict
        }
    
    def _validate_temporal_integrity(self, df: pd.DataFrame) -> dict:
        """Validate temporal alignment and check for leakage"""
        issues = []
        
        # Check for proper temporal features
        t1_features = [col for col in df.columns if col.endswith('_t1')]
        
        if 'target_forward' not in df.columns:
            issues.append("Forward target missing")
            return {'status': 'FAIL', 'issues': issues}
        
        if len(t1_features) == 0:
            issues.append("No T-1 features found")
            return {'status': 'FAIL', 'issues': issues}
        
        # Simple leakage tests
        if len(df) > 1000 and len(t1_features) > 0:
            feature_mean = df[t1_features].fillna(0).mean(axis=1)
            target = df['target_forward']
            
            # Shuffle test
            shuffled_target = target.sample(frac=1, random_state=42).reset_index(drop=True)
            shuffle_ic, _ = spearmanr(feature_mean, shuffled_target)
            
            if abs(shuffle_ic) > 0.05:
                issues.append(f"High shuffle IC: {shuffle_ic:.4f}")
            
            # Future shift test
            if 'Ticker' in df.columns:
                shifted_target = df.groupby('Ticker')['target_forward'].shift(-5)
                valid_mask = ~shifted_target.isnull()
                
                if valid_mask.sum() > 100:
                    shift_ic, _ = spearmanr(feature_mean[valid_mask], shifted_target[valid_mask])
                    if abs(shift_ic) > 0.03:
                        issues.append(f"High future shift IC: {shift_ic:.4f}")
        
        print(f"      T-1 features: {len(t1_features)}, Leakage tests: {len(issues)} issues")
        
        return {
            'status': 'PASS' if len(issues) == 0 else 'WARNING',
            'issues': issues,
            't1_features': len(t1_features)
        }
    
    def train_and_validate_models(self, data_validation: dict) -> dict:
        """Train and validate all models with data-specific checks"""
        print("\n🎯 TRAINING & VALIDATING ALL MODELS")
        print("=" * 50)
        
        if not data_validation['ready_for_modeling']:
            print("   ❌ Data not ready for modeling")
            return {'status': 'FAILED', 'reason': 'data_not_ready'}
        
        # Load processed data
        df = pd.read_parquet(self.processed_dir / "train_institutional.parquet")
        feature_cols = [col for col in df.columns if col.endswith('_t1')]
        target_col = 'target_forward'
        
        # Prepare data with cleaning
        X = self._prepare_features_for_training(df, feature_cols)
        y = df[target_col].fillna(0)
        
        # Split data
        val_split_idx = int(len(df) * 0.7)
        X_train, X_val = X.iloc[:val_split_idx], X.iloc[val_split_idx:]
        y_train, y_val = y.iloc[:val_split_idx], y.iloc[val_split_idx:]
        
        print(f"   📊 Training: {len(X_train)} samples, Validation: {len(X_val)} samples")
        
        model_results = {}
        
        # Train each model with specific data validation
        model_results['ridge'] = self._train_ridge_with_validation(X_train, X_val, y_train, y_val, data_validation)
        model_results['lasso'] = self._train_lasso_with_validation(X_train, X_val, y_train, y_val, data_validation)
        model_results['lightgbm'] = self._train_lightgbm_with_validation(X_train, X_val, y_train, y_val, data_validation)
        
        # Model selection
        approved_models = [name for name, result in model_results.items() 
                         if result.get('institutional_approved', False)]
        
        print(f"\n   📊 MODEL TRAINING SUMMARY:")
        for name, result in model_results.items():
            status = "✅ APPROVED" if result.get('institutional_approved', False) else "❌ NOT APPROVED"
            ic = result.get('validation_ic', 'N/A')
            print(f"      {name.title()}: IC={ic:.4f if ic != 'N/A' else ic} {status}")
        
        print(f"   🏛️ Institutionally approved models: {len(approved_models)}/3")
        
        self.validation_results['model_results'] = model_results
        
        return {
            'status': 'SUCCESS',
            'model_results': model_results,
            'approved_models': approved_models,
            'total_approved': len(approved_models)
        }
    
    def _prepare_features_for_training(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """Prepare features with proper cleaning"""
        X = df[feature_cols].copy()
        
        # Remove extreme outliers (cap at 99.5th percentile)
        for col in feature_cols:
            values = X[col]
            lower_bound = values.quantile(0.005)
            upper_bound = values.quantile(0.995)
            X[col] = values.clip(lower=lower_bound, upper=upper_bound)
        
        # Fill missing values
        X = X.fillna(0)
        
        return X
    
    def _train_ridge_with_validation(self, X_train, X_val, y_train, y_val, data_validation) -> dict:
        """Train Ridge with data-specific validation"""
        print("\n      🎯 RIDGE REGRESSION:")
        
        # Ridge works well with most data
        suitable_features = data_validation['feature_quality']['model_suitability']['ridge']
        
        if suitable_features < 5:
            print(f"         ❌ Insufficient suitable features: {suitable_features}")
            return {'status': 'INSUFFICIENT_FEATURES', 'institutional_approved': False}
        
        # Scale and train
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Find best alpha
        best_alpha, best_ic = 10.0, -np.inf
        for alpha in [0.1, 1.0, 10.0, 100.0]:
            ridge = Ridge(alpha=alpha, random_state=42)
            ridge.fit(X_train_scaled, y_train)
            
            val_pred = ridge.predict(X_val_scaled)
            ic, _ = spearmanr(y_val, val_pred)
            
            if not np.isnan(ic) and abs(ic) > abs(best_ic):
                best_ic = ic
                best_alpha = alpha
        
        # Train final model
        final_ridge = Ridge(alpha=best_alpha, random_state=42)
        final_ridge.fit(X_train_scaled, y_train)
        
        val_pred = final_ridge.predict(X_val_scaled)
        val_ic, _ = spearmanr(y_val, val_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        
        # Institutional assessment
        institutional_approved = (0.005 <= abs(val_ic) <= 0.08)
        
        print(f"         Alpha: {best_alpha}, IC: {val_ic:.4f}")
        print(f"         Institutional: {'✅ APPROVED' if institutional_approved else '❌ NOT APPROVED'}")
        
        return {
            'model': final_ridge,
            'scaler': scaler,
            'validation_ic': val_ic,
            'validation_mse': val_mse,
            'best_alpha': best_alpha,
            'institutional_approved': institutional_approved,
            'status': 'SUCCESS'
        }
    
    def _train_lasso_with_validation(self, X_train, X_val, y_train, y_val, data_validation) -> dict:
        """Train Lasso/ElasticNet with data-specific validation"""
        print("\n      📊 LASSO/ELASTICNET:")
        
        suitable_features = data_validation['feature_quality']['model_suitability']['lasso']
        
        if suitable_features < 5:
            print(f"         ❌ Insufficient suitable features: {suitable_features}")
            return {'status': 'INSUFFICIENT_FEATURES', 'institutional_approved': False}
        
        # Use RobustScaler for Lasso (less sensitive to outliers)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Try ElasticNet with different ratios
        best_model, best_ic, best_params = None, -np.inf, None
        
        for l1_ratio in [0.1, 0.5, 0.9]:
            for alpha in [1e-5, 1e-4, 1e-3]:
                try:
                    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, 
                                     random_state=42, selection='random')
                    model.fit(X_train_scaled, y_train)
                    
                    selected_features = np.sum(np.abs(model.coef_) > 1e-8)
                    
                    if selected_features >= 3:
                        val_pred = model.predict(X_val_scaled)
                        ic, _ = spearmanr(y_val, val_pred)
                        
                        if not np.isnan(ic) and abs(ic) > abs(best_ic):
                            best_ic = ic
                            best_model = model
                            best_params = {'alpha': alpha, 'l1_ratio': l1_ratio, 'n_features': selected_features}
                
                except Exception:
                    continue
        
        if best_model is not None:
            institutional_approved = (0.005 <= abs(best_ic) <= 0.08)
            print(f"         Best params: {best_params}")
            print(f"         IC: {best_ic:.4f}")
            print(f"         Institutional: {'✅ APPROVED' if institutional_approved else '❌ NOT APPROVED'}")
            
            return {
                'model': best_model,
                'scaler': scaler,
                'validation_ic': best_ic,
                'best_params': best_params,
                'institutional_approved': institutional_approved,
                'status': 'SUCCESS'
            }
        else:
            print(f"         ❌ All parameter combinations failed")
            return {'status': 'FAILED', 'institutional_approved': False}
    
    def _train_lightgbm_with_validation(self, X_train, X_val, y_train, y_val, data_validation) -> dict:
        """Train LightGBM with data-specific validation"""
        print("\n      🌟 LIGHTGBM:")
        
        suitable_features = data_validation['feature_quality']['model_suitability']['lightgbm']
        
        if suitable_features < 5:
            print(f"         ❌ Insufficient suitable features: {suitable_features}")
            return {'status': 'INSUFFICIENT_FEATURES', 'institutional_approved': False}
        
        # Ultra-conservative parameters based on data characteristics
        params = {
            'objective': 'regression',
            'metric': 'l2',
            'num_leaves': 3,
            'learning_rate': 0.001,
            'feature_fraction': 0.5,
            'bagging_fraction': 0.5,
            'min_data_in_leaf': max(500, len(X_train) // 20),
            'lambda_l1': 20.0,
            'lambda_l2': 20.0,
            'max_depth': 2,
            'verbose': -1,
            'random_state': 42
        }
        
        try:
            train_data = lgb.Dataset(X_train.fillna(0), label=y_train.fillna(0))
            val_data = lgb.Dataset(X_val.fillna(0), label=y_val.fillna(0), reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=20,
                callbacks=[lgb.early_stopping(3), lgb.log_evaluation(0)]
            )
            
            val_pred = model.predict(X_val.fillna(0))
            
            # Handle NaN predictions
            if np.any(np.isnan(val_pred)):
                val_pred = np.nan_to_num(val_pred, nan=0.0)
            
            ic, _ = spearmanr(y_val, val_pred)
            
            if np.isnan(ic):
                print(f"         ❌ IC calculation failed (NaN)")
                return {'status': 'FAILED', 'institutional_approved': False}
            
            institutional_approved = (0.005 <= abs(ic) <= 0.08)
            
            print(f"         IC: {ic:.4f}, Iterations: {model.best_iteration}")
            print(f"         Institutional: {'✅ APPROVED' if institutional_approved else '❌ NOT APPROVED'}")
            
            return {
                'model': model,
                'validation_ic': ic,
                'best_iteration': model.best_iteration,
                'institutional_approved': institutional_approved,
                'status': 'SUCCESS'
            }
            
        except Exception as e:
            print(f"         ❌ Training failed: {str(e)}")
            return {'status': 'FAILED', 'institutional_approved': False, 'error': str(e)}
    
    def generate_complete_institutional_report(self) -> dict:
        """Generate complete institutional report"""
        print("\n🏛️ COMPLETE INSTITUTIONAL REPORT")
        print("=" * 70)
        
        # Run complete system validation
        data_validation = self.comprehensive_data_validation()
        model_validation = self.train_and_validate_models(data_validation)
        
        # Generate final assessment
        total_checks = 5  # Data validation sections
        passed_data_checks = sum([
            data_validation['raw_data']['status'] == 'PASS',
            data_validation['processed_data']['status'] == 'PASS',
            data_validation['feature_quality']['status'] == 'PASS',
            data_validation['target_analysis']['status'] == 'PASS',
            data_validation['temporal_integrity']['status'] == 'PASS'
        ])
        
        approved_models = model_validation.get('total_approved', 0)
        
        # Final institutional status
        if passed_data_checks >= 4 and approved_models >= 1:
            if approved_models >= 2:
                institutional_status = "INSTITUTIONAL APPROVED"
                status_icon = "🟢"
            else:
                institutional_status = "CONDITIONAL APPROVAL"
                status_icon = "🟡"
        else:
            institutional_status = "NOT APPROVED"
            status_icon = "🔴"
        
        final_report = {
            'timestamp': datetime.now().isoformat(),
            'institutional_status': institutional_status,
            'data_validation': data_validation,
            'model_validation': model_validation,
            'summary': {
                'data_checks_passed': f"{passed_data_checks}/5",
                'approved_models': f"{approved_models}/3",
                'overall_status': institutional_status
            }
        }
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.validation_dir / f"complete_institutional_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        print(f"\n{status_icon} FINAL INSTITUTIONAL STATUS: {institutional_status}")
        print(f"   📊 Data validation: {passed_data_checks}/5 checks passed")
        print(f"   🎯 Model validation: {approved_models}/3 models approved")
        print(f"   📄 Report saved: {report_file}")
        
        return final_report

def main():
    """Main system execution"""
    system = CompleteInstitutionalSystem()
    results = system.generate_complete_institutional_report()
    
    institutional_ready = results['institutional_status'] in ['INSTITUTIONAL APPROVED', 'CONDITIONAL APPROVAL']
    
    if institutional_ready:
        print(f"\n🎉 COMPLETE INSTITUTIONAL SYSTEM: ✅ APPROVED FOR DEPLOYMENT")
    else:
        print(f"\n🔧 COMPLETE INSTITUTIONAL SYSTEM: ❌ REQUIRES FIXES")
    
    return institutional_ready

if __name__ == "__main__":
    success = main()