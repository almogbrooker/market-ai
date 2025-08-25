#!/usr/bin/env python3
"""
FINAL INSTITUTIONAL CHECKLIST
==============================
Comprehensive institutional validation for the organized trading system
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

class FinalInstitutionalChecklist:
    """Complete institutional checklist for organized system"""
    
    def __init__(self):
        print("🏛️ FINAL INSTITUTIONAL CHECKLIST")
        print("=" * 60)
        
        # Organized paths
        self.base_dir = Path("../artifacts")
        self.data_dir = self.base_dir
        self.models_dir = self.base_dir / "models"
        self.processed_dir = self.base_dir / "processed"
        self.validation_dir = self.base_dir / "validation"
        
        # Results tracking
        self.passed_checks = []
        self.warnings = []
        self.critical_failures = []
        
        # Institutional thresholds
        self.thresholds = {
            'min_ic': 0.005,
            'max_ic': 0.080,
            'psi_threshold': 0.25,
            'min_samples': 1000,
            'max_missing_pct': 20.0,
            'universe_size': 24
        }
    
    def section_1_system_organization(self) -> dict:
        """Validate system organization and paths"""
        print("\n📁 SECTION 1: SYSTEM ORGANIZATION")
        print("-" * 50)
        
        # Check directory structure
        required_dirs = {
            'base': self.base_dir,
            'models': self.models_dir,
            'processed': self.processed_dir,
            'validation': self.validation_dir
        }
        
        for dir_name, dir_path in required_dirs.items():
            if dir_path.exists():
                print(f"   ✅ {dir_name.title()} directory: {dir_path}")
                self.passed_checks.append(f"{dir_name.title()} directory exists")
            else:
                print(f"   ❌ {dir_name.title()} directory missing: {dir_path}")
                self.critical_failures.append(f"{dir_name.title()} directory missing")
        
        # Check data files in correct locations
        raw_data_file = self.data_dir / "ds_train.parquet"
        processed_data_file = self.processed_dir / "train_institutional.parquet"
        feature_config_file = self.processed_dir / "feature_config.json"
        
        data_files = {
            'Raw training data': raw_data_file,
            'Processed training data': processed_data_file,
            'Feature configuration': feature_config_file
        }
        
        for file_name, file_path in data_files.items():
            if file_path.exists():
                print(f"   ✅ {file_name}: {file_path}")
                self.passed_checks.append(f"{file_name} in correct location")
            else:
                print(f"   ❌ {file_name} missing: {file_path}")
                self.critical_failures.append(f"{file_name} missing")
        
        return {
            'directories_exist': all(d.exists() for d in required_dirs.values()),
            'data_files_exist': all(f.exists() for f in data_files.values())
        }
    
    def section_2_data_validation(self) -> dict:
        """Validate data quality and structure"""
        print("\n📊 SECTION 2: DATA VALIDATION")
        print("-" * 50)
        
        # Load raw data
        try:
            raw_data = pd.read_parquet(self.data_dir / "ds_train.parquet")
            raw_data['Date'] = pd.to_datetime(raw_data['Date'])
            
            print(f"   📄 Raw data: {len(raw_data):,} rows, {len(raw_data.columns)} columns")
            
            # Check universe size
            actual_tickers = raw_data['Ticker'].nunique()
            if actual_tickers == self.thresholds['universe_size']:
                print(f"   ✅ Universe size: {actual_tickers} tickers (expected: {self.thresholds['universe_size']})")
                self.passed_checks.append("Correct universe size")
            else:
                print(f"   ⚠️ Universe size: {actual_tickers} tickers (expected: {self.thresholds['universe_size']})")
                self.warnings.append(f"Universe size mismatch: {actual_tickers}")
            
            # Check date range
            date_range_days = (raw_data['Date'].max() - raw_data['Date'].min()).days
            print(f"   📅 Date range: {raw_data['Date'].min()} to {raw_data['Date'].max()} ({date_range_days} days)")
            
            if date_range_days >= 500:  # At least ~2 years
                print(f"   ✅ Adequate historical data")
                self.passed_checks.append("Adequate historical data")
            else:
                print(f"   ⚠️ Limited historical data: {date_range_days} days")
                self.warnings.append(f"Limited historical data: {date_range_days} days")
            
            # Check target column
            if 'target_1d' in raw_data.columns:
                target = raw_data['target_1d'].dropna()
                target_mean = target.mean()
                target_std = target.std()
                
                print(f"   🎯 Target stats: mean={target_mean:.6f}, std={target_std:.4f}")
                
                # Realistic daily returns check
                if 0.01 <= target_std <= 0.05:  # 1-5% daily vol
                    print(f"   ✅ Target volatility realistic for daily equity returns")
                    self.passed_checks.append("Realistic target distribution")
                else:
                    print(f"   ⚠️ Target volatility: {target_std:.4f}")
                    self.warnings.append(f"Unusual target volatility: {target_std:.4f}")
            
        except Exception as e:
            print(f"   ❌ Raw data validation failed: {str(e)}")
            self.critical_failures.append(f"Raw data validation failed: {str(e)}")
            return {'status': 'FAILED'}
        
        # Load processed data
        try:
            processed_data = pd.read_parquet(self.processed_dir / "train_institutional.parquet")
            print(f"   📊 Processed data: {len(processed_data):,} rows, {len(processed_data.columns)} columns")
            
            retention_rate = len(processed_data) / len(raw_data) * 100
            print(f"   🔧 Data retention: {retention_rate:.1f}%")
            
            if retention_rate >= 50:  # At least 50% retention
                print(f"   ✅ Adequate data retention")
                self.passed_checks.append("Adequate data retention")
            else:
                print(f"   ⚠️ Low data retention: {retention_rate:.1f}%")
                self.warnings.append(f"Low data retention: {retention_rate:.1f}%")
            
        except Exception as e:
            print(f"   ❌ Processed data validation failed: {str(e)}")
            self.critical_failures.append(f"Processed data validation failed: {str(e)}")
        
        # Check feature configuration
        try:
            with open(self.processed_dir / "feature_config.json", 'r') as f:
                feature_config = json.load(f)
            
            selected_features = feature_config.get('selected_features', [])
            print(f"   🎯 Features configured: {len(selected_features)}")
            
            if 10 <= len(selected_features) <= 30:  # Reasonable feature count
                print(f"   ✅ Appropriate feature count")
                self.passed_checks.append("Appropriate feature count")
            else:
                print(f"   ⚠️ Feature count: {len(selected_features)}")
                self.warnings.append(f"Feature count: {len(selected_features)}")
                
        except Exception as e:
            print(f"   ❌ Feature config validation failed: {str(e)}")
            self.critical_failures.append(f"Feature config validation failed: {str(e)}")
        
        return {'status': 'COMPLETED'}
    
    def section_3_model_validation(self) -> dict:
        """Validate trained models"""
        print("\n🎯 SECTION 3: MODEL VALIDATION")
        print("-" * 50)
        
        # Find all models
        model_dirs = [d for d in self.models_dir.iterdir() if d.is_dir()]
        
        if not model_dirs:
            print(f"   ❌ No models found in {self.models_dir}")
            self.critical_failures.append("No models found")
            return {'status': 'FAILED'}
        
        print(f"   📊 Found {len(model_dirs)} model(s)")
        
        approved_models = []
        
        for model_dir in model_dirs:
            model_name = model_dir.name
            print(f"\n   🔍 Validating model: {model_name}")
            
            # Check model card
            model_card_path = model_dir / "model_card.json"
            if not model_card_path.exists():
                print(f"      ❌ Model card missing")
                self.critical_failures.append(f"Model card missing: {model_name}")
                continue
            
            try:
                with open(model_card_path, 'r') as f:
                    model_card = json.load(f)
                
                # Extract performance metrics
                validation_ic = model_card.get('performance', {}).get('validation_ic', 0.0)
                model_type = model_card.get('model_info', {}).get('model_type', 'unknown')
                institutional_approved = model_card.get('deployment_ready', False)
                
                print(f"      📊 Model type: {model_type}")
                print(f"      📈 Validation IC: {validation_ic:.4f}")
                print(f"      🏛️ Institutional approved: {'✅' if institutional_approved else '❌'}")
                
                # Validate IC against thresholds
                if self.thresholds['min_ic'] <= abs(validation_ic) <= self.thresholds['max_ic']:
                    print(f"      ✅ IC within institutional range")
                    self.passed_checks.append(f"Model {model_name} IC acceptable")
                elif abs(validation_ic) < self.thresholds['min_ic']:
                    print(f"      ⚠️ IC below minimum threshold")
                    self.warnings.append(f"Model {model_name} low IC: {validation_ic:.4f}")
                else:
                    print(f"      ⚠️ IC above maximum threshold (potential overfitting)")
                    self.warnings.append(f"Model {model_name} high IC: {validation_ic:.4f}")
                
                # Check model files exist
                if model_type in ['ridge', 'lasso']:
                    model_file = model_dir / "model.pkl"
                    scaler_file = model_dir / "scaler.pkl"
                    features_file = model_dir / "features.json"
                    
                    required_files = [model_file, scaler_file, features_file]
                    missing_files = [f for f in required_files if not f.exists()]
                    
                    if missing_files:
                        print(f"      ❌ Missing files: {[f.name for f in missing_files]}")
                        self.critical_failures.append(f"Model {model_name} missing files")
                    else:
                        print(f"      ✅ All required files present")
                        self.passed_checks.append(f"Model {model_name} files complete")
                
                # Test model loading
                try:
                    model = joblib.load(model_dir / "model.pkl")
                    scaler = joblib.load(model_dir / "scaler.pkl")
                    
                    with open(model_dir / "features.json", 'r') as f:
                        features = json.load(f)
                    
                    # Test prediction
                    test_input = np.random.randn(1, len(features))
                    test_scaled = scaler.transform(test_input)
                    test_pred = model.predict(test_scaled)
                    
                    print(f"      ✅ Model loading and prediction test passed")
                    self.passed_checks.append(f"Model {model_name} loads correctly")
                    
                    if institutional_approved:
                        approved_models.append({
                            'name': model_name,
                            'ic': validation_ic,
                            'type': model_type
                        })
                    
                except Exception as e:
                    print(f"      ❌ Model loading failed: {str(e)}")
                    self.critical_failures.append(f"Model {model_name} loading failed")
                
            except Exception as e:
                print(f"      ❌ Model validation failed: {str(e)}")
                self.critical_failures.append(f"Model {model_name} validation failed")
        
        # Summary
        print(f"\n   📊 MODEL VALIDATION SUMMARY:")
        print(f"      Total models: {len(model_dirs)}")
        print(f"      Approved models: {len(approved_models)}")
        
        if approved_models:
            best_model = max(approved_models, key=lambda x: abs(x['ic']))
            print(f"      Best approved model: {best_model['name']} (IC: {best_model['ic']:.4f})")
            self.passed_checks.append("At least one approved model available")
        else:
            print(f"      ❌ No approved models available")
            self.critical_failures.append("No approved models available")
        
        return {
            'status': 'COMPLETED',
            'total_models': len(model_dirs),
            'approved_models': len(approved_models),
            'best_model': approved_models[0] if approved_models else None
        }
    
    def section_4_temporal_alignment_validation(self) -> dict:
        """Validate temporal alignment and leakage prevention"""
        print("\n⏰ SECTION 4: TEMPORAL ALIGNMENT VALIDATION")
        print("-" * 50)
        
        try:
            # Load processed data for alignment testing
            processed_data = pd.read_parquet(self.processed_dir / "train_institutional.parquet")
            processed_data['Date'] = pd.to_datetime(processed_data['Date'])
            
            # Check for proper T-1 → T+1 features
            t1_features = [col for col in processed_data.columns if col.endswith('_t1')]
            
            if 'target_forward' in processed_data.columns:
                print(f"   ✅ Forward target found: target_forward")
                print(f"   📊 T-1 features found: {len(t1_features)}")
                self.passed_checks.append("Proper temporal alignment structure")
            else:
                print(f"   ❌ Forward target not found")
                self.critical_failures.append("Forward target missing")
                return {'status': 'FAILED'}
            
            # Simple leakage test - shuffle test
            if len(t1_features) > 0 and len(processed_data) > 1000:
                feature_mean = processed_data[t1_features].fillna(0).mean(axis=1)
                target = processed_data['target_forward']
                
                # Normal correlation
                normal_ic, _ = spearmanr(feature_mean, target)
                
                # Shuffled target correlation (should be near zero)
                shuffled_target = target.sample(frac=1, random_state=42).reset_index(drop=True)
                shuffle_ic, _ = spearmanr(feature_mean, shuffled_target)
                
                print(f"   📊 Normal IC: {normal_ic:.4f}")
                print(f"   🎲 Shuffle IC: {shuffle_ic:.4f}")
                
                if abs(shuffle_ic) < 0.05:  # Should be near zero
                    print(f"   ✅ Shuffle test passed (no obvious leakage)")
                    self.passed_checks.append("Shuffle test passed")
                else:
                    print(f"   ⚠️ Shuffle test concern: {shuffle_ic:.4f}")
                    self.warnings.append(f"High shuffle IC: {shuffle_ic:.4f}")
                
                # Time shift test
                shifted_target = processed_data.groupby('Ticker')['target_forward'].shift(-5)  # 5-day shift
                valid_mask = ~shifted_target.isnull()
                
                if valid_mask.sum() > 100:
                    shift_ic, _ = spearmanr(feature_mean[valid_mask], shifted_target[valid_mask])
                    print(f"   ⏩ +5d shift IC: {shift_ic:.4f}")
                    
                    if abs(shift_ic) < 0.03:  # Should degrade
                        print(f"   ✅ Time shift test passed")
                        self.passed_checks.append("Time shift test passed")
                    else:
                        print(f"   ⚠️ High future shift IC: {shift_ic:.4f}")
                        self.warnings.append(f"High future shift IC: {shift_ic:.4f}")
            
        except Exception as e:
            print(f"   ❌ Temporal alignment validation failed: {str(e)}")
            self.critical_failures.append(f"Temporal alignment validation failed: {str(e)}")
            return {'status': 'FAILED'}
        
        return {'status': 'COMPLETED'}
    
    def section_5_institutional_compliance(self) -> dict:
        """Final institutional compliance check"""
        print("\n🏛️ SECTION 5: INSTITUTIONAL COMPLIANCE")
        print("-" * 50)
        
        # Compliance scoring
        total_checks = len(self.passed_checks) + len(self.warnings) + len(self.critical_failures)
        pass_rate = len(self.passed_checks) / total_checks * 100 if total_checks > 0 else 0
        
        print(f"   📊 Total checks performed: {total_checks}")
        print(f"   ✅ Passed checks: {len(self.passed_checks)} ({pass_rate:.1f}%)")
        print(f"   ⚠️ Warnings: {len(self.warnings)}")
        print(f"   ❌ Critical failures: {len(self.critical_failures)}")
        
        # Determine compliance level
        if len(self.critical_failures) == 0:
            if len(self.warnings) <= 3:
                compliance_level = "INSTITUTIONAL APPROVED"
                compliance_status = "✅"
            else:
                compliance_level = "CONDITIONAL APPROVAL"
                compliance_status = "🟡"
        else:
            compliance_level = "NOT APPROVED"
            compliance_status = "❌"
        
        print(f"\n   {compliance_status} COMPLIANCE LEVEL: {compliance_level}")
        
        # Print details if issues exist
        if self.critical_failures:
            print(f"\n   ❌ CRITICAL FAILURES:")
            for failure in self.critical_failures:
                print(f"      • {failure}")
        
        if self.warnings:
            print(f"\n   ⚠️ WARNINGS:")
            for warning in self.warnings:
                print(f"      • {warning}")
        
        compliance_result = {
            'level': compliance_level,
            'pass_rate': pass_rate,
            'critical_failures': len(self.critical_failures),
            'warnings': len(self.warnings),
            'approved': len(self.critical_failures) == 0
        }
        
        return compliance_result
    
    def generate_final_report(self) -> dict:
        """Generate comprehensive final institutional report"""
        
        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.validation_dir / f"institutional_checklist_{timestamp}.json"
        
        final_report = {
            'timestamp': datetime.now().isoformat(),
            'checklist_type': 'institutional_equity_trading',
            'system_paths': {
                'base_dir': str(self.base_dir),
                'models_dir': str(self.models_dir),
                'processed_dir': str(self.processed_dir),
                'validation_dir': str(self.validation_dir)
            },
            'validation_summary': {
                'total_checks': len(self.passed_checks) + len(self.warnings) + len(self.critical_failures),
                'passed_checks': len(self.passed_checks),
                'warnings': len(self.warnings),
                'critical_failures': len(self.critical_failures)
            },
            'details': {
                'passed_checks': self.passed_checks,
                'warnings': self.warnings,
                'critical_failures': self.critical_failures
            }
        }
        
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved: {report_file}")
        
        return final_report
    
    def run_complete_checklist(self) -> dict:
        """Run complete institutional checklist"""
        print("🚀 RUNNING COMPLETE INSTITUTIONAL CHECKLIST")
        print("=" * 80)
        
        try:
            # Run all sections
            section_1 = self.section_1_system_organization()
            section_2 = self.section_2_data_validation()
            section_3 = self.section_3_model_validation()
            section_4 = self.section_4_temporal_alignment_validation()
            section_5 = self.section_5_institutional_compliance()
            
            # Generate final report
            final_report = self.generate_final_report()
            final_report['compliance'] = section_5
            final_report['sections'] = {
                'system_organization': section_1,
                'data_validation': section_2,
                'model_validation': section_3,
                'temporal_alignment': section_4,
                'institutional_compliance': section_5
            }
            
            print(f"\n🎉 INSTITUTIONAL CHECKLIST COMPLETE")
            print("=" * 60)
            
            if section_5['approved']:
                print(f"🟢 FINAL STATUS: {section_5['level']}")
                print(f"✅ SYSTEM READY FOR INSTITUTIONAL DEPLOYMENT")
            else:
                print(f"🔴 FINAL STATUS: {section_5['level']}")
                print(f"🔧 ADDRESS CRITICAL ISSUES BEFORE DEPLOYMENT")
            
            return final_report
            
        except Exception as e:
            print(f"❌ Checklist execution failed: {str(e)}")
            self.critical_failures.append(f"Checklist execution failed: {str(e)}")
            return {'status': 'FAILED', 'error': str(e)}

def main():
    """Main checklist execution"""
    checklist = FinalInstitutionalChecklist()
    results = checklist.run_complete_checklist()
    
    if results.get('compliance', {}).get('approved', False):
        print(f"\n🎯 INSTITUTIONAL CHECKLIST: ✅ APPROVED")
        return True
    else:
        print(f"\n🚨 INSTITUTIONAL CHECKLIST: ❌ NOT APPROVED")
        return False

if __name__ == "__main__":
    success = main()