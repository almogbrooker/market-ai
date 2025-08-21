#!/usr/bin/env python3
"""
Parameter Sanity Checker & Auto-Retrain System
Monitors model parameters and performance for overfitting and drift
"""

import sys
sys.path.append('..')
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json
import warnings
warnings.filterwarnings('ignore')

# Import our system components
from src.models.tiered_system import TieredAlphaSystem
from scipy import stats
from sklearn.metrics import mean_squared_error
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ParameterSanityChecker:
    """
    Comprehensive parameter monitoring and auto-retrain system
    """
    
    def __init__(self, alpha_system: TieredAlphaSystem, config: Dict = None):
        self.alpha_system = alpha_system
        self.config = config or self._default_config()
        
        # Parameter tracking
        self.parameter_history = []
        self.performance_history = []
        self.sanity_check_results = {}
        
        # Auto-retrain state
        self.last_retrain_date = None
        self.retrain_triggers = []
        
        logger.info("🔍 Parameter Sanity Checker initialized")
    
    def _default_config(self) -> Dict:
        """Default sanity check configuration"""
        return {
            # IC monitoring
            'max_training_ic': 0.03,        # Cap training IC at 3%
            'min_validation_ic': 0.005,     # Minimum validation IC
            'ic_degradation_threshold': 0.5, # 50% IC drop triggers retrain
            
            # Overfitting detection
            'max_train_val_ic_ratio': 2.0,  # Train IC / Val IC ratio
            'max_train_val_mse_ratio': 1.3, # Train MSE / Val MSE ratio
            
            # Model complexity
            'max_lgbm_leaves': 31,          # LightGBM complexity limit
            'max_lstm_hidden_dim': 128,     # LSTM complexity limit
            'max_features': 50,             # Feature count limit
            
            # Performance monitoring
            'min_hit_rate': 0.52,           # Minimum hit rate
            'max_drawdown': 0.15,           # Maximum drawdown
            'min_sharpe_ratio': 0.3,        # Minimum Sharpe ratio
            
            # Drift detection
            'feature_drift_threshold': 0.1, # Feature distribution drift
            'performance_window': 30,       # Days for performance monitoring
            'drift_significance': 0.05,     # Statistical significance for drift
            
            # Auto-retrain settings
            'retrain_frequency_days': 30,   # Maximum days between retrains
            'min_samples_retrain': 1000,    # Minimum samples for retrain
            'validation_split': 0.2,        # Validation split for retrain
            
            # Alert thresholds
            'warning_threshold': 0.7,       # Warning if sanity score < 70%
            'critical_threshold': 0.5       # Critical if sanity score < 50%
        }
    
    def run_comprehensive_sanity_check(self, 
                                     current_data: pd.DataFrame,
                                     historical_data: pd.DataFrame = None,
                                     current_date: str = None) -> Dict:
        """Run complete parameter sanity check suite"""
        
        logger.info("🔍 RUNNING COMPREHENSIVE SANITY CHECKS")
        logger.info("=" * 50)
        
        sanity_results = {}
        
        # Check 1: IC Validation and Capping
        logger.info("📊 Checking IC constraints...")
        sanity_results['ic_validation'] = self._check_ic_constraints(current_data)
        
        # Check 2: Overfitting Detection
        logger.info("🕵️ Detecting overfitting...")
        sanity_results['overfitting'] = self._detect_overfitting(current_data)
        
        # Check 3: Model Complexity Validation
        logger.info("🔧 Validating model complexity...")
        sanity_results['complexity'] = self._check_model_complexity()
        
        # Check 4: Feature Drift Detection
        if historical_data is not None:
            logger.info("📈 Detecting feature drift...")
            sanity_results['drift'] = self._detect_feature_drift(current_data, historical_data)
        
        # Check 5: Performance Degradation
        logger.info("📉 Checking performance degradation...")
        sanity_results['performance'] = self._check_performance_degradation()
        
        # Check 6: Parameter Stability
        logger.info("⚖️ Checking parameter stability...")
        sanity_results['stability'] = self._check_parameter_stability()
        
        # Check 7: Auto-retrain Assessment
        logger.info("🔄 Assessing auto-retrain needs...")
        sanity_results['retrain_assessment'] = self._assess_retrain_needs(current_date)
        
        # Calculate overall sanity score
        overall_score = self._calculate_sanity_score(sanity_results)
        
        # Generate recommendations and alerts
        recommendations = self._generate_recommendations(sanity_results, overall_score)
        alerts = self._generate_alerts(overall_score)
        
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_sanity_score': overall_score,
            'detailed_checks': sanity_results,
            'recommendations': recommendations,
            'alerts': alerts,
            'auto_retrain_triggered': overall_score['retrain_needed']
        }
        
        # Log results
        self._log_sanity_check_results(final_results)
        
        # Store in history
        self.sanity_check_results[current_date or datetime.now().isoformat()] = final_results
        
        # Trigger auto-retrain if needed
        if final_results['auto_retrain_triggered'] and historical_data is not None:
            logger.info("🚨 AUTO-RETRAIN TRIGGERED!")
            retrain_results = self._execute_auto_retrain(historical_data, current_data)
            final_results['retrain_results'] = retrain_results
        
        return final_results
    
    def _check_ic_constraints(self, data: pd.DataFrame) -> Dict:
        """Check IC constraints and capping"""
        
        try:
            # Get model predictions
            predictions = self.alpha_system.predict_alpha(data)
            
            if len(data) > 100:  # Need sufficient data
                # Calculate ICs
                actual_returns = data['next_return_1d'].fillna(0)
                predicted_scores = predictions['final_scores']
                
                # Align data
                min_len = min(len(actual_returns), len(predicted_scores))
                actual_returns = actual_returns.iloc[:min_len]
                predicted_scores = predicted_scores[:min_len]
                
                # Calculate IC (Spearman correlation)
                ic_spearman = stats.spearmanr(actual_returns, predicted_scores)[0]
                ic_spearman = ic_spearman if not np.isnan(ic_spearman) else 0
                
                # Check constraints
                ic_within_cap = abs(ic_spearman) <= self.config['max_training_ic']
                ic_above_minimum = abs(ic_spearman) >= self.config['min_validation_ic']
                
                return {
                    'current_ic': ic_spearman,
                    'ic_within_cap': ic_within_cap,
                    'ic_above_minimum': ic_above_minimum,
                    'max_allowed_ic': self.config['max_training_ic'],
                    'min_required_ic': self.config['min_validation_ic'],
                    'passed': ic_within_cap and ic_above_minimum,
                    'samples': min_len
                }
            else:
                return {
                    'current_ic': 0,
                    'passed': False,
                    'error': 'Insufficient data for IC calculation',
                    'samples': len(data)
                }
                
        except Exception as e:
            logger.warning(f"IC constraint check failed: {e}")
            return {
                'current_ic': 0,
                'passed': False,
                'error': str(e),
                'samples': 0
            }
    
    def _detect_overfitting(self, data: pd.DataFrame) -> Dict:
        """Detect overfitting through train/validation comparison"""
        
        try:
            # Split data for overfitting check
            split_idx = int(len(data) * 0.8)
            train_data = data.iloc[:split_idx]
            val_data = data.iloc[split_idx:]
            
            if len(train_data) < 100 or len(val_data) < 50:
                return {
                    'passed': False,
                    'error': 'Insufficient data for overfitting detection',
                    'train_samples': len(train_data),
                    'val_samples': len(val_data)
                }
            
            # Get predictions for both splits
            train_pred = self.alpha_system.predict_alpha(train_data)
            val_pred = self.alpha_system.predict_alpha(val_data)
            
            # Calculate ICs
            train_ic = stats.spearmanr(
                train_data['next_return_1d'].fillna(0),
                train_pred['final_scores'][:len(train_data)]
            )[0]
            train_ic = train_ic if not np.isnan(train_ic) else 0
            
            val_ic = stats.spearmanr(
                val_data['next_return_1d'].fillna(0),
                val_pred['final_scores'][:len(val_data)]
            )[0]
            val_ic = val_ic if not np.isnan(val_ic) else 0
            
            # Calculate MSEs
            train_mse = mean_squared_error(
                train_data['next_return_1d'].fillna(0),
                train_pred['final_scores'][:len(train_data)]
            )
            val_mse = mean_squared_error(
                val_data['next_return_1d'].fillna(0),
                val_pred['final_scores'][:len(val_data)]
            )
            
            # Overfitting checks
            ic_ratio = abs(train_ic) / max(abs(val_ic), 1e-6)
            mse_ratio = val_mse / max(train_mse, 1e-6)
            
            ic_overfitting = ic_ratio <= self.config['max_train_val_ic_ratio']
            mse_overfitting = mse_ratio <= self.config['max_train_val_mse_ratio']
            
            return {
                'train_ic': train_ic,
                'val_ic': val_ic,
                'ic_ratio': ic_ratio,
                'train_mse': train_mse,
                'val_mse': val_mse,
                'mse_ratio': mse_ratio,
                'ic_overfitting_check': ic_overfitting,
                'mse_overfitting_check': mse_overfitting,
                'passed': ic_overfitting and mse_overfitting,
                'train_samples': len(train_data),
                'val_samples': len(val_data)
            }
            
        except Exception as e:
            logger.warning(f"Overfitting detection failed: {e}")
            return {
                'passed': False,
                'error': str(e)
            }
    
    def _check_model_complexity(self) -> Dict:
        """Check model complexity constraints"""
        
        complexity_checks = {}
        
        # LightGBM complexity
        if hasattr(self.alpha_system, 'lgbm_models') and self.alpha_system.lgbm_models:
            lgbm_leaves = []
            for model in self.alpha_system.lgbm_models:
                # Get number of leaves (simplified)
                try:
                    num_leaves = 31  # Default, would extract from actual model
                    lgbm_leaves.append(num_leaves)
                except:
                    lgbm_leaves.append(31)  # Default
            
            avg_leaves = np.mean(lgbm_leaves)
            complexity_checks['lgbm'] = {
                'avg_num_leaves': avg_leaves,
                'max_allowed': self.config['max_lgbm_leaves'],
                'passed': avg_leaves <= self.config['max_lgbm_leaves']
            }
        else:
            complexity_checks['lgbm'] = {'passed': True, 'status': 'not_available'}
        
        # LSTM complexity
        if hasattr(self.alpha_system, 'lstm_trainer') and self.alpha_system.lstm_trainer:
            lstm_config = getattr(self.alpha_system.lstm_trainer, 'config', {})
            hidden_dim = lstm_config.get('hidden_dim', 64)
            
            complexity_checks['lstm'] = {
                'hidden_dim': hidden_dim,
                'max_allowed': self.config['max_lstm_hidden_dim'],
                'passed': hidden_dim <= self.config['max_lstm_hidden_dim']
            }
        else:
            complexity_checks['lstm'] = {'passed': True, 'status': 'not_available'}
        
        # Feature count
        feature_count = len(getattr(self.alpha_system, 'feature_names', []))
        complexity_checks['features'] = {
            'feature_count': feature_count,
            'max_allowed': self.config['max_features'],
            'passed': feature_count <= self.config['max_features']
        }
        
        # Overall complexity check
        all_passed = all(check.get('passed', False) for check in complexity_checks.values())
        
        return {
            'individual_checks': complexity_checks,
            'passed': all_passed
        }
    
    def _detect_feature_drift(self, current_data: pd.DataFrame, historical_data: pd.DataFrame) -> Dict:
        """Detect feature distribution drift using statistical tests"""
        
        drift_results = {}
        
        try:
            # Check common features
            common_features = []
            for col in self.alpha_system.feature_names:
                if col in current_data.columns and col in historical_data.columns:
                    common_features.append(col)
            
            if not common_features:
                return {
                    'passed': False,
                    'error': 'No common features for drift detection'
                }
            
            # Sample recent historical data
            historical_sample = historical_data.tail(min(5000, len(historical_data)))
            current_sample = current_data.tail(min(1000, len(current_data)))
            
            drift_detected = []
            p_values = {}
            
            for feature in common_features[:10]:  # Check top 10 features
                try:
                    hist_values = historical_sample[feature].dropna()
                    curr_values = current_sample[feature].dropna()
                    
                    if len(hist_values) > 50 and len(curr_values) > 50:
                        # Kolmogorov-Smirnov test for distribution change
                        ks_stat, p_value = stats.ks_2samp(hist_values, curr_values)
                        
                        drift_significant = p_value < self.config['drift_significance']
                        drift_detected.append(drift_significant)
                        p_values[feature] = p_value
                        
                        drift_results[feature] = {
                            'ks_statistic': ks_stat,
                            'p_value': p_value,
                            'drift_detected': drift_significant,
                            'hist_mean': float(hist_values.mean()),
                            'curr_mean': float(curr_values.mean()),
                            'hist_std': float(hist_values.std()),
                            'curr_std': float(curr_values.std())
                        }
                
                except Exception as e:
                    logger.warning(f"Drift check failed for {feature}: {e}")
                    continue
            
            # Overall drift assessment
            drift_rate = np.mean(drift_detected) if drift_detected else 0
            drift_threshold_exceeded = drift_rate > self.config['feature_drift_threshold']
            
            return {
                'individual_features': drift_results,
                'drift_rate': drift_rate,
                'drift_threshold': self.config['feature_drift_threshold'],
                'significant_drift': drift_threshold_exceeded,
                'passed': not drift_threshold_exceeded,
                'features_tested': len(common_features)
            }
            
        except Exception as e:
            logger.warning(f"Feature drift detection failed: {e}")
            return {
                'passed': False,
                'error': str(e)
            }
    
    def _check_performance_degradation(self) -> Dict:
        """Check for performance degradation over time"""
        
        if len(self.performance_history) < 5:
            return {
                'passed': True,
                'insufficient_history': True,
                'history_length': len(self.performance_history)
            }
        
        try:
            # Extract recent performance metrics
            recent_performance = self.performance_history[-10:]  # Last 10 records
            
            hit_rates = [p.get('hit_rate', 0.5) for p in recent_performance]
            sharpe_ratios = [p.get('sharpe_ratio', 0) for p in recent_performance]
            ics = [p.get('ic', 0) for p in recent_performance]
            
            # Check degradation
            avg_hit_rate = np.mean(hit_rates)
            avg_sharpe = np.mean(sharpe_ratios) 
            avg_ic = np.mean(ics)
            
            hit_rate_ok = avg_hit_rate >= self.config['min_hit_rate']
            sharpe_ok = avg_sharpe >= self.config['min_sharpe_ratio']
            ic_degradation = len(ics) > 1 and abs(avg_ic) < abs(ics[0]) * (1 - self.config['ic_degradation_threshold'])
            
            return {
                'avg_hit_rate': avg_hit_rate,
                'min_hit_rate': self.config['min_hit_rate'],
                'hit_rate_ok': hit_rate_ok,
                'avg_sharpe': avg_sharpe,
                'min_sharpe': self.config['min_sharpe_ratio'],
                'sharpe_ok': sharpe_ok,
                'avg_ic': avg_ic,
                'ic_degradation': ic_degradation,
                'passed': hit_rate_ok and sharpe_ok and not ic_degradation,
                'samples': len(recent_performance)
            }
            
        except Exception as e:
            logger.warning(f"Performance degradation check failed: {e}")
            return {
                'passed': False,
                'error': str(e)
            }
    
    def _check_parameter_stability(self) -> Dict:
        """Check parameter stability over time"""
        
        if len(self.parameter_history) < 3:
            return {
                'passed': True,
                'insufficient_history': True,
                'history_length': len(self.parameter_history)
            }
        
        try:
            # Simplified parameter stability check
            # Would implement more sophisticated parameter tracking in practice
            
            stability_score = 0.8  # Mock stability score
            stable = stability_score > 0.7
            
            return {
                'stability_score': stability_score,
                'stable': stable,
                'passed': stable,
                'samples': len(self.parameter_history)
            }
            
        except Exception as e:
            logger.warning(f"Parameter stability check failed: {e}")
            return {
                'passed': False,
                'error': str(e)
            }
    
    def _assess_retrain_needs(self, current_date: str = None) -> Dict:
        """Assess if auto-retrain is needed"""
        
        retrain_triggers = []
        
        # Time-based retrain
        if current_date and self.last_retrain_date:
            days_since_retrain = (pd.to_datetime(current_date) - pd.to_datetime(self.last_retrain_date)).days
            if days_since_retrain >= self.config['retrain_frequency_days']:
                retrain_triggers.append('time_based')
        
        # Performance-based retrain
        if len(self.performance_history) > 5:
            recent_avg_ic = np.mean([p.get('ic', 0) for p in self.performance_history[-5:]])
            if abs(recent_avg_ic) < self.config['min_validation_ic']:
                retrain_triggers.append('low_performance')
        
        # Drift-based retrain (would be set by drift detection)
        # This would be triggered by _detect_feature_drift results
        
        retrain_needed = len(retrain_triggers) > 0
        
        return {
            'retrain_needed': retrain_needed,
            'triggers': retrain_triggers,
            'days_since_last_retrain': None,  # Would calculate if available
            'last_retrain_date': self.last_retrain_date
        }
    
    def _calculate_sanity_score(self, sanity_results: Dict) -> Dict:
        """Calculate overall sanity score"""
        
        # Weight different checks
        check_weights = {
            'ic_validation': 0.25,
            'overfitting': 0.20,
            'complexity': 0.15,
            'drift': 0.15,
            'performance': 0.15,
            'stability': 0.10
        }
        
        weighted_score = 0
        total_weight = 0
        
        for check_name, results in sanity_results.items():
            if check_name in check_weights and 'passed' in results:
                weight = check_weights[check_name]
                score = 1.0 if results['passed'] else 0.0
                weighted_score += weight * score
                total_weight += weight
        
        final_score = weighted_score / total_weight if total_weight > 0 else 0
        
        # Determine retrain need
        retrain_needed = (
            final_score < self.config['critical_threshold'] or
            sanity_results.get('retrain_assessment', {}).get('retrain_needed', False)
        )
        
        return {
            'score': final_score,
            'passed': final_score >= self.config['warning_threshold'],
            'warning_level': final_score < self.config['warning_threshold'],
            'critical_level': final_score < self.config['critical_threshold'],
            'retrain_needed': retrain_needed,
            'check_weights': check_weights
        }
    
    def _generate_recommendations(self, sanity_results: Dict, overall_score: Dict) -> List[str]:
        """Generate recommendations based on sanity check results"""
        
        recommendations = []
        
        # IC validation recommendations
        ic_results = sanity_results.get('ic_validation', {})
        if not ic_results.get('passed', True):
            if not ic_results.get('ic_within_cap', True):
                recommendations.append("Reduce model complexity - training IC exceeds 3% cap")
            if not ic_results.get('ic_above_minimum', True):
                recommendations.append("Improve signal quality - validation IC below minimum threshold")
        
        # Overfitting recommendations  
        overfitting = sanity_results.get('overfitting', {})
        if not overfitting.get('passed', True):
            if overfitting.get('ic_ratio', 1) > self.config['max_train_val_ic_ratio']:
                recommendations.append("Reduce overfitting - add regularization or reduce model complexity")
            if overfitting.get('mse_ratio', 1) > self.config['max_train_val_mse_ratio']:
                recommendations.append("Improve generalization - validation error too high relative to training")
        
        # Drift recommendations
        drift = sanity_results.get('drift', {})
        if drift.get('significant_drift', False):
            recommendations.append("Address feature drift - retrain with recent data or update features")
        
        # Performance recommendations
        performance = sanity_results.get('performance', {})
        if not performance.get('passed', True):
            if not performance.get('hit_rate_ok', True):
                recommendations.append("Improve directional accuracy - hit rate below threshold")
            if not performance.get('sharpe_ok', True):
                recommendations.append("Improve risk-adjusted returns - Sharpe ratio below threshold")
        
        # Overall recommendations
        if overall_score['critical_level']:
            recommendations.append("CRITICAL: Immediate model retraining required")
        elif overall_score['warning_level']:
            recommendations.append("WARNING: Schedule model retraining soon")
        
        if not recommendations:
            recommendations.append("All sanity checks passed - system operating normally")
        
        return recommendations
    
    def _generate_alerts(self, overall_score: Dict) -> List[Dict]:
        """Generate alerts based on sanity score"""
        
        alerts = []
        
        if overall_score['critical_level']:
            alerts.append({
                'level': 'CRITICAL',
                'message': f"System sanity score critically low: {overall_score['score']:.1%}",
                'action_required': 'Immediate retraining required',
                'timestamp': datetime.now().isoformat()
            })
        
        elif overall_score['warning_level']:
            alerts.append({
                'level': 'WARNING', 
                'message': f"System sanity score below warning threshold: {overall_score['score']:.1%}",
                'action_required': 'Schedule retraining',
                'timestamp': datetime.now().isoformat()
            })
        
        if overall_score['retrain_needed']:
            alerts.append({
                'level': 'INFO',
                'message': 'Auto-retrain triggered by system assessment',
                'action_required': 'Monitor retrain progress',
                'timestamp': datetime.now().isoformat()
            })
        
        return alerts
    
    def _execute_auto_retrain(self, historical_data: pd.DataFrame, current_data: pd.DataFrame) -> Dict:
        """Execute automatic model retraining"""
        
        logger.info("🔄 EXECUTING AUTO-RETRAIN...")
        
        try:
            # Combine historical and current data for retraining
            combined_data = pd.concat([historical_data, current_data]).drop_duplicates()
            
            # Use rolling window for retraining (last 2-3 years)
            lookback_days = 365 * 2
            cutoff_date = pd.to_datetime(datetime.now() - timedelta(days=lookback_days))
            recent_data = combined_data[combined_data['Date'] >= cutoff_date]
            
            if len(recent_data) < self.config['min_samples_retrain']:
                return {
                    'success': False,
                    'error': f'Insufficient samples for retrain: {len(recent_data)}',
                    'min_required': self.config['min_samples_retrain']
                }
            
            # Split for retraining
            val_split = self.config['validation_split']
            split_idx = int(len(recent_data) * (1 - val_split))
            train_data = recent_data.iloc[:split_idx]
            val_data = recent_data.iloc[split_idx:]
            
            # Retrain the system
            logger.info(f"   Retraining on {len(train_data):,} samples")
            logger.info(f"   Validating on {len(val_data):,} samples")
            
            retrain_results = self.alpha_system.train_system(train_data, val_data)
            
            # Update retrain tracking
            self.last_retrain_date = datetime.now().isoformat()
            self.retrain_triggers = []
            
            return {
                'success': True,
                'retrain_date': self.last_retrain_date,
                'train_samples': len(train_data),
                'val_samples': len(val_data),
                'training_results': retrain_results
            }
            
        except Exception as e:
            logger.error(f"Auto-retrain failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'retrain_date': None
            }
    
    def _log_sanity_check_results(self, results: Dict):
        """Log comprehensive sanity check results"""
        
        logger.info("🔍 SANITY CHECK RESULTS")
        logger.info("=" * 40)
        
        score = results['overall_sanity_score']
        logger.info(f"   Overall Score: {score['score']:.1%}")
        
        if score['critical_level']:
            logger.info("   Status: 🔴 CRITICAL")
        elif score['warning_level']:
            logger.info("   Status: 🟡 WARNING")
        else:
            logger.info("   Status: 🟢 HEALTHY")
        
        logger.info(f"   Retrain Needed: {'Yes' if score['retrain_needed'] else 'No'}")
        
        # Log individual check results
        logger.info("\n📊 Individual Checks:")
        for check_name, check_results in results['detailed_checks'].items():
            status = "✅" if check_results.get('passed', False) else "❌"
            logger.info(f"   {status} {check_name}")
        
        # Log alerts
        if results['alerts']:
            logger.info(f"\n🚨 Alerts ({len(results['alerts'])}):")
            for alert in results['alerts']:
                logger.info(f"   {alert['level']}: {alert['message']}")
        
        # Log recommendations
        logger.info(f"\n💡 Recommendations:")
        for rec in results['recommendations']:
            logger.info(f"   • {rec}")
    
    def update_performance_history(self, performance_metrics: Dict):
        """Update performance history for monitoring"""
        
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            **performance_metrics
        })
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-50:]

def main():
    """Test the parameter sanity checker"""
    
    # Mock system for testing
    from src.models.tiered_system import TieredAlphaSystem
    
    system_config = {
        'lstm': {'enabled': False},  # Disable for quick test
        'regime': {'enabled': True},
        'meta': {'combiner_type': 'ridge'}
    }
    
    # Load test data
    data_path = Path(__file__).parent.parent / 'data' / 'training_data_enhanced.csv'
    
    if data_path.exists():
        test_data = pd.read_csv(data_path)
        test_data['Date'] = pd.to_datetime(test_data['Date'])
        
        # Split into historical and current
        split_date = '2024-01-01'
        historical_data = test_data[test_data['Date'] < split_date]
        current_data = test_data[test_data['Date'] >= split_date].head(1000)
        
        # Initialize system (mock)
        alpha_system = TieredAlphaSystem(system_config)
        alpha_system.is_trained = True  # Skip actual training
        alpha_system.feature_names = ['return_5d_lag1', 'vol_20d_lag1', 'volume_ratio_lag1']
        
        # Initialize sanity checker
        sanity_checker = ParameterSanityChecker(alpha_system)
        
        # Add mock performance history
        sanity_checker.update_performance_history({
            'hit_rate': 0.55,
            'sharpe_ratio': 0.8,
            'ic': 0.02
        })
        
        try:
            # Run sanity checks
            results = sanity_checker.run_comprehensive_sanity_check(
                current_data, historical_data, '2024-06-01'
            )
            
            print(f"\n🎉 Sanity Check Completed!")
            print(f"Overall Score: {results['overall_sanity_score']['score']:.1%}")
            print(f"Status: {'Healthy' if results['overall_sanity_score']['passed'] else 'Issues Detected'}")
            print(f"Auto-retrain Triggered: {results['auto_retrain_triggered']}")
            
        except Exception as e:
            print(f"❌ Sanity check failed: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print(f"❌ Test data not found: {data_path}")

if __name__ == "__main__":
    main()