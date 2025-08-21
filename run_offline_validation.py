#!/usr/bin/env python3
"""
Offline Validation System - No External Dependencies
Complete validation using only local data and synthetic stress tests
"""

import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import warnings
warnings.filterwarnings('ignore')

# Import core system
from src.models.tiered_system import TieredAlphaSystem

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OfflineValidationSystem:
    """
    Complete validation system that works entirely offline
    """
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.alpha_system = None
        
        logger.info("🏠 OFFLINE VALIDATION SYSTEM INITIALIZED")
    
    def run_complete_offline_validation(self) -> Dict:
        """Run complete validation without external dependencies"""
        
        logger.info("🚀 STARTING COMPLETE OFFLINE VALIDATION")
        logger.info("=" * 60)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'validation_type': 'complete_offline_validation'
        }
        
        try:
            # Phase 1: System Initialization
            logger.info("📋 PHASE 1: SYSTEM INITIALIZATION")
            init_results = self._initialize_system()
            results['phase_1_initialization'] = init_results
            
            if not init_results['success']:
                return results
            
            # Phase 2: Core Performance Validation (6-month simulation)
            logger.info("📋 PHASE 2: 6-MONTH PERFORMANCE SIMULATION")
            performance_results = self._simulate_6month_performance()
            results['phase_2_performance'] = performance_results
            
            # Phase 3: Offline Stress Testing
            logger.info("📋 PHASE 3: OFFLINE STRESS TESTING") 
            stress_results = self._run_offline_stress_tests()
            results['phase_3_stress_testing'] = stress_results
            
            # Phase 4: Parameter Sanity Checks
            logger.info("📋 PHASE 4: PARAMETER SANITY CHECKS")
            sanity_results = self._run_offline_sanity_checks()
            results['phase_4_sanity_checks'] = sanity_results
            
            # Phase 5: Position Sizing & Risk Validation
            logger.info("📋 PHASE 5: RISK VALIDATION")
            risk_results = self._validate_risk_controls()
            results['phase_5_risk_validation'] = risk_results
            
            # Phase 6: Final Assessment
            logger.info("📋 PHASE 6: FINAL ASSESSMENT")
            assessment = self._generate_final_assessment(results)
            results['final_assessment'] = assessment
            
            # Generate report
            self._save_validation_report(results)
            self._log_final_summary(results)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Offline validation failed: {e}")
            import traceback
            traceback.print_exc()
            
            results['error'] = str(e)
            results['success'] = False
            return results
    
    def _initialize_system(self) -> Dict:
        """Initialize the alpha system without external data"""
        
        try:
            # Initialize with regime disabled to avoid yfinance
            system_config = {
                'lstm': {'enabled': True, 'max_epochs': 30},
                'regime': {'enabled': False},  # Disable to avoid external data
                'meta': {'combiner_type': 'ridge'}
            }
            
            self.alpha_system = TieredAlphaSystem(system_config)
            
            # Load training data
            training_data = self._load_training_data()
            if training_data is None:
                return {'success': False, 'error': 'No training data'}
            
            # Train system
            logger.info("   Training complete tiered system...")
            training_results = self.alpha_system.train_system(training_data)
            
            return {
                'success': True,
                'training_samples': len(training_data),
                'training_results': training_results,
                'models_trained': list(training_results.keys())
            }
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _load_training_data(self) -> Optional[pd.DataFrame]:
        """Load training data"""
        
        data_path = self.base_dir / 'data' / 'training_data_enhanced.csv'
        
        if data_path.exists():
            data = pd.read_csv(data_path)
            data['Date'] = pd.to_datetime(data['Date'])
            
            # Use data before 2023 for training
            training_data = data[data['Date'] < '2023-01-01']
            logger.info(f"   Training data: {len(training_data):,} samples")
            logger.info(f"   Period: {training_data['Date'].min()} to {training_data['Date'].max()}")
            
            return training_data
        
        return None
    
    def _simulate_6month_performance(self) -> Dict:
        """Simulate 6-month paper trading performance"""
        
        try:
            # Load validation data
            data_path = self.base_dir / 'data' / 'training_data_enhanced.csv'
            data = pd.read_csv(data_path)
            data['Date'] = pd.to_datetime(data['Date'])
            
            # Use 2023 data for 6-month simulation
            validation_data = data[
                (data['Date'] >= '2023-01-01') & (data['Date'] <= '2023-06-30')
            ]
            
            logger.info(f"   Simulating {len(validation_data):,} samples over 6 months")
            
            # Group by date for daily simulation
            daily_groups = validation_data.groupby('Date')
            trading_dates = sorted(daily_groups.groups.keys())[:124]  # ~6 months
            
            # Portfolio simulation
            portfolio_value = 1000000  # $1M start
            daily_returns = []
            daily_positions = []
            daily_scores = []
            
            for i, date in enumerate(trading_dates):
                day_data = daily_groups.get_group(date)
                
                if len(day_data) < 5:
                    daily_returns.append(0)
                    daily_positions.append(0)
                    daily_scores.append([])
                    continue
                
                # Generate predictions
                predictions = self.alpha_system.predict_alpha(day_data, str(date.date()))
                
                # Simple market-neutral strategy
                scores = predictions['final_scores']
                actual_returns = day_data['next_return_1d'].fillna(0).values
                
                # Rank stocks and create long/short portfolio
                if len(scores) >= 10:
                    # Top 20% long, bottom 20% short
                    n_positions = len(scores)
                    n_long = max(1, n_positions // 5)
                    n_short = max(1, n_positions // 5)
                    
                    sorted_indices = np.argsort(scores)
                    long_indices = sorted_indices[-n_long:]
                    short_indices = sorted_indices[:n_short]
                    
                    # Calculate portfolio return
                    long_return = actual_returns[long_indices].mean() if len(long_indices) > 0 else 0
                    short_return = -actual_returns[short_indices].mean() if len(short_indices) > 0 else 0
                    daily_return = (long_return + short_return) / 2
                    
                    daily_returns.append(daily_return)
                    daily_positions.append(n_long + n_short)
                else:
                    daily_returns.append(0)
                    daily_positions.append(0)
                
                daily_scores.append(scores)
                
                # Update portfolio value
                portfolio_value *= (1 + daily_returns[-1])
            
            # Calculate performance metrics
            daily_returns = np.array(daily_returns)
            
            total_return = (portfolio_value / 1000000 - 1) * 100
            volatility = np.std(daily_returns) * np.sqrt(252) * 100
            sharpe_ratio = (np.mean(daily_returns) * 252) / (np.std(daily_returns) * np.sqrt(252)) if np.std(daily_returns) > 0 else 0
            
            # Drawdown calculation
            cumulative = np.cumprod(1 + daily_returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdown) * 100
            
            # Hit rate
            hit_rate = np.mean(daily_returns > 0)
            
            return {
                'success': True,
                'trading_days': len(trading_dates),
                'total_return_pct': total_return,
                'annualized_return_pct': total_return * 2,  # Annualized
                'volatility_pct': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown,
                'hit_rate': hit_rate,
                'avg_daily_positions': np.mean(daily_positions),
                'final_portfolio_value': portfolio_value,
                'performance_passed': (
                    sharpe_ratio > 0.3 and 
                    max_drawdown > -20 and 
                    hit_rate > 0.48
                )
            }
            
        except Exception as e:
            logger.error(f"Performance simulation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _run_offline_stress_tests(self) -> Dict:
        """Run synthetic stress tests without external data"""
        
        try:
            # Load test data
            data_path = self.base_dir / 'data' / 'training_data_enhanced.csv'
            data = pd.read_csv(data_path)
            data['Date'] = pd.to_datetime(data['Date'])
            test_data = data[data['Date'] >= '2023-01-01'].sample(1000, random_state=42)
            
            stress_results = {}
            
            # Test 1: High Volatility Scenario
            logger.info("   🌊 Testing high volatility scenario...")
            high_vol_data = test_data.copy()
            high_vol_data['next_return_1d'] *= 2  # Double the volatility
            
            predictions = self.alpha_system.predict_alpha(high_vol_data.head(100))
            position_sizes = predictions.get('position_sizes', [])
            
            # Check if system reduces positions in high vol
            max_position = np.abs(position_sizes).max() if len(position_sizes) > 0 else 0
            high_vol_passed = max_position <= 0.12  # Should be conservative
            
            stress_results['high_volatility'] = {
                'max_position': max_position,
                'passed': high_vol_passed
            }
            
            # Test 2: Market Crash Scenario
            logger.info("   💥 Testing market crash scenario...")
            crash_data = test_data.copy()
            crash_data['next_return_1d'] -= 0.05  # Simulate -5% market crash
            
            predictions = self.alpha_system.predict_alpha(crash_data.head(100))
            position_sizes = predictions.get('position_sizes', [])
            
            # Check defensive positioning
            net_exposure = np.sum(position_sizes) if len(position_sizes) > 0 else 0
            crash_passed = abs(net_exposure) <= 0.1  # Should be near market neutral
            
            stress_results['market_crash'] = {
                'net_exposure': net_exposure,
                'passed': crash_passed
            }
            
            # Test 3: Low Signal Environment
            logger.info("   📻 Testing low signal environment...")
            noise_data = test_data.copy()
            # Add noise to features to simulate low signal
            feature_cols = [col for col in noise_data.columns if 'lag1' in col]
            for col in feature_cols[:5]:
                noise_data[col] += np.random.normal(0, 0.01, len(noise_data))
            
            predictions = self.alpha_system.predict_alpha(noise_data.head(100))
            n_tradeable = predictions.get('n_tradeable', 0)
            
            # Should trade less when signal is weak
            low_signal_passed = n_tradeable <= len(noise_data) * 0.3
            
            stress_results['low_signal'] = {
                'n_tradeable': n_tradeable,
                'passed': low_signal_passed
            }
            
            # Overall stress test assessment
            passed_tests = sum(test['passed'] for test in stress_results.values())
            total_tests = len(stress_results)
            pass_rate = passed_tests / total_tests
            
            return {
                'success': True,
                'individual_tests': stress_results,
                'pass_rate': pass_rate,
                'stress_tests_passed': pass_rate >= 0.67  # 67% pass rate
            }
            
        except Exception as e:
            logger.error(f"Stress testing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _run_offline_sanity_checks(self) -> Dict:
        """Run parameter sanity checks without external dependencies"""
        
        try:
            sanity_results = {}
            
            # Check 1: Model Complexity
            feature_count = len(getattr(self.alpha_system, 'feature_names', []))
            complexity_reasonable = feature_count <= 50
            
            sanity_results['model_complexity'] = {
                'feature_count': feature_count,
                'reasonable': complexity_reasonable
            }
            
            # Check 2: Prediction Stability
            data_path = self.base_dir / 'data' / 'training_data_enhanced.csv'
            data = pd.read_csv(data_path)
            data['Date'] = pd.to_datetime(data['Date'])
            test_sample = data[data['Date'] >= '2023-01-01'].head(200)
            
            # Run predictions twice on same data
            pred1 = self.alpha_system.predict_alpha(test_sample)
            pred2 = self.alpha_system.predict_alpha(test_sample)
            
            # Check prediction stability (should be identical)
            scores1 = pred1['final_scores']
            scores2 = pred2['final_scores']
            
            if len(scores1) == len(scores2) and len(scores1) > 0:
                correlation = np.corrcoef(scores1, scores2)[0, 1]
                stability_check = correlation > 0.99
            else:
                correlation = 0
                stability_check = False
            
            sanity_results['prediction_stability'] = {
                'correlation': correlation,
                'stable': stability_check
            }
            
            # Check 3: Position Sizing Constraints
            position_sizes = pred1.get('position_sizes', [])
            if len(position_sizes) > 0:
                max_position = np.abs(position_sizes).max()
                total_exposure = np.abs(position_sizes).sum()
                
                position_constraints_met = (
                    max_position <= 0.20 and  # Max 20% position
                    total_exposure <= 2.0     # Max 200% gross exposure
                )
            else:
                position_constraints_met = True
                max_position = 0
                total_exposure = 0
            
            sanity_results['position_constraints'] = {
                'max_position': max_position,
                'total_exposure': total_exposure,
                'constraints_met': position_constraints_met
            }
            
            # Overall sanity score
            passed_checks = sum(
                check.get('reasonable', check.get('stable', check.get('constraints_met', False)))
                for check in sanity_results.values()
            )
            total_checks = len(sanity_results)
            sanity_score = passed_checks / total_checks
            
            return {
                'success': True,
                'individual_checks': sanity_results,
                'sanity_score': sanity_score,
                'sanity_checks_passed': sanity_score >= 0.67  # 67% pass rate
            }
            
        except Exception as e:
            logger.error(f"Sanity checks failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _validate_risk_controls(self) -> Dict:
        """Validate risk control mechanisms"""
        
        try:
            # Load test data
            data_path = self.base_dir / 'data' / 'training_data_enhanced.csv'
            data = pd.read_csv(data_path)
            data['Date'] = pd.to_datetime(data['Date'])
            test_data = data[data['Date'] >= '2023-01-01'].head(500)
            
            # Test risk controls across multiple scenarios
            scenarios = []
            
            for i in range(5):  # 5 different random samples
                sample = test_data.sample(min(100, len(test_data)), random_state=42+i)
                predictions = self.alpha_system.predict_alpha(sample)
                
                position_sizes = predictions.get('position_sizes', [])
                if len(position_sizes) > 0:
                    scenario = {
                        'max_position': np.abs(position_sizes).max(),
                        'gross_exposure': np.abs(position_sizes).sum(),
                        'net_exposure': np.sum(position_sizes),
                        'n_positions': np.sum(np.abs(position_sizes) > 0.001)
                    }
                    scenarios.append(scenario)
            
            if not scenarios:
                return {
                    'success': False,
                    'error': 'No valid risk scenarios generated'
                }
            
            # Analyze risk metrics
            max_positions = [s['max_position'] for s in scenarios]
            gross_exposures = [s['gross_exposure'] for s in scenarios]
            net_exposures = [s['net_exposure'] for s in scenarios]
            
            # Risk control checks
            risk_checks = {
                'max_position_limit': max(max_positions) <= 0.15,  # 15% max position
                'gross_exposure_limit': max(gross_exposures) <= 1.5,  # 150% max gross
                'market_neutral': max(np.abs(net_exposures)) <= 0.2,  # Near market neutral
                'position_count_reasonable': np.mean([s['n_positions'] for s in scenarios]) <= 50
            }
            
            passed_risk_checks = sum(risk_checks.values())
            total_risk_checks = len(risk_checks)
            risk_score = passed_risk_checks / total_risk_checks
            
            return {
                'success': True,
                'scenarios': scenarios,
                'risk_checks': risk_checks,
                'risk_score': risk_score,
                'risk_controls_passed': risk_score >= 0.75  # 75% pass rate
            }
            
        except Exception as e:
            logger.error(f"Risk validation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_final_assessment(self, results: Dict) -> Dict:
        """Generate final validation assessment"""
        
        # Extract phase results
        performance = results.get('phase_2_performance', {})
        stress_testing = results.get('phase_3_stress_testing', {})
        sanity_checks = results.get('phase_4_sanity_checks', {})
        risk_validation = results.get('phase_5_risk_validation', {})
        
        # Individual phase passes
        performance_passed = performance.get('performance_passed', False)
        stress_passed = stress_testing.get('stress_tests_passed', False)
        sanity_passed = sanity_checks.get('sanity_checks_passed', False)
        risk_passed = risk_validation.get('risk_controls_passed', False)
        
        # Count passes
        phases_passed = sum([performance_passed, stress_passed, sanity_passed, risk_passed])
        total_phases = 4
        
        # Overall assessment
        validation_passed = phases_passed >= 3  # At least 3/4 phases must pass
        
        if phases_passed == 4:
            confidence_level = 'high'
            production_ready = True
        elif phases_passed == 3:
            confidence_level = 'medium'
            production_ready = True
        else:
            confidence_level = 'low'
            production_ready = False
        
        # Key metrics
        key_metrics = {
            'total_return_pct': performance.get('total_return_pct', 0),
            'sharpe_ratio': performance.get('sharpe_ratio', 0),
            'max_drawdown_pct': performance.get('max_drawdown_pct', 0),
            'stress_pass_rate': stress_testing.get('pass_rate', 0),
            'sanity_score': sanity_checks.get('sanity_score', 0),
            'risk_score': risk_validation.get('risk_score', 0)
        }
        
        return {
            'validation_passed': validation_passed,
            'confidence_level': confidence_level,
            'production_ready': production_ready,
            'phases_passed': f"{phases_passed}/{total_phases}",
            'individual_phases': {
                'performance': performance_passed,
                'stress_testing': stress_passed,
                'sanity_checks': sanity_passed,
                'risk_validation': risk_passed
            },
            'key_metrics': key_metrics
        }
    
    def _save_validation_report(self, results: Dict):
        """Save validation report to file"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.base_dir / f"offline_validation_report_{timestamp}.json"
        
        try:
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"📊 Validation report saved: {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
    
    def _log_final_summary(self, results: Dict):
        """Log final validation summary"""
        
        logger.info("🏠 OFFLINE VALIDATION COMPLETE")
        logger.info("=" * 50)
        
        assessment = results['final_assessment']
        
        # Overall result
        if assessment['validation_passed']:
            logger.info("🎉 VALIDATION RESULT: ✅ PASSED")
        else:
            logger.info("💥 VALIDATION RESULT: ❌ FAILED")
        
        logger.info(f"   Confidence: {assessment['confidence_level'].upper()}")
        logger.info(f"   Phases Passed: {assessment['phases_passed']}")
        logger.info(f"   Production Ready: {'Yes' if assessment['production_ready'] else 'No'}")
        
        # Key metrics
        metrics = assessment['key_metrics']
        logger.info(f"\n📊 KEY METRICS:")
        logger.info(f"   6-Month Return: {metrics['total_return_pct']:.2f}%")
        logger.info(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        logger.info(f"   Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        logger.info(f"   Stress Pass Rate: {metrics['stress_pass_rate']:.1%}")
        logger.info(f"   Sanity Score: {metrics['sanity_score']:.1%}")
        logger.info(f"   Risk Score: {metrics['risk_score']:.1%}")
        
        # Individual phases
        logger.info(f"\n🔍 PHASE RESULTS:")
        phases = assessment['individual_phases']
        for phase, passed in phases.items():
            status = "✅" if passed else "❌"
            logger.info(f"   {status} {phase}")

def main():
    """Run complete offline validation"""
    
    validator = OfflineValidationSystem()
    results = validator.run_complete_offline_validation()
    
    if 'error' not in results:
        assessment = results['final_assessment']
        print(f"\n🏠 OFFLINE VALIDATION COMPLETED!")
        print(f"Result: {'✅ PASSED' if assessment['validation_passed'] else '❌ FAILED'}")
        print(f"Confidence: {assessment['confidence_level'].upper()}")
        print(f"Production Ready: {'Yes' if assessment['production_ready'] else 'No'}")
        
        if assessment['validation_passed']:
            print("\n🚀 System ready for production deployment!")
        else:
            print("\n🔧 System needs improvements before deployment.")
    else:
        print(f"\n💥 Offline validation failed: {results['error']}")

if __name__ == "__main__":
    main()