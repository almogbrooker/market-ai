#!/usr/bin/env python3
"""
Quick Validation System - Streamlined for Speed
Focuses on core validation metrics without external data dependencies
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

class QuickValidationSystem:
    """
    Quick validation focusing on core performance metrics
    """
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.alpha_system = None
        
        logger.info("⚡ QUICK VALIDATION SYSTEM INITIALIZED")
    
    def run_quick_validation(self) -> Dict:
        """Run streamlined validation focusing on key metrics"""
        
        logger.info("🚀 STARTING QUICK VALIDATION")
        logger.info("=" * 50)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'validation_type': 'quick_validation'
        }
        
        try:
            # Step 1: Initialize and train system
            logger.info("🔧 Step 1: System Initialization")
            init_results = self._initialize_system()
            results['initialization'] = init_results
            
            if not init_results['success']:
                return results
            
            # Step 2: Load validation data
            logger.info("📊 Step 2: Loading Validation Data")
            data_results = self._load_validation_data()
            results['data_loading'] = data_results
            
            if not data_results['success']:
                return results
            
            train_data = data_results['train_data']
            test_data = data_results['test_data']
            
            # Step 3: Core performance validation
            logger.info("📈 Step 3: Core Performance Validation")
            performance_results = self._validate_core_performance(train_data, test_data)
            results['performance'] = performance_results
            
            # Step 4: Quick overfitting check
            logger.info("🔍 Step 4: Overfitting Detection")
            overfitting_results = self._check_overfitting(train_data, test_data)
            results['overfitting'] = overfitting_results
            
            # Step 5: Position sizing validation
            logger.info("⚖️ Step 5: Position Sizing Validation")
            position_results = self._validate_position_sizing(test_data)
            results['position_sizing'] = position_results
            
            # Step 6: Generate assessment
            logger.info("🎯 Step 6: Overall Assessment")
            assessment = self._generate_assessment(results)
            results['overall_assessment'] = assessment
            
            # Log summary
            self._log_validation_summary(results)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Quick validation failed: {e}")
            import traceback
            traceback.print_exc()
            
            results['error'] = str(e)
            results['success'] = False
            return results
    
    def _initialize_system(self) -> Dict:
        """Initialize the alpha system"""
        
        try:
            # Initialize system
            system_config = {
                'lstm': {'enabled': True, 'max_epochs': 20},  # Faster training
                'regime': {'enabled': False},  # Skip regime to avoid yfinance
                'meta': {'combiner_type': 'ridge'}
            }
            
            self.alpha_system = TieredAlphaSystem(system_config)
            
            # Load training data
            training_data = self._load_training_data()
            if training_data is None:
                return {'success': False, 'error': 'No training data'}
            
            # Train system
            logger.info("   Training system...")
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
            
            return training_data
        
        return None
    
    def _load_validation_data(self) -> Dict:
        """Load and split validation data"""
        
        try:
            data_path = self.base_dir / 'data' / 'training_data_enhanced.csv'
            
            if not data_path.exists():
                return {'success': False, 'error': 'No data file found'}
            
            data = pd.read_csv(data_path)
            data['Date'] = pd.to_datetime(data['Date'])
            
            # Split data
            train_cutoff = '2022-01-01'
            test_start = '2023-01-01'
            
            train_data = data[data['Date'] < train_cutoff]
            test_data = data[data['Date'] >= test_start]
            
            logger.info(f"   Train: {len(train_data):,} samples ({train_data['Date'].min()} to {train_data['Date'].max()})")
            logger.info(f"   Test:  {len(test_data):,} samples ({test_data['Date'].min()} to {test_data['Date'].max()})")
            
            return {
                'success': True,
                'train_data': train_data,
                'test_data': test_data,
                'train_samples': len(train_data),
                'test_samples': len(test_data)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _validate_core_performance(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict:
        """Validate core model performance"""
        
        try:
            results = {}
            
            # Sample data for quick testing
            train_sample = train_data.sample(min(5000, len(train_data)), random_state=42)
            test_sample = test_data.sample(min(2000, len(test_data)), random_state=42)
            
            # Get predictions
            logger.info("   Generating train predictions...")
            train_predictions = self.alpha_system.predict_alpha(train_sample)
            
            logger.info("   Generating test predictions...")  
            test_predictions = self.alpha_system.predict_alpha(test_sample)
            
            # Calculate Information Coefficients
            train_actual = train_sample['next_return_1d'].fillna(0).values
            train_pred = train_predictions['final_scores']
            
            test_actual = test_sample['next_return_1d'].fillna(0).values
            test_pred = test_predictions['final_scores']
            
            # Align lengths
            min_train_len = min(len(train_actual), len(train_pred))
            min_test_len = min(len(test_actual), len(test_pred))
            
            train_ic = np.corrcoef(train_actual[:min_train_len], train_pred[:min_train_len])[0, 1]
            test_ic = np.corrcoef(test_actual[:min_test_len], test_pred[:min_test_len])[0, 1]
            
            # Handle NaN correlations
            train_ic = train_ic if not np.isnan(train_ic) else 0
            test_ic = test_ic if not np.isnan(test_ic) else 0
            
            # Calculate hit rates (directional accuracy)
            train_hit_rate = np.mean((train_actual[:min_train_len] > 0) == (train_pred[:min_train_len] > 0))
            test_hit_rate = np.mean((test_actual[:min_test_len] > 0) == (test_pred[:min_test_len] > 0))
            
            results = {
                'train_ic': train_ic,
                'test_ic': test_ic,
                'train_hit_rate': train_hit_rate,
                'test_hit_rate': test_hit_rate,
                'train_samples': min_train_len,
                'test_samples': min_test_len,
                'ic_ratio': abs(train_ic) / max(abs(test_ic), 1e-6),
                'performance_passed': (
                    abs(test_ic) > 0.005 and  # Minimum IC threshold
                    test_hit_rate > 0.51 and  # Better than random
                    abs(train_ic) / max(abs(test_ic), 1e-6) < 3.0  # Not extreme overfitting
                )
            }
            
            logger.info(f"   Train IC: {train_ic:.4f}, Test IC: {test_ic:.4f}")
            logger.info(f"   Train Hit Rate: {train_hit_rate:.3f}, Test Hit Rate: {test_hit_rate:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Performance validation failed: {e}")
            return {'error': str(e), 'performance_passed': False}
    
    def _check_overfitting(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict:
        """Quick overfitting detection"""
        
        try:
            # Use smaller samples for speed
            train_sample = train_data.sample(min(2000, len(train_data)), random_state=43)
            test_sample = test_data.sample(min(1000, len(test_data)), random_state=43)
            
            # Get predictions
            train_pred = self.alpha_system.predict_alpha(train_sample)
            test_pred = self.alpha_system.predict_alpha(test_sample)
            
            # Calculate MSEs
            train_actual = train_sample['next_return_1d'].fillna(0)
            test_actual = test_sample['next_return_1d'].fillna(0)
            
            min_train_len = min(len(train_actual), len(train_pred['final_scores']))
            min_test_len = min(len(test_actual), len(test_pred['final_scores']))
            
            train_mse = np.mean((train_actual.iloc[:min_train_len] - train_pred['final_scores'][:min_train_len]) ** 2)
            test_mse = np.mean((test_actual.iloc[:min_test_len] - test_pred['final_scores'][:min_test_len]) ** 2)
            
            mse_ratio = test_mse / max(train_mse, 1e-10)
            
            # Overfitting checks
            overfitting_detected = mse_ratio > 2.0  # Test MSE more than 2x train MSE
            
            return {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'mse_ratio': mse_ratio,
                'overfitting_detected': overfitting_detected,
                'overfitting_passed': not overfitting_detected
            }
            
        except Exception as e:
            logger.error(f"Overfitting check failed: {e}")
            return {'error': str(e), 'overfitting_passed': False}
    
    def _validate_position_sizing(self, test_data: pd.DataFrame) -> Dict:
        """Validate position sizing is reasonable"""
        
        try:
            # Sample for testing
            test_sample = test_data.sample(min(500, len(test_data)), random_state=44)
            
            predictions = self.alpha_system.predict_alpha(test_sample)
            position_sizes = predictions.get('position_sizes', [])
            
            if len(position_sizes) == 0:
                return {'error': 'No position sizes generated', 'sizing_passed': False}
            
            max_position = np.abs(position_sizes).max()
            mean_position = np.abs(position_sizes).mean()
            std_position = np.abs(position_sizes).std()
            
            # Position sizing checks
            reasonable_max = max_position <= 0.20  # Max 20% position
            reasonable_mean = mean_position <= 0.10  # Average position reasonable
            not_too_volatile = std_position <= 0.05  # Not too much variation
            
            sizing_passed = reasonable_max and reasonable_mean and not_too_volatile
            
            return {
                'max_position': max_position,
                'mean_position': mean_position,
                'std_position': std_position,
                'reasonable_max': reasonable_max,
                'reasonable_mean': reasonable_mean,
                'not_too_volatile': not_too_volatile,
                'sizing_passed': sizing_passed,
                'n_positions': len(position_sizes)
            }
            
        except Exception as e:
            logger.error(f"Position sizing validation failed: {e}")
            return {'error': str(e), 'sizing_passed': False}
    
    def _generate_assessment(self, results: Dict) -> Dict:
        """Generate overall validation assessment"""
        
        # Extract key results
        performance = results.get('performance', {})
        overfitting = results.get('overfitting', {})
        position_sizing = results.get('position_sizing', {})
        
        # Individual checks
        performance_passed = performance.get('performance_passed', False)
        overfitting_passed = overfitting.get('overfitting_passed', False)
        sizing_passed = position_sizing.get('sizing_passed', False)
        
        # Overall assessment
        checks_passed = sum([performance_passed, overfitting_passed, sizing_passed])
        validation_passed = checks_passed >= 2  # At least 2 out of 3
        
        # Confidence level
        if checks_passed == 3:
            confidence = 'high'
        elif checks_passed == 2:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        # Key metrics
        test_ic = performance.get('test_ic', 0)
        test_hit_rate = performance.get('test_hit_rate', 0.5)
        max_position = position_sizing.get('max_position', 0)
        
        assessment = {
            'validation_passed': validation_passed,
            'confidence_level': confidence,
            'checks_passed': f"{checks_passed}/3",
            'individual_checks': {
                'performance': performance_passed,
                'overfitting': overfitting_passed,
                'position_sizing': sizing_passed
            },
            'key_metrics': {
                'test_ic': test_ic,
                'test_hit_rate': test_hit_rate,
                'max_position': max_position
            },
            'production_ready': validation_passed and confidence in ['medium', 'high']
        }
        
        return assessment
    
    def _log_validation_summary(self, results: Dict):
        """Log validation summary"""
        
        logger.info("⚡ QUICK VALIDATION SUMMARY")
        logger.info("=" * 40)
        
        assessment = results['overall_assessment']
        
        # Overall result
        if assessment['validation_passed']:
            logger.info("🎉 VALIDATION RESULT: ✅ PASSED")
        else:
            logger.info("💥 VALIDATION RESULT: ❌ FAILED")
        
        logger.info(f"   Confidence: {assessment['confidence_level'].upper()}")
        logger.info(f"   Checks Passed: {assessment['checks_passed']}")
        logger.info(f"   Production Ready: {'Yes' if assessment['production_ready'] else 'No'}")
        
        # Key metrics
        metrics = assessment['key_metrics']
        logger.info(f"\n📊 KEY METRICS:")
        logger.info(f"   Test IC: {metrics['test_ic']:.4f}")
        logger.info(f"   Hit Rate: {metrics['test_hit_rate']:.3f}")
        logger.info(f"   Max Position: {metrics['max_position']:.3f}")
        
        # Individual checks
        logger.info(f"\n🔍 INDIVIDUAL CHECKS:")
        checks = assessment['individual_checks']
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            logger.info(f"   {status} {check}")

def main():
    """Run quick validation"""
    
    validator = QuickValidationSystem()
    results = validator.run_quick_validation()
    
    if 'error' not in results:
        assessment = results['overall_assessment']
        print(f"\n⚡ QUICK VALIDATION COMPLETED!")
        print(f"Result: {'✅ PASSED' if assessment['validation_passed'] else '❌ FAILED'}")
        print(f"Confidence: {assessment['confidence_level'].upper()}")
        print(f"Production Ready: {'Yes' if assessment['production_ready'] else 'No'}")
        
        if assessment['validation_passed']:
            print("\n🚀 System shows promise for full 6-month validation!")
        else:
            print("\n🔧 System needs improvements before full validation.")
    else:
        print(f"\n💥 Quick validation failed: {results['error']}")

if __name__ == "__main__":
    main()