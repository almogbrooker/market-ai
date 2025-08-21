#!/usr/bin/env python3
"""
Complete 6-Month Validation Pipeline Deployment
Orchestrates the entire validation system with all components
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

# Import all validation components
from validation.paper_trading_engine import PaperTradingEngine
from validation.daily_logger import ValidationLogger
from validation.regime_stress_tester import RegimeStressTester
from validation.parameter_sanity_checker import ParameterSanityChecker
from validation.research_sandbox import ResearchSandbox

# Import main system
from src.models.tiered_system import TieredAlphaSystem

# Setup logging
import os
logs_dir = Path(__file__).parent / 'validation' / 'logs'
logs_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / 'validation_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CompleteValidationPipeline:
    """
    Complete 6-Month Validation Pipeline
    Orchestrates all validation components for comprehensive system testing
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self._load_config(config_file)
        self.base_dir = Path(__file__).parent
        self.validation_dir = self.base_dir / "validation"
        
        # Initialize components
        self.alpha_system = None
        self.paper_trader = None
        self.daily_logger = None
        self.stress_tester = None
        self.sanity_checker = None
        self.research_sandbox = None
        
        # Results tracking
        self.validation_results = {}
        self.daily_reports = []
        
        logger.info("🚀 COMPLETE VALIDATION PIPELINE INITIALIZED")
        logger.info("=" * 60)
    
    def _load_config(self, config_file: Optional[str] = None) -> Dict:
        """Load validation configuration"""
        
        default_config = {
            'validation_period': {
                'start_date': '2023-01-01',
                'end_date': '2023-06-30',
                'oos_period': '6_months'
            },
            'paper_trading': {
                'initial_capital': 1000000,
                'start_date': '2023-01-01',
                'end_date': '2023-06-30',
                'kill_switches': {
                    'vix_threshold': 35,
                    'max_turnover': 0.8,
                    'max_drawdown': 0.15
                },
                'validation_gates': {
                    'min_sharpe': 0.5,
                    'max_drawdown_pct': -15,
                    'min_days': 120
                }
            },
            'stress_testing': {
                'vix_shock_levels': [25, 30, 35, 45, 60],
                'market_crash_scenarios': [-0.05, -0.10, -0.20],
                'min_pass_rate': 0.75
            },
            'parameter_monitoring': {
                'max_training_ic': 0.03,
                'ic_degradation_threshold': 0.5,
                'retrain_frequency_days': 30,
                'warning_threshold': 0.7,
                'critical_threshold': 0.5
            },
            'research_models': {
                'enabled': True,
                'run_experiments': True,
                'models_e_f_testing': True
            },
            'reporting': {
                'daily_reports': True,
                'weekly_summaries': True,
                'monthly_deep_dives': True,
                'final_report': True
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file) as f:
                    user_config = json.load(f)
                # Merge configs (simplified)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config file {config_file}: {e}")
        
        return default_config
    
    def initialize_validation_system(self) -> Dict:
        """Initialize the complete validation system"""
        
        logger.info("🔧 INITIALIZING VALIDATION SYSTEM COMPONENTS")
        logger.info("=" * 50)
        
        initialization_results = {}
        
        try:
            # Step 1: Initialize Alpha System
            logger.info("🤖 Initializing Alpha System...")
            system_config = {
                'lstm': {'enabled': True, 'max_epochs': 50},
                'regime': {'enabled': True},
                'meta': {'combiner_type': 'ridge'}
            }
            
            self.alpha_system = TieredAlphaSystem(system_config)
            
            # Load training data for initial system training
            training_data = self._load_training_data()
            if training_data is not None:
                # Train the system before validation
                logger.info("   Training alpha system for validation...")
                training_results = self.alpha_system.train_system(training_data)
                initialization_results['alpha_system'] = {
                    'status': 'trained',
                    'results': training_results
                }
            else:
                initialization_results['alpha_system'] = {
                    'status': 'failed',
                    'error': 'No training data available'
                }
            
            # Step 2: Initialize Paper Trading Engine
            logger.info("📊 Initializing Paper Trading Engine...")
            self.paper_trader = PaperTradingEngine(self.config['paper_trading'])
            initialization_results['paper_trading'] = {'status': 'initialized'}
            
            # Step 3: Initialize Daily Logger
            logger.info("📝 Initializing Daily Logger...")
            logs_dir = self.validation_dir / "logs"
            self.daily_logger = ValidationLogger(logs_dir)
            initialization_results['daily_logger'] = {'status': 'initialized'}
            
            # Step 4: Initialize Stress Tester
            logger.info("🧪 Initializing Stress Tester...")
            if self.alpha_system and self.alpha_system.is_trained:
                self.stress_tester = RegimeStressTester(
                    self.alpha_system, 
                    self.config['stress_testing']
                )
                initialization_results['stress_tester'] = {'status': 'initialized'}
            else:
                initialization_results['stress_tester'] = {
                    'status': 'failed', 
                    'error': 'Alpha system not trained'
                }
            
            # Step 5: Initialize Parameter Sanity Checker
            logger.info("🔍 Initializing Parameter Sanity Checker...")
            if self.alpha_system:
                self.sanity_checker = ParameterSanityChecker(
                    self.alpha_system,
                    self.config['parameter_monitoring']
                )
                initialization_results['sanity_checker'] = {'status': 'initialized'}
            else:
                initialization_results['sanity_checker'] = {
                    'status': 'failed',
                    'error': 'Alpha system not available'
                }
            
            # Step 6: Initialize Research Sandbox
            logger.info("🧪 Initializing Research Sandbox...")
            if self.config['research_models']['enabled']:
                self.research_sandbox = ResearchSandbox(self.config.get('research_models', {}))
                initialization_results['research_sandbox'] = {'status': 'initialized'}
            else:
                initialization_results['research_sandbox'] = {'status': 'disabled'}
            
            logger.info("✅ VALIDATION SYSTEM INITIALIZATION COMPLETE")
            
            # Overall initialization status
            all_critical_initialized = all([
                self.alpha_system is not None,
                self.paper_trader is not None,
                self.daily_logger is not None
            ])
            
            initialization_results['overall'] = {
                'success': all_critical_initialized,
                'critical_components_ready': all_critical_initialized,
                'optional_components': {
                    'stress_tester': self.stress_tester is not None,
                    'sanity_checker': self.sanity_checker is not None,
                    'research_sandbox': self.research_sandbox is not None
                }
            }
            
            return initialization_results
            
        except Exception as e:
            logger.error(f"❌ INITIALIZATION FAILED: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'overall': {'success': False, 'error': str(e)},
                'initialization_failed': True
            }
    
    def _load_training_data(self) -> Optional[pd.DataFrame]:
        """Load training data for system initialization"""
        
        data_path = self.base_dir / 'data' / 'training_data_enhanced.csv'
        
        if data_path.exists():
            try:
                data = pd.read_csv(data_path)
                data['Date'] = pd.to_datetime(data['Date'])
                
                # Use pre-validation period for training
                validation_start = pd.to_datetime(self.config['validation_period']['start_date'])
                training_data = data[data['Date'] < validation_start]
                
                logger.info(f"   Loaded {len(training_data):,} training samples")
                logger.info(f"   Training period: {training_data['Date'].min()} to {training_data['Date'].max()}")
                
                return training_data
                
            except Exception as e:
                logger.error(f"Failed to load training data: {e}")
                return None
        else:
            logger.warning(f"Training data not found: {data_path}")
            return None
    
    def run_complete_6month_validation(self) -> Dict:
        """Run the complete 6-month validation pipeline"""
        
        logger.info("🚀 STARTING COMPLETE 6-MONTH VALIDATION PIPELINE")
        logger.info("=" * 60)
        
        pipeline_start_time = datetime.now()
        
        try:
            # Phase 1: System Initialization
            logger.info("📋 PHASE 1: SYSTEM INITIALIZATION")
            init_results = self.initialize_validation_system()
            
            if not init_results['overall']['success']:
                return {
                    'success': False,
                    'phase_failed': 'initialization',
                    'error': init_results.get('overall', {}).get('error', 'Unknown initialization error')
                }
            
            # Phase 2: Paper Trading Validation
            logger.info("📋 PHASE 2: 6-MONTH PAPER TRADING VALIDATION")
            paper_trading_results = self.paper_trader.run_6month_validation()
            
            # Phase 3: Stress Testing
            logger.info("📋 PHASE 3: REGIME STRESS TESTING")
            if self.stress_tester:
                validation_data = self._load_validation_data()
                if validation_data is not None:
                    stress_test_results = self.stress_tester.run_complete_stress_test(validation_data)
                else:
                    stress_test_results = {'success': False, 'error': 'No validation data'}
            else:
                stress_test_results = {'success': False, 'error': 'Stress tester not initialized'}
            
            # Phase 4: Parameter Sanity Checks
            logger.info("📋 PHASE 4: PARAMETER SANITY CHECKS")
            if self.sanity_checker and validation_data is not None:
                # Run sanity checks on validation data
                training_data = self._load_training_data()
                sanity_results = self.sanity_checker.run_comprehensive_sanity_check(
                    validation_data.head(1000),  # Current data sample
                    training_data,               # Historical data
                    self.config['validation_period']['end_date']
                )
            else:
                sanity_results = {'success': False, 'error': 'Sanity checker not available'}
            
            # Phase 5: Research Models Testing (Optional)
            logger.info("📋 PHASE 5: RESEARCH MODELS (E/F) TESTING")
            research_results = {'status': 'disabled'}
            
            if self.config['research_models']['run_experiments'] and self.research_sandbox:
                try:
                    training_data = self._load_training_data()
                    if training_data is not None and validation_data is not None:
                        research_results = self.research_sandbox.run_research_experiment(
                            training_data.tail(2000),  # Recent training sample
                            validation_data.head(500), # Validation sample
                            "validation_pipeline_experiment"
                        )
                    else:
                        research_results = {'status': 'failed', 'error': 'No data for research experiments'}
                except Exception as e:
                    logger.warning(f"Research experiments failed: {e}")
                    research_results = {'status': 'failed', 'error': str(e)}
            
            # Phase 6: Generate Comprehensive Report
            logger.info("📋 PHASE 6: COMPREHENSIVE VALIDATION REPORT")
            
            pipeline_end_time = datetime.now()
            pipeline_duration = pipeline_end_time - pipeline_start_time
            
            comprehensive_results = {
                'validation_metadata': {
                    'pipeline_version': '1.0',
                    'start_time': pipeline_start_time.isoformat(),
                    'end_time': pipeline_end_time.isoformat(),
                    'duration_hours': pipeline_duration.total_seconds() / 3600,
                    'validation_period': self.config['validation_period'],
                },
                
                'phase_results': {
                    '1_initialization': init_results,
                    '2_paper_trading': paper_trading_results,
                    '3_stress_testing': stress_test_results,
                    '4_sanity_checks': sanity_results,
                    '5_research_models': research_results
                },
                
                'overall_assessment': self._generate_overall_assessment(
                    paper_trading_results, 
                    stress_test_results, 
                    sanity_results,
                    research_results
                ),
                
                'recommendations': self._generate_final_recommendations(
                    paper_trading_results,
                    stress_test_results, 
                    sanity_results,
                    research_results
                ),
                
                'next_steps': self._generate_next_steps(
                    paper_trading_results,
                    stress_test_results,
                    sanity_results
                )
            }
            
            # Save comprehensive report
            self._save_comprehensive_report(comprehensive_results)
            
            # Log final summary
            self._log_final_validation_summary(comprehensive_results)
            
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"❌ VALIDATION PIPELINE FAILED: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'pipeline_failed': True,
                'timestamp': datetime.now().isoformat()
            }
    
    def _load_validation_data(self) -> Optional[pd.DataFrame]:
        """Load out-of-sample validation data"""
        
        data_path = self.base_dir / 'data' / 'training_data_enhanced.csv'
        
        if data_path.exists():
            try:
                data = pd.read_csv(data_path)
                data['Date'] = pd.to_datetime(data['Date'])
                
                # Filter to validation period
                start_date = pd.to_datetime(self.config['validation_period']['start_date'])
                end_date = pd.to_datetime(self.config['validation_period']['end_date'])
                
                validation_data = data[
                    (data['Date'] >= start_date) & (data['Date'] <= end_date)
                ]
                
                logger.info(f"   Loaded {len(validation_data):,} validation samples")
                return validation_data
                
            except Exception as e:
                logger.error(f"Failed to load validation data: {e}")
                return None
        else:
            logger.warning("No validation data available")
            return None
    
    def _generate_overall_assessment(self, 
                                   paper_trading: Dict,
                                   stress_testing: Dict,
                                   sanity_checks: Dict,
                                   research: Dict) -> Dict:
        """Generate overall validation assessment"""
        
        assessment = {
            'validation_passed': False,
            'confidence_level': 'low',
            'production_ready': False,
            'key_metrics': {},
            'critical_issues': [],
            'strengths': []
        }
        
        try:
            # Paper trading assessment
            if 'performance_metrics' in paper_trading:
                metrics = paper_trading['performance_metrics']
                sharpe = metrics.get('sharpe_ratio', 0)
                max_dd = metrics.get('max_drawdown_pct', -100)
                total_return = metrics.get('total_return_pct', 0)
                
                assessment['key_metrics'].update({
                    'sharpe_ratio': sharpe,
                    'max_drawdown_pct': max_dd,
                    'total_return_pct': total_return
                })
                
                # Performance gates
                if sharpe >= 0.5 and max_dd >= -15:
                    assessment['strengths'].append("Paper trading performance meets minimum thresholds")
                else:
                    assessment['critical_issues'].append(f"Poor paper trading metrics: Sharpe {sharpe:.2f}, DD {max_dd:.1f}%")
            
            # Stress testing assessment
            if stress_testing.get('test_summary', {}).get('stress_test_passed', False):
                assessment['strengths'].append("System passes regime stress tests")
            else:
                assessment['critical_issues'].append("System fails regime stress tests")
            
            # Sanity checks assessment
            sanity_score = sanity_checks.get('overall_sanity_score', {}).get('score', 0)
            if sanity_score >= 0.7:
                assessment['strengths'].append(f"Parameter sanity score healthy: {sanity_score:.1%}")
            else:
                assessment['critical_issues'].append(f"Parameter sanity concerns: {sanity_score:.1%}")
            
            # Overall determination
            paper_passed = paper_trading.get('validation_passed', False)
            stress_passed = stress_testing.get('test_summary', {}).get('stress_test_passed', False)
            sanity_passed = sanity_score >= 0.7
            
            passed_count = sum([paper_passed, stress_passed, sanity_passed])
            
            if passed_count >= 3:
                assessment['validation_passed'] = True
                assessment['confidence_level'] = 'high'
                assessment['production_ready'] = True
            elif passed_count >= 2:
                assessment['validation_passed'] = True
                assessment['confidence_level'] = 'medium'
                assessment['production_ready'] = False
            else:
                assessment['validation_passed'] = False
                assessment['confidence_level'] = 'low'
                assessment['production_ready'] = False
            
        except Exception as e:
            logger.warning(f"Assessment generation failed: {e}")
            assessment['error'] = str(e)
        
        return assessment
    
    def _generate_final_recommendations(self,
                                      paper_trading: Dict,
                                      stress_testing: Dict,
                                      sanity_checks: Dict,
                                      research: Dict) -> List[str]:
        """Generate final recommendations based on all validation results"""
        
        recommendations = []
        
        try:
            # Paper trading recommendations
            if 'performance_metrics' in paper_trading:
                metrics = paper_trading['performance_metrics']
                if metrics.get('sharpe_ratio', 0) < 0.5:
                    recommendations.append("CRITICAL: Improve risk-adjusted returns before production deployment")
                
                if metrics.get('max_drawdown_pct', -100) < -15:
                    recommendations.append("CRITICAL: Implement stronger drawdown controls")
            
            # Stress testing recommendations
            if not stress_testing.get('test_summary', {}).get('stress_test_passed', True):
                recommendations.append("Address regime stress test failures - improve VIX response and crash positioning")
            
            # Sanity check recommendations
            sanity_recs = sanity_checks.get('recommendations', [])
            recommendations.extend(sanity_recs[:2])  # Top 2 sanity recommendations
            
            # Research model recommendations
            if research.get('combined_analysis', {}).get('implementation_priority') == 'high':
                recommendations.append("Consider integrating Models E/F for additional alpha generation")
            
            # General recommendations
            if not recommendations:
                recommendations.append("System validation successful - proceed with production deployment planning")
            
        except Exception as e:
            logger.warning(f"Recommendation generation failed: {e}")
            recommendations.append("Review validation results manually due to processing error")
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _generate_next_steps(self,
                           paper_trading: Dict,
                           stress_testing: Dict,
                           sanity_checks: Dict) -> List[str]:
        """Generate concrete next steps"""
        
        next_steps = []
        
        # Determine validation outcome
        overall_passed = (
            paper_trading.get('validation_passed', False) and
            stress_testing.get('test_summary', {}).get('stress_test_passed', False) and
            sanity_checks.get('overall_sanity_score', {}).get('passed', False)
        )
        
        if overall_passed:
            next_steps = [
                "✅ Validation Complete: Begin production deployment preparation",
                "Set up real-time data feeds and broker API integration",
                "Implement live trading infrastructure with kill switches",
                "Start with small capital allocation (e.g., $100K) for live testing",
                "Monitor performance daily for first month of live trading"
            ]
        else:
            next_steps = [
                "❌ Validation Failed: Address critical issues before re-validation",
                "Focus on primary failure points identified in assessment",
                "Retrain models with additional regularization if overfitting detected",
                "Re-run 6-month validation after implementing fixes",
                "Consider extending validation period to 12 months for more confidence"
            ]
        
        return next_steps
    
    def _save_comprehensive_report(self, results: Dict):
        """Save comprehensive validation report"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.validation_dir / "logs" / f"comprehensive_validation_report_{timestamp}.json"
        
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"📊 Comprehensive report saved: {report_path}")
            
            # Also save summary report
            summary_path = self.validation_dir / "logs" / f"validation_summary_{timestamp}.txt"
            self._save_summary_report(results, summary_path)
            
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
    
    def _save_summary_report(self, results: Dict, summary_path: Path):
        """Save human-readable summary report"""
        
        try:
            with open(summary_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("COMPLETE 6-MONTH VALIDATION PIPELINE REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                # Metadata
                metadata = results['validation_metadata']
                f.write(f"Validation Period: {metadata['validation_period']['start_date']} to {metadata['validation_period']['end_date']}\n")
                f.write(f"Pipeline Duration: {metadata['duration_hours']:.1f} hours\n")
                f.write(f"Timestamp: {metadata['end_time']}\n\n")
                
                # Overall Assessment
                assessment = results['overall_assessment']
                f.write("OVERALL ASSESSMENT:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Validation Passed: {'✅ YES' if assessment['validation_passed'] else '❌ NO'}\n")
                f.write(f"Confidence Level: {assessment['confidence_level'].upper()}\n")
                f.write(f"Production Ready: {'✅ YES' if assessment['production_ready'] else '❌ NO'}\n\n")
                
                # Key Metrics
                if assessment.get('key_metrics'):
                    f.write("KEY METRICS:\n")
                    f.write("-" * 40 + "\n")
                    for metric, value in assessment['key_metrics'].items():
                        f.write(f"{metric}: {value:.3f}\n")
                    f.write("\n")
                
                # Recommendations
                f.write("RECOMMENDATIONS:\n")
                f.write("-" * 40 + "\n")
                for i, rec in enumerate(results['recommendations'], 1):
                    f.write(f"{i}. {rec}\n")
                f.write("\n")
                
                # Next Steps
                f.write("NEXT STEPS:\n")
                f.write("-" * 40 + "\n")
                for i, step in enumerate(results['next_steps'], 1):
                    f.write(f"{i}. {step}\n")
                
            logger.info(f"📝 Summary report saved: {summary_path}")
            
        except Exception as e:
            logger.error(f"Failed to save summary report: {e}")
    
    def _log_final_validation_summary(self, results: Dict):
        """Log final validation pipeline summary"""
        
        logger.info("🏁 COMPLETE VALIDATION PIPELINE FINISHED")
        logger.info("=" * 60)
        
        assessment = results['overall_assessment']
        metadata = results['validation_metadata']
        
        # Overall result
        if assessment['validation_passed']:
            logger.info("🎉 VALIDATION RESULT: ✅ PASSED")
            logger.info(f"   Confidence Level: {assessment['confidence_level'].upper()}")
            logger.info(f"   Production Ready: {'Yes' if assessment['production_ready'] else 'No'}")
        else:
            logger.info("💥 VALIDATION RESULT: ❌ FAILED")
            logger.info("   System requires improvements before production deployment")
        
        # Key metrics
        if assessment.get('key_metrics'):
            logger.info("\n📊 KEY PERFORMANCE METRICS:")
            for metric, value in assessment['key_metrics'].items():
                logger.info(f"   {metric}: {value:.3f}")
        
        # Duration
        logger.info(f"\n⏱️ Pipeline Duration: {metadata['duration_hours']:.1f} hours")
        
        # Critical issues
        if assessment.get('critical_issues'):
            logger.info(f"\n🚨 CRITICAL ISSUES ({len(assessment['critical_issues'])}):")
            for issue in assessment['critical_issues']:
                logger.info(f"   • {issue}")
        
        # Strengths
        if assessment.get('strengths'):
            logger.info(f"\n💪 STRENGTHS ({len(assessment['strengths'])}):")
            for strength in assessment['strengths']:
                logger.info(f"   • {strength}")
        
        # Top recommendations
        logger.info(f"\n💡 TOP RECOMMENDATIONS:")
        for i, rec in enumerate(results['recommendations'][:3], 1):
            logger.info(f"   {i}. {rec}")
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 Complete validation report saved in validation/logs/")
        logger.info("🚀 Ready for next phase based on validation outcome!")

def main():
    """Run the complete validation pipeline"""
    
    # Create validation pipeline
    pipeline = CompleteValidationPipeline()
    
    # Run complete 6-month validation
    results = pipeline.run_complete_6month_validation()
    
    if results.get('success', True):  # Default to True if not specified
        print("\n🎉 VALIDATION PIPELINE COMPLETED SUCCESSFULLY!")
        
        assessment = results.get('overall_assessment', {})
        if assessment.get('validation_passed', False):
            print("✅ System PASSED validation - ready for production consideration")
        else:
            print("❌ System FAILED validation - improvements required")
        
        print(f"\nDetailed results saved in validation/logs/")
        
    else:
        print("\n💥 VALIDATION PIPELINE FAILED!")
        print(f"Error: {results.get('error', 'Unknown error')}")
        print("Check logs for detailed error information")

if __name__ == "__main__":
    main()