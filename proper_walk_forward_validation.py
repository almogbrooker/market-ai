#!/usr/bin/env python3
"""
Proper Walk-Forward Validation - No Look-Ahead Bias
Train only on past data, test on future data sequentially
"""

import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Dict, List

# Import system components
from src.models.tiered_system import TieredAlphaSystem
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def proper_walk_forward_validation():
    """
    Proper walk-forward validation with no look-ahead bias:
    1. Use ONLY original dataset (no IC fixes using future data)
    2. Train on expanding window of past data
    3. Test on next month of data
    4. Never use future data for any calculations
    """
    
    logger.info("🚀 PROPER WALK-FORWARD VALIDATION")
    logger.info("=" * 60)
    logger.info("NO LOOK-AHEAD BIAS - ONLY PAST DATA FOR TRAINING")
    logger.info("=" * 60)
    
    # Load ORIGINAL dataset (not the "fixed" one)
    original_data_path = Path(__file__).parent / 'data' / 'training_data_2020_2024_complete.csv'
    if not original_data_path.exists():
        # Fallback to enhanced dataset but be clear about the issue
        original_data_path = Path(__file__).parent / 'data' / 'training_data_enhanced.csv'
        logger.warning("Using enhanced dataset - may contain look-ahead bias in features")
    
    data = pd.read_csv(original_data_path)
    data['Date'] = pd.to_datetime(data['Date'])
    
    logger.info(f"Loaded dataset: {len(data):,} samples")
    logger.info(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    
    # Define validation periods (no overlap, no future data)
    validation_periods = [
        {'train_end': '2022-12-31', 'test_start': '2023-01-01', 'test_end': '2023-03-31', 'name': 'Q1_2023'},
        {'train_end': '2023-03-31', 'test_start': '2023-04-01', 'test_end': '2023-06-30', 'name': 'Q2_2023'}, 
        {'train_end': '2023-06-30', 'test_start': '2023-07-01', 'test_end': '2023-09-30', 'name': 'Q3_2023'},
        {'train_end': '2023-09-30', 'test_start': '2023-10-01', 'test_end': '2023-12-31', 'name': 'Q4_2023'},
        {'train_end': '2023-12-31', 'test_start': '2024-01-01', 'test_end': '2024-03-31', 'name': 'Q1_2024'},
        {'train_end': '2024-03-31', 'test_start': '2024-04-01', 'test_end': '2024-06-30', 'name': 'Q2_2024'},
    ]
    
    results = []
    portfolio_value = 1000000  # Start with $1M
    
    for i, period in enumerate(validation_periods):
        logger.info(f"\n📅 PERIOD {i+1}: {period['name']}")
        logger.info(f"   Train: up to {period['train_end']}")
        logger.info(f"   Test: {period['test_start']} to {period['test_end']}")
        
        # Get training data (expanding window - ONLY PAST DATA)
        train_data = data[data['Date'] <= period['train_end']]
        test_data = data[
            (data['Date'] >= period['test_start']) & 
            (data['Date'] <= period['test_end'])
        ]
        
        logger.info(f"   Training samples: {len(train_data):,}")
        logger.info(f"   Testing samples: {len(test_data):,}")
        
        if len(train_data) < 1000 or len(test_data) < 100:
            logger.warning(f"   Insufficient data for {period['name']}")
            continue
        
        # Train system on ONLY past data
        try:
            logger.info("   🏋️ Training system...")
            
            system_config = {
                'lstm': {'enabled': True, 'max_epochs': 30},
                'regime': {'enabled': False},  # Avoid external data
                'meta': {'combiner_type': 'ridge'}
            }
            
            alpha_system = TieredAlphaSystem(system_config)
            
            # Use subset for training speed
            train_subset = train_data.sample(min(5000, len(train_data)), random_state=42)
            training_results = alpha_system.train_system(train_subset)
            
            logger.info(f"   ✅ Training complete")
            
            # Test on out-of-sample data
            logger.info("   🔮 Testing predictions...")
            
            # Group test data by date for daily trading simulation
            daily_groups = test_data.groupby('Date')
            daily_returns = []
            
            for date, day_data in daily_groups:
                if len(day_data) < 5:
                    daily_returns.append(0)
                    continue
                
                try:
                    # Generate predictions for this day
                    predictions = alpha_system.predict_alpha(day_data)
                    
                    scores = predictions.get('final_scores', [])
                    actual_returns = day_data['next_return_1d'].fillna(0).values
                    
                    if len(scores) >= 10 and len(actual_returns) == len(scores):
                        # Simple long/short strategy
                        n_positions = len(scores)
                        n_long = max(1, n_positions // 5)  # Top 20%
                        n_short = max(1, n_positions // 5)  # Bottom 20%
                        
                        sorted_indices = np.argsort(scores)
                        long_indices = sorted_indices[-n_long:]
                        short_indices = sorted_indices[:n_short]
                        
                        # Portfolio return for the day
                        long_return = actual_returns[long_indices].mean()
                        short_return = -actual_returns[short_indices].mean()  # Short
                        daily_return = (long_return + short_return) / 2
                        
                        daily_returns.append(daily_return)
                    else:
                        daily_returns.append(0)
                        
                except Exception as e:
                    logger.warning(f"   Prediction failed for {date}: {e}")
                    daily_returns.append(0)
            
            # Calculate period performance
            daily_returns = np.array(daily_returns)
            period_return = np.sum(daily_returns)
            
            # Calculate IC on test data
            all_scores = []
            all_returns = []
            
            for date, day_data in daily_groups:
                if len(day_data) >= 5:
                    try:
                        predictions = alpha_system.predict_alpha(day_data)
                        scores = predictions.get('final_scores', [])
                        returns = day_data['next_return_1d'].fillna(0).values
                        
                        if len(scores) == len(returns) and len(scores) > 0:
                            all_scores.extend(scores)
                            all_returns.extend(returns)
                    except:
                        continue
            
            if len(all_scores) > 10:
                period_ic = spearmanr(all_scores, all_returns)[0]
                if np.isnan(period_ic):
                    period_ic = 0
            else:
                period_ic = 0
            
            # Update portfolio value
            portfolio_value *= (1 + period_return)
            
            period_result = {
                'period': period['name'],
                'train_end': period['train_end'],
                'test_start': period['test_start'],
                'test_end': period['test_end'],
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                'period_return': period_return,
                'period_ic': period_ic,
                'portfolio_value': portfolio_value,
                'trading_days': len(daily_returns),
                'hit_rate': np.mean(daily_returns > 0) if len(daily_returns) > 0 else 0
            }
            
            results.append(period_result)
            
            logger.info(f"   📊 Period Results:")
            logger.info(f"      Return: {period_return*100:+.2f}%")
            logger.info(f"      IC: {period_ic:+.6f}")
            logger.info(f"      Portfolio Value: ${portfolio_value:,.0f}")
            
        except Exception as e:
            logger.error(f"   ❌ Period {period['name']} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final results
    if results:
        total_return = (portfolio_value / 1000000 - 1) * 100
        avg_ic = np.mean([r['period_ic'] for r in results])
        avg_hit_rate = np.mean([r['hit_rate'] for r in results])
        
        logger.info("\n🎉 PROPER VALIDATION COMPLETE!")
        logger.info("=" * 50)
        logger.info(f"Total Return: {total_return:+.2f}%")
        logger.info(f"Final Portfolio: ${portfolio_value:,.0f}")
        logger.info(f"Average IC: {avg_ic:+.6f}")
        logger.info(f"Average Hit Rate: {avg_hit_rate:.1%}")
        logger.info(f"Periods Tested: {len(results)}")
        
        # Assessment
        if avg_ic > 0.002 and total_return > 5:
            logger.info("✅ VALIDATION PASSED - No Look-Ahead Bias")
        elif avg_ic > 0 and total_return > 0:
            logger.info("⚠️ VALIDATION MARGINAL - Positive but weak")
        else:
            logger.info("❌ VALIDATION FAILED - Negative performance")
        
        return {
            'total_return_pct': total_return,
            'final_portfolio_value': portfolio_value,
            'average_ic': avg_ic,
            'average_hit_rate': avg_hit_rate,
            'periods_tested': len(results),
            'period_results': results,
            'validation_passed': avg_ic > 0.002 and total_return > 5
        }
    
    else:
        logger.error("❌ NO VALID RESULTS")
        return {'validation_passed': False, 'error': 'No valid periods tested'}

def main():
    """Run proper validation"""
    
    results = proper_walk_forward_validation()
    
    if results.get('validation_passed', False):
        print(f"\n🎉 PROPER VALIDATION PASSED!")
        print(f"Total Return: {results['total_return_pct']:+.2f}%")
        print(f"Average IC: {results['average_ic']:+.6f}")
        print("System validated without look-ahead bias!")
    else:
        print(f"\n❌ PROPER VALIDATION FAILED")
        print("System does not work on true out-of-sample data")
        if 'total_return_pct' in results:
            print(f"Return: {results['total_return_pct']:+.2f}%")
            print(f"IC: {results['average_ic']:+.6f}")

if __name__ == "__main__":
    main()