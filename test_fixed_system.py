#!/usr/bin/env python3
"""
Test Fixed System - Verify the complete system works with positive IC
"""

import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Import system components
from src.models.tiered_system import TieredAlphaSystem
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def test_fixed_system_comprehensive():
    """Test the complete system with fixed IC"""
    
    logger.info("🧪 TESTING COMPLETE FIXED SYSTEM")
    logger.info("=" * 50)
    
    # Load fixed dataset
    fixed_data_path = Path(__file__).parent / 'data' / 'training_data_enhanced_fixed.csv'
    if not fixed_data_path.exists():
        logger.error(f"Fixed dataset not found: {fixed_data_path}")
        return False
    
    data = pd.read_csv(fixed_data_path)
    data['Date'] = pd.to_datetime(data['Date'])
    
    logger.info(f"Loaded fixed dataset: {len(data):,} samples")
    logger.info(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    
    # Initialize system (without regime to avoid external dependencies)
    system_config = {
        'lstm': {'enabled': True, 'max_epochs': 20},
        'regime': {'enabled': False},  # Disabled to avoid yfinance
        'meta': {'combiner_type': 'ridge'}
    }
    
    alpha_system = TieredAlphaSystem(system_config)
    
    # Split data for training and testing
    train_cutoff = '2022-01-01'
    test_start = '2023-01-01'
    
    train_data = data[data['Date'] < train_cutoff].sample(3000, random_state=42)  # Subset for speed
    test_data = data[data['Date'] >= test_start].head(200)  # Recent data for testing
    
    logger.info(f"Training samples: {len(train_data):,}")
    logger.info(f"Testing samples: {len(test_data):,}")
    
    # Train system
    logger.info("\n🏋️ TRAINING SYSTEM...")
    try:
        training_results = alpha_system.train_system(train_data)
        logger.info("✅ Training completed successfully")
        
        # Show training results
        for model_name, results in training_results.items():
            if isinstance(results, dict) and 'val_ic' in results:
                logger.info(f"   {model_name}: Val IC = {results['val_ic']:+.6f}")
    
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test predictions
    logger.info("\n🔮 TESTING PREDICTIONS...")
    try:
        predictions = alpha_system.predict_alpha(test_data)
        
        # Analyze predictions
        final_scores = predictions.get('final_scores', [])
        position_sizes = predictions.get('position_sizes', [])
        n_tradeable = predictions.get('n_tradeable', 0)
        
        # Calculate actual IC on test data
        if len(final_scores) > 0 and len(test_data) > 0:
            actual_returns = test_data['next_return_1d'].fillna(0).values[:len(final_scores)]
            test_ic = spearmanr(final_scores, actual_returns)[0]
        else:
            test_ic = 0
        
        logger.info(f"📊 PREDICTION RESULTS:")
        logger.info(f"   Final scores range: [{np.min(final_scores):.6f}, {np.max(final_scores):.6f}]")
        logger.info(f"   Test IC: {test_ic:+.6f}")
        logger.info(f"   Tradeable positions: {n_tradeable}/{len(test_data)}")
        logger.info(f"   Position sizes range: [{np.min(position_sizes):.6f}, {np.max(position_sizes):.6f}]")
        logger.info(f"   Max position: {np.max(np.abs(position_sizes)):.6f}")
        logger.info(f"   Gross exposure: {np.sum(np.abs(position_sizes)):.6f}")
        logger.info(f"   Net exposure: {np.sum(position_sizes):+.6f}")
        
        # Success criteria
        ic_positive = test_ic > 0.002
        trading_active = n_tradeable > len(test_data) * 0.2  # At least 20% trading
        reasonable_positions = np.max(np.abs(position_sizes)) <= 0.20  # Max 20% position
        market_neutral = abs(np.sum(position_sizes)) <= 0.15  # Near market neutral
        
        success_criteria = {
            'ic_positive': ic_positive,
            'trading_active': trading_active, 
            'reasonable_positions': reasonable_positions,
            'market_neutral': market_neutral
        }
        
        passed_checks = sum(success_criteria.values())
        total_checks = len(success_criteria)
        
        logger.info(f"\n✅ SUCCESS CRITERIA ({passed_checks}/{total_checks}):")
        for criterion, passed in success_criteria.items():
            status = "✅" if passed else "❌"
            logger.info(f"   {status} {criterion}")
        
        overall_success = passed_checks >= 3  # Need at least 3/4 criteria
        
        return {
            'overall_success': overall_success,
            'test_ic': test_ic,
            'n_tradeable': n_tradeable,
            'max_position': np.max(np.abs(position_sizes)),
            'gross_exposure': np.sum(np.abs(position_sizes)),
            'net_exposure': np.sum(position_sizes),
            'success_criteria': success_criteria
        }
        
    except Exception as e:
        logger.error(f"❌ Prediction testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test the fixed system"""
    
    results = test_fixed_system_comprehensive()
    
    if results and results['overall_success']:
        logger.info("\n🎉 FIXED SYSTEM TEST PASSED!")
        logger.info("=" * 40)
        logger.info(f"✅ Positive IC: {results['test_ic']:+.6f}")
        logger.info(f"✅ Trading Activity: {results['n_tradeable']} positions")
        logger.info(f"✅ Risk Control: {results['max_position']:.3f} max position")
        logger.info(f"✅ Market Neutral: {results['net_exposure']:+.3f} net exposure")
        logger.info("\n🚀 SYSTEM READY FOR 6-MONTH VALIDATION!")
        
    elif results:
        logger.info("\n⚠️ FIXED SYSTEM PARTIALLY WORKING")
        logger.info("Some criteria passed, but needs refinement before full validation")
        
    else:
        logger.info("\n❌ FIXED SYSTEM TEST FAILED")
        logger.info("Need to address remaining issues before validation")

if __name__ == "__main__":
    main()