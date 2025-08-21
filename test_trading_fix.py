#!/usr/bin/env python3
"""
Test the trading fix to ensure positions are generated
"""

import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Import core system
from src.models.tiered_system import TieredAlphaSystem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def test_trading_fix():
    """Test if the trading fix works"""
    
    logger.info("🧪 TESTING TRADING FIX")
    logger.info("=" * 30)
    
    # Initialize system without regime (to avoid yfinance issues)
    system_config = {
        'lstm': {'enabled': True, 'max_epochs': 10},
        'regime': {'enabled': False},
        'meta': {'combiner_type': 'ridge'}
    }
    
    alpha_system = TieredAlphaSystem(system_config)
    
    # Load and train on small sample
    data_path = Path(__file__).parent / 'data' / 'training_data_enhanced.csv'
    data = pd.read_csv(data_path)
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Quick training
    train_data = data[data['Date'] < '2022-01-01'].sample(2000, random_state=42)
    logger.info(f"Training on {len(train_data)} samples...")
    
    alpha_system.train_system(train_data)
    
    # Test prediction on new data
    test_data = data[data['Date'] >= '2023-01-01'].head(100)
    logger.info(f"Testing on {len(test_data)} samples...")
    
    predictions = alpha_system.predict_alpha(test_data)
    
    # Check results
    n_tradeable = predictions.get('n_tradeable', 0)
    position_sizes = predictions.get('position_sizes', [])
    final_scores = predictions.get('final_scores', [])
    
    logger.info(f"📊 RESULTS:")
    logger.info(f"   Final scores range: [{np.min(final_scores):.6f}, {np.max(final_scores):.6f}]")
    logger.info(f"   Tradeable positions: {n_tradeable}/{len(test_data)}")
    logger.info(f"   Position sizes range: [{np.min(position_sizes):.6f}, {np.max(position_sizes):.6f}]")
    logger.info(f"   Max absolute position: {np.max(np.abs(position_sizes)):.6f}")
    
    if n_tradeable > 0:
        logger.info("✅ FIX SUCCESSFUL - System is generating trades!")
        return True
    else:
        logger.info("❌ FIX FAILED - Still no trades generated")
        return False

if __name__ == "__main__":
    success = test_trading_fix()
    if success:
        print("\n🎉 Trading fix successful! System ready for validation.")
    else:
        print("\n🔧 Still need more adjustments to enable trading.")