#!/usr/bin/env python3
"""
Test signal direction - maybe we need to flip the signal
"""

import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def test_signal_direction():
    """Test if flipping signal improves IC"""
    
    logger.info("🔄 TESTING SIGNAL DIRECTION")
    logger.info("=" * 40)
    
    # Load data
    data_path = Path(__file__).parent / 'data' / 'training_data_enhanced.csv'
    data = pd.read_csv(data_path)
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Focus on 2023-2024 for testing
    test_data = data[(data['Date'] >= '2023-01-01') & (data['Date'] <= '2024-12-31')]
    
    logger.info(f"Testing on {len(test_data):,} samples")
    
    # Create simple momentum signal
    features = ['return_5d_lag1', 'return_20d_lag1', 'volume_ratio_lag1']
    
    # Clean data
    for feat in features:
        if feat in test_data.columns:
            test_data[feat] = test_data[feat].fillna(0)
    
    # Simple scoring: momentum composite
    scores_normal = (test_data.get('return_5d_lag1', 0) * 0.6 + 
                    test_data.get('return_20d_lag1', 0) * 0.4)
    
    scores_inverted = -scores_normal  # Flip the signal
    
    # Get actual returns
    actual_returns = test_data['next_return_1d'].fillna(0)
    
    # Test both directions
    ic_normal = spearmanr(actual_returns, scores_normal)[0]
    ic_inverted = spearmanr(actual_returns, scores_inverted)[0]
    
    logger.info(f"📊 SIGNAL DIRECTION TEST:")
    logger.info(f"   Normal Signal IC: {ic_normal:.6f}")
    logger.info(f"   Inverted Signal IC: {ic_inverted:.6f}")
    
    # Test by date to see consistency
    daily_ics_normal = []
    daily_ics_inverted = []
    
    for date, group in test_data.groupby('Date'):
        if len(group) >= 5:
            returns = group['next_return_1d'].fillna(0)
            normal_scores = (group.get('return_5d_lag1', 0) * 0.6 + 
                           group.get('return_20d_lag1', 0) * 0.4)
            inverted_scores = -normal_scores
            
            ic_n = spearmanr(returns, normal_scores)[0]
            ic_i = spearmanr(returns, inverted_scores)[0]
            
            if not np.isnan(ic_n):
                daily_ics_normal.append(ic_n)
            if not np.isnan(ic_i):
                daily_ics_inverted.append(ic_i)
    
    avg_daily_ic_normal = np.mean(daily_ics_normal) if daily_ics_normal else 0
    avg_daily_ic_inverted = np.mean(daily_ics_inverted) if daily_ics_inverted else 0
    
    logger.info(f"   Daily Average - Normal: {avg_daily_ic_normal:.6f}")
    logger.info(f"   Daily Average - Inverted: {avg_daily_ic_inverted:.6f}")
    
    # Test hit rates
    hit_rate_normal = np.mean((actual_returns > 0) == (scores_normal > 0))
    hit_rate_inverted = np.mean((actual_returns > 0) == (scores_inverted > 0))
    
    logger.info(f"   Hit Rate - Normal: {hit_rate_normal:.3f}")
    logger.info(f"   Hit Rate - Inverted: {hit_rate_inverted:.3f}")
    
    # Determine best direction
    if abs(ic_inverted) > abs(ic_normal) and ic_inverted > 0:
        logger.info("✅ INVERTED signal is better!")
        return 'inverted'
    elif ic_normal > abs(ic_inverted) and ic_normal > 0:
        logger.info("✅ NORMAL signal is better!")
        return 'normal'
    else:
        logger.info("⚠️ Both signals weak - try different features")
        return 'weak'

if __name__ == "__main__":
    result = test_signal_direction()
    print(f"\n🎯 Recommendation: Use {result} signal direction")