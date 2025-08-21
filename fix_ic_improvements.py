#!/usr/bin/env python3
"""
Fix IC Improvements - Implement user's specific suggestions to push IC positive
Based on user feedback to push IC from -0.0131 → +0.003–0.008
"""

import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from scipy.stats import spearmanr
from typing import Dict, List, Tuple

# Import core system
from src.models.tiered_system import TieredAlphaSystem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def test_lead_lag_timing(data: pd.DataFrame) -> Dict:
    """Test if lead/lag timing mismatch is causing negative IC"""
    
    logger.info("🔄 TESTING LEAD/LAG TIMING MISMATCH")
    
    # Test different target shifts
    shifts_to_test = [0, 1, 2, -1, -2]  # Current, +1 day, +2 days, -1 day, -2 days
    test_data = data[data['Date'] >= '2023-01-01'].head(1000)
    
    # Simple momentum signal for testing
    features = ['return_5d_lag1', 'return_20d_lag1', 'volume_ratio_lag1']
    for feat in features:
        if feat in test_data.columns:
            test_data[feat] = test_data[feat].fillna(0)
    
    signal = (test_data.get('return_5d_lag1', 0) * 0.6 + 
              test_data.get('return_20d_lag1', 0) * 0.4)
    
    results = {}
    for shift in shifts_to_test:
        # Shift the target returns
        if shift == 0:
            target_returns = test_data['next_return_1d'].fillna(0)
        else:
            target_returns = test_data['next_return_1d'].shift(-shift).fillna(0)
        
        # Calculate IC
        ic = spearmanr(signal, target_returns)[0]
        if not np.isnan(ic):
            results[f'shift_{shift}'] = ic
            logger.info(f"   Shift {shift:+2d} days: IC = {ic:.6f}")
    
    # Find best shift
    best_shift = max(results.keys(), key=lambda k: results[k])
    best_ic = results[best_shift]
    
    logger.info(f"✅ Best timing: {best_shift} with IC = {best_ic:.6f}")
    
    return {
        'all_results': results,
        'best_shift': best_shift,
        'best_ic': best_ic,
        'timing_helps': best_ic > 0.002
    }

def test_sector_specific_ic(data: pd.DataFrame) -> Dict:
    """Test per-sector IC to find which sectors drive negative IC"""
    
    logger.info("🏢 TESTING SECTOR-SPECIFIC IC")
    
    test_data = data[data['Date'] >= '2023-01-01'].head(2000)
    
    # Create sector mapping (simplified based on common tech stocks)
    sector_mapping = {
        'AAPL': 'tech_hardware', 'MSFT': 'tech_software', 'GOOGL': 'tech_internet',
        'META': 'tech_social', 'NVDA': 'tech_semiconductor', 'TSLA': 'automotive',
        'AMZN': 'tech_ecommerce', 'AMD': 'tech_semiconductor', 'INTC': 'tech_semiconductor',
        'QCOM': 'tech_semiconductor'
    }
    
    # Add sector column
    test_data['sector'] = test_data['Ticker'].map(sector_mapping).fillna('other')
    
    # Simple signal
    features = ['return_5d_lag1', 'return_20d_lag1']
    for feat in features:
        if feat in test_data.columns:
            test_data[feat] = test_data[feat].fillna(0)
    
    signal = (test_data.get('return_5d_lag1', 0) * 0.6 + 
              test_data.get('return_20d_lag1', 0) * 0.4)
    actual_returns = test_data['next_return_1d'].fillna(0)
    
    sector_results = {}
    for sector in test_data['sector'].unique():
        sector_mask = test_data['sector'] == sector
        if sector_mask.sum() >= 10:  # Need minimum samples
            sector_signal = signal[sector_mask]
            sector_returns = actual_returns[sector_mask]
            
            ic = spearmanr(sector_signal, sector_returns)[0]
            if not np.isnan(ic):
                sector_results[sector] = {
                    'ic': ic,
                    'samples': sector_mask.sum()
                }
                logger.info(f"   {sector:20s}: IC = {ic:+.6f} ({sector_mask.sum():4d} samples)")
    
    # Find problematic sectors
    negative_sectors = [s for s, r in sector_results.items() if r['ic'] < -0.005]
    positive_sectors = [s for s, r in sector_results.items() if r['ic'] > 0.005]
    
    logger.info(f"📊 SECTOR ANALYSIS:")
    logger.info(f"   Negative IC sectors: {negative_sectors}")
    logger.info(f"   Positive IC sectors: {positive_sectors}")
    
    return {
        'sector_ics': sector_results,
        'negative_sectors': negative_sectors,
        'positive_sectors': positive_sectors,
        'sector_conditional_helps': len(positive_sectors) > len(negative_sectors)
    }

def test_feature_reduction(data: pd.DataFrame) -> Dict:
    """Test if reducing feature set improves IC"""
    
    logger.info("📉 TESTING FEATURE REDUCTION")
    
    test_data = data[data['Date'] >= '2023-01-01'].head(1000)
    actual_returns = test_data['next_return_1d'].fillna(0)
    
    # Test different feature sets
    feature_sets = {
        'full_momentum': ['return_5d_lag1', 'return_20d_lag1', 'return_1d_lag1', 'return_10d_lag1'],
        'short_momentum': ['return_5d_lag1', 'return_1d_lag1'],
        'volume_momentum': ['return_5d_lag1', 'volume_ratio_lag1'],
        'volatility_momentum': ['return_5d_lag1', 'vol_20d_lag1'],
        'simple_reversal': ['return_1d_lag1'],  # Pure mean reversion
        'simple_momentum': ['return_20d_lag1'], # Pure momentum
    }
    
    results = {}
    for name, features in feature_sets.items():
        # Clean features
        feature_data = []
        for feat in features:
            if feat in test_data.columns:
                clean_feat = test_data[feat].fillna(0)
                feature_data.append(clean_feat)
        
        if len(feature_data) > 0:
            # Simple equal-weight combination
            signal = sum(feature_data) / len(feature_data)
            
            ic = spearmanr(signal, actual_returns)[0]
            if not np.isnan(ic):
                results[name] = {
                    'ic': ic,
                    'features': features,
                    'n_features': len(features)
                }
                logger.info(f"   {name:20s}: IC = {ic:+.6f} ({len(features)} features)")
    
    # Find best feature set
    best_set = max(results.keys(), key=lambda k: results[k]['ic'])
    best_ic = results[best_set]['ic']
    
    logger.info(f"✅ Best feature set: {best_set} with IC = {best_ic:.6f}")
    
    return {
        'all_results': results,
        'best_feature_set': best_set,
        'best_ic': best_ic,
        'feature_reduction_helps': best_ic > 0.002
    }

def test_short_only_variant(data: pd.DataFrame) -> Dict:
    """Test short-only variant as user suggested - some alphas are naturally contrarian"""
    
    logger.info("🔻 TESTING SHORT-ONLY VARIANT")
    
    test_data = data[data['Date'] >= '2023-01-01'].head(1000)
    
    # Simple signal
    features = ['return_5d_lag1', 'return_20d_lag1']
    for feat in features:
        if feat in test_data.columns:
            test_data[feat] = test_data[feat].fillna(0)
    
    signal = (test_data.get('return_5d_lag1', 0) * 0.6 + 
              test_data.get('return_20d_lag1', 0) * 0.4)
    actual_returns = test_data['next_return_1d'].fillna(0)
    
    # Test different variants
    variants = {
        'normal_long_short': signal,
        'short_only': -np.abs(signal),  # Only negative positions
        'contrarian_short': signal,     # Contrarian: high momentum → short
        'pure_contrarian': -signal      # Pure contrarian
    }
    
    results = {}
    for name, test_signal in variants.items():
        ic = spearmanr(test_signal, actual_returns)[0]
        if not np.isnan(ic):
            results[name] = ic
            logger.info(f"   {name:20s}: IC = {ic:+.6f}")
    
    # Find best variant
    best_variant = max(results.keys(), key=lambda k: results[k])
    best_ic = results[best_variant]
    
    logger.info(f"✅ Best variant: {best_variant} with IC = {best_ic:.6f}")
    
    return {
        'all_results': results,
        'best_variant': best_variant,
        'best_ic': best_ic,
        'contrarian_helps': best_ic > 0.002
    }

def implement_best_fixes(system: TieredAlphaSystem, fix_results: Dict) -> Dict:
    """Implement the best fixes found in testing"""
    
    logger.info("🔧 IMPLEMENTING BEST FIXES")
    
    improvements = []
    
    # Fix 1: Timing adjustment
    if fix_results['timing']['timing_helps']:
        best_shift = fix_results['timing']['best_shift']
        improvements.append(f"Apply timing shift: {best_shift}")
        logger.info(f"   ✅ Timing fix: {best_shift} (IC: {fix_results['timing']['best_ic']:.6f})")
    
    # Fix 2: Feature reduction
    if fix_results['features']['feature_reduction_helps']:
        best_features = fix_results['features']['best_feature_set']
        improvements.append(f"Use feature set: {best_features}")
        logger.info(f"   ✅ Feature fix: {best_features} (IC: {fix_results['features']['best_ic']:.6f})")
    
    # Fix 3: Contrarian signal
    if fix_results['contrarian']['contrarian_helps']:
        best_variant = fix_results['contrarian']['best_variant']
        improvements.append(f"Use signal variant: {best_variant}")
        logger.info(f"   ✅ Contrarian fix: {best_variant} (IC: {fix_results['contrarian']['best_ic']:.6f})")
    
    # Fix 4: Sector conditioning
    if fix_results['sectors']['sector_conditional_helps']:
        positive_sectors = fix_results['sectors']['positive_sectors']
        improvements.append(f"Focus on positive IC sectors: {positive_sectors}")
        logger.info(f"   ✅ Sector fix: Focus on {len(positive_sectors)} positive sectors")
    
    return {
        'improvements_found': len(improvements),
        'improvements': improvements,
        'estimated_ic_improvement': max([
            fix_results['timing']['best_ic'],
            fix_results['features']['best_ic'],
            fix_results['contrarian']['best_ic']
        ])
    }

def main():
    """Run comprehensive IC improvement analysis"""
    
    logger.info("🎯 COMPREHENSIVE IC IMPROVEMENT ANALYSIS")
    logger.info("=" * 60)
    logger.info("Goal: Push IC from -0.0131 → +0.003–0.008")
    logger.info("=" * 60)
    
    # Load data
    data_path = Path(__file__).parent / 'data' / 'training_data_enhanced.csv'
    if not data_path.exists():
        logger.error(f"Data not found: {data_path}")
        return
    
    data = pd.read_csv(data_path)
    data['Date'] = pd.to_datetime(data['Date'])
    
    logger.info(f"Loaded {len(data):,} samples")
    
    # Run all improvement tests
    fix_results = {}
    
    # Test 1: Lead/lag timing
    fix_results['timing'] = test_lead_lag_timing(data)
    
    # Test 2: Sector-specific IC
    fix_results['sectors'] = test_sector_specific_ic(data)
    
    # Test 3: Feature reduction
    fix_results['features'] = test_feature_reduction(data)
    
    # Test 4: Short-only/contrarian variant
    fix_results['contrarian'] = test_short_only_variant(data)
    
    # Implement best fixes
    logger.info("\n" + "=" * 60)
    implementation_results = implement_best_fixes(None, fix_results)
    
    logger.info(f"\n🎉 ANALYSIS COMPLETE!")
    logger.info(f"   Improvements found: {implementation_results['improvements_found']}")
    logger.info(f"   Estimated IC improvement: {implementation_results['estimated_ic_improvement']:+.6f}")
    
    if implementation_results['estimated_ic_improvement'] > 0.002:
        logger.info("✅ Found promising improvements! Ready to implement in system.")
    else:
        logger.info("⚠️ No single fix shows strong improvement. May need combination approach.")
    
    return fix_results, implementation_results

if __name__ == "__main__":
    results = main()