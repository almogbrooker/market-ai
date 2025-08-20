#!/usr/bin/env python3
"""
Market Regime Detection for Systematic Bias
Detects bull/bear/neutral markets to adjust strategy bias
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """
    Detect market regimes to adjust trading bias
    """
    
    def __init__(self):
        self.regime_history = []
        
    def detect_regime(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add market regime features to data
        """
        
        logger.info("🌊 Detecting market regimes...")
        
        data = data.copy()
        
        # Calculate market-wide indicators
        data = self._add_market_indicators(data)
        
        # Detect regime for each date
        data = self._classify_regime(data)
        
        logger.info("✅ Market regime detection complete")
        return data
    
    def _add_market_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market-wide trend indicators"""
        
        # Calculate daily market return (equal weight) 
        daily_market = data.groupby('Date')['Return_1D'].mean().reset_index()
        daily_market.columns = ['Date', 'market_return']
        
        # Market trend indicators
        daily_market['market_sma_10'] = daily_market['market_return'].rolling(10).mean()
        daily_market['market_sma_20'] = daily_market['market_return'].rolling(20).mean()
        daily_market['market_vol'] = daily_market['market_return'].rolling(20).std()
        
        # Market momentum
        daily_market['market_momentum'] = daily_market['market_return'].rolling(5).mean()
        
        # VIX regime (from existing data)
        if 'VIX' in data.columns:
            vix_data = data.groupby('Date')['VIX'].first().reset_index()
            daily_market = daily_market.merge(vix_data, on='Date', how='left')
        else:
            daily_market['VIX'] = 20.0  # Default VIX value
        
        # Merge back to main data
        data = data.merge(daily_market, on='Date', how='left')
        
        return data
    
    def _classify_regime(self, data: pd.DataFrame) -> pd.DataFrame:
        """Classify market regime for each date"""
        
        data['regime'] = 'neutral'  # Default
        data['regime_score'] = 0.0
        data['bull_bias'] = 0.0
        
        # Bull market conditions
        bull_conditions = (
            (data['market_sma_10'] > 0.001) &  # Positive 10-day trend
            (data['market_sma_10'] > data['market_sma_20']) &  # Short > long term
            (data['VIX'] < 25) &  # Low fear
            (data['market_momentum'] > 0)  # Positive momentum
        )
        
        # Bear market conditions  
        bear_conditions = (
            (data['market_sma_10'] < -0.001) &  # Negative 10-day trend
            (data['market_sma_10'] < data['market_sma_20']) &  # Short < long term
            (data['VIX'] > 30) &  # High fear
            (data['market_momentum'] < 0)  # Negative momentum
        )
        
        # Assign regimes
        data.loc[bull_conditions, 'regime'] = 'bull'
        data.loc[bear_conditions, 'regime'] = 'bear'
        
        # Calculate regime strength scores
        data.loc[bull_conditions, 'regime_score'] = (
            data.loc[bull_conditions, 'market_momentum'] * 10 +  # Momentum weight
            np.maximum(0, 25 - data.loc[bull_conditions, 'VIX']) / 25  # VIX weight
        )
        
        data.loc[bear_conditions, 'regime_score'] = (
            data.loc[bear_conditions, 'market_momentum'] * -10 +  # Negative momentum
            np.minimum(0, 25 - data.loc[bear_conditions, 'VIX']) / 25  # High VIX penalty
        )
        
        # Bull bias (how much to favor long positions)
        data['bull_bias'] = 0.0
        data.loc[data['regime'] == 'bull', 'bull_bias'] = np.clip(
            data.loc[data['regime'] == 'bull', 'regime_score'] * 0.2, 0, 0.3
        )
        data.loc[data['regime'] == 'bear', 'bull_bias'] = np.clip(
            data.loc[data['regime'] == 'bear', 'regime_score'] * 0.2, -0.3, 0
        )
        
        # Log regime distribution
        regime_counts = data['regime'].value_counts()
        logger.info(f"📊 Regime distribution: {dict(regime_counts)}")
        
        return data

def main():
    """Test market regime detector"""
    
    print("🌊 Testing Market Regime Detector")
    print("=" * 50)
    
    # Load test data
    test_data = pd.read_parquet('artifacts/test_data.parquet')
    
    # Apply regime detection
    detector = MarketRegimeDetector()
    test_data_with_regime = detector.detect_regime(test_data)
    
    # Show regime analysis
    print(f"\n📊 Regime Analysis for Test Period:")
    regime_summary = test_data_with_regime.groupby('Date').agg({
        'regime': 'first',
        'bull_bias': 'first',
        'market_return': 'first',
        'VIX': 'first'
    }).reset_index()
    
    regime_counts = regime_summary['regime'].value_counts()
    print(f"   Regime distribution: {dict(regime_counts)}")
    print(f"   Avg bull bias: {regime_summary['bull_bias'].mean():.3f}")
    print(f"   Bull days: {len(regime_summary[regime_summary['regime'] == 'bull'])}")
    print(f"   Bear days: {len(regime_summary[regime_summary['regime'] == 'bear'])}")
    
    # Show sample dates
    print(f"\n📅 Sample Regime Classifications:")
    sample = regime_summary.head(10)[['Date', 'regime', 'bull_bias', 'market_return', 'VIX']]
    print(sample.to_string(index=False))

if __name__ == "__main__":
    main()