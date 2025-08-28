#!/usr/bin/env python3
"""
MARKET REGIME DETECTOR
=====================
Detects bull/bear markets, volatility regimes, and market conditions
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class MarketRegimeDetector:
    """Detect market regimes and conditions for trading strategy adjustment"""
    
    def __init__(self):
        self.market_tickers = {
            'SPY': 'S&P 500 ETF',
            'QQQ': 'NASDAQ 100 ETF', 
            '^VIX': 'Volatility Index',
            'TLT': '20+ Year Treasury ETF',
            'GLD': 'Gold ETF',
            # 'DXY': 'Dollar Index'  # Skip DXY as it's not available on Yahoo
        }
        
        self.regime_thresholds = {
            'bull_trend_days': 20,      # Days above trend for bull market
            'bear_trend_days': 20,      # Days below trend for bear market  
            'high_vol_vix': 25,         # VIX level for high volatility
            'low_vol_vix': 15,          # VIX level for low volatility
            'momentum_lookback': 60     # Days for momentum calculation
        }
        
    def get_market_data(self, lookback_days: int = 252) -> pd.DataFrame:
        """Fetch market data for regime analysis"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        market_data = {}
        
        for ticker, description in self.market_tickers.items():
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    # Extract Close prices as Series
                    close_prices = data['Close']
                    if isinstance(close_prices, pd.DataFrame):
                        close_prices = close_prices.iloc[:, 0]  # Take first column if DataFrame
                    market_data[ticker] = close_prices
                    
            except Exception as e:
                print(f"⚠️ Failed to fetch {ticker}: {e}")
                
        if not market_data:
            return pd.DataFrame()
            
        # Create DataFrame with proper index alignment
        try:
            df = pd.DataFrame(market_data)
        except Exception as e:
            print(f"❌ DataFrame creation error: {e}")
            return pd.DataFrame()
        
        if df.empty:
            return df
        
        # Forward fill missing values and drop empty rows
        df = df.ffill().dropna(how='all')
        return df
    
    def detect_trend_regime(self, prices: pd.Series, ma_period: int = 50) -> Dict:
        """Detect bull/bear trend regime"""
        
        # Calculate moving average
        ma = prices.rolling(ma_period).mean()
        
        # Current vs moving average
        current_price = prices.iloc[-1]
        current_ma = ma.iloc[-1]
        
        # Days above/below MA
        above_ma = (prices > ma).rolling(self.regime_thresholds['bull_trend_days']).sum()
        below_ma = (prices < ma).rolling(self.regime_thresholds['bear_trend_days']).sum()
        
        # Trend strength (% above MA over period)
        trend_strength = (prices.iloc[-20:] / ma.iloc[-20:] - 1).mean()
        
        # Momentum
        momentum_60d = (current_price / prices.iloc[-self.regime_thresholds['momentum_lookback']] - 1)
        momentum_20d = (current_price / prices.iloc[-20] - 1)
        
        # Classify regime
        if above_ma.iloc[-1] >= self.regime_thresholds['bull_trend_days'] * 0.8:
            regime = 'BULL'
        elif below_ma.iloc[-1] >= self.regime_thresholds['bear_trend_days'] * 0.8:
            regime = 'BEAR'
        else:
            regime = 'SIDEWAYS'
            
        return {
            'regime': regime,
            'trend_strength': trend_strength,
            'momentum_60d': momentum_60d,
            'momentum_20d': momentum_20d,
            'price_vs_ma': current_price / current_ma - 1,
            'days_above_ma': above_ma.iloc[-1],
            'confidence': min(1.0, abs(trend_strength) * 5)  # 0-1 confidence score
        }
    
    def detect_volatility_regime(self, vix_prices: pd.Series) -> Dict:
        """Detect volatility regime"""
        
        current_vix = vix_prices.iloc[-1]
        avg_vix_20d = vix_prices.iloc[-20:].mean()
        avg_vix_60d = vix_prices.iloc[-60:].mean()
        
        # VIX percentile over last year
        vix_percentile = (vix_prices <= current_vix).mean()
        
        # Volatility trend
        vix_trend = (current_vix - avg_vix_20d) / avg_vix_20d
        
        # Classify regime
        if current_vix > self.regime_thresholds['high_vol_vix']:
            vol_regime = 'HIGH_VOL'
        elif current_vix < self.regime_thresholds['low_vol_vix']:
            vol_regime = 'LOW_VOL'
        else:
            vol_regime = 'NORMAL_VOL'
            
        return {
            'vol_regime': vol_regime,
            'current_vix': current_vix,
            'vix_20d_avg': avg_vix_20d,
            'vix_60d_avg': avg_vix_60d,
            'vix_percentile': vix_percentile,
            'vix_trend': vix_trend,
            'fear_level': 'HIGH' if current_vix > 30 else 'MEDIUM' if current_vix > 20 else 'LOW'
        }
    
    def detect_sector_rotation(self, spy_prices: pd.Series, qqq_prices: pd.Series, tlt_prices: pd.Series) -> Dict:
        """Detect sector rotation patterns"""
        
        # Tech vs Broad Market (QQQ vs SPY)
        tech_outperformance = (qqq_prices.pct_change(20).iloc[-1] - spy_prices.pct_change(20).iloc[-1])
        
        # Risk-on vs Risk-off (Stocks vs Bonds)
        risk_on_score = (spy_prices.pct_change(20).iloc[-1] - tlt_prices.pct_change(20).iloc[-1])
        
        return {
            'tech_vs_broad': tech_outperformance,
            'risk_on_score': risk_on_score,
            'tech_leadership': 'TECH_LEADS' if tech_outperformance > 0.02 else 'BROAD_LEADS' if tech_outperformance < -0.02 else 'NEUTRAL',
            'risk_sentiment': 'RISK_ON' if risk_on_score > 0.05 else 'RISK_OFF' if risk_on_score < -0.05 else 'NEUTRAL'
        }
    
    def get_market_regime(self) -> Dict:
        """Get comprehensive market regime analysis"""
        
        print("📊 Analyzing market regime...")
        
        try:
            # Fetch market data
            market_data = self.get_market_data()
            
            if market_data.empty:
                return {'error': 'Failed to fetch market data'}
            
            # Analyze different components
            regime_analysis = {
                'timestamp': datetime.now().isoformat(),
                'data_period': f"{len(market_data)} days"
            }
            
            # SPY trend analysis
            if 'SPY' in market_data.columns:
                spy_analysis = self.detect_trend_regime(market_data['SPY'])
                regime_analysis['spy_regime'] = spy_analysis
                
            # QQQ trend analysis  
            if 'QQQ' in market_data.columns:
                qqq_analysis = self.detect_trend_regime(market_data['QQQ'])
                regime_analysis['qqq_regime'] = qqq_analysis
                
            # VIX volatility analysis
            if '^VIX' in market_data.columns:
                vix_analysis = self.detect_volatility_regime(market_data['^VIX'])
                regime_analysis['volatility_regime'] = vix_analysis
                
            # Sector rotation analysis
            if all(ticker in market_data.columns for ticker in ['SPY', 'QQQ', 'TLT']):
                rotation_analysis = self.detect_sector_rotation(
                    market_data['SPY'], 
                    market_data['QQQ'], 
                    market_data['TLT']
                )
                regime_analysis['sector_rotation'] = rotation_analysis
                
            # Overall market assessment
            regime_analysis['overall_regime'] = self._synthesize_regime(regime_analysis)
            
            return regime_analysis
            
        except Exception as e:
            return {'error': f'Market regime analysis failed: {e}'}
    
    def _synthesize_regime(self, analysis: Dict) -> Dict:
        """Synthesize overall market regime from components"""
        
        regimes = []
        confidences = []
        
        # Collect regime signals
        if 'spy_regime' in analysis:
            regimes.append(analysis['spy_regime']['regime'])
            confidences.append(analysis['spy_regime']['confidence'])
            
        if 'qqq_regime' in analysis:
            regimes.append(analysis['qqq_regime']['regime'])
            confidences.append(analysis['qqq_regime']['confidence'])
            
        # Consensus regime
        if regimes:
            regime_counts = pd.Series(regimes).value_counts()
            consensus_regime = regime_counts.index[0]
            consensus_strength = regime_counts.iloc[0] / len(regimes)
        else:
            consensus_regime = 'UNKNOWN'
            consensus_strength = 0.0
            
        # Risk assessment
        vol_regime = analysis.get('volatility_regime', {}).get('vol_regime', 'NORMAL_VOL')
        fear_level = analysis.get('volatility_regime', {}).get('fear_level', 'MEDIUM')
        
        return {
            'consensus_regime': consensus_regime,
            'consensus_strength': consensus_strength,
            'volatility_regime': vol_regime,
            'fear_level': fear_level,
            'trading_environment': self._classify_trading_environment(consensus_regime, vol_regime),
            'recommended_exposure': self._recommend_exposure(consensus_regime, vol_regime, fear_level)
        }
    
    def _classify_trading_environment(self, trend_regime: str, vol_regime: str) -> str:
        """Classify overall trading environment"""
        
        if trend_regime == 'BULL' and vol_regime == 'LOW_VOL':
            return 'GOLDILOCKS'  # Perfect conditions
        elif trend_regime == 'BULL' and vol_regime == 'HIGH_VOL':
            return 'VOLATILE_BULL'  # Bull market but choppy
        elif trend_regime == 'BEAR' and vol_regime == 'HIGH_VOL':
            return 'CRISIS'  # Worst conditions
        elif trend_regime == 'BEAR' and vol_regime == 'LOW_VOL':
            return 'GRINDING_BEAR'  # Slow decline
        elif trend_regime == 'SIDEWAYS':
            return 'RANGE_BOUND'  # Choppy sideways
        else:
            return 'TRANSITIONAL'  # Changing conditions
    
    def _recommend_exposure(self, trend_regime: str, vol_regime: str, fear_level: str) -> float:
        """Recommend portfolio exposure based on regime"""
        
        base_exposure = 0.30  # Default 30%
        
        # Adjust for trend
        if trend_regime == 'BULL':
            base_exposure *= 1.2  # Increase in bull market
        elif trend_regime == 'BEAR':
            base_exposure *= 0.6  # Reduce in bear market
            
        # Adjust for volatility
        if vol_regime == 'HIGH_VOL':
            base_exposure *= 0.7  # Reduce in high vol
        elif vol_regime == 'LOW_VOL':
            base_exposure *= 1.1  # Increase in low vol
            
        # Adjust for fear
        if fear_level == 'HIGH':
            base_exposure *= 0.5  # Dramatic reduction in crisis
        elif fear_level == 'LOW':
            base_exposure *= 1.2  # Increase when complacent
            
        return min(0.50, max(0.10, base_exposure))  # Cap between 10-50%

def main():
    """Test market regime detector"""
    detector = MarketRegimeDetector()
    regime = detector.get_market_regime()
    
    print("\n🔍 MARKET REGIME ANALYSIS")
    print("=" * 50)
    
    if 'error' in regime:
        print(f"❌ {regime['error']}")
        return
        
    # Print key findings
    overall = regime.get('overall_regime', {})
    print(f"📊 Consensus Regime: {overall.get('consensus_regime', 'Unknown')}")
    print(f"💪 Strength: {overall.get('consensus_strength', 0):.1%}")
    print(f"📈 Trading Environment: {overall.get('trading_environment', 'Unknown')}")
    print(f"🎯 Recommended Exposure: {overall.get('recommended_exposure', 0.3):.1%}")
    
    # Detailed analysis
    if 'spy_regime' in regime:
        spy = regime['spy_regime']
        print(f"\n📈 SPY Analysis:")
        print(f"   Regime: {spy['regime']}")
        print(f"   60d Momentum: {spy['momentum_60d']:+.1%}")
        print(f"   Confidence: {spy['confidence']:.1%}")
        
    if 'volatility_regime' in regime:
        vol = regime['volatility_regime']
        print(f"\n📊 Volatility Analysis:")
        print(f"   Regime: {vol['vol_regime']}")
        print(f"   Current VIX: {vol['current_vix']:.1f}")
        print(f"   Fear Level: {vol['fear_level']}")

if __name__ == "__main__":
    main()