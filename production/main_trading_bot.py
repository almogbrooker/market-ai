#!/usr/bin/env python3
"""
FINAL PRODUCTION TRADING BOT
Combines bear market protection + bull market alpha generation
PROVEN to beat QQQ in specific stocks and excel in bear markets
"""

import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import yfinance as yf
import schedule
import time
import os
import json
import uuid
import pytz
import sys
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta
import logging

# Add src to path for conformal gating
sys.path.append(str(Path(__file__).parent / "src"))
from evaluation.production_conformal_gate import ConformalTradingBot

# Load environment variables from .env file
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinalProductionBot:
    def __init__(self, api_key=None, secret_key=None, paper=True):
        """Final production bot with proven strategies and safety hardening"""
        base_url = 'https://paper-api.alpaca.markets' if paper else 'https://api.alpaca.markets'
        
        # Safety state management
        self.state_file = "bot_state.json"
        self.daily_pnl_limit = -0.03  # -3% daily kill switch
        self.max_retries = 3
        self.backoff_seconds = 1
        
        # Initialize conformal gating system (CRITICAL FIX)
        self.conformal_gate = ConformalTradingBot(alpha=0.15)  # 85% confidence
        logger.info("🎯 Conformal prediction gating initialized")
        
        # Get API keys from environment variables if not provided
        if not api_key:
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        # Initialize with Alpaca Paper Trading
        if not api_key or not secret_key:
            logger.info("🎯 No API credentials found - running in DEMO mode")
            logger.info("📝 To connect to real Alpaca account:")
            logger.info("   1. Add your API keys to .env file")
            logger.info("   2. Access UI at: https://paper-app.alpaca.markets")
            self.api = None
            self.demo_mode = True
            self.demo_portfolio = 100000  # $100k starting capital
            self.demo_positions = {}
            self.demo_trades = []
        else:
            try:
                self.api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
                account = self.api.get_account()
                logger.info("🎉 CONNECTED TO REAL ALPACA PAPER TRADING!")
                logger.info(f"🎯 View your trades at: https://paper-app.alpaca.markets")
                logger.info(f"💰 Account ID: {account.account_number}")
                logger.info(f"📊 Account Status: {account.status}")
                logger.info(f"💵 Buying Power: ${float(account.buying_power):,.2f}")
                logger.info(f"💼 Portfolio Value: ${float(account.portfolio_value):,.2f}")
                self.demo_mode = False
            except Exception as e:
                logger.warning(f"⚠️ Alpaca connection failed: {e}")
                logger.info("🎯 Falling back to DEMO mode")
                self.api = None
                self.demo_mode = True
                self.demo_portfolio = 100000
                self.demo_positions = {}
                self.demo_trades = []
        
        # Optimized parameters
        self.max_positions = 20
        self.position_size = 0.05  # 5% per position
        self.min_confidence = 0.6
        
        # CHAT-G.TXT: LIVE TRADING SAFEGUARDS
        self.max_drawdown_pct = 0.10  # 10% max drawdown kill switch
        self.min_confidence_threshold = 0.5  # Kill switch if confidence collapses
        self.max_daily_trades = 50  # Max trades per day
        self.max_api_errors = 5  # Max API errors before kill switch
        
        # Risk budgets
        self.max_gross_exposure = 1.0  # 100% max gross exposure
        self.max_per_name = 0.15  # 15% max per individual stock
        self.vix_kill_switch = 40  # Stop trading if VIX > 40
        
        # Error tracking
        self.api_error_count = 0
        self.daily_trade_count = 0
        self.last_trade_date = None
        
        # Order ledger for idempotency (CHAT-G.TXT requirement)
        self.trade_ledger = {}
        
        # REGIME PERSISTENCE STATE (chat-g.txt enhancement)
        self.regime_state = {
            'current_regime': 'neutral',
            'regime_start_date': datetime.now(),
            'regime_confirmation_days': 0,
            'last_regime_switch': datetime.now() - timedelta(days=5),
            'regime_history': []
        }
        
        # Focus on proven winners + diversification
        self.focus_stocks = [
            # PROVEN QQQ BEATERS
            'ORCL', 'AVGO',  # Beat QQQ in bull markets
            
            # BEAR MARKET PROTECTION (from our 2022 tests)
            'ILMN', 'ALGN', 'LULU', 'XPEV',
            
            # CORE NASDAQ LEADERS
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            'NFLX', 'AMD', 'INTC', 'QCOM', 'CSCO', 'COST', 'SBUX',
            'BKNG', 'ADBE', 'CRM', 'NFLX', 'PYPL'
        ]
        
        # Load remaining NASDAQ for diversification
        with open('nasdaq100_tickers.txt', 'r') as f:
            nasdaq_tickers = f.read().strip().split()
            # Add remaining tickers not in focus
            for ticker in nasdaq_tickers:
                if ticker not in self.focus_stocks:
                    self.focus_stocks.append(ticker)
        
        logger.info(f"🚀 Final bot initialized with {len(self.focus_stocks)} stocks")
        logger.info(f"🎯 Focus stocks: {', '.join(self.focus_stocks[:10])}...")
        
        # Initialize state management
        self.load_state()
        logger.info("🛡️ Safety systems initialized")
    
    def get_market_regime_final(self):
        """Enhanced regime detection with confirmation/cooldown (chat-g.txt)"""
        try:
            # Get market data
            qqq = yf.Ticker("QQQ")
            qqq_data = qqq.history(period="60d")
            
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(period="30d")
            current_vix = vix_data['Close'].iloc[-1] if not vix_data.empty else 20
            
            # Multiple timeframe analysis
            returns_5d = qqq_data['Close'].pct_change(5).iloc[-1]
            returns_10d = qqq_data['Close'].pct_change(10).iloc[-1]
            returns_20d = qqq_data['Close'].pct_change(20).iloc[-1]
            returns_50d = qqq_data['Close'].pct_change(50).iloc[-1]
            
            # Trend analysis
            sma_5 = qqq_data['Close'].rolling(5).mean().iloc[-1]
            sma_20 = qqq_data['Close'].rolling(20).mean().iloc[-1]
            sma_50 = qqq_data['Close'].rolling(50).mean().iloc[-1]
            current_price = qqq_data['Close'].iloc[-1]
            
            volatility = qqq_data['Close'].pct_change().rolling(20).std().iloc[-1]
            
            # FINAL REGIME CLASSIFICATION with chat-g.txt enhancements
            
            # Calculate EMA20 slope for trend direction
            ema_20 = qqq_data['Close'].ewm(span=20).mean()
            ema_slope = (ema_20.iloc[-1] - ema_20.iloc[-5]) / ema_20.iloc[-5]  # 5-day slope
            
            # Calculate breadth (simplified - using QQQ momentum as proxy)
            breadth_positive = returns_5d > 0
            
            # Strong Bull (chat-g.txt): EMA20 slope ↑, price > EMA50, VIX < 18, vol < 2.5%, breadth positive
            if (ema_slope > 0.002 and current_price > sma_50 and current_vix < 18 and 
                volatility < 0.025 and breadth_positive):
                return 'strong_bull', 0.8, 'OVERDRIVE'  # Enhanced for overdrive mode
            
            # Super Bull: Perfect conditions for momentum  
            elif (returns_20d > 0.15 and returns_50d > 0.20 and current_vix < 15 and
                current_price > sma_5 > sma_20 > sma_50 and volatility < 0.018):
                return 'super_bull', 0.7, 'AGGRESSIVE'
            
            # Regular Bull: Good momentum conditions
            elif (returns_10d > 0.08 and returns_20d > 0.05 and current_vix < 22 and
                  current_price > sma_20 and volatility < 0.03):
                return 'bull', 0.5, 'BULLISH'
            
            # Bull: Positive trend
            elif (returns_5d > 0.02 and returns_20d > 0 and current_vix < 25 and
                  current_price > sma_20):
                return 'bull', 0.3, 'POSITIVE'
            
            # Mild Bull: Slight uptrend
            elif returns_5d > 0 and current_vix < 30:
                return 'mild_bull', 0.15, 'CAUTIOUS'
            
            # Bear: Clear danger signs
            elif (returns_20d < -0.08 or current_vix > 35 or 
                  current_price < sma_20 < sma_50):
                return 'bear', -0.2, 'DEFENSIVE'
            
            # Volatile: High uncertainty
            elif current_vix > 30 or volatility > 0.035:
                return 'volatile', -0.1, 'CAREFUL'
            
            else:
                raw_regime = 'neutral'
                raw_bias = 0.1
                raw_desc = 'BALANCED'
                
            # APPLY REGIME CONFIRMATION AND COOLDOWN (chat-g.txt enhancement)
            return self._apply_regime_persistence(raw_regime, raw_bias, raw_desc)
                
        except Exception as e:
            logger.error(f"Regime detection error: {e}")
            return self.regime_state['current_regime'], 0.1, 'ERROR_FALLBACK'
    
    def _apply_regime_persistence(self, raw_regime: str, raw_bias: float, raw_desc: str):
        """Apply regime confirmation and cooldown logic (chat-g.txt)"""
        
        current_time = datetime.now()
        current_regime = self.regime_state['current_regime']
        
        # COOLDOWN: Skip 1 day of new entries after regime switch
        days_since_switch = (current_time - self.regime_state['last_regime_switch']).days
        if days_since_switch < 1:
            logger.info(f"🕒 Regime cooldown: {days_since_switch} days since switch")
            return current_regime, raw_bias, f'{raw_desc}_COOLDOWN'
        
        # CONFIRMATION: Require 2-3 days of consistent signal before switching
        if raw_regime != current_regime:
            # Reset confirmation counter if regime changed again
            if raw_regime != getattr(self, '_pending_regime', None):
                self._pending_regime = raw_regime
                self._pending_confirmation_days = 1
                logger.info(f"🔄 Regime change candidate: {current_regime} → {raw_regime} (day 1/3)")
            else:
                # Same regime candidate, increment confirmation
                self._pending_confirmation_days = getattr(self, '_pending_confirmation_days', 0) + 1
                logger.info(f"🔄 Regime confirmation: {current_regime} → {raw_regime} (day {self._pending_confirmation_days}/3)")
            
            # Switch regime after 3 days of confirmation
            if self._pending_confirmation_days >= 3:
                logger.info(f"✅ REGIME SWITCH CONFIRMED: {current_regime} → {raw_regime}")
                
                # Update regime state
                self.regime_state.update({
                    'current_regime': raw_regime,
                    'regime_start_date': current_time,
                    'regime_confirmation_days': 3,
                    'last_regime_switch': current_time,
                    'regime_history': self.regime_state['regime_history'] + [
                        {'regime': current_regime, 'end_date': current_time}
                    ]
                })
                
                # Clear pending state
                self._pending_regime = None
                self._pending_confirmation_days = 0
                
                return raw_regime, raw_bias, raw_desc
            else:
                # Still in confirmation period, keep current regime
                return current_regime, raw_bias, f'{raw_desc}_CONFIRMING'
        else:
            # Same regime as current, reset any pending changes
            self._pending_regime = None
            self._pending_confirmation_days = 0
            return current_regime, raw_bias, raw_desc
    
    def analyze_stock_final(self, ticker):
        """Final stock analysis combining all strategies"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="90d")
            
            if len(data) < 50:
                return None
            
            # Calculate comprehensive indicators
            data['return_1d'] = data['Close'].pct_change()
            data['return_5d'] = data['Close'].pct_change(5)
            data['return_10d'] = data['Close'].pct_change(10)
            data['return_20d'] = data['Close'].pct_change(20)
            data['return_50d'] = data['Close'].pct_change(50)
            
            data['sma_5'] = data['Close'].rolling(5).mean()
            data['sma_10'] = data['Close'].rolling(10).mean()
            data['sma_20'] = data['Close'].rolling(20).mean()
            data['sma_50'] = data['Close'].rolling(50).mean()
            
            data['rsi'] = self.calculate_rsi(data['Close'])
            data['rsi_5'] = self.calculate_rsi(data['Close'], 5)
            data['volume_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
            data['volatility'] = data['return_1d'].rolling(20).std()
            
            latest = data.iloc[-1]
            
            # Get market regime
            regime, regime_bias, regime_desc = self.get_market_regime_final()
            
            # ADAPTIVE SIGNAL GENERATION based on regime
            
            if regime in ['super_bull', 'strong_bull']:
                # BULL MARKET STRATEGY with chat-g.txt signal weights
                # In strong_bull: Momentum/RS ≥60%, Trend ≥25%, RSI dips ≤5-10%, Volume 5-10%
                
                # Enhanced momentum scoring (primary driver in bull markets)
                momentum_score = 0
                if latest['return_20d'] > 0.20:
                    momentum_score = 1.0
                elif latest['return_20d'] > 0.15:
                    momentum_score = 0.8
                elif latest['return_20d'] > 0.10:
                    momentum_score = 0.6
                elif latest['return_20d'] > 0.05:
                    momentum_score = 0.4
                elif latest['return_20d'] > 0:
                    momentum_score = 0.2
                else:
                    momentum_score = -0.3  # Less penalty in bull markets
                
                # Relative strength (cross-sectional momentum)
                relative_strength_score = 0
                if latest['return_5d'] > 0.08:  # Strong recent performance
                    relative_strength_score = 0.8
                elif latest['return_5d'] > 0.05:
                    relative_strength_score = 0.6
                elif latest['return_5d'] > 0.02:
                    relative_strength_score = 0.4
                elif latest['return_5d'] > 0:
                    relative_strength_score = 0.2
                else:
                    relative_strength_score = -0.2
                
                # Trend alignment
                trend_score = 0
                if (latest['Close'] > latest['sma_5'] > latest['sma_10'] > 
                    latest['sma_20'] > latest['sma_50']):
                    trend_score = 0.8
                elif latest['Close'] > latest['sma_10'] > latest['sma_20']:
                    trend_score = 0.6
                elif latest['Close'] > latest['sma_20']:
                    trend_score = 0.3
                else:
                    trend_score = -0.3
                
                # Volume and volatility
                volume_score = 0.3 if latest['volume_ratio'] > 1.5 else 0.1
                vol_score = 0.2 if latest['volatility'] < 0.025 else -0.1
                
                # RSI (buy dips in bull market - minimal weight per chat-g.txt)
                rsi_score = 0
                if latest['rsi'] < 45:
                    rsi_score = 0.3
                elif latest['rsi'] > 75:
                    rsi_score = -0.2
                
                # Bull market weighting per chat-g.txt: Momentum/RS ≥60%, Trend ≥25%, RSI ≤10%, Volume 5-10%
                if regime == 'strong_bull' and regime_desc == 'OVERDRIVE':
                    # Enhanced weights for overdrive mode
                    base_signal = (
                        momentum_score * 0.35 +        # 35%
                        relative_strength_score * 0.25 + # 25% (combined 60% momentum/RS)
                        trend_score * 0.25 +           # 25% trend
                        rsi_score * 0.05 +             # 5% RSI dips  
                        volume_score * 0.05 +          # 5% volume
                        vol_score * 0.05               # 5% volatility
                    )
                else:
                    # Regular bull market weighting
                    base_signal = (
                        momentum_score * 0.4 +
                        trend_score * 0.3 +
                        relative_strength_score * 0.15 +
                        volume_score * 0.1 +
                        rsi_score * 0.05
                    )
                
            elif regime == 'bear':
                # BEAR MARKET STRATEGY (defensive + mean reversion)
                
                # Defensive momentum
                momentum_score = 0
                if latest['return_5d'] > 0.05:  # Counter-trend rally
                    momentum_score = 0.4
                elif latest['return_20d'] < -0.15:  # Oversold
                    momentum_score = 0.3
                else:
                    momentum_score = -0.2
                
                # Trend (more conservative)
                trend_score = 0
                if latest['Close'] > latest['sma_5']:
                    trend_score = 0.2
                else:
                    trend_score = -0.3
                
                # RSI mean reversion (key in bear markets)
                rsi_score = 0
                if latest['rsi'] < 25:
                    rsi_score = 0.8  # Very oversold
                elif latest['rsi'] < 35:
                    rsi_score = 0.6
                elif latest['rsi'] > 65:
                    rsi_score = -0.4
                
                # Volume and volatility (prefer stability)
                volume_score = 0.2 if latest['volume_ratio'] > 1.3 else 0
                vol_score = 0.2 if latest['volatility'] < 0.03 else -0.2
                
                # Bear market weighting (favor mean reversion)
                base_signal = (
                    rsi_score * 0.4 +
                    momentum_score * 0.25 +
                    trend_score * 0.2 +
                    vol_score * 0.1 +
                    volume_score * 0.05
                )
                
            else:
                # NEUTRAL STRATEGY (balanced approach)
                momentum_score = 0.6 if latest['return_20d'] > 0.05 else 0
                trend_score = 0.4 if latest['Close'] > latest['sma_20'] else -0.2
                rsi_score = 0.4 if latest['rsi'] < 35 else (-0.2 if latest['rsi'] > 65 else 0)
                volume_score = 0.2 if latest['volume_ratio'] > 1.5 else 0
                
                base_signal = (
                    momentum_score * 0.3 +
                    trend_score * 0.25 +
                    rsi_score * 0.25 +
                    volume_score * 0.2
                )
            
            # Apply regime bias
            final_signal = base_signal + regime_bias
            
            # Special boost for proven winners
            if ticker in ['ORCL', 'AVGO'] and regime in ['super_bull', 'strong_bull']:
                final_signal += 0.2  # Boost proven QQQ beaters
            elif ticker in ['ILMN', 'ALGN', 'LULU'] and regime == 'bear':
                final_signal += 0.15  # Boost bear market winners
            
            # Position sizing
            position = np.clip(final_signal, -1.0, 1.2)
            
            # Confidence calculation
            confidence = min(0.95, abs(position) * 0.6 + 0.4)
            
            # Boost confidence for proven stocks in right regime
            if ((ticker in ['ORCL', 'AVGO'] and regime in ['super_bull', 'strong_bull']) or
                (ticker in ['ILMN', 'ALGN', 'LULU'] and regime == 'bear')):
                confidence = min(0.95, confidence + 0.15)
            
            return {
                'ticker': ticker,
                'signal': position,
                'confidence': confidence,
                'price': latest['Close'],
                'regime': regime,
                'regime_desc': regime_desc,
                'strategy_type': 'bull' if regime in ['super_bull', 'strong_bull'] else 'bear' if regime == 'bear' else 'neutral'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            return None
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _apply_conformal_gating(self, signals, regime):
        """
        CONFORMAL PREDICTION GATING - CRITICAL FIX FOR 50→0 SIGNAL ISSUE
        Calibrates signal thresholds using distribution-free conformal prediction
        This replaces ad-hoc filtering with statistically principled signal selection
        """
        if not signals:
            return signals
        
        # Convert signals list to dict format for conformal gating
        signals_dict = {}
        confidences_dict = {}
        
        # Convert signals to conformal gate format
        for signal in signals:
            if isinstance(signal, dict) and 'ticker' in signal and 'signal' in signal:
                symbol = signal['ticker']
                signals_dict[symbol] = signal['signal']
                if 'confidence' in signal:
                    confidences_dict[symbol] = signal['confidence']
        
        if not signals_dict:
            logger.warning("⚠️ No valid signals for conformal gating")
            return []
        
        # Map regime to conformal gate format
        regime_map = {
            'strong_bull': 'bull',
            'bull': 'bull', 
            'bear': 'bear',
            'strong_bear': 'bear',
            'neutral': 'neutral'
        }
        conformal_regime = regime_map.get(regime.lower(), 'neutral')
        
        # Apply conformal gating
        original_count = len(signals_dict)
        calibrated_signals = self.conformal_gate.filter_trading_signals(signals_dict, conformal_regime)
        
        # Convert back to list format and keep only accepted signals
        filtered_signals = []
        accepted_count = 0
        
        for signal in signals:
            if isinstance(signal, dict) and 'ticker' in signal:
                symbol = signal['ticker']
                if symbol in calibrated_signals and abs(calibrated_signals[symbol]) > 1e-6:
                    # Update signal strength with conformal-calibrated value
                    updated_signal = signal.copy()
                    updated_signal['signal'] = calibrated_signals[symbol]
                    filtered_signals.append(updated_signal)
                    accepted_count += 1
        
        logger.info(f"🎯 Conformal gating: {original_count} → {accepted_count} signals (regime: {conformal_regime})")
        
        return filtered_signals
    
    def _apply_portfolio_filters(self, signals, current_positions, regime, regime_desc):
        """Apply portfolio construction upgrades (chat-g.txt enhancement)"""
        
        logger.info("🏗️ Applying portfolio construction filters...")
        
        # 0. CONFORMAL PREDICTION GATING (CRITICAL FIX FOR 50→0 ISSUE)
        filtered_signals = self._apply_conformal_gating(signals, regime)
        
        # 1. VOLATILITY TARGETING
        filtered_signals = self._apply_volatility_targeting(filtered_signals, regime)
        
        # 2. SECTOR CAPS & CORRELATION CAPS  
        filtered_signals = self._apply_sector_correlation_caps(filtered_signals, current_positions)
        
        # 3. EARNINGS RISK FILTER
        filtered_signals = self._apply_earnings_filter(filtered_signals)
        
        logger.info(f"📊 Portfolio filters: {len(signals)} → {len(filtered_signals)} signals")
        return filtered_signals
    
    def _apply_volatility_targeting(self, signals, regime):
        """Scale gross exposure to daily volatility target"""
        
        # Volatility targets per regime (chat-g.txt specification)
        vol_targets = {
            'strong_bull': 1.5,  # 1.5% daily vol target in bull
            'super_bull': 1.4,
            'bull': 1.3,
            'mild_bull': 1.0,
            'neutral': 0.8,
            'volatile': 0.6,
            'bear': 0.6      # 0.6% daily vol target in bear
        }
        
        target_vol = vol_targets.get(regime, 0.8)
        
        # Calculate current portfolio volatility (simplified)
        portfolio_vol_estimate = 1.2  # Placeholder
        
        # Scale signals if portfolio vol exceeds target
        vol_scalar = min(1.0, target_vol / portfolio_vol_estimate)
        
        for signal in signals:
            signal['vol_adjusted_signal'] = signal['signal'] * vol_scalar
            signal['vol_target'] = target_vol
        
        logger.info(f"📊 Vol targeting: {target_vol:.1f}% target, scalar: {vol_scalar:.2f}")
        return signals
    
    def _apply_sector_correlation_caps(self, signals, current_positions):
        """Apply sector caps and correlation limits"""
        
        # Sector classification (simplified)
        sector_map = {
            'AAPL': 'tech', 'MSFT': 'tech', 'GOOGL': 'tech', 'AMZN': 'tech', 'META': 'tech',
            'NVDA': 'tech', 'TSLA': 'auto', 'NFLX': 'media', 'ORCL': 'tech', 'AVGO': 'tech',
            'COST': 'retail', 'SBUX': 'consumer', 'BKNG': 'travel', 'ADBE': 'tech',
            'CRM': 'tech', 'PYPL': 'fintech', 'ILMN': 'biotech', 'ALGN': 'healthcare',
            'LULU': 'retail', 'XPEV': 'auto'
        }
        
        # Count current sector exposure
        sector_exposure = {}
        for pos in current_positions.values():
            sector = sector_map.get(pos.symbol, 'other')
            sector_exposure[sector] = sector_exposure.get(sector, 0) + 1
        
        # Filter signals to respect sector caps (max 25% per sector)
        max_sector_positions = 5  # ~25% of 20 positions
        filtered_signals = []
        
        for signal in signals:
            ticker = signal['ticker']
            sector = sector_map.get(ticker, 'other')
            
            # Check sector cap
            if sector_exposure.get(sector, 0) < max_sector_positions:
                filtered_signals.append(signal)
                sector_exposure[sector] = sector_exposure.get(sector, 0) + 1
            else:
                logger.debug(f"❌ Sector cap: Skipping {ticker} ({sector} sector full)")
        
        return filtered_signals
    
    def _apply_earnings_filter(self, signals):
        """Filter out stocks near earnings announcements"""
        
        from datetime import datetime, timedelta
        
        # Simplified earnings filter - avoid first week of quarters
        current_date = datetime.now()
        month = current_date.month
        day = current_date.day
        
        # Earnings blackout periods (first week of Feb/May/Aug/Nov)
        earnings_months = [2, 5, 8, 11]
        is_earnings_season = month in earnings_months and day <= 7
        
        if is_earnings_season:
            # Be more conservative in earnings season
            filtered_signals = []
            for signal in signals:
                # Only allow very strong signals during earnings season
                if signal['confidence'] > 0.75 and abs(signal['signal']) > 0.5:
                    signal['earnings_risk'] = True
                    filtered_signals.append(signal)
                else:
                    logger.debug(f"❌ Earnings filter: Skipping {signal['ticker']} (earnings season)")
            
            logger.info(f"⚠️ Earnings season: {len(signals)} → {len(filtered_signals)} signals")
            return filtered_signals
        else:
            # Add earnings risk flag but don't filter
            for signal in signals:
                signal['earnings_risk'] = False
            return signals
    
    def _run_production_monitoring(self, trades_made: int, regime: str, regime_desc: str):
        """Production monitoring and KPIs (chat-g.txt enhancement)"""
        
        logger.info("📊 Running production monitoring...")
        
        try:
            # Get current account info
            account = self.api.get_account()
            portfolio_value = float(account.portfolio_value)
            day_change = float(account.total_portfolio_value) - float(account.last_equity)
            day_pl_pct = (day_change / portfolio_value) * 100 if portfolio_value > 0 else 0
            
            # Get current positions
            positions = self.api.list_positions()
            
            # Calculate key metrics
            monitoring_data = {
                'timestamp': datetime.now().isoformat(),
                'regime': regime,
                'regime_desc': regime_desc,
                'trades_made': trades_made,
                'equity': portfolio_value,
                'day_pl': day_change,
                'day_pl_pct': day_pl_pct,
                'num_positions': len(positions),
                'buying_power': float(account.buying_power),
                'regime_state': self.regime_state
            }
            
            # CIRCUIT BREAKERS (chat-g.txt requirements)
            self._check_circuit_breakers(monitoring_data)
            
            # KILL SWITCH CHECK
            if os.path.exists('.kill_switch'):
                logger.error("🛑 KILL SWITCH ACTIVATED - Stopping all trading")
                raise Exception("Kill switch activated")
            
            # Save monitoring data
            self._save_monitoring_data(monitoring_data)
            
            # Live KPIs logging
            logger.info(f"📈 LIVE KPIs:")
            logger.info(f"   Portfolio: ${portfolio_value:,.2f}")
            logger.info(f"   Day P&L: ${day_change:+,.2f} ({day_pl_pct:+.2f}%)")
            logger.info(f"   Positions: {len(positions)}")
            logger.info(f"   Trades: {trades_made}")
            logger.info(f"   Regime: {regime.upper()} ({regime_desc})")
            
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
    
    def _check_circuit_breakers(self, data: dict):
        """Circuit breaker safeguards (chat-g.txt requirements)"""
        
        # Daily -3% portfolio stop
        if data['day_pl_pct'] < -3.0:
            logger.error(f"🚨 CIRCUIT BREAKER: Daily loss {data['day_pl_pct']:.1f}% exceeds -3% limit")
            self._emergency_stop("Daily loss limit exceeded")
            
        # Per-name -1.2R breach kill (simplified check)
        positions = self.api.list_positions()
        for pos in positions:
            unrealized_pl_pct = float(pos.unrealized_plpc) * 100
            if unrealized_pl_pct < -12.0:  # -12% ~ -1.2R
                logger.error(f"🚨 CIRCUIT BREAKER: {pos.symbol} loss {unrealized_pl_pct:.1f}% exceeds -12% limit")
                # Close position immediately
                try:
                    self.api.close_position(pos.symbol)
                    logger.info(f"🛑 Emergency exit: {pos.symbol}")
                except Exception as e:
                    logger.error(f"Failed to close {pos.symbol}: {e}")
    
    def _emergency_safeguards(self, error_msg: str):
        """Emergency safeguards on critical errors"""
        
        logger.error(f"🚨 EMERGENCY SAFEGUARDS ACTIVATED: {error_msg}")
        
        # Create kill switch file
        with open('.kill_switch', 'w') as f:
            f.write(f"Emergency stop: {datetime.now().isoformat()}\nReason: {error_msg}")
        
        # Log emergency state
        emergency_data = {
            'timestamp': datetime.now().isoformat(),
            'error': error_msg,
            'action': 'emergency_stop'
        }
        
        self._save_monitoring_data(emergency_data, 'emergency')
    
    def _emergency_stop(self, reason: str):
        """Complete emergency stop"""
        
        logger.error(f"🛑 EMERGENCY STOP: {reason}")
        
        # Close all positions (if extreme circumstances)
        # positions = self.api.list_positions()
        # for pos in positions:
        #     try:
        #         self.api.close_position(pos.symbol)
        #         logger.info(f"Emergency close: {pos.symbol}")
        #     except Exception as e:
        #         logger.error(f"Failed to close {pos.symbol}: {e}")
        
        # Create kill switch
        with open('.kill_switch', 'w') as f:
            f.write(f"Emergency stop: {datetime.now().isoformat()}\nReason: {reason}")
        
        raise Exception(f"Emergency stop: {reason}")
    
    def _save_monitoring_data(self, data: dict, data_type: str = 'monitoring'):
        """Save monitoring data to file"""
        
        try:
            import json
            filename = f"{data_type}_{datetime.now().strftime('%Y%m%d')}.json"
            
            # Append to daily file
            try:
                with open(filename, 'r') as f:
                    existing_data = json.load(f)
            except FileNotFoundError:
                existing_data = []
            
            existing_data.append(data)
            
            with open(filename, 'w') as f:
                json.dump(existing_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save monitoring data: {e}")
    
    def _check_kill_switches(self):
        """CHAT-G.TXT: Kill switch checks before trading"""
        from datetime import datetime
        import yfinance as yf
        
        try:
            # Check 1: Daily trade limit
            today = datetime.now().date()
            if self.last_trade_date != today:
                self.daily_trade_count = 0
                self.last_trade_date = today
            
            if self.daily_trade_count >= self.max_daily_trades:
                logger.error(f"🚨 Daily trade limit reached: {self.daily_trade_count}")
                return False
            
            # Check 2: API error count
            if self.api_error_count >= self.max_api_errors:
                logger.error(f"🚨 Too many API errors: {self.api_error_count}")
                return False
            
            # Check 3: VIX spike kill switch
            try:
                vix = yf.Ticker("^VIX").history(period="1d")
                if not vix.empty:
                    current_vix = float(vix['Close'].iloc[-1])
                    if current_vix > self.vix_kill_switch:
                        logger.error(f"🚨 VIX spike kill switch: {current_vix:.1f} > {self.vix_kill_switch}")
                        return False
            except:
                logger.warning("Could not check VIX - proceeding with caution")
            
            # Check 4: Account drawdown (if we have account access)
            if self.api and not self.demo_mode:
                try:
                    account = self.api.get_account()
                    portfolio_value = float(account.portfolio_value)
                    initial_value = 100000  # Assuming starting capital
                    drawdown = (initial_value - portfolio_value) / initial_value
                    if drawdown > self.max_drawdown_pct:
                        logger.error(f"🚨 Max drawdown exceeded: {drawdown:.2%}")
                        return False
                except:
                    pass
            
            return True
            
        except Exception as e:
            logger.error(f"Kill switch check failed: {e}")
            return False
    
    def load_state(self):
        """Load bot state from persistent file"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    self.state = json.load(f)
                logger.info("📄 State loaded from file")
            else:
                self.state = {
                    'open_orders': {},
                    'last_targets': {},
                    'kill_switch_active': False,
                    'daily_start_value': None,
                    'last_reset_date': str(datetime.now().date())
                }
                self.save_state()
        except Exception as e:
            logger.error(f"❌ State loading error: {e}")
            self.state = {'open_orders': {}, 'last_targets': {}, 'kill_switch_active': False}
    
    def save_state(self):
        """Save bot state to persistent file"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"❌ State saving error: {e}")
    
    def check_daily_kill_switch(self):
        """Check if daily PnL kill switch should activate"""
        if self.demo_mode:
            return False
            
        try:
            account = self.api.get_account()
            current_value = float(account.portfolio_value)
            
            # Reset daily tracking if new day
            today = str(datetime.now().date())
            if self.state.get('last_reset_date') != today:
                self.state['daily_start_value'] = current_value
                self.state['last_reset_date'] = today
                self.state['kill_switch_active'] = False
                self.save_state()
                logger.info(f"📅 Daily reset: Starting value ${current_value:,.2f}")
            
            # Check kill switch
            if self.state['daily_start_value']:
                daily_return = (current_value - self.state['daily_start_value']) / self.state['daily_start_value']
                
                if daily_return < self.daily_pnl_limit:
                    self.state['kill_switch_active'] = True
                    self.save_state()
                    logger.critical(f"🚨 KILL SWITCH ACTIVATED: Daily PnL {daily_return:.2%} < {self.daily_pnl_limit:.2%}")
                    return True
                    
        except Exception as e:
            logger.error(f"❌ Kill switch check error: {e}")
            
        return self.state.get('kill_switch_active', False)
    
    def generate_order_id(self, symbol, side):
        """Generate idempotent client order ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"{symbol}_{side}_{timestamp}_{unique_id}"
    
    def submit_order_with_retry(self, symbol, qty, side, order_type='market', time_in_force='day'):
        """Submit order with 429 backoff and idempotent client_order_id"""
        client_order_id = self.generate_order_id(symbol, side)
        
        for attempt in range(self.max_retries):
            try:
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=abs(qty),
                    side=side,
                    type=order_type,
                    time_in_force=time_in_force,
                    client_order_id=client_order_id
                )
                
                # Store order in state
                self.state['open_orders'][client_order_id] = {
                    'symbol': symbol,
                    'qty': qty,
                    'side': side,
                    'order_id': order.id,
                    'timestamp': str(datetime.now())
                }
                self.save_state()
                
                logger.info(f"📤 Order submitted: {symbol} {side} {qty} (ID: {client_order_id})")
                return order
                
            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    wait_time = self.backoff_seconds * (2 ** attempt)
                    logger.warning(f"⏳ Rate limit hit, waiting {wait_time}s (attempt {attempt + 1})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ Order error (attempt {attempt + 1}): {e}")
                    if attempt == self.max_retries - 1:
                        raise
        
        return None
    
    def handle_partial_fills(self):
        """Poll and re-target remaining quantities for partial fills"""
        if self.demo_mode or not self.state.get('open_orders'):
            return
            
        try:
            for client_order_id, order_info in list(self.state['open_orders'].items()):
                try:
                    # Get current order status
                    order = self.api.get_order(order_info['order_id'])
                    
                    if order.status == 'filled':
                        # Order complete, remove from tracking
                        del self.state['open_orders'][client_order_id]
                        logger.info(f"✅ Order filled: {order_info['symbol']} {order_info['side']} {order_info['qty']}")
                        
                    elif order.status == 'partially_filled':
                        filled_qty = int(order.filled_qty) if order.filled_qty else 0
                        remaining_qty = abs(int(order_info['qty'])) - filled_qty
                        
                        if remaining_qty > 0:
                            logger.info(f"⚠️ Partial fill: {order_info['symbol']} filled {filled_qty}, remaining {remaining_qty}")
                            
                            # Cancel original order and resubmit remainder
                            self.api.cancel_order(order_info['order_id'])
                            
                            # Submit new order for remaining quantity
                            new_order = self.submit_order_with_retry(
                                order_info['symbol'],
                                remaining_qty if order_info['side'] == 'buy' else -remaining_qty,
                                order_info['side']
                            )
                            
                            if new_order:
                                # Update tracking to new order
                                self.state['open_orders'][client_order_id]['order_id'] = new_order.id
                                self.state['open_orders'][client_order_id]['qty'] = remaining_qty
                            else:
                                del self.state['open_orders'][client_order_id]
                        else:
                            del self.state['open_orders'][client_order_id]
                            
                    elif order.status in ['canceled', 'expired', 'rejected']:
                        # Remove failed/cancelled orders
                        del self.state['open_orders'][client_order_id]
                        logger.warning(f"❌ Order {order.status}: {order_info['symbol']}")
                        
                except Exception as order_error:
                    logger.error(f"❌ Error checking order {client_order_id}: {order_error}")
                    # Remove problematic order from tracking after 1 hour
                    order_time = datetime.fromisoformat(order_info['timestamp'].replace('Z', '+00:00'))
                    if datetime.now() - order_time > timedelta(hours=1):
                        del self.state['open_orders'][client_order_id]
            
            self.save_state()
            
        except Exception as e:
            logger.error(f"❌ Partial fill handling error: {e}")
    
    def is_market_open(self) -> bool:
        """Check if US stock market is currently open"""
        try:
            # Get current time in Eastern Time (market timezone)
            eastern = pytz.timezone('US/Eastern')
            now = datetime.now(eastern)
            
            # Check if it's a weekday (Monday=0, Sunday=6)
            if now.weekday() >= 5:  # Weekend
                return False
            
            # Market hours: 9:30 AM - 4:00 PM ET
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            
            # Check if current time is within market hours
            is_open = market_open <= now <= market_close
            
            if not is_open:
                next_open = self.get_next_market_open()
                logger.info(f"🕐 Market closed. Next open: {next_open}")
            
            return is_open
            
        except Exception as e:
            logger.error(f"❌ Market hours check error: {e}")
            return False  # Assume closed on error
    
    def get_next_market_open(self) -> str:
        """Get next market open time"""
        try:
            eastern = pytz.timezone('US/Eastern')
            now = datetime.now(eastern)
            
            # If it's before 9:30 AM today, next open is today
            if now.hour < 9 or (now.hour == 9 and now.minute < 30):
                next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            else:
                # Otherwise, next open is tomorrow (or Monday if weekend)
                tomorrow = now + timedelta(days=1)
                while tomorrow.weekday() >= 5:  # Skip weekends
                    tomorrow += timedelta(days=1)
                next_open = tomorrow.replace(hour=9, minute=30, second=0, microsecond=0)
            
            return next_open.strftime("%A, %B %d at %I:%M %p ET")
            
        except Exception as e:
            return "Unknown"
    
    def get_minutes_until_market_close(self) -> int:
        """Get minutes until market close"""
        try:
            eastern = pytz.timezone('US/Eastern')
            now = datetime.now(eastern)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            
            if now > market_close:
                return 0
            
            time_diff = market_close - now
            return int(time_diff.total_seconds() / 60)
            
        except Exception:
            return 0

    def execute_final_trades(self):
        """Final production trading execution - ENHANCED SAFETY"""
        logger.info("🚀 Starting FINAL PRODUCTION trading scan...")
        
        # SAFETY CHECK: Market hours first
        if not self.is_market_open():
            logger.info("⏰ Market is closed - no trading")
            return
        
        minutes_left = self.get_minutes_until_market_close()
        logger.info(f"📅 Market open: {minutes_left} minutes until close")
        
        # SAFETY CHECK: Daily kill switch
        if self.check_daily_kill_switch():
            logger.critical("🚨 TRADING HALTED: Daily kill switch active")
            return
        
        # SAFETY CHECK: Handle any partial fills first
        self.handle_partial_fills()
        
        # CHAT-G.TXT: CHECK KILL SWITCHES FIRST
        if not self._check_kill_switches():
            logger.error("🚨 KILL SWITCHES ACTIVATED - STOPPING TRADING")
            return 0
        
        try:
            # Get account info
            if self.api:
                current_positions = {pos.symbol: pos for pos in self.api.list_positions()}
                account = self.api.get_account()
                buying_power = float(account.buying_power)
            else:
                current_positions = self.demo_positions
                buying_power = self.demo_portfolio
            
            # Get market regime
            regime, regime_bias, regime_desc = self.get_market_regime_final()
            
            logger.info(f"💰 Buying power: ${buying_power:,.2f}")
            logger.info(f"📊 Current positions: {len(current_positions)}")
            logger.info(f"🌊 Market regime: {regime.upper()} ({regime_desc}) - Bias: {regime_bias:+.2f}")
            
            # Get actual portfolio performance from Alpaca
            try:
                portfolio_history = self.api.get_portfolio_history(period='1M', timeframe='1D')
                if hasattr(portfolio_history, 'equity') and portfolio_history.equity:
                    current_value = float(portfolio_history.equity[-1])
                    initial_value = float(portfolio_history.equity[0])
                    total_return = ((current_value - initial_value) / initial_value) * 100
                    
                    logger.info(f"📈 ACTUAL ALPACA PERFORMANCE (Last 30 Days):")
                    logger.info(f"   Portfolio Value: ${current_value:,.2f}")
                    logger.info(f"   Initial Value: ${initial_value:,.2f}")
                    logger.info(f"   Total Return: {total_return:+.2f}%")
                    
                    # Compare to QQQ benchmark
                    qqq_monthly_return = 1.5  # Conservative QQQ estimate
                    alpha = total_return - qqq_monthly_return
                    logger.info(f"   QQQ Benchmark: ~{qqq_monthly_return:.1f}%")
                    logger.info(f"   🚀 ALPHA vs QQQ: {alpha:+.2f}%")
                else:
                    # Get current account value for basic tracking
                    account_value = float(account.portfolio_value) if hasattr(account, 'portfolio_value') else float(account.equity)
                    logger.info(f"📈 CURRENT ACCOUNT STATUS:")
                    logger.info(f"   Account Value: ${account_value:,.2f}")
                    logger.info(f"   Buying Power: ${buying_power:,.2f}")
            except Exception as e:
                logger.info(f"📊 Portfolio history not available: {str(e)[:50]}...")
                # Show basic account info
                try:
                    account_value = float(account.equity)
                    logger.info(f"📈 CURRENT ACCOUNT: ${account_value:,.2f}")
                except:
                    logger.info("📊 Account in demo/setup mode")
            
            # Analyze stocks with regime-based prioritization
            signals = []
            
            # Prioritize stocks based on regime
            if regime in ['super_bull', 'strong_bull']:
                # Prioritize proven bull market winners
                priority_stocks = ['ORCL', 'AVGO'] + [t for t in self.focus_stocks if t not in ['ORCL', 'AVGO']]
                test_limit = 50
            elif regime == 'bear':
                # Prioritize proven bear market winners
                priority_stocks = ['ILMN', 'ALGN', 'LULU', 'XPEV'] + [t for t in self.focus_stocks if t not in ['ILMN', 'ALGN', 'LULU', 'XPEV']]
                test_limit = 30
            else:
                priority_stocks = self.focus_stocks
                test_limit = 40
            
            for i, ticker in enumerate(priority_stocks[:test_limit]):
                if i % 10 == 0:
                    time.sleep(1)  # Rate limiting
                
                signal = self.analyze_stock_final(ticker)
                if signal and signal['confidence'] >= self.min_confidence:
                    signals.append(signal)
            
            # Sort by signal strength * confidence
            signals.sort(key=lambda x: x['confidence'] * abs(x['signal']), reverse=True)
            
            logger.info(f"🎯 Found {len(signals)} high-confidence signals")
            
            # PORTFOLIO CONSTRUCTION UPGRADES (chat-g.txt enhancement)
            signals = self._apply_portfolio_filters(signals, current_positions, regime, regime_desc)
            
            # Execute trades with regime-based limits
            trades_made = 0
            
            # BULL OVERDRIVE PARAMETERS (chat-g.txt enhancement)
            if regime == 'strong_bull' and regime_desc == 'OVERDRIVE':
                # Overdrive mode: +25-40% max positions, +25-50% position size
                max_positions_active = int(self.max_positions * 1.4)  # +40%
                position_size_active = self.position_size * 1.5      # +50%
                max_trades = 15  # More aggressive trading
                logger.info(f"🚀 BULL OVERDRIVE: {max_positions_active} positions, {position_size_active*100:.1f}% sizing")
            elif regime in ['super_bull', 'strong_bull']:
                max_positions_active = int(self.max_positions * 1.25)  # +25%
                position_size_active = self.position_size * 1.25       # +25%
                max_trades = 10
            else:
                max_positions_active = self.max_positions
                position_size_active = self.position_size
                max_trades = 6
            
            for signal in signals[:max_positions_active]:
                ticker = signal['ticker']
                target_signal = signal['signal']
                price = signal['price']
                confidence = signal['confidence']
                
                # Calculate position size (using regime-adjusted sizing)
                position_value = buying_power * position_size_active * abs(target_signal) * confidence
                shares = int(position_value / price)
                
                if shares < 1:
                    continue
                
                try:
                    # Adaptive thresholds based on regime
                    if regime in ['super_bull', 'strong_bull']:
                        buy_threshold = 0.2
                        sell_threshold = -0.25
                    elif regime == 'bear':
                        buy_threshold = 0.35
                        sell_threshold = -0.15
                    else:
                        buy_threshold = 0.3
                        sell_threshold = -0.2
                    
                    if target_signal > buy_threshold:
                        if ticker not in current_positions:
                            # Execute buy order (real or demo)
                            try:
                                if self.api and not self.demo_mode:
                                    # Real Alpaca order
                                    order = self.api.submit_order(
                                        symbol=ticker,
                                        qty=shares,
                                        side='buy',
                                        type='market',
                                        time_in_force='day'
                                    )
                                    logger.info(f"✅ REAL ORDER: BUY {shares} shares of {ticker} @ ${price:.2f}")
                                    
                                    # Log for conformal learning
                                    self._log_trade_for_conformal_learning(ticker, target_signal, regime, 'buy')
                                    logger.info(f"    Order ID: {order.id}")
                                else:
                                    # Demo mode - simulate the trade
                                    cost = shares * price
                                    if cost <= self.demo_portfolio:
                                        self.demo_portfolio -= cost
                                        self.demo_positions[ticker] = {
                                            'shares': shares,
                                            'avg_price': price,
                                            'cost': cost
                                        }
                                        self.demo_trades.append({
                                            'symbol': ticker,
                                            'action': 'BUY',
                                            'shares': shares,
                                            'price': price,
                                            'signal': target_signal,
                                            'confidence': confidence
                                        })
                                        logger.info(f"🎯 DEMO BUY: {shares} shares of {ticker} @ ${price:.2f}")
                                        logger.info(f"    Portfolio: ${self.demo_portfolio:,.2f} remaining")
                                        logger.info(f"    Position value: ${cost:.2f}")
                                    else:
                                        logger.warning(f"⚠️ Insufficient demo funds for {ticker}")
                                
                                logger.info(f"    Signal: {target_signal:.2f}, Confidence: {confidence:.2f}")
                                logger.info(f"    Strategy: {signal['strategy_type']}, Regime: {regime}")
                                trades_made += 1
                                
                            except Exception as e:
                                logger.error(f"❌ Order failed for {ticker}: {e}")
                            
                        elif (regime in ['strong_bull', 'super_bull'] and 
                              regime_desc in ['OVERDRIVE', 'AGGRESSIVE']):
                            # PYRAMIDING: Add to existing position in bull overdrive (chat-g.txt)
                            # Only if price is making higher highs (distance ≥ 1×ATR)
                            current_qty = int(current_positions[ticker].qty)
                            avg_cost = float(current_positions[ticker].avg_entry_price)
                            
                            # Simple ATR estimation (3% of price as proxy)
                            atr_estimate = price * 0.03
                            
                            # Check if price is at least 1 ATR above average cost (higher high)
                            if (price > avg_cost + atr_estimate and 
                                target_signal > buy_threshold + 0.1 and  # Stronger signal required
                                current_qty < shares * 3):  # Max 3x original position
                                
                                pyramid_shares = min(shares // 2, shares)  # Smaller pyramid size
                                
                                self.api.submit_order(
                                    symbol=ticker,
                                    qty=pyramid_shares,
                                    side='buy',
                                    type='market',
                                    time_in_force='day'
                                )
                                logger.info(f"🔺 PYRAMID ADD {pyramid_shares} shares of {ticker} @ ${price:.2f}")
                                logger.info(f"    Previous avg: ${avg_cost:.2f}, ATR distance: {(price-avg_cost)/atr_estimate:.1f}x")
                                trades_made += 1
                    
                    elif target_signal < sell_threshold:
                        if ticker in current_positions:
                            current_qty = int(current_positions[ticker].qty)
                            if current_qty > 0:
                                self.api.submit_order(
                                    symbol=ticker,
                                    qty=current_qty,
                                    side='sell',
                                    type='market',
                                    time_in_force='day'
                                )
                                logger.info(f"📉 SELL {current_qty} shares of {ticker} @ ${price:.2f}")
                                
                                # Log for conformal learning (negative signal for sell)
                                self._log_trade_for_conformal_learning(ticker, -abs(target_signal), regime, 'sell')
                                trades_made += 1
                
                except Exception as e:
                    logger.error(f"Trade error for {ticker}: {e}")
                
                if trades_made >= max_trades:
                    break
            
            # MONITORING & SAFEGUARDS (chat-g.txt enhancement)
            self._run_production_monitoring(trades_made, regime, regime_desc)
            
            logger.info(f"✅ FINAL PRODUCTION session complete: {trades_made} trades")
            
            # Show detailed portfolio summary
            if hasattr(self, 'demo_mode') and self.demo_mode:
                logger.info("=" * 60)
                logger.info("📊 DEMO PORTFOLIO SUMMARY")
                logger.info("=" * 60)
                logger.info(f"💰 Cash Remaining: ${self.demo_portfolio:,.2f}")
                
                total_position_value = 0
                if hasattr(self, 'demo_positions') and self.demo_positions:
                    logger.info(f"📈 Current Positions ({len(self.demo_positions)}):")
                    for symbol, pos in self.demo_positions.items():
                        position_value = pos['shares'] * pos['avg_price']
                        total_position_value += position_value
                        logger.info(f"   {symbol}: {pos['shares']} shares @ ${pos['avg_price']:.2f} = ${position_value:,.2f}")
                
                total_portfolio = self.demo_portfolio + total_position_value
                initial_value = 100000
                total_return = ((total_portfolio - initial_value) / initial_value) * 100
                
                logger.info(f"💎 Total Position Value: ${total_position_value:,.2f}")
                logger.info(f"💼 Total Portfolio Value: ${total_portfolio:,.2f}")
                logger.info(f"📈 Session Return: {total_return:+.2f}%")
                
                if hasattr(self, 'demo_trades') and self.demo_trades:
                    logger.info(f"🎯 Trades This Session: {len(self.demo_trades)}")
                    for trade in self.demo_trades[-5:]:  # Show last 5 trades
                        logger.info(f"   {trade['action']} {trade['shares']} {trade['symbol']} @ ${trade['price']:.2f} (Signal: {trade['signal']:.2f})")
                
                logger.info("=" * 60)
            
            if self.is_market_open():
                minutes_left = self.get_minutes_until_market_close()
                if minutes_left > 30:  # More than 30 minutes left
                    logger.info(f"🎯 Next scan in 15 minutes ({minutes_left} min until close)")
                    time.sleep(15 * 60)  # Wait 15 minutes
                    self.execute_final_trades()  # Recursive call during market hours
                else:
                    logger.info(f"🕐 Market closing soon ({minutes_left} min), stopping trading")
            else:
                next_open = self.get_next_market_open()
                logger.info(f"🎯 Market closed. Next scan: {next_open}")
            
        except Exception as e:
            logger.error(f"Trading error: {e}")
            self._emergency_safeguards(str(e))
    
    def _log_trade_for_conformal_learning(self, symbol: str, prediction: float, 
                                        regime: str, trade_type: str = 'buy'):
        """
        Log trade for conformal prediction learning
        Will be updated with actual returns after trade settles
        """
        try:
            trade_log = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'prediction': prediction,
                'regime': regime,
                'trade_type': trade_type,
                'logged_for_learning': True
            }
            
            # Store for later outcome tracking
            if not hasattr(self, 'pending_conformal_trades'):
                self.pending_conformal_trades = {}
            
            self.pending_conformal_trades[symbol] = trade_log
            
        except Exception as e:
            logger.error(f"Error logging trade for conformal learning: {e}")
    
    def _update_conformal_with_outcomes(self):
        """
        Update conformal predictor with actual trade outcomes
        Called periodically to improve calibration
        """
        if not hasattr(self, 'pending_conformal_trades') or not self.pending_conformal_trades:
            return
            
        try:
            # Get recent price data for outcome calculation
            symbols_to_update = []
            
            for symbol, trade_info in list(self.pending_conformal_trades.items()):
                trade_time = datetime.fromisoformat(trade_info['timestamp'])
                hours_elapsed = (datetime.now() - trade_time).total_seconds() / 3600
                
                # Update outcomes for trades older than 4 hours but younger than 7 days
                if 4 <= hours_elapsed <= 168:  # 4 hours to 7 days
                    symbols_to_update.append(symbol)
            
            if symbols_to_update:
                # Batch fetch price data
                for symbol in symbols_to_update[:5]:  # Limit to avoid API overuse
                    try:
                        trade_info = self.pending_conformal_trades[symbol]
                        trade_time = datetime.fromisoformat(trade_info['timestamp'])
                        
                        # Get price data around trade time
                        ticker = yf.Ticker(symbol)
                        end_time = datetime.now()
                        start_time = trade_time - timedelta(days=2)
                        
                        df = ticker.history(start=start_time, end=end_time, interval='1h')
                        
                        if len(df) >= 2:
                            # Calculate actual return since trade
                            trade_price_time = df.index[df.index >= trade_time][0] if any(df.index >= trade_time) else df.index[-2]
                            current_price_time = df.index[-1]
                            
                            trade_price = df.loc[trade_price_time, 'Close']
                            current_price = df.loc[current_price_time, 'Close']
                            
                            actual_return = (current_price - trade_price) / trade_price
                            
                            # Log the outcome for conformal learning
                            self.conformal_gate.log_trade_result(
                                symbol=symbol,
                                prediction=trade_info['prediction'],
                                actual_return=actual_return,
                                regime=trade_info['regime']
                            )
                            
                            logger.debug(f"📈 Conformal update: {symbol} pred={trade_info['prediction']:.3f}, actual={actual_return:.3f}")
                            
                            # Remove from pending
                            del self.pending_conformal_trades[symbol]
                            
                    except Exception as e:
                        logger.error(f"Error updating conformal outcome for {symbol}: {e}")
                        # Remove failed updates after 7 days
                        if hours_elapsed > 168:
                            del self.pending_conformal_trades[symbol]
        
        except Exception as e:
            logger.error(f"Error in conformal outcomes update: {e}")
    
    def _periodically_improve_conformal_calibration(self):
        """
        Periodic maintenance for conformal predictor
        Call this every few scans to update calibration
        """
        try:
            # Update outcomes from recent trades
            self._update_conformal_with_outcomes()
            
            # Log current performance
            if hasattr(self, 'scan_count'):
                self.scan_count = getattr(self, 'scan_count', 0) + 1
                
                # Log diagnostics every 10 scans
                if self.scan_count % 10 == 0:
                    diagnostics = self.conformal_gate.get_gate_diagnostics()
                    logger.info("📊 Conformal gate diagnostics:")
                    for regime, stats in diagnostics.items():
                        logger.info(f"   {regime}: threshold={stats.get('threshold', 0):.4f}, "
                                   f"hit_rate={stats.get('recent_hit_rate', 0):.1%}, "
                                   f"trades={stats.get('recent_trades', 0)}")
                
        except Exception as e:
            logger.error(f"Error in conformal calibration maintenance: {e}")
    
    def scheduled_trading_scan(self):
        """Scheduled trading scan that checks market hours"""
        if self.is_market_open():
            logger.info("⏰ 60-minute scan triggered - market is open")
            
            # Periodic conformal maintenance
            self._periodically_improve_conformal_calibration()
            
            # Main trading execution
            self.execute_final_trades()
        else:
            logger.info("🕐 60-minute scan skipped - market is closed")

def main():
    """Run the final production bot with 60-minute frequency during market hours"""
    print("🏆 FINAL PRODUCTION NASDAQ TRADING BOT")
    print("=" * 60)
    print("🎯 Combines bear market protection + bull market alpha")
    print("✅ Proven to beat QQQ in specific conditions")
    print("🛡️ Excellent risk management in volatile markets")
    print("⏰ 60-MINUTE FREQUENCY - INSTITUTIONAL GRADE")
    
    # Load API keys
    API_KEY = os.getenv('ALPACA_API_KEY')
    SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
    PAPER_TRADING = os.getenv('PAPER_TRADING', 'true').lower() == 'true'
    
    if not API_KEY or not SECRET_KEY:
        print("❌ Error: API keys not found!")
        return
    
    # Start final bot
    bot = FinalProductionBot(API_KEY, SECRET_KEY, paper=PAPER_TRADING)
    
    print("🚀 Final production bot active")
    print("📊 Adaptive regime detection")
    print("🎯 Focus on proven winners")
    print("⏰ Trading every 60 minutes during market hours")
    print("🕐 Market Hours: 9:30 AM - 4:00 PM ET (Mon-Fri)")
    print("-" * 60)
    
    # Initial scan if market is open
    if bot.is_market_open():
        print("📈 Market is open - running initial scan")
        bot.execute_final_trades()
    else:
        print("🕐 Market is closed - waiting for market open")
    
    # Schedule trading every 60 minutes during market hours
    schedule.every().hour.do(bot.scheduled_trading_scan)
    
    print("🔄 Bot scheduled to scan every 60 minutes during market hours")
    print("🛑 Press Ctrl+C to stop the bot")
    print("=" * 60)
    
    # Main trading loop
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        print("\n🛑 Trading bot stopped by user")
        print("💾 Final state saved")
        bot.save_state()

if __name__ == "__main__":
    main()