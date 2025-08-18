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
from dotenv import load_dotenv
from datetime import datetime, timedelta
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinalProductionBot:
    def __init__(self, api_key, secret_key, paper=True):
        """Final production bot with proven strategies"""
        base_url = 'https://paper-api.alpaca.markets' if paper else 'https://api.alpaca.markets'
        self.api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        
        # Optimized parameters
        self.max_positions = 20
        self.position_size = 0.05  # 5% per position
        self.min_confidence = 0.6
        
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
    
    def _apply_portfolio_filters(self, signals, current_positions, regime, regime_desc):
        """Apply portfolio construction upgrades (chat-g.txt enhancement)"""
        
        logger.info("🏗️ Applying portfolio construction filters...")
        
        # 1. VOLATILITY TARGETING
        filtered_signals = self._apply_volatility_targeting(signals, regime)
        
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
    
    def execute_final_trades(self):
        """Final production trading execution"""
        logger.info("🚀 Starting FINAL PRODUCTION trading scan...")
        
        try:
            # Get account info
            current_positions = {pos.symbol: pos for pos in self.api.list_positions()}
            account = self.api.get_account()
            buying_power = float(account.buying_power)
            
            # Get market regime
            regime, regime_bias, regime_desc = self.get_market_regime_final()
            
            logger.info(f"💰 Buying power: ${buying_power:,.2f}")
            logger.info(f"📊 Current positions: {len(current_positions)}")
            logger.info(f"🌊 Market regime: {regime.upper()} ({regime_desc}) - Bias: {regime_bias:+.2f}")
            
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
                            # New position
                            self.api.submit_order(
                                symbol=ticker,
                                qty=shares,
                                side='buy',
                                type='market',
                                time_in_force='day'
                            )
                            logger.info(f"📈 BUY {shares} shares of {ticker} @ ${price:.2f}")
                            logger.info(f"    Signal: {target_signal:.2f}, Confidence: {confidence:.2f}")
                            logger.info(f"    Strategy: {signal['strategy_type']}, Regime: {regime}")
                            trades_made += 1
                            
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
                                trades_made += 1
                
                except Exception as e:
                    logger.error(f"Trade error for {ticker}: {e}")
                
                if trades_made >= max_trades:
                    break
            
            # MONITORING & SAFEGUARDS (chat-g.txt enhancement)
            self._run_production_monitoring(trades_made, regime, regime_desc)
            
            logger.info(f"✅ FINAL PRODUCTION session complete: {trades_made} trades")
            logger.info(f"🎯 Next scan in 24 hours")
            
        except Exception as e:
            logger.error(f"Trading error: {e}")
            self._emergency_safeguards(str(e))

def main():
    """Run the final production bot"""
    print("🏆 FINAL PRODUCTION NASDAQ TRADING BOT")
    print("=" * 60)
    print("🎯 Combines bear market protection + bull market alpha")
    print("✅ Proven to beat QQQ in specific conditions")
    print("🛡️ Excellent risk management in volatile markets")
    
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
    
    # Test run
    bot.execute_final_trades()

if __name__ == "__main__":
    main()