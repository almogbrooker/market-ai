#!/usr/bin/env python3
"""
OPTIMIZED TRADING BOT WITH FREQUENCY TESTING
Tests different trading frequencies (15min, 30min, 60min) against QQQ benchmark
Based on institutional best practices for algorithmic trading
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
from dotenv import load_dotenv
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys

# Add src to path for model imports
sys.path.append(str(Path(__file__).parent / "src"))

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedTradingBot:
    """
    Optimized trading bot with configurable frequency and QQQ benchmark comparison
    Implements institutional best practices for frequency trading
    """
    
    def __init__(self, api_key=None, secret_key=None, paper=True, trading_frequency=60):
        """
        Initialize optimized trading bot
        
        Args:
            trading_frequency: Minutes between trading scans (15, 30, or 60)
        """
        self.trading_frequency = trading_frequency
        
        # Safety state management
        self.state_file = f"bot_state_{trading_frequency}min.json"
        self.daily_pnl_limit = -0.03  # -3% daily kill switch
        self.max_retries = 3
        self.backoff_seconds = 1
        
        # Performance tracking for comparison
        self.performance_log = f"performance_{trading_frequency}min.json"
        self.trades_history = []
        self.portfolio_values = []
        self.qqq_benchmark = []
        
        # Alpaca API setup
        base_url = 'https://paper-api.alpaca.markets' if paper else 'https://api.alpaca.markets'
        
        if not api_key:
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not api_key or not secret_key:
            logger.info(f"🎯 No API credentials - running {trading_frequency}min frequency in DEMO mode")
            self.api = None
            self.demo_mode = True
        else:
            try:
                self.api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
                account = self.api.get_account()
                logger.info(f"✅ Connected to Alpaca ({trading_frequency}min frequency)")
                logger.info(f"💰 Buying Power: ${float(account.buying_power):,.2f}")
                self.demo_mode = False
            except Exception as e:
                logger.warning(f"⚠️ Could not connect to Alpaca: {e}")
                self.demo_mode = True
        
        # Trading parameters optimized for frequency
        if trading_frequency == 15:
            # High frequency - more conservative
            self.position_size = 0.05  # 5% per position
            self.min_confidence = 0.75  # Higher confidence threshold
            self.max_positions = 8
        elif trading_frequency == 30:
            # Medium frequency - balanced
            self.position_size = 0.075  # 7.5% per position
            self.min_confidence = 0.65
            self.max_positions = 10
        else:  # 60 minutes
            # Lower frequency - institutional approach
            self.position_size = 0.10  # 10% per position
            self.min_confidence = 0.60
            self.max_positions = 12
        
        # Stock universe - focus on liquid names
        self.stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD',
            'INTC', 'QCOM', 'AVGO', 'TXN', 'ORCL', 'CRM', 'ADBE', 'NOW'
        ]
        
        # Initialize state
        self.load_state()
        
        logger.info(f"🚀 Optimized Trading Bot Initialized")
        logger.info(f"⏰ Trading Frequency: {trading_frequency} minutes")
        logger.info(f"💼 Position Size: {self.position_size:.1%}")
        logger.info(f"🎯 Min Confidence: {self.min_confidence:.1%}")
        logger.info(f"📈 Max Positions: {self.max_positions}")
    
    def is_market_open(self) -> bool:
        """Check if market is currently open (9:30 AM - 4:00 PM ET, Mon-Fri)"""
        et_tz = pytz.timezone('US/Eastern')
        now_et = datetime.now(et_tz)
        
        # Check if weekday (0 = Monday, 6 = Sunday)
        if now_et.weekday() >= 5:  # Saturday or Sunday
            return False
        
        # Check if within trading hours (9:30 AM - 4:00 PM ET)
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now_et <= market_close
    
    def load_state(self):
        """Load bot state from file"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.open_orders = state.get('open_orders', {})
                    self.last_targets = state.get('last_targets', {})
                    self.kill_switch_active = state.get('kill_switch_active', False)
                    self.daily_start_value = state.get('daily_start_value', None)
                    self.last_reset_date = state.get('last_reset_date', str(datetime.now().date()))
                    logger.info(f"📁 State loaded from {self.state_file}")
            else:
                self.open_orders = {}
                self.last_targets = {}
                self.kill_switch_active = False
                self.daily_start_value = None
                self.last_reset_date = str(datetime.now().date())
        except Exception as e:
            logger.error(f"❌ Error loading state: {e}")
            self.open_orders = {}
            self.last_targets = {}
            self.kill_switch_active = False
            self.daily_start_value = None
            self.last_reset_date = str(datetime.now().date())
    
    def save_state(self):
        """Save bot state to file"""
        try:
            state = {
                'open_orders': self.open_orders,
                'last_targets': self.last_targets,
                'kill_switch_active': self.kill_switch_active,
                'daily_start_value': self.daily_start_value,
                'last_reset_date': self.last_reset_date
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Error saving state: {e}")
    
    def check_daily_kill_switch(self) -> bool:
        """Check if daily kill switch should be activated"""
        if self.demo_mode:
            return False
        
        try:
            # Reset daily tracking if new day
            today = str(datetime.now().date())
            if today != self.last_reset_date:
                self.daily_start_value = None
                self.kill_switch_active = False
                self.last_reset_date = today
                logger.info(f"🔄 Daily reset: {today}")
            
            # Get current portfolio value
            account = self.api.get_account()
            current_value = float(account.portfolio_value)
            
            # Set starting value if not set
            if self.daily_start_value is None:
                self.daily_start_value = current_value
                logger.info(f"📊 Daily start value: ${current_value:,.2f}")
            
            # Check kill switch
            daily_pnl = (current_value - self.daily_start_value) / self.daily_start_value
            
            if daily_pnl <= self.daily_pnl_limit:
                self.kill_switch_active = True
                logger.warning(f"🚨 DAILY KILL SWITCH ACTIVATED: {daily_pnl:.2%}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking kill switch: {e}")
            return False
    
    def get_enhanced_features(self, symbol: str) -> dict:
        """Get enhanced technical features for a symbol"""
        try:
            # Download recent data
            stock = yf.Ticker(symbol)
            df = stock.history(period="3mo", interval="1d")
            
            if len(df) < 50:
                return {}
            
            # Calculate technical indicators
            latest = df.iloc[-1]
            
            # Price momentum
            df['return_1d'] = df['Close'].pct_change(1)
            df['return_5d'] = df['Close'].pct_change(5)
            df['return_20d'] = df['Close'].pct_change(20)
            
            # Moving averages
            df['sma_5'] = df['Close'].rolling(5).mean()
            df['sma_10'] = df['Close'].rolling(10).mean()
            df['sma_20'] = df['Close'].rolling(20).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Volume indicators
            df['volume_sma'] = df['Volume'].rolling(20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma']
            
            # Volatility
            df['volatility'] = df['return_1d'].rolling(20).std() * np.sqrt(252)
            
            # Return latest features
            latest_row = df.iloc[-1]
            
            features = {
                'return_1d': latest_row['return_1d'],
                'return_5d': latest_row['return_5d'],
                'return_20d': latest_row['return_20d'],
                'price_vs_sma5': latest['Close'] / latest_row['sma_5'] - 1,
                'price_vs_sma10': latest['Close'] / latest_row['sma_10'] - 1,
                'price_vs_sma20': latest['Close'] / latest_row['sma_20'] - 1,
                'rsi': latest_row['rsi'],
                'volume_ratio': latest_row['volume_ratio'],
                'volatility': latest_row['volatility'],
                'close_price': latest['Close']
            }
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error getting features for {symbol}: {e}")
            return {}
    
    def generate_signals(self) -> dict:
        """Generate trading signals based on frequency-optimized strategy"""
        signals = {}
        
        for symbol in self.stocks:
            try:
                features = self.get_enhanced_features(symbol)
                if not features:
                    continue
                
                # Frequency-specific signal generation
                if self.trading_frequency == 15:
                    # High frequency - momentum focused
                    signal = self._generate_hf_signal(features)
                elif self.trading_frequency == 30:
                    # Medium frequency - balanced
                    signal = self._generate_mf_signal(features)
                else:  # 60 minutes
                    # Low frequency - fundamental + technical
                    signal = self._generate_lf_signal(features)
                
                if abs(signal) >= self.min_confidence:
                    signals[symbol] = {
                        'signal': signal,
                        'confidence': abs(signal),
                        'features': features
                    }
                    
            except Exception as e:
                logger.error(f"❌ Error generating signal for {symbol}: {e}")
        
        return signals
    
    def _generate_hf_signal(self, features: dict) -> float:
        """High frequency signal (15 min) - momentum focused"""
        signal = 0.0
        
        # Short-term momentum (60% weight)
        momentum_score = 0
        if features['return_1d'] > 0.02:
            momentum_score = 0.4
        elif features['return_1d'] < -0.02:
            momentum_score = -0.4
        
        # Price vs SMA5 (30% weight)
        sma_score = 0
        if features['price_vs_sma5'] > 0.01:
            sma_score = 0.3
        elif features['price_vs_sma5'] < -0.01:
            sma_score = -0.3
        
        # RSI mean reversion (10% weight)
        rsi_score = 0
        if features['rsi'] < 30:
            rsi_score = 0.1
        elif features['rsi'] > 70:
            rsi_score = -0.1
        
        signal = momentum_score * 0.6 + sma_score * 0.3 + rsi_score * 0.1
        return signal
    
    def _generate_mf_signal(self, features: dict) -> float:
        """Medium frequency signal (30 min) - balanced approach"""
        signal = 0.0
        
        # Medium-term momentum (40% weight)
        momentum_score = 0
        if features['return_5d'] > 0.05:
            momentum_score = 0.3
        elif features['return_5d'] < -0.05:
            momentum_score = -0.3
        
        # Trend (30% weight)
        trend_score = 0
        if features['price_vs_sma10'] > 0.02:
            trend_score = 0.2
        elif features['price_vs_sma10'] < -0.02:
            trend_score = -0.2
        
        # RSI (20% weight)
        rsi_score = 0
        if features['rsi'] < 35:
            rsi_score = 0.15
        elif features['rsi'] > 65:
            rsi_score = -0.15
        
        # Volume confirmation (10% weight)
        volume_score = 0
        if features['volume_ratio'] > 1.5:
            volume_score = 0.05
        
        signal = momentum_score * 0.4 + trend_score * 0.3 + rsi_score * 0.2 + volume_score * 0.1
        return signal
    
    def _generate_lf_signal(self, features: dict) -> float:
        """Low frequency signal (60 min) - institutional approach"""
        signal = 0.0
        
        # Long-term trend (35% weight)
        trend_score = 0
        if features['price_vs_sma20'] > 0.03:
            trend_score = 0.25
        elif features['price_vs_sma20'] < -0.03:
            trend_score = -0.25
        
        # Medium-term momentum (30% weight)
        momentum_score = 0
        if features['return_20d'] > 0.10:
            momentum_score = 0.2
        elif features['return_20d'] < -0.10:
            momentum_score = -0.2
        
        # RSI positioning (25% weight)
        rsi_score = 0
        if features['rsi'] < 40:
            rsi_score = 0.18
        elif features['rsi'] > 60:
            rsi_score = -0.18
        
        # Volatility adjustment (10% weight)
        vol_score = 0
        if features['volatility'] < 0.20:  # Low volatility
            vol_score = 0.05
        elif features['volatility'] > 0.40:  # High volatility
            vol_score = -0.05
        
        signal = trend_score * 0.35 + momentum_score * 0.3 + rsi_score * 0.25 + vol_score * 0.1
        return signal
    
    def execute_trades(self, signals: dict):
        """Execute trades based on signals"""
        if self.kill_switch_active:
            logger.warning("🚨 Kill switch active - no trading")
            return
        
        if not self.is_market_open():
            logger.info("🕐 Market closed - no trading")
            return
        
        if self.demo_mode:
            logger.info(f"🎯 DEMO MODE - Would trade {len(signals)} signals")
            for symbol, signal_data in signals.items():
                logger.info(f"  {symbol}: {signal_data['signal']:.3f} (conf: {signal_data['confidence']:.3f})")
            return
        
        # Execute actual trades in live mode
        self._execute_live_trades(signals)
    
    def _execute_live_trades(self, signals: dict):
        """Execute live trades with Alpaca"""
        try:
            account = self.api.get_account()
            buying_power = float(account.buying_power)
            
            for symbol, signal_data in signals.items():
                signal = signal_data['signal']
                confidence = signal_data['confidence']
                
                # Calculate position size
                trade_value = buying_power * self.position_size * confidence
                
                if trade_value < 100:  # Minimum trade size
                    continue
                
                # Get current price
                latest_price = signal_data['features']['close_price']
                shares = int(trade_value / latest_price)
                
                if shares == 0:
                    continue
                
                # Determine side
                side = 'buy' if signal > 0 else 'sell'
                
                # Submit order
                try:
                    order = self.api.submit_order(
                        symbol=symbol,
                        qty=shares,
                        side=side,
                        type='market',
                        time_in_force='day',
                        client_order_id=str(uuid.uuid4())
                    )
                    
                    logger.info(f"📈 {side.upper()} {shares} {symbol} @ ${latest_price:.2f} (conf: {confidence:.3f})")
                    
                    # Track trade
                    self.trades_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'side': side,
                        'shares': shares,
                        'price': latest_price,
                        'confidence': confidence,
                        'frequency': f"{self.trading_frequency}min"
                    })
                    
                except Exception as e:
                    logger.error(f"❌ Error submitting order for {symbol}: {e}")
            
            self.save_state()
            
        except Exception as e:
            logger.error(f"❌ Error executing trades: {e}")
    
    def log_performance(self):
        """Log performance metrics for comparison"""
        try:
            if self.demo_mode:
                return
            
            # Get current portfolio value
            account = self.api.get_account()
            portfolio_value = float(account.portfolio_value)
            
            # Get QQQ for benchmark
            qqq = yf.Ticker("QQQ")
            qqq_data = qqq.history(period="1d", interval="1m")
            qqq_price = qqq_data['Close'].iloc[-1] if len(qqq_data) > 0 else 0
            
            # Log data point
            data_point = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_value': portfolio_value,
                'qqq_price': qqq_price,
                'frequency': f"{self.trading_frequency}min"
            }
            
            # Save to performance log
            try:
                if os.path.exists(self.performance_log):
                    with open(self.performance_log, 'r') as f:
                        performance_data = json.load(f)
                else:
                    performance_data = []
                
                performance_data.append(data_point)
                
                with open(self.performance_log, 'w') as f:
                    json.dump(performance_data, f, indent=2)
                    
            except Exception as e:
                logger.error(f"❌ Error saving performance: {e}")
                
        except Exception as e:
            logger.error(f"❌ Error logging performance: {e}")
    
    def scan_and_trade(self):
        """Main trading logic called on schedule"""
        try:
            logger.info(f"🔍 Scanning markets ({self.trading_frequency}min frequency)")
            
            # Check kill switch
            if self.check_daily_kill_switch():
                return
            
            # Generate signals
            signals = self.generate_signals()
            logger.info(f"📊 Generated {len(signals)} signals")
            
            # Execute trades
            self.execute_trades(signals)
            
            # Log performance
            self.log_performance()
            
            self.save_state()
            
        except Exception as e:
            logger.error(f"❌ Error in scan_and_trade: {e}")
    
    def start_trading(self):
        """Start the trading bot with market hours awareness"""
        logger.info(f"🚀 Starting Optimized Trading Bot ({self.trading_frequency}min frequency)")
        logger.info(f"⏰ Market Hours: 9:30 AM - 4:00 PM ET (Mon-Fri)")
        
        # Initial scan if market is open
        if self.is_market_open():
            logger.info("📈 Market is open - running initial scan")
            self.scan_and_trade()
        else:
            logger.info("🕐 Market is closed - waiting for open")
        
        # Schedule trading based on frequency
        if self.trading_frequency == 15:
            schedule.every(15).minutes.do(self._scheduled_scan)
        elif self.trading_frequency == 30:
            schedule.every(30).minutes.do(self._scheduled_scan)
        else:  # 60 minutes
            schedule.every().hour.do(self._scheduled_scan)
        
        # Main loop
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info(f"🛑 Trading bot stopped ({self.trading_frequency}min)")
    
    def _scheduled_scan(self):
        """Wrapper for scheduled scans that checks market hours"""
        if self.is_market_open():
            self.scan_and_trade()
        else:
            logger.info("🕐 Market closed - skipping scan")

def compare_frequencies():
    """Compare performance across different trading frequencies"""
    logger.info("📊 TRADING FREQUENCY COMPARISON ANALYSIS")
    logger.info("=" * 60)
    
    frequencies = [15, 30, 60]
    performance_data = {}
    
    for freq in frequencies:
        performance_file = f"performance_{freq}min.json"
        
        if os.path.exists(performance_file):
            with open(performance_file, 'r') as f:
                data = json.load(f)
                performance_data[freq] = data
        else:
            logger.warning(f"⚠️ No performance data found for {freq}min frequency")
    
    if not performance_data:
        logger.warning("⚠️ No performance data available for comparison")
        return
    
    # Analyze and compare
    for freq, data in performance_data.items():
        if len(data) < 2:
            continue
        
        start_value = data[0]['portfolio_value']
        end_value = data[-1]['portfolio_value']
        total_return = (end_value - start_value) / start_value
        
        # QQQ comparison
        start_qqq = data[0]['qqq_price']
        end_qqq = data[-1]['qqq_price']
        qqq_return = (end_qqq - start_qqq) / start_qqq if start_qqq > 0 else 0
        
        alpha = total_return - qqq_return
        
        logger.info(f"📈 {freq}min Frequency:")
        logger.info(f"   Portfolio Return: {total_return:.2%}")
        logger.info(f"   QQQ Return: {qqq_return:.2%}")
        logger.info(f"   Alpha vs QQQ: {alpha:.2%}")
        logger.info(f"   Data Points: {len(data)}")
        logger.info("-" * 40)

def main():
    """Main entry point with frequency selection"""
    print("🤖 OPTIMIZED TRADING BOT - FREQUENCY TESTING")
    print("=" * 60)
    print("📊 Institutional-grade frequency optimization")
    print("🎯 Tests 15min, 30min, and 60min intervals vs QQQ")
    print()
    
    # Get configuration
    frequency = int(os.getenv('TRADING_FREQUENCY', '60'))
    
    print(f"⏰ Trading Frequency: {frequency} minutes")
    
    if frequency == 15:
        print("🔥 HIGH FREQUENCY: Momentum-focused, 5% positions, 75% confidence")
    elif frequency == 30:
        print("⚖️ MEDIUM FREQUENCY: Balanced approach, 7.5% positions, 65% confidence")
    else:
        print("🏛️ LOW FREQUENCY: Institutional approach, 10% positions, 60% confidence")
    
    print("-" * 60)
    
    # Initialize and start bot
    bot = OptimizedTradingBot(trading_frequency=frequency)
    
    try:
        bot.start_trading()
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("📊 GENERATING FREQUENCY COMPARISON REPORT")
        compare_frequencies()

if __name__ == "__main__":
    main()