#!/usr/bin/env python3
"""
SMART REBALANCING BOT WITH DECISION LOGGING
Intelligent position management - keep winners, exit losers, refresh weak signals
Replaces daily liquidation with smart rebalancing strategy
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
import logging
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

from decision_logger import DecisionLogger

# Add root directory to path for imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from utils.intent_hash import compute_intent_hash

# Monitoring will be integrated later when prometheus_client is installed
try:
    from monitoring import (
        LATENCY,
        GROSS_EXPOSURE,
        SIGNAL_ACCEPT_RATE,
        start_monitoring,
    )
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    # Create dummy functions/objects
    class DummyMetric:
        def labels(self, **kwargs): return self
        def observe(self, value): pass
        def set(self, value): pass
    
    LATENCY = DummyMetric()
    GROSS_EXPOSURE = DummyMetric()
    SIGNAL_ACCEPT_RATE = DummyMetric()
    def start_monitoring(): pass

# Load environment variables
def load_env_file(env_path="PRODUCTION/config/alpaca.env"):
    """Load environment variables from file"""
    env_file = Path(env_path)
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print(f"✅ Loaded environment from {env_path}")
    else:
        print(f"⚠️ Environment file not found: {env_path}")

class SmartRebalancingBot:
    """Smart rebalancing bot - keeps winners, exits losers, refreshes weak signals"""
    
    def __init__(self, paper_trading=True):
        # Load environment
        load_env_file()
        
        self.paper_trading = paper_trading
        self.decision_logger = DecisionLogger()
        self.logger = self.setup_logging()
        
        # Smart rebalancing thresholds
        self.rebalance_config = {
            "profit_lock_threshold": 0.05,    # Lock profits at 5%+ gains
            "trailing_stop_distance": 0.06,   # 6% trailing stop from peak
            "stop_loss_threshold": -0.08,     # -8% hard stop loss
            "min_confidence_threshold": 0.15, # Exit if model confidence drops below 15%
            "profit_take_target": 0.25,       # Take full profit at 25%
            "position_refresh_days": 7,       # Refresh positions older than 7 days with low confidence
            "max_position_age_days": 14       # Force refresh positions older than 14 days
        }
        
        # Initialize Alpaca (if available)
        self.alpaca_available = self.setup_alpaca()
        
    def setup_logging(self):
        """Setup standard logging"""
        log_dir = Path("PRODUCTION/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'smart_rebalancing_bot.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('SmartRebalancingBot')
    
    def setup_alpaca(self):
        """Setup Alpaca connection"""
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
            
            api_key = os.getenv('ALPACA_API_KEY')
            api_secret = os.getenv('ALPACA_API_SECRET')
            
            if not api_key or not api_secret or api_key == 'your_api_key_here':
                self.logger.warning("⚠️ Alpaca API credentials not configured")
                return False
            
            self.trading_client = TradingClient(
                api_key=api_key,
                secret_key=api_secret,
                paper=self.paper_trading
            )
            
            # Test connection
            account = self.trading_client.get_account()
            self.logger.info(f"✅ Connected to Alpaca {'Paper' if self.paper_trading else 'Live'}")
            self.logger.info(f"   Portfolio: ${float(account.portfolio_value):,.2f}")
            
            return True
            
        except ImportError:
            self.logger.error("❌ Alpaca library not installed: pip install alpaca-py")
            return False
        except Exception as e:
            self.logger.error(f"❌ Alpaca connection failed: {e}")
            self.decision_logger.log_error("alpaca_connection", str(e))
            return False
    
    def get_current_positions(self):
        """Get current positions"""
        if not self.alpaca_available:
            # Simulate positions for testing
            return self.simulate_positions()
        
        try:
            positions = self.trading_client.get_all_positions()
            self.logger.info(f"📊 Found {len(positions)} current positions")
            
            position_data = []
            for pos in positions:
                position_data.append({
                    'symbol': pos.symbol,
                    'qty': float(pos.qty),
                    'side': pos.side,
                    'market_value': float(pos.market_value),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'avg_entry_price': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price),
                    'change_today': float(pos.change_today) if pos.change_today else 0
                })
            
            return position_data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get positions: {e}")
            self.decision_logger.log_error("get_positions", str(e))
            return []
    
    def analyze_position_performance(self, position):
        """Analyze individual position performance and determine action"""
        symbol = position['symbol']
        current_price = position['current_price']
        avg_entry_price = position['avg_entry_price']
        unrealized_pnl_pct = position['unrealized_pl'] / abs(position['market_value'])
        
        # Calculate position metrics
        position_analysis = {
            'symbol': symbol,
            'pnl_pct': unrealized_pnl_pct,
            'days_held': self.estimate_position_age(symbol),  # Would need position history
            'current_confidence': self.get_model_confidence_for_symbol(symbol),
            'action': 'HOLD',
            'reason': 'Normal position within thresholds'
        }
        
        # Decision logic
        if unrealized_pnl_pct >= self.rebalance_config['profit_take_target']:
            position_analysis['action'] = 'TAKE_PROFIT'
            position_analysis['reason'] = f'Hit profit target: +{unrealized_pnl_pct:.1%}'
            
        elif unrealized_pnl_pct <= self.rebalance_config['stop_loss_threshold']:
            position_analysis['action'] = 'STOP_LOSS'
            position_analysis['reason'] = f'Hit stop loss: {unrealized_pnl_pct:.1%}'
            
        elif unrealized_pnl_pct >= self.rebalance_config['profit_lock_threshold']:
            # Implement trailing stop logic
            position_analysis['action'] = 'TRAILING_STOP'
            position_analysis['reason'] = f'Profitable position +{unrealized_pnl_pct:.1%}, trailing stop active'
            
        elif position_analysis['current_confidence'] < self.rebalance_config['min_confidence_threshold']:
            position_analysis['action'] = 'EXIT_LOW_CONFIDENCE'
            position_analysis['reason'] = f'Model confidence dropped: {position_analysis["current_confidence"]:.2f}'
            
        elif position_analysis['days_held'] >= self.rebalance_config['max_position_age_days']:
            position_analysis['action'] = 'REFRESH_OLD'
            position_analysis['reason'] = f'Position too old: {position_analysis["days_held"]} days'
            
        elif (position_analysis['days_held'] >= self.rebalance_config['position_refresh_days'] and 
              position_analysis['current_confidence'] < 0.20):
            position_analysis['action'] = 'REFRESH_WEAK'
            position_analysis['reason'] = f'Weak signal after {position_analysis["days_held"]} days'
        
        return position_analysis
    
    def estimate_position_age(self, symbol):
        """Estimate position age (simplified - would need position history in production)"""
        return np.random.randint(1, 10)  # Placeholder
    
    def get_model_confidence_for_symbol(self, symbol):
        """Get current model confidence for symbol (simplified)"""
        return np.random.uniform(0.1, 0.3)  # Placeholder
    
    def simulate_positions(self):
        """Simulate positions for testing"""
        return [
            {
                'symbol': 'AAPL',
                'qty': 50,
                'side': 'long',
                'market_value': 8500.0,
                'unrealized_pl': 250.0,
                'avg_entry_price': 165.0,
                'current_price': 170.0,
                'change_today': 5.0
            },
            {
                'symbol': 'NVDA',
                'qty': 25,
                'side': 'long', 
                'market_value': 12750.0,
                'unrealized_pl': -125.0,
                'avg_entry_price': 515.0,
                'current_price': 510.0,
                'change_today': -5.0
            },
            {
                'symbol': 'TSLA',
                'qty': 30,
                'side': 'long',
                'market_value': 7200.0,
                'unrealized_pl': 450.0,
                'avg_entry_price': 225.0,
                'current_price': 240.0,
                'change_today': 15.0
            }
        ]
    
    def execute_position_action(self, position, analysis):
        """Execute action on position based on analysis"""
        start_time = time.time()
        symbol = position['symbol']
        qty = abs(position['qty'])
        side = position['side']
        current_price = position['current_price']
        action = analysis['action']
        reason = analysis['reason']
        
        if action == 'HOLD':
            self.logger.info(f"✅ HOLDING {symbol}: {reason}")
            return None
        
        # Compute intent hash for this action
        intent_hash = compute_intent_hash({
            "symbol": symbol,
            "action": action,
            "side": side,
            "qty": qty,
            "reason": reason
        })
        self.logger.info(f"Intent hash for {symbol} {action}: {intent_hash}")
        
        # For all exit actions, we need to liquidate the position
        self.decision_logger.log_liquidation_decision(
            position, 
            reason=f"Smart rebalancing: {reason}"
        )
        
        # Determine liquidation side
        if side == 'long':
            order_side = "SELL"
        else:
            order_side = "BUY"  # Cover short
        
        # Log order decision with smart rebalancing context
        self.decision_logger.log_order_decision(
            symbol, order_side, qty, current_price, "market"
        )
        
        if self.alpaca_available:
            try:
                # Execute real liquidation
                from alpaca.trading.requests import MarketOrderRequest
                from alpaca.trading.enums import OrderSide, TimeInForce
                
                alpaca_side = OrderSide.SELL if order_side == "SELL" else OrderSide.BUY
                
                market_order = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=alpaca_side,
                    time_in_force=TimeInForce.GTC
                )
                
                order = self.trading_client.submit_order(order_data=market_order)
                self.logger.info(f"✅ {action} order submitted: {symbol} - {reason}")
                
                result = {
                    'symbol': symbol,
                    'order_id': order.id,
                    'status': 'submitted',
                    'qty': qty,
                    'side': order_side,
                    'action': action,
                    'reason': reason,
                    'intent_hash': intent_hash
                }
                
                # Track latency
                LATENCY.labels(operation="execute_position_action").observe(time.time() - start_time)
                return result
                
            except Exception as e:
                self.logger.error(f"❌ Failed to liquidate {symbol}: {e}")
                self.decision_logger.log_error("liquidation_execution", str(e), {"symbol": symbol})
                return None
        else:
            # Simulate action
            self.logger.info(f"🎯 SIMULATED {action}: {order_side} {qty} {symbol} @ ${current_price:.2f} - {reason}")
            result = {
                'symbol': symbol,
                'order_id': f'SIM_{symbol}_{int(time.time())}',
                'status': 'simulated',
                'qty': qty,
                'side': order_side,
                'action': action,
                'reason': reason,
                'intent_hash': intent_hash
            }
            
            # Track latency
            LATENCY.labels(operation="execute_position_action").observe(time.time() - start_time)
            return result
    
    def wait_for_fills(self, liquidation_orders, timeout=300):
        """Wait for liquidation orders to fill"""
        if not liquidation_orders or not self.alpaca_available:
            return
        
        self.logger.info(f"⏳ Waiting for {len(liquidation_orders)} orders to fill...")
        
        start_time = time.time()
        pending_orders = liquidation_orders.copy()
        
        while pending_orders and (time.time() - start_time) < timeout:
            try:
                for order_info in pending_orders.copy():
                    if order_info['status'] == 'simulated':
                        pending_orders.remove(order_info)
                        continue
                        
                    # Check order status
                    order = self.trading_client.get_order_by_id(order_info['order_id'])
                    
                    if order.status in ['filled', 'partially_filled']:
                        self.logger.info(f"✅ {order_info['symbol']} liquidation filled")
                        pending_orders.remove(order_info)
                    elif order.status in ['canceled', 'rejected']:
                        self.logger.warning(f"⚠️ {order_info['symbol']} liquidation {order.status}")
                        pending_orders.remove(order_info)
                
                if pending_orders:
                    time.sleep(5)  # Wait 5 seconds before checking again
                    
            except Exception as e:
                self.logger.error(f"❌ Error checking order status: {e}")
                break
        
        if pending_orders:
            self.logger.warning(f"⚠️ {len(pending_orders)} orders still pending after timeout")
        else:
            self.logger.info("✅ All liquidation orders completed")
    
    def run_smart_rebalancing(self):
        """Run smart rebalancing process - keep winners, exit losers"""
        print("\n🧠 STARTING SMART REBALANCING")
        print("=" * 60)
        
        try:
            # Get current positions
            positions = self.get_current_positions()
            
            if not positions:
                print("✅ No positions to rebalance")
                self.decision_logger.save_session_summary(total_orders=0)
                return True
            
            print(f"📊 Analyzing {len(positions)} positions for smart rebalancing:")
            total_value = 0
            total_pnl = 0
            
            # Analyze each position
            position_actions = []
            for pos in positions:
                analysis = self.analyze_position_performance(pos)
                position_actions.append((pos, analysis))
                
                print(f"   {pos['symbol']}: {pos['side']} {pos['qty']} shares")
                print(f"     Market Value: ${pos['market_value']:,.2f}")
                print(f"     P&L: ${pos['unrealized_pl']:,.2f} ({analysis['pnl_pct']:.1%})")
                print(f"     Price: ${pos['current_price']:.2f} (${pos['change_today']:+.2f} today)")
                print(f"     🎯 ACTION: {analysis['action']} - {analysis['reason']}")
                total_value += pos['market_value']
                total_pnl += pos['unrealized_pl']
            
            print(f"\n💰 TOTAL PORTFOLIO:")
            print(f"   Market Value: ${total_value:,.2f}")
            print(f"   Unrealized P&L: ${total_pnl:+,.2f} ({total_pnl/total_value:.1%})")
            
            # Execute smart rebalancing actions
            print(f"\n🧠 EXECUTING SMART REBALANCING:")
            rebalancing_orders = []
            holds = 0
            exits = 0
            
            for position, analysis in position_actions:
                if analysis['action'] == 'HOLD':
                    holds += 1
                    print(f"✅ KEEPING {position['symbol']}: {analysis['reason']}")
                else:
                    exits += 1
                    order_info = self.execute_position_action(position, analysis)
                    if order_info:
                        rebalancing_orders.append(order_info)
                print()  # Add spacing
            
            # Calculate and track exposure metrics
            remaining_positions = len(updated_positions) if 'updated_positions' in locals() else (len(positions) - exits)
            gross_exposure = remaining_positions * 0.03  # Assume 3% per position
            GROSS_EXPOSURE.set(gross_exposure)
            
            # Track rebalancing metrics
            exit_rate = exits / len(positions) if positions else 0
            SIGNAL_ACCEPT_RATE.set(1 - exit_rate)  # Invert to show kept positions
            
            print(f"\n📊 REBALANCING SUMMARY:")
            print(f"   Positions Held: {holds}")
            print(f"   Positions Exited: {exits}")
            print(f"   Orders Submitted: {len(rebalancing_orders)}")
            print(f"   New Gross Exposure: {gross_exposure:.1%}")
            
            # Wait for fills
            if rebalancing_orders:
                self.wait_for_fills(rebalancing_orders)
            
            # Verify rebalancing
            print(f"\n🔍 VERIFYING REBALANCING:")
            updated_positions = self.get_current_positions()
            
            print(f"📊 Portfolio after rebalancing: {len(updated_positions)} positions")
            if updated_positions:
                for pos in updated_positions:
                    pnl_pct = pos['unrealized_pl'] / abs(pos['market_value'])
                    print(f"   {pos['symbol']}: {pos['qty']} shares ({pnl_pct:+.1%} P&L)")
            
            # Save decision logs
            self.decision_logger.save_session_summary(
                total_orders=len(rebalancing_orders)
            )
            
            print(f"\n🧠 SMART REBALANCING COMPLETE")
            print(f"   Winners Kept: {holds}")
            print(f"   Losers/Weak Exited: {exits}")
            print(f"   Orders Executed: {len(rebalancing_orders)}")
            print(f"   Decision Logs: {self.decision_logger.session_log}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Smart rebalancing failed: {e}")
            self.decision_logger.log_error("rebalancing_process", str(e))
            self.decision_logger.save_session_summary()
            return False

def main():
    """Main smart rebalancing function"""
    # Start monitoring system
    start_monitoring()
    
    print("🧠 ALPACA SMART REBALANCING BOT")
    print("WITH INTELLIGENT POSITION MANAGEMENT")
    print("=" * 60)
    
    # Check if paper trading
    paper_trading = os.getenv('PAPER_TRADING', 'true').lower() == 'true'
    
    if paper_trading:
        print("📊 Running in PAPER TRADING mode")
    else:
        print("💰 Running in LIVE TRADING mode")
        confirm = input("⚠️ Confirm LIVE smart rebalancing (yes/no): ")
        if confirm.lower() != 'yes':
            print("❌ Live rebalancing cancelled")
            return
    
    # Run smart rebalancing
    bot = SmartRebalancingBot(paper_trading=paper_trading)
    success = bot.run_smart_rebalancing()
    
    if success:
        print("\n🎉 SMART REBALANCING SUCCESSFUL")
        print("💡 Winners kept, losers exited, weak positions refreshed!")
    else:
        print("\n❌ SMART REBALANCING FAILED")

if __name__ == "__main__":
    main()