#!/usr/bin/env python3
"""
LIQUIDATION BOT WITH DECISION LOGGING
Sell all positions with detailed decision logging for each action
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

class LiquidationBot:
    """Bot to liquidate all positions with detailed logging"""
    
    def __init__(self, paper_trading=True):
        # Load environment
        load_env_file()
        
        self.paper_trading = paper_trading
        self.decision_logger = DecisionLogger()
        self.logger = self.setup_logging()
        
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
                logging.FileHandler(log_dir / 'liquidation_bot.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('LiquidationBot')
    
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
    
    def liquidate_position(self, position):
        """Liquidate a single position with decision logging"""
        symbol = position['symbol']
        qty = abs(position['qty'])
        side = position['side']
        current_price = position['current_price']
        
        # Log liquidation decision
        self.decision_logger.log_liquidation_decision(
            position, 
            reason="Daily reset - liquidate all positions before new signals"
        )
        
        # Determine liquidation side
        if side == 'long':
            order_side = "SELL"
        else:
            order_side = "BUY"  # Cover short
        
        # Log order decision
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
                self.logger.info(f"✅ Liquidation order submitted: {symbol}")
                
                return {
                    'symbol': symbol,
                    'order_id': order.id,
                    'status': 'submitted',
                    'qty': qty,
                    'side': order_side
                }
                
            except Exception as e:
                self.logger.error(f"❌ Failed to liquidate {symbol}: {e}")
                self.decision_logger.log_error("liquidation_execution", str(e), {"symbol": symbol})
                return None
        else:
            # Simulate liquidation
            self.logger.info(f"🎯 SIMULATED LIQUIDATION: {order_side} {qty} {symbol} @ ${current_price:.2f}")
            return {
                'symbol': symbol,
                'order_id': f'SIM_{symbol}_{int(time.time())}',
                'status': 'simulated',
                'qty': qty,
                'side': order_side
            }
    
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
    
    def run_liquidation(self):
        """Run complete liquidation process"""
        print("\n🔥 STARTING COMPLETE LIQUIDATION")
        print("=" * 60)
        
        try:
            # Get current positions
            positions = self.get_current_positions()
            
            if not positions:
                print("✅ No positions to liquidate")
                self.decision_logger.save_session_summary(total_orders=0)
                return True
            
            print(f"📊 Found {len(positions)} positions to liquidate:")
            total_value = 0
            total_pnl = 0
            
            for pos in positions:
                print(f"   {pos['symbol']}: {pos['side']} {pos['qty']} shares")
                print(f"     Market Value: ${pos['market_value']:,.2f}")
                print(f"     P&L: ${pos['unrealized_pl']:,.2f}")
                print(f"     Price: ${pos['current_price']:.2f} (${pos['change_today']:+.2f} today)")
                total_value += pos['market_value']
                total_pnl += pos['unrealized_pl']
            
            print(f"\n💰 TOTAL PORTFOLIO:")
            print(f"   Market Value: ${total_value:,.2f}")
            print(f"   Unrealized P&L: ${total_pnl:+,.2f}")
            
            # Execute liquidations
            print(f"\n🎯 EXECUTING LIQUIDATIONS:")
            liquidation_orders = []
            
            for position in positions:
                order_info = self.liquidate_position(position)
                if order_info:
                    liquidation_orders.append(order_info)
                print()  # Add spacing
            
            # Wait for fills
            if liquidation_orders:
                self.wait_for_fills(liquidation_orders)
            
            # Verify liquidation
            print(f"\n🔍 VERIFYING LIQUIDATION:")
            remaining_positions = self.get_current_positions()
            
            if remaining_positions:
                print(f"⚠️ {len(remaining_positions)} positions still remain:")
                for pos in remaining_positions:
                    print(f"   {pos['symbol']}: {pos['qty']} shares")
            else:
                print("✅ All positions successfully liquidated")
            
            # Save decision logs
            self.decision_logger.save_session_summary(
                total_orders=len(liquidation_orders)
            )
            
            print(f"\n📊 LIQUIDATION COMPLETE")
            print(f"   Orders Executed: {len(liquidation_orders)}")
            print(f"   Decision Logs: {self.decision_logger.session_log}")
            
            return len(remaining_positions) == 0
            
        except Exception as e:
            self.logger.error(f"❌ Liquidation failed: {e}")
            self.decision_logger.log_error("liquidation_process", str(e))
            self.decision_logger.save_session_summary()
            return False

def main():
    """Main liquidation function"""
    print("🔥 ALPACA LIQUIDATION BOT")
    print("WITH DETAILED DECISION LOGGING")
    print("=" * 60)
    
    # Check if paper trading
    paper_trading = os.getenv('PAPER_TRADING', 'true').lower() == 'true'
    
    if paper_trading:
        print("📊 Running in PAPER TRADING mode")
    else:
        print("💰 Running in LIVE TRADING mode")
        confirm = input("⚠️ Confirm LIVE liquidation (yes/no): ")
        if confirm.lower() != 'yes':
            print("❌ Live liquidation cancelled")
            return
    
    # Run liquidation
    bot = LiquidationBot(paper_trading=paper_trading)
    success = bot.run_liquidation()
    
    if success:
        print("\n🎉 LIQUIDATION SUCCESSFUL")
    else:
        print("\n❌ LIQUIDATION FAILED")

if __name__ == "__main__":
    main()