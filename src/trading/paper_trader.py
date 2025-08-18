#!/usr/bin/env python3
"""
Paper Trading with Alpaca Integration
Implements Phase 5 from chat-g-2.txt: Live paper trading with real market data
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from pathlib import Path
import time
import json
from datetime import datetime, timedelta
import asyncio
import signal
import sys

try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import REST, TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logging.warning("Alpaca Trade API not available. Install with: pip install alpaca-trade-api")

logger = logging.getLogger(__name__)

class PaperTrader:
    """
    Live paper trading with Alpaca integration
    """
    
    def __init__(self, model_dir: str, max_gross: float = 0.6, max_per_name: float = 0.08,
                 api_key: str = None, secret_key: str = None, base_url: str = None):
        
        self.model_dir = Path(model_dir)
        self.max_gross = max_gross  # Maximum gross exposure
        self.max_per_name = max_per_name  # Maximum per-name exposure
        
        # Alpaca credentials (use environment variables in production)
        self.api_key = api_key or "YOUR_ALPACA_API_KEY"
        self.secret_key = secret_key or "YOUR_ALPACA_SECRET_KEY"
        self.base_url = base_url or "https://paper-api.alpaca.markets"  # Paper trading URL
        
        # Initialize Alpaca API
        self.api = None
        self.account = None
        self._initialize_alpaca()
        
        # Trading state
        self.positions = {}
        self.orders = {}
        self.portfolio_value = 0.0
        self.is_trading = False
        
        # Risk management
        self.daily_loss_limit = 0.05  # 5% daily loss limit
        self.position_timeout = 20  # Days to hold position

        # Load calibrated regime thresholds
        self.regime_thresholds = self._load_regime_thresholds()
        self.quality_gate = self.regime_thresholds.get('quality_gate', 0.95)
        
        logger.info(f"📈 PaperTrader initialized: {max_gross:.1%} max gross, {max_per_name:.1%} per name")
    
    def _initialize_alpaca(self):
        """Initialize Alpaca API connection"""
        
        if not ALPACA_AVAILABLE:
            logger.warning("⚠️ Alpaca API not available, using simulation mode")
            return
        
        try:
            self.api = REST(
                key_id=self.api_key,
                secret_key=self.secret_key,
                base_url=self.base_url,
                api_version='v2'
            )
            
            # Test connection
            self.account = self.api.get_account()
            logger.info(f"✅ Connected to Alpaca: {self.account.status}")
            logger.info(f"   Portfolio Value: ${float(self.account.portfolio_value):,.2f}")
            logger.info(f"   Buying Power: ${float(self.account.buying_power):,.2f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Alpaca: {e}")
            logger.info("Running in simulation mode")
            self.api = None

    def _load_regime_thresholds(self) -> Dict[str, float]:
        """Load regime thresholds from the model directory."""

        path = self.model_dir / "regime_thresholds.json"
        if path.exists():
            try:
                with open(path, "r") as f:
                    thresholds = json.load(f)
                logger.info(f"📥 Loaded regime thresholds from {path}")
                return thresholds
            except Exception as e:
                logger.warning(f"Could not load regime thresholds: {e}")
        else:
            logger.warning("regime_thresholds.json not found; using default thresholds")
        return {}

    def start_trading(self):
        """Start the paper trading loop"""
        
        logger.info("🚀 Starting paper trading...")
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.is_trading = True
        
        try:
            # Run the main trading loop
            asyncio.run(self._trading_loop())
        except KeyboardInterrupt:
            logger.info("Trading stopped by user")
        except Exception as e:
            logger.error(f"Trading loop error: {e}")
        finally:
            self._cleanup()
    
    async def _trading_loop(self):
        """Main trading loop"""
        
        logger.info("🔄 Starting trading loop...")
        
        while self.is_trading:
            try:
                # Check if market is open
                if not self._is_market_open():
                    logger.info("📅 Market is closed, waiting...")
                    await asyncio.sleep(300)  # Wait 5 minutes
                    continue
                
                # Update portfolio information
                self._update_portfolio_info()
                
                # Check and manage existing positions
                self._manage_positions()
                
                # Check for new trading signals
                signals = self._get_trading_signals()
                
                # Execute new trades
                if signals:
                    self._execute_trades(signals)
                
                # Risk management checks
                self._risk_management_checks()
                
                # Log status
                self._log_trading_status()
                
                # Wait before next iteration (trade every 15 minutes during market hours)
                await asyncio.sleep(900)
                
            except Exception as e:
                logger.error(f"Trading loop iteration failed: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    def _is_market_open(self) -> bool:
        """Check if the market is currently open"""
        
        if self.api:
            try:
                clock = self.api.get_clock()
                return clock.is_open
            except Exception as e:
                logger.warning(f"Could not check market status: {e}")
        
        # Fallback: simple time-based check (9:30 AM - 4:00 PM ET, weekdays)
        now = datetime.now()
        if now.weekday() >= 5:  # Weekend
            return False
        
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def _update_portfolio_info(self):
        """Update current portfolio information"""
        
        if self.api:
            try:
                self.account = self.api.get_account()
                self.portfolio_value = float(self.account.portfolio_value)
                
                # Get current positions
                positions = self.api.list_positions()
                self.positions = {pos.symbol: pos for pos in positions}
                
            except Exception as e:
                logger.warning(f"Could not update portfolio info: {e}")
        else:
            # Simulation mode
            self.portfolio_value = 100000.0  # Mock portfolio value
    
    def _manage_positions(self):
        """Manage existing positions - check for exits"""
        
        for symbol, position in self.positions.items():
            try:
                # Get current price
                current_price = self._get_current_price(symbol)
                if not current_price:
                    continue
                
                # Check if position should be closed
                should_close = self._should_close_position(symbol, position, current_price)
                
                if should_close:
                    self._close_position(symbol, position)
                    
            except Exception as e:
                logger.error(f"Error managing position {symbol}: {e}")
    
    def _should_close_position(self, symbol: str, position, current_price: float) -> bool:
        """Determine if a position should be closed"""
        
        entry_price = float(position.avg_entry_price)
        quantity = float(position.qty)
        side = position.side
        
        # Calculate current P&L percentage
        if side == 'long':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
        
        # Take profit at 4% (equivalent to 4x ATR for typical stock)
        if pnl_pct >= 0.04:
            logger.info(f"📈 Taking profit on {symbol}: {pnl_pct:.2%}")
            return True
        
        # Stop loss at -2.5% (equivalent to 2.5x ATR)
        if pnl_pct <= -0.025:
            logger.info(f"📉 Stop loss on {symbol}: {pnl_pct:.2%}")
            return True
        
        # Time-based exit (20 days)
        # Note: In production, would track entry dates properly
        # For now, use a simple heuristic
        
        return False
    
    def _get_trading_signals(self) -> List[Dict]:
        """Get trading signals from the model"""
        
        # In production, this would:
        # 1. Fetch latest market data
        # 2. Run feature engineering (technical indicators, LLM sentiment, etc.)
        # 3. Run trained models to get predictions
        # 4. Generate trading signals
        
        # For now, generate mock signals for demonstration
        if not self._should_generate_signals():
            return []
        
        mock_signals = [
            {
                'symbol': 'AAPL',
                'side': 'buy',
                'prob': 0.76,
                'prob_diff': 0.52,
                'quality': 0.97,
                'expected_return': 0.02,
                'current_price': self._get_current_price('AAPL')
            },
            {
                'symbol': 'MSFT',
                'side': 'sell',
                'prob': 0.70,
                'prob_diff': 0.40,
                'quality': 0.96,
                'expected_return': -0.015,
                'current_price': self._get_current_price('MSFT')
            }
        ]

        # Apply regime thresholds and quality gate
        filtered_signals = []
        for s in mock_signals:
            thr_key = f"{s['side']}_prob"
            threshold = self.regime_thresholds.get(thr_key, 0.5)
            if s['prob'] >= threshold and s.get('quality', 1.0) >= self.quality_gate:
                filtered_signals.append(s)

        logger.info(f"🎯 Generated {len(filtered_signals)} trading signals")
        return filtered_signals
    
    def _should_generate_signals(self) -> bool:
        """Check if we should generate new signals (not too frequently)"""
        
        # Only generate signals at market open and a few times during the day
        now = datetime.now()
        signal_times = [
            now.replace(hour=9, minute=35),   # 5 min after open
            now.replace(hour=11, minute=0),   # Mid morning
            now.replace(hour=14, minute=0),   # Early afternoon
        ]
        
        for signal_time in signal_times:
            if abs((now - signal_time).total_seconds()) < 300:  # Within 5 minutes
                return True
        
        return False
    
    def _execute_trades(self, signals: List[Dict]):
        """Execute trading signals"""
        
        logger.info(f"🔄 Executing {len(signals)} trades...")
        
        for signal in signals:
            try:
                symbol = signal['symbol']
                side = signal['side']
                prob_diff = signal['prob_diff']
                current_price = signal['current_price']

                if not current_price:
                    logger.warning(f"No price available for {symbol}")
                    continue

                # Calculate position size
                position_size = self._calculate_position_size(symbol, prob_diff, current_price)
                
                if position_size == 0:
                    logger.info(f"Skipping {symbol}: position size too small")
                    continue
                
                # Check if we already have a position in this symbol
                if symbol in self.positions:
                    logger.info(f"Already have position in {symbol}, skipping")
                    continue
                
                # Submit order
                success = self._submit_order(symbol, side, position_size)
                
                if success:
                    logger.info(f"✅ Order submitted: {side} {position_size} shares of {symbol}")
                else:
                    logger.warning(f"❌ Failed to submit order for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error executing trade for {signal['symbol']}: {e}")
    
    def _calculate_position_size(self, symbol: str, prob_diff: float, price: float) -> int:
        """Calculate appropriate position size"""

        # Base position size as percentage of portfolio
        base_position_pct = 0.02  # 2% of portfolio per position

        # Adjust by calibrated probability difference
        prob_adjusted_pct = base_position_pct * prob_diff

        # Calculate dollar amount
        position_value = self.portfolio_value * prob_adjusted_pct
        
        # Convert to shares (must be whole number)
        shares = int(position_value / price)
        
        # Check maximum position size constraint
        max_position_value = self.portfolio_value * self.max_per_name
        max_shares = int(max_position_value / price)
        
        return min(shares, max_shares)
    
    def _submit_order(self, symbol: str, side: str, quantity: int) -> bool:
        """Submit order to Alpaca"""
        
        if self.api:
            try:
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side=side,
                    type='market',
                    time_in_force='day'
                )
                
                self.orders[order.id] = order
                logger.info(f"📝 Order submitted: {order.id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to submit order: {e}")
                return False
        else:
            # Simulation mode
            logger.info(f"📝 [SIMULATION] {side} {quantity} shares of {symbol}")
            return True
    
    def _close_position(self, symbol: str, position):
        """Close an existing position"""
        
        if self.api:
            try:
                # Submit closing order
                order = self.api.close_position(symbol)
                logger.info(f"📝 Position closed: {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to close position {symbol}: {e}")
        else:
            # Simulation mode
            logger.info(f"📝 [SIMULATION] Closed position: {symbol}")
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        
        if self.api:
            try:
                # Get latest trade
                trade = self.api.get_latest_trade(symbol)
                return float(trade.price)
                
            except Exception as e:
                logger.warning(f"Could not get price for {symbol}: {e}")
                return None
        else:
            # Mock prices for simulation
            mock_prices = {
                'AAPL': 150.0,
                'MSFT': 300.0,
                'GOOGL': 2500.0,
                'AMZN': 3000.0,
                'TSLA': 800.0
            }
            return mock_prices.get(symbol, 100.0)
    
    def _risk_management_checks(self):
        """Perform risk management checks"""
        
        # Check daily loss limit
        if self.account:
            daily_pnl = float(self.account.equity) - float(self.account.last_equity)
            daily_pnl_pct = daily_pnl / float(self.account.last_equity)
            
            if daily_pnl_pct < -self.daily_loss_limit:
                logger.warning(f"🚨 Daily loss limit exceeded: {daily_pnl_pct:.2%}")
                # In production, might stop trading for the day
        
        # Check gross exposure
        total_exposure = 0
        for position in self.positions.values():
            if hasattr(position, 'market_value'):
                total_exposure += abs(float(position.market_value))
        
        gross_exposure = total_exposure / self.portfolio_value
        
        if gross_exposure > self.max_gross:
            logger.warning(f"🚨 Gross exposure too high: {gross_exposure:.2%}")
            # In production, might reduce positions
    
    def _log_trading_status(self):
        """Log current trading status"""
        
        num_positions = len(self.positions)
        logger.info(f"📊 Portfolio: ${self.portfolio_value:,.2f}, Positions: {num_positions}")
        
        if self.positions:
            total_pnl = 0
            for symbol, position in self.positions.items():
                if hasattr(position, 'unrealized_pl'):
                    pnl = float(position.unrealized_pl)
                    total_pnl += pnl
                    logger.info(f"   {symbol}: {position.side} {position.qty} @ ${position.avg_entry_price}, P&L: ${pnl:.2f}")
            
            logger.info(f"   Total Unrealized P&L: ${total_pnl:.2f}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("📪 Received shutdown signal, stopping trading...")
        self.is_trading = False
    
    def _cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up...")
        
        # Cancel any pending orders
        if self.api:
            try:
                orders = self.api.list_orders(status='open')
                for order in orders:
                    self.api.cancel_order(order.id)
                    logger.info(f"Cancelled order: {order.id}")
            except Exception as e:
                logger.error(f"Error cancelling orders: {e}")
        
        logger.info("✅ Paper trader stopped")

def main():
    """Test the paper trader"""
    
    print("📈 Testing Paper Trader")
    print("=" * 50)
    
    if not ALPACA_AVAILABLE:
        print("⚠️ Alpaca Trade API not available")
        print("Install with: pip install alpaca-trade-api")
        print("Running in simulation mode...")
    
    # Create paper trader
    trader = PaperTrader(
        model_dir="artifacts/models/best",
        max_gross=0.6,
        max_per_name=0.08
    )
    
    print("📝 Note: Set up Alpaca API credentials:")
    print("   - Get paper trading account at alpaca.markets")
    print("   - Set API_KEY and SECRET_KEY environment variables")
    print("   - Or pass credentials to PaperTrader constructor")
    
    # Start trading (will run until interrupted)
    trader.start_trading()

if __name__ == "__main__":
    main()