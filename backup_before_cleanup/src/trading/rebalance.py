#!/usr/bin/env python3
"""
Target-Weights Rebalancer
Converts target weights [-1,1] to integer share diffs vs current positions
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RebalanceOrder:
    """Represents a rebalancing order"""
    symbol: str
    current_shares: int
    target_shares: int
    share_diff: int
    side: str  # 'buy' or 'sell'
    current_weight: float
    target_weight: float
    priority: float  # Higher = more important to execute

class TargetWeightsRebalancer:
    """
    Rebalances portfolio to target weights with safety constraints
    """
    
    def __init__(self, max_gross_exposure: float = 0.95, min_trade_value: float = 100.0):
        """
        Args:
            max_gross_exposure: Maximum gross exposure as fraction of equity
            min_trade_value: Minimum trade value to avoid tiny trades
        """
        self.max_gross_exposure = max_gross_exposure
        self.min_trade_value = min_trade_value
        
    def calculate_rebalance_orders(
        self,
        target_weights: Dict[str, float],
        current_positions: Dict[str, int],
        current_prices: Dict[str, float],
        portfolio_value: float,
        open_orders: Optional[Dict] = None
    ) -> List[RebalanceOrder]:
        """
        Calculate rebalancing orders to achieve target weights
        
        Args:
            target_weights: Dict of {symbol: weight} where weight ∈ [-1, 1]
            current_positions: Dict of {symbol: shares} (negative for shorts)
            current_prices: Dict of {symbol: price}
            portfolio_value: Current portfolio value
            open_orders: Optional dict of pending orders to account for
            
        Returns:
            List of RebalanceOrder objects sorted by priority
        """
        orders = []
        
        # Calculate current weights
        current_weights = self._calculate_current_weights(
            current_positions, current_prices, portfolio_value
        )
        
        # Account for pending orders
        if open_orders:
            target_weights, current_positions = self._adjust_for_pending_orders(
                target_weights, current_positions, open_orders, current_prices
            )
        
        # Calculate target portfolio value accounting for max exposure
        available_capital = portfolio_value * self.max_gross_exposure
        
        # Calculate gross exposure of targets
        target_gross = sum(abs(weight) for weight in target_weights.values())
        if target_gross > self.max_gross_exposure:
            # Scale down all weights proportionally
            scale_factor = self.max_gross_exposure / target_gross
            target_weights = {k: v * scale_factor for k, v in target_weights.items()}
            logger.info(f"📊 Scaled target weights by {scale_factor:.3f} to fit exposure limit")
        
        # Calculate target positions for each symbol
        for symbol, target_weight in target_weights.items():
            if symbol not in current_prices:
                logger.warning(f"⚠️ No price data for {symbol}, skipping")
                continue
                
            price = current_prices[symbol]
            current_shares = current_positions.get(symbol, 0)
            current_weight = current_weights.get(symbol, 0.0)
            
            # Calculate target shares
            target_value = target_weight * available_capital
            target_shares = int(target_value / price) if price > 0 else 0
            
            # Calculate share difference
            share_diff = target_shares - current_shares
            
            # Skip if change is too small
            trade_value = abs(share_diff * price)
            if trade_value < self.min_trade_value:
                continue
            
            # Determine side
            side = 'buy' if share_diff > 0 else 'sell'
            
            # Calculate priority (larger weight changes = higher priority)
            weight_change = abs(target_weight - current_weight)
            priority = weight_change * trade_value
            
            order = RebalanceOrder(
                symbol=symbol,
                current_shares=current_shares,
                target_shares=target_shares,
                share_diff=share_diff,
                side=side,
                current_weight=current_weight,
                target_weight=target_weight,
                priority=priority
            )
            
            orders.append(order)
        
        # Handle positions to close (not in target weights)
        for symbol, current_shares in current_positions.items():
            if symbol not in target_weights and current_shares != 0:
                if symbol not in current_prices:
                    logger.warning(f"⚠️ No price data for {symbol} to close, skipping")
                    continue
                    
                price = current_prices[symbol]
                current_weight = current_weights.get(symbol, 0.0)
                trade_value = abs(current_shares * price)
                
                if trade_value >= self.min_trade_value:
                    order = RebalanceOrder(
                        symbol=symbol,
                        current_shares=current_shares,
                        target_shares=0,
                        share_diff=-current_shares,
                        side='sell' if current_shares > 0 else 'buy',
                        current_weight=current_weight,
                        target_weight=0.0,
                        priority=trade_value  # High priority to close positions
                    )
                    orders.append(order)
        
        # Sort by priority (descending)
        orders.sort(key=lambda x: x.priority, reverse=True)
        
        logger.info(f"📋 Generated {len(orders)} rebalancing orders")
        return orders
    
    def _calculate_current_weights(
        self,
        positions: Dict[str, int],
        prices: Dict[str, float],
        portfolio_value: float
    ) -> Dict[str, float]:
        """Calculate current portfolio weights"""
        weights = {}
        
        for symbol, shares in positions.items():
            if symbol in prices and portfolio_value > 0:
                position_value = shares * prices[symbol]
                weights[symbol] = position_value / portfolio_value
            else:
                weights[symbol] = 0.0
                
        return weights
    
    def _adjust_for_pending_orders(
        self,
        target_weights: Dict[str, float],
        current_positions: Dict[str, int],
        open_orders: Dict,
        prices: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, int]]:
        """Adjust calculations to account for pending orders"""
        adjusted_positions = current_positions.copy()
        
        for order_id, order_info in open_orders.items():
            symbol = order_info['symbol']
            qty = order_info['qty']
            side = order_info['side']
            
            # Assume pending orders will execute
            if side == 'buy':
                adjusted_positions[symbol] = adjusted_positions.get(symbol, 0) + abs(qty)
            else:  # sell
                adjusted_positions[symbol] = adjusted_positions.get(symbol, 0) - abs(qty)
        
        return target_weights, adjusted_positions
    
    def cancel_stale_orders(self, api, open_orders: Dict, max_age_minutes: int = 30):
        """Cancel orders older than max_age_minutes"""
        from datetime import datetime, timedelta
        
        cancelled_orders = []
        cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
        
        for client_order_id, order_info in open_orders.items():
            try:
                order_time = datetime.fromisoformat(order_info['timestamp'].replace('Z', '+00:00'))
                
                if order_time < cutoff_time:
                    api.cancel_order(order_info['order_id'])
                    cancelled_orders.append(client_order_id)
                    logger.info(f"🚫 Cancelled stale order: {order_info['symbol']} (age: {datetime.now() - order_time})")
                    
            except Exception as e:
                logger.error(f"❌ Error cancelling order {client_order_id}: {e}")
        
        return cancelled_orders
    
    def validate_rebalance_orders(
        self,
        orders: List[RebalanceOrder],
        portfolio_value: float,
        max_single_trade_pct: float = 0.2
    ) -> List[RebalanceOrder]:
        """Validate and filter rebalancing orders for safety"""
        validated_orders = []
        
        for order in orders:
            trade_value = abs(order.share_diff) * \
                         (abs(order.current_weight - order.target_weight) * portfolio_value) / \
                         abs(order.share_diff) if order.share_diff != 0 else 0
            
            trade_pct = trade_value / portfolio_value if portfolio_value > 0 else 0
            
            # Skip trades that are too large
            if trade_pct > max_single_trade_pct:
                logger.warning(f"⚠️ Skipping large trade: {order.symbol} ({trade_pct:.1%} of portfolio)")
                continue
            
            # Skip if target weight is unreasonable
            if abs(order.target_weight) > 0.5:  # No single position > 50%
                logger.warning(f"⚠️ Skipping extreme weight: {order.symbol} ({order.target_weight:.1%})")
                continue
            
            validated_orders.append(order)
        
        logger.info(f"✅ Validated {len(validated_orders)}/{len(orders)} rebalancing orders")
        return validated_orders

def demo_rebalancer():
    """Demo the rebalancer functionality"""
    print("🧪 Target-Weights Rebalancer Demo")
    print("=" * 50)
    
    # Sample data
    target_weights = {
        'AAPL': 0.25,   # 25% long
        'MSFT': 0.20,   # 20% long  
        'GOOGL': 0.15,  # 15% long
        'TSLA': -0.10,  # 10% short
        'META': 0.05    # 5% long
    }
    
    current_positions = {
        'AAPL': 100,    # Currently own 100 shares
        'MSFT': 0,      # No position
        'GOOGL': 50,    # Own 50 shares  
        'TSLA': 25,     # Own 25 shares (want to short)
        'NVDA': 75      # Position to close (not in targets)
    }
    
    current_prices = {
        'AAPL': 175.0,
        'MSFT': 350.0,
        'GOOGL': 140.0,
        'TSLA': 250.0,
        'META': 300.0,
        'NVDA': 900.0
    }
    
    portfolio_value = 100000.0  # $100K portfolio
    
    # Initialize rebalancer
    rebalancer = TargetWeightsRebalancer(max_gross_exposure=0.95)
    
    # Calculate orders
    orders = rebalancer.calculate_rebalance_orders(
        target_weights=target_weights,
        current_positions=current_positions,
        current_prices=current_prices,
        portfolio_value=portfolio_value
    )
    
    # Display results
    print(f"\\n📊 Portfolio Value: ${portfolio_value:,}")
    print(f"📋 Generated {len(orders)} rebalancing orders:")
    print()
    
    for i, order in enumerate(orders, 1):
        print(f"{i}. {order.symbol}:")
        print(f"   Current: {order.current_shares} shares ({order.current_weight:+.1%})")
        print(f"   Target:  {order.target_shares} shares ({order.target_weight:+.1%})")
        print(f"   Action:  {order.side.upper()} {abs(order.share_diff)} shares")
        print(f"   Priority: {order.priority:.0f}")
        print()

if __name__ == "__main__":
    demo_rebalancer()