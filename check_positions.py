#!/usr/bin/env python3
"""
Quick position checker
"""

import os
from pathlib import Path

# Load environment
env_file = Path("PRODUCTION/config/alpaca.env")
if env_file.exists():
    with open(env_file, 'r') as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

try:
    from alpaca.trading.client import TradingClient
    
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_API_SECRET')
    
    client = TradingClient(
        api_key=api_key,
        secret_key=api_secret,
        paper=True
    )
    
    # Get account info
    account = client.get_account()
    print(f"💰 Account Value: ${float(account.portfolio_value):,.2f}")
    print(f"💵 Buying Power: ${float(account.buying_power):,.2f}")
    
    # Get positions
    positions = client.get_all_positions()
    print(f"\n📊 Current Positions: {len(positions)}")
    
    if positions:
        total_value = 0
        for pos in positions:
            value = float(pos.market_value)
            pnl = float(pos.unrealized_pl)
            total_value += value
            print(f"  {pos.symbol}: {pos.qty} shares = ${value:,.2f} (P&L: ${pnl:+,.2f})")
        
        print(f"\n💰 Total Position Value: ${total_value:,.2f}")
    else:
        print("✅ No positions - all liquidated!")
    
    # Check recent orders
    orders = client.get_orders()
    recent_orders = [o for o in orders if o.status in ['filled', 'new', 'partially_filled']]
    
    print(f"\n📋 Recent Orders: {len(recent_orders)}")
    for order in recent_orders[:10]:  # Last 10 orders
        print(f"  {order.symbol} {order.side} {order.qty} - {order.status}")

except Exception as e:
    print(f"Error: {e}")