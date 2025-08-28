#!/usr/bin/env python3
import alpaca_trade_api as tradeapi
import os

# Load credentials from .env
with open('/home/almog/market-ai/.env') as f:
    for line in f:
        if '=' in line and not line.startswith('#'):
            key, value = line.strip().split('=', 1)
            os.environ[key] = value

api = tradeapi.REST(
    os.getenv('ALPACA_API_KEY'),
    os.getenv('ALPACA_SECRET_KEY'), 
    'https://paper-api.alpaca.markets',
    api_version='v2'
)

print("🔍 ALPACA ACCOUNT STATUS")
print("=" * 40)

# Check current positions
positions = api.list_positions()
print(f'📊 Current positions: {len(positions)}')
if positions:
    for pos in positions:
        pnl = float(pos.unrealized_pl)
        pnl_pct = float(pos.unrealized_plpc) * 100
        print(f'  {pos.symbol}: {pos.qty} shares, ${float(pos.market_value):,.2f} (P&L: ${pnl:+.2f}, {pnl_pct:+.1f}%)')
else:
    print('  No current positions')

# Check account
account = api.get_account()
print(f'\n💰 Account Summary:')
print(f'  Equity: ${float(account.equity):,.2f}')
print(f'  Cash: ${float(account.cash):,.2f}')
print(f'  Buying power: ${float(account.buying_power):,.2f}')

# Check recent orders
orders = api.list_orders(status='all', limit=10)
print(f'\n📋 Recent orders: {len(orders)}')
if orders:
    for order in orders[:5]:  # Show last 5
        qty_str = f"{order.qty} shares" if order.qty else f"${float(order.notional):,.0f}"
        date_str = str(order.submitted_at)[:10]
        print(f'  {date_str} {order.symbol}: {order.side.upper()} {qty_str} ({order.status})')
else:
    print('  No recent orders')

print(f'\n✅ Account is ready for trading!')