#!/bin/bash

# Setup script for Alpaca API environment
echo "🚀 Setting up Alpaca API environment for professional trading..."

# Check if environment variables are set
if [ -z "$ALPACA_API_KEY" ] || [ -z "$ALPACA_SECRET_KEY" ]; then
    echo "❌ Alpaca credentials not found in environment"
    echo "Please set the following environment variables:"
    echo "export ALPACA_API_KEY='your_api_key_here'"
    echo "export ALPACA_SECRET_KEY='your_secret_key_here'"
    echo ""
    echo "You can get these from:"
    echo "Paper Trading: https://app.alpaca.markets/paper/dashboard/overview"
    echo "Live Trading: https://app.alpaca.markets/brokerage/dashboard/overview"
    echo ""
    echo "Add these to your ~/.bashrc or ~/.zshrc for persistence:"
    echo "echo 'export ALPACA_API_KEY=\"your_api_key\"' >> ~/.bashrc"
    echo "echo 'export ALPACA_SECRET_KEY=\"your_secret_key\"' >> ~/.bashrc"
    echo "source ~/.bashrc"
    exit 1
fi

# Install required packages
echo "📦 Installing required packages..."
pip install alpaca-trade-api pandas numpy

# Verify credentials work
echo "🔍 Testing Alpaca API connection..."
python3 << EOF
import os
import alpaca_trade_api as tradeapi

try:
    api = tradeapi.REST(
        os.getenv('ALPACA_API_KEY'),
        os.getenv('ALPACA_SECRET_KEY'),
        'https://paper-api.alpaca.markets',
        api_version='v2'
    )
    
    account = api.get_account()
    print(f"✅ Connection successful!")
    print(f"   Account ID: {account.id}")
    print(f"   Equity: \${float(account.equity):,.2f}")
    print(f"   Buying Power: \${float(account.buying_power):,.2f}")
    print(f"   Status: {account.status}")
    
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("Please check your API credentials")
    exit(1)
EOF

echo ""
echo "🎉 Alpaca environment setup complete!"
echo "You can now run the trading bot with:"
echo "python live_trading_bot.py --paper  # Paper trading mode"
echo "python live_trading_bot.py --live   # Live trading mode (BE CAREFUL!)"