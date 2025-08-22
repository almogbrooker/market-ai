#!/usr/bin/env python3
"""
LIVE TRADING SETUP
Setup script for Alpaca live trading with daily retraining
"""

import os
import sys
from pathlib import Path
import subprocess

def setup_environment():
    """Setup trading environment"""
    print("🏛️ LIVE TRADING SETUP")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    print(f"✅ Python version: {sys.version}")
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    dependencies = [
        "alpaca-py",
        "schedule",
        "yfinance"  # Backup data source
    ]
    
    for dep in dependencies:
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Installed {dep}")
            else:
                print(f"❌ Failed to install {dep}")
                return False
        except Exception as e:
            print(f"❌ Error installing {dep}: {e}")
            return False
    
    print("✅ All dependencies installed")
    return True

def setup_alpaca_credentials():
    """Setup Alpaca API credentials"""
    print("\n🔑 ALPACA API SETUP")
    print("-" * 30)
    
    # Check existing credentials
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_API_SECRET')
    
    if api_key and api_secret:
        print("✅ Alpaca credentials found in environment")
        return True
    
    print("Please get your Alpaca API credentials:")
    print("1. Go to https://app.alpaca.markets/")
    print("2. Create account / Log in")
    print("3. Go to 'API Keys' section")
    print("4. Generate new API key")
    
    print("\n📝 Enter your credentials:")
    api_key = input("Alpaca API Key: ").strip()
    api_secret = input("Alpaca API Secret: ").strip()
    
    if not api_key or not api_secret:
        print("❌ API credentials required")
        return False
    
    # Save to environment file
    env_file = Path("PRODUCTION/config/alpaca.env")
    env_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(env_file, 'w') as f:
        f.write(f"ALPACA_API_KEY={api_key}\n")
        f.write(f"ALPACA_API_SECRET={api_secret}\n")
        f.write(f"PAPER_TRADING=true\n")
    
    print(f"✅ Credentials saved to {env_file}")
    print("💡 To use in terminal: source PRODUCTION/config/alpaca.env")
    
    return True

def test_alpaca_connection():
    """Test Alpaca API connection"""
    print("\n🔌 Testing Alpaca connection...")
    
    try:
        # Source environment variables
        env_file = Path("PRODUCTION/config/alpaca.env")
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
        
        # Test connection
        from alpaca.trading.client import TradingClient
        
        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_API_SECRET')
        
        if not api_key or not api_secret:
            print("❌ API credentials not found")
            return False
        
        client = TradingClient(
            api_key=api_key,
            secret_key=api_secret,
            paper=True  # Always test with paper trading first
        )
        
        account = client.get_account()
        print(f"✅ Connected to Alpaca Paper Trading")
        print(f"   Account: {account.account_number}")
        print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def create_startup_scripts():
    """Create startup scripts"""
    print("\n📜 Creating startup scripts...")
    
    # Paper trading startup script
    paper_script = Path("PRODUCTION/start_paper_trading.sh")
    with open(paper_script, 'w') as f:
        f.write("""#!/bin/bash
# Start Paper Trading System
echo "🚀 Starting Paper Trading System"
cd "$(dirname "$0")/.."
source PRODUCTION/config/alpaca.env
export PAPER_TRADING=true
python PRODUCTION/bots/trading_scheduler.py
""")
    paper_script.chmod(0o755)
    
    # Live trading startup script
    live_script = Path("PRODUCTION/start_live_trading.sh")
    with open(live_script, 'w') as f:
        f.write("""#!/bin/bash
# Start Live Trading System
echo "💰 Starting LIVE Trading System"
cd "$(dirname "$0")/.."
source PRODUCTION/config/alpaca.env
export PAPER_TRADING=false
python PRODUCTION/bots/trading_scheduler.py
""")
    live_script.chmod(0o755)
    
    # Manual trading script
    manual_script = Path("PRODUCTION/run_manual_trade.sh")
    with open(manual_script, 'w') as f:
        f.write("""#!/bin/bash
# Manual Trading Execution
echo "🎯 Manual Trading Execution"
cd "$(dirname "$0")/.."
source PRODUCTION/config/alpaca.env
python PRODUCTION/bots/alpaca_trader.py
""")
    manual_script.chmod(0o755)
    
    print(f"✅ Created {paper_script}")
    print(f"✅ Created {live_script}")
    print(f"✅ Created {manual_script}")
    
    return True

def main():
    """Main setup function"""
    success = True
    
    # Setup steps
    if not setup_environment():
        success = False
    
    if not setup_alpaca_credentials():
        success = False
    
    if not test_alpaca_connection():
        success = False
    
    if not create_startup_scripts():
        success = False
    
    if success:
        print("\n🎉 SETUP COMPLETE!")
        print("=" * 60)
        print("📊 To start paper trading:")
        print("   ./PRODUCTION/start_paper_trading.sh")
        print("")
        print("💰 To start live trading:")
        print("   ./PRODUCTION/start_live_trading.sh")
        print("")
        print("🎯 Manual single trade:")
        print("   ./PRODUCTION/run_manual_trade.sh")
        print("")
        print("🔧 Manual retraining:")
        print("   python PRODUCTION/bots/daily_retrain.py")
        print("")
        print("📊 System audit:")
        print("   python PRODUCTION/tools/institutional_audit_system.py")
        print("=" * 60)
    else:
        print("\n❌ SETUP FAILED")
        print("Please fix the errors and run setup again")

if __name__ == "__main__":
    main()