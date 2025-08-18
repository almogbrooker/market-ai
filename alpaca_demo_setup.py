#!/usr/bin/env python3
"""
ALPACA PAPER TRADING DEMO SETUP
Get real Alpaca UI access to see bot trading activity
"""

import os
from pathlib import Path

def create_alpaca_demo():
    """Set up Alpaca paper trading demo with UI access"""
    
    print("🚀 ALPACA PAPER TRADING DEMO SETUP")
    print("=" * 60)
    print()
    
    print("📝 STEP 1: CREATE ALPACA ACCOUNT")
    print("   1. Go to: https://alpaca.markets")
    print("   2. Click 'Sign Up' (FREE)")
    print("   3. Complete registration")
    print("   4. Verify email")
    print()
    
    print("🔑 STEP 2: GET PAPER TRADING API KEYS")
    print("   1. Login to your Alpaca account")
    print("   2. Go to: https://app.alpaca.markets/paper/dashboard")
    print("   3. Click 'Generate API Keys' or 'View API Keys'")
    print("   4. Create new paper trading keys")
    print("   5. Copy your API Key and Secret Key")
    print()
    
    print("⚙️ STEP 3: CONFIGURE BOT")
    print("   Option A - Environment Variables:")
    print("   export ALPACA_API_KEY='your_paper_api_key_here'")
    print("   export ALPACA_SECRET_KEY='your_paper_secret_key_here'")
    print()
    print("   Option B - .env File:")
    env_content = """ALPACA_API_KEY=your_paper_api_key_here
ALPACA_SECRET_KEY=your_paper_secret_key_here"""
    
    env_file = Path(".env")
    with open(env_file, "w") as f:
        f.write(env_content)
    
    print(f"   Created .env file: {env_file.absolute()}")
    print("   Edit this file and add your real API keys")
    print()
    
    print("🎯 STEP 4: ACCESS ALPACA UI")
    print("   • Paper Trading Dashboard: https://paper-app.alpaca.markets")
    print("   • View your portfolio, positions, orders, and performance")
    print("   • See all bot trades in real-time")
    print("   • Track performance vs benchmarks")
    print()
    
    print("🤖 STEP 5: RUN THE BOT")
    print("   python final_production_bot.py")
    print()
    print("   The bot will:")
    print("   ✅ Connect to your Alpaca paper account")
    print("   ✅ Make real paper trades (no real money)")
    print("   ✅ Show all activity in Alpaca UI")
    print("   ✅ Track performance vs QQQ")
    print()
    
    print("📊 WHAT YOU'LL SEE IN ALPACA UI:")
    print("   • Portfolio value changes")
    print("   • Individual stock positions")
    print("   • Order history with timestamps")
    print("   • Performance charts vs S&P 500")
    print("   • Real-time P&L tracking")
    print()
    
    print("🎮 DEMO MODE (NO API KEYS NEEDED)")
    print("   If you don't have API keys yet, the bot will run in demo mode:")
    print("   • Simulates realistic trading")
    print("   • Shows detailed trade decisions")
    print("   • Tracks portfolio performance")
    print("   • No actual orders placed")
    print()
    
    print("✅ READY TO TRADE!")
    print("=" * 60)

def create_sample_run():
    """Create sample run script"""
    
    run_script = """#!/usr/bin/env python3
# Sample run script for Alpaca bot

import os
from final_production_bot import FinalProductionBot

# Option 1: Use environment variables
api_key = os.getenv('ALPACA_API_KEY')
secret_key = os.getenv('ALPACA_SECRET_KEY')

# Option 2: Direct input (replace with your keys)
# api_key = "your_paper_api_key_here"
# secret_key = "your_paper_secret_key_here"

# Initialize and run bot
bot = FinalProductionBot(api_key=api_key, secret_key=secret_key, paper=True)
bot.execute_final_trades()
"""
    
    with open("run_alpaca_bot.py", "w") as f:
        f.write(run_script)
    
    print(f"📄 Created run_alpaca_bot.py - Edit this file with your API keys")

def main():
    create_alpaca_demo()
    create_sample_run()
    
    print("\n🚀 QUICK START:")
    print("1. Get API keys from https://alpaca.markets")
    print("2. Edit .env file with your keys")
    print("3. Run: python final_production_bot.py")
    print("4. View trades at: https://paper-app.alpaca.markets")

if __name__ == "__main__":
    main()