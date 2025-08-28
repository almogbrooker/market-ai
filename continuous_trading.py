#!/usr/bin/env python3
"""
CONTINUOUS TRADING DAEMON
========================
Runs the trading bot continuously during market hours with proper scheduling
"""

import time
import schedule
import subprocess
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import logging
import signal
import json

class TradingDaemon:
    """Continuous trading daemon with market hours awareness"""
    
    def __init__(self):
        self.setup_logging()
        self.running = True
        self.bot_process = None
        
        # Trading schedule (EST/EDT)
        self.market_open = "09:30"
        self.market_close = "16:00"
        
        print("🤖 CONTINUOUS TRADING DAEMON")
        print("=" * 40)
        print(f"📅 Market hours: {self.market_open} - {self.market_close} EST")
        print("🔄 Bot will run every 5 minutes during market hours")
        
    def setup_logging(self):
        """Setup logging"""
        log_dir = Path("../artifacts/logs/daemon")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"daemon_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def is_market_hours(self):
        """Check if market is currently open"""
        import pytz
        
        # Get current time in EST/EDT
        utc_now = datetime.utcnow()
        est = pytz.timezone('US/Eastern')
        est_now = utc_now.replace(tzinfo=pytz.UTC).astimezone(est)
        
        # Skip weekends
        if est_now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
            
        # Check time in EST/EDT
        current_time = est_now.strftime("%H:%M")
        is_open = self.market_open <= current_time <= self.market_close
        
        print(f"🕐 EST time: {current_time}, Market open: {is_open}")
        return is_open
        
    def run_trading_bot(self):
        """Execute single trading bot run"""
        if not self.is_market_hours():
            print(f"🕐 Outside market hours ({datetime.now().strftime('%H:%M')}), skipping...")
            return
            
        print(f"\n🚀 Running trading bot at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Load environment variables from .env file
            env = os.environ.copy()
            env_file = "/home/almog/market-ai/.env"
            if os.path.exists(env_file):
                print(f"📋 Loading credentials from {env_file}")
                with open(env_file) as f:
                    for line in f:
                        if '=' in line and not line.startswith('#'):
                            key, value = line.strip().split('=', 1)
                            env[key] = value
                print(f"✅ Loaded ALPACA_API_KEY: {env.get('ALPACA_API_KEY', 'NOT FOUND')}")
            else:
                print(f"❌ .env file not found at {env_file}")
            
            # Run the trading bot with environment variables
            result = subprocess.run([
                sys.executable, 
                "src/live_trading_bot.py",
                "--paper"  # Keep in paper mode for safety
            ], 
            cwd="/home/almog/market-ai",
            capture_output=True, 
            text=True,
            timeout=300,  # 5 minute timeout
            env=env
            )
            
            if result.returncode == 0:
                print("✅ Trading bot completed successfully")
                self.logger.info("Trading bot run completed successfully")
            else:
                print(f"❌ Trading bot failed: {result.stderr}")
                self.logger.error(f"Trading bot failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("⏰ Trading bot timed out after 5 minutes")
            self.logger.warning("Trading bot execution timed out")
        except Exception as e:
            print(f"💥 Unexpected error: {e}")
            self.logger.error(f"Unexpected error in trading bot: {e}")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.running = False
        if self.bot_process:
            self.bot_process.terminate()
            
    def start_daemon(self):
        """Start the continuous trading daemon"""
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Schedule trading bot runs every 5 minutes during market hours
        schedule.every(5).minutes.do(self.run_trading_bot)
        
        print(f"🔄 Daemon started, will run trading bot every 5 minutes during market hours")
        print("Press Ctrl+C to stop gracefully")
        
        # Main loop
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            print("\n🛑 Keyboard interrupt received")
        finally:
            print("👋 Daemon shutting down...")
            self.logger.info("Trading daemon shut down")
            
    def run_once_now(self):
        """Run trading bot immediately (for testing)"""
        print("🧪 Running trading bot once immediately...")
        self.run_trading_bot()

def main():
    daemon = TradingDaemon()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        # Run once for testing
        daemon.run_once_now()
    else:
        # Start continuous daemon
        daemon.start_daemon()

if __name__ == "__main__":
    main()