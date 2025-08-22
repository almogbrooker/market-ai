#!/usr/bin/env python3
"""
FULLY AUTOMATED TRADING SYSTEM
Runs continuously with automatic retraining and decision logging
"""

import os
import time
import schedule
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
import json

from jsonschema import validate
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
from pydantic import ValidationError as PydanticValidationError

from PRODUCTION.config.settings import Settings


def load_live_trading_config() -> dict:
    """Load and validate live trading configuration against its JSON schema."""
    config_path = Path("PRODUCTION/config/live_trading_config.json")
    schema_path = Path("PRODUCTION/config/live_trading_config.schema.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    with open(schema_path, "r") as f:
        schema = json.load(f)
    validate(instance=config, schema=schema)
    return config

class AutomatedTradingSystem:
    """Fully automated trading system with continuous operation"""
    
    def __init__(self):
        self.setup_environment()
        self.last_retrain_date = None
        self.running = True
        
    def setup_environment(self):
        """Load environment, validate configuration, and check setup"""
        try:
            self.settings = Settings()
            # ensure secrets are available to subprocesses without logging
            os.environ["ALPACA_API_KEY"] = self.settings.alpaca_api_key.get_secret_value()
            os.environ["ALPACA_API_SECRET"] = self.settings.alpaca_api_secret.get_secret_value()
            os.environ["PAPER_TRADING"] = "true" if self.settings.paper_trading else "false"
            os.environ["MAX_POSITION_SIZE"] = str(self.settings.max_position_size)
            os.environ["BASELINE_EXPOSURE"] = str(self.settings.baseline_exposure)
            os.environ["MAX_EXPOSURE"] = str(self.settings.max_exposure)
        except PydanticValidationError as e:
            raise RuntimeError(f"Settings validation error: {e}") from e

        try:
            self.live_config = load_live_trading_config()
        except JSONSchemaValidationError as e:
            raise RuntimeError(f"Live trading config validation error: {e.message}") from e

        print("🏛️ AUTOMATED TRADING SYSTEM STARTED")
        print(f"⏰ Time: {datetime.now()}")
        print(f"📊 Mode: {'Paper Trading' if self.settings.paper_trading else 'Live Trading'}")
        
    def is_trading_day(self):
        """Check if today is a trading day"""
        return datetime.now().weekday() < 5  # Mon-Fri
    
    def is_market_hours(self):
        """Check if market is open (9:30 AM - 4:00 PM EST)"""
        now = datetime.now()
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now <= market_close
    
    def should_retrain_model(self):
        """Check if model needs retraining (daily)"""
        today = datetime.now().date()
        
        # Check if already retrained today
        if self.last_retrain_date == today:
            return False
            
        # Check if it's a trading day
        if not self.is_trading_day():
            return False
            
        # Check if it's before market open (retrain at 6 AM)
        now = datetime.now()
        retrain_time = now.replace(hour=6, minute=0, second=0, microsecond=0)
        
        return now >= retrain_time and self.last_retrain_date != today
    
    def run_daily_retrain(self):
        """Execute daily model retraining"""
        print(f"\n🔄 DAILY RETRAINING - {datetime.now()}")
        print("=" * 50)
        
        try:
            result = subprocess.run([
                sys.executable, "PRODUCTION/bots/daily_retrain.py"
            ], capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            if result.returncode == 0:
                print("✅ Model retraining completed successfully")
                self.last_retrain_date = datetime.now().date()
                return True
            else:
                print(f"❌ Retraining failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Retraining timed out after 30 minutes")
            return False
        except Exception as e:
            print(f"❌ Retraining error: {e}")
            return False
    
    def run_system_audit(self):
        """Run system health check"""
        print(f"\n🏛️ SYSTEM AUDIT - {datetime.now()}")
        print("=" * 50)
        
        try:
            result = subprocess.run([
                sys.executable, "PRODUCTION/tools/institutional_audit_system.py"
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                # Check audit results
                audit_path = Path("institutional_audit_report.json")
                if audit_path.exists():
                    with open(audit_path, 'r') as f:
                        audit_data = json.load(f)
                    
                    success_rate = audit_data.get("success_rate", 0)
                    print(f"✅ System audit: {success_rate:.1%} success rate")
                    
                    if success_rate >= 0.8:
                        return True
                    else:
                        print("⚠️ Audit success rate below 80% - system needs attention")
                        return False
                else:
                    print("⚠️ Audit completed but no report found")
                    return False
            else:
                print(f"❌ Audit failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Audit error: {e}")
            return False
    
    def run_trading_cycle(self):
        """Execute trading cycle with liquidation and new positions"""
        print(f"\n🎯 TRADING CYCLE - {datetime.now()}")
        print("=" * 50)
        
        try:
            # Step 1: Liquidate all positions
            print("🔥 Step 1: Liquidating all positions...")
            liquidation_result = subprocess.run([
                sys.executable, "PRODUCTION/bots/liquidation_bot.py"
            ], capture_output=True, text=True, timeout=600)
            
            if liquidation_result.returncode != 0:
                print(f"❌ Liquidation failed: {liquidation_result.stderr}")
                return False
            
            print("✅ Liquidation completed")
            
            # Step 2: Wait a moment for orders to settle
            print("⏳ Waiting 30 seconds for orders to settle...")
            time.sleep(30)
            
            # Step 3: Generate new signals and place orders
            print("🎯 Step 2: Generating new signals and placing orders...")
            trading_result = subprocess.run([
                sys.executable, "PRODUCTION/bots/alpaca_trader.py"
            ], capture_output=True, text=True, timeout=600)
            
            if trading_result.returncode == 0:
                print("✅ Trading cycle completed successfully")
                return True
            else:
                print(f"❌ Trading failed: {trading_result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Trading cycle error: {e}")
            return False
    
    def check_positions_status(self):
        """Quick check of current positions"""
        try:
            from alpaca.trading.client import TradingClient
            
            client = TradingClient(
                api_key=self.settings.alpaca_api_key.get_secret_value(),
                secret_key=self.settings.alpaca_api_secret.get_secret_value(),
                paper=self.settings.paper_trading,
            )
            
            account = client.get_account()
            positions = client.get_all_positions()
            
            print(f"💰 Portfolio: ${float(account.portfolio_value):,.2f}")
            print(f"📊 Positions: {len(positions)}")
            
            if positions:
                total_value = 0
                for pos in positions[:5]:  # Show top 5
                    value = float(pos.market_value)
                    total_value += value
                    print(f"  {pos.symbol}: ${value:,.0f}")
                if len(positions) > 5:
                    print(f"  ... and {len(positions)-5} more")
                    
        except Exception as e:
            print(f"❌ Position check failed: {e}")
    
    def daily_morning_routine(self):
        """Complete morning routine before trading"""
        if not self.is_trading_day():
            print("📅 Non-trading day - skipping morning routine")
            return True
            
        print(f"\n🌅 MORNING ROUTINE - {datetime.now()}")
        print("=" * 60)
        
        success = True
        
        # 1. Retrain model if needed
        if self.should_retrain_model():
            if not self.run_daily_retrain():
                print("❌ Model retraining failed - continuing with existing model")
                # Don't fail completely, use existing model
        else:
            print("ℹ️ Model retraining not needed today")
        
        # 2. Run system audit
        if not self.run_system_audit():
            print("⚠️ System audit failed - review needed")
            success = False
        
        print(f"{'✅' if success else '⚠️'} Morning routine completed")
        return success
    
    def trading_session(self):
        """Execute trading session"""
        if not self.is_trading_day():
            print("📅 Non-trading day - no trading")
            return
            
        if not self.is_market_hours():
            print("🕐 Outside market hours - no trading")
            return
            
        # Check if we already traded today
        trading_time = datetime.now().replace(hour=9, minute=45, second=0, microsecond=0)
        if datetime.now() < trading_time:
            print("🕘 Waiting for 9:45 AM trading time")
            return
            
        # Execute trading cycle
        success = self.run_trading_cycle()
        
        if success:
            print("✅ Trading session completed successfully")
        else:
            print("❌ Trading session failed")
    
    def evening_monitoring(self):
        """End of day monitoring"""
        if not self.is_trading_day():
            return
            
        print(f"\n🌙 EVENING MONITORING - {datetime.now()}")
        print("=" * 50)
        
        # Check positions status
        self.check_positions_status()
        
        # Run drift monitoring
        try:
            result = subprocess.run([
                sys.executable, "PRODUCTION/tools/drift_monitoring_system.py"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ Drift monitoring completed")
            else:
                print("⚠️ Drift monitoring had issues")
                
        except Exception as e:
            print(f"❌ Monitoring error: {e}")
    
    def run_continuous(self):
        """Main continuous operation"""
        print("🚀 STARTING CONTINUOUS AUTOMATED TRADING")
        print("⏹️ Press Ctrl+C to stop")
        print("=" * 60)
        
        # Set up schedule
        schedule.every().day.at("06:00").do(self.daily_morning_routine)
        schedule.every().day.at("09:45").do(self.trading_session)
        schedule.every().day.at("16:15").do(self.evening_monitoring)
        
        # Show schedule
        print("📅 AUTOMATED SCHEDULE:")
        print("   06:00 - Daily model retraining + system audit")
        print("   09:45 - Trading session (liquidate + new positions)")
        print("   16:15 - Evening monitoring and drift check")
        print("   Real-time position monitoring every 5 minutes")
        print()
        
        last_status_check = datetime.now()
        
        try:
            while self.running:
                # Run scheduled tasks
                schedule.run_pending()
                
                # Status check every 5 minutes
                if datetime.now() - last_status_check > timedelta(minutes=5):
                    if self.is_trading_day() and self.is_market_hours():
                        print(f"📊 {datetime.now().strftime('%H:%M')} - ", end="")
                        self.check_positions_status()
                    last_status_check = datetime.now()
                
                # Sleep for 1 minute
                time.sleep(60)
                
        except KeyboardInterrupt:
            print("\n⏹️ Automated trading stopped by user")
            self.running = False
        except Exception as e:
            print(f"\n❌ System error: {e}")
            print("🔄 Restarting in 60 seconds...")
            time.sleep(60)

def main():
    """Main function"""
    # Check if system is already running
    if Path("trading_system.lock").exists():
        print("❌ Trading system already running")
        print("Remove trading_system.lock file if this is incorrect")
        return
    
    # Create lock file
    with open("trading_system.lock", 'w') as f:
        f.write(f"Started: {datetime.now()}\nPID: {os.getpid()}")
    
    try:
        system = AutomatedTradingSystem()
        system.run_continuous()
    finally:
        # Remove lock file
        if Path("trading_system.lock").exists():
            Path("trading_system.lock").unlink()

if __name__ == "__main__":
    main()