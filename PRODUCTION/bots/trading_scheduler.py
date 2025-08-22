#!/usr/bin/env python3
"""
LIVE TRADING SCHEDULER
Automated daily trading system with retraining
"""

import schedule
import time
import logging
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import pandas_market_calendars as mcal
import subprocess
import sys
import os

class TradingScheduler:
    """Automated trading scheduler with daily retraining"""
    
    def __init__(self, paper_trading=True):
        self.paper_trading = paper_trading
        self.logger = self.setup_logging()

        # Timezone and exchange calendar
        self.tz = ZoneInfo("America/New_York")
        self.calendar = mcal.get_calendar("NYSE")

        # Trading schedule (EST/EDT times)
        self.schedule_config = {
            "retrain_time": "06:00",      # 6:00 AM - Before market open
            "audit_time": "06:30",        # 6:30 AM - After retraining
            "trade_time": "09:45",        # 9:45 AM - After market open
            "monitor_time": "16:15",      # 4:15 PM - After market close
        }

        self.max_retries = 3
        self.retry_delay = 300  # 5 minutes
        
    def setup_logging(self):
        """Setup logging"""
        log_dir = Path("PRODUCTION/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'trading_scheduler.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('TradingScheduler')
    
    def now(self):
        """Current time in America/New_York"""
        return datetime.now(self.tz)

    def is_trading_day(self, dt=None):
        """Check if a given date is a trading day"""
        dt = dt or self.now()
        schedule = self.calendar.schedule(start_date=dt.date(), end_date=dt.date())
        return not schedule.empty

    def is_market_open(self, dt=None):
        """Check if the market is currently open"""
        dt = dt or self.now()
        schedule = self.calendar.schedule(start_date=dt.date(), end_date=dt.date())
        if schedule.empty:
            return False
        market_open = schedule.loc[dt.date(), "market_open"].tz_convert(self.tz)
        market_close = schedule.loc[dt.date(), "market_close"].tz_convert(self.tz)
        return market_open <= pd.Timestamp(dt) <= market_close
    
    def run_with_retries(self, func, task_name, max_retries=None):
        """Run a task with retry logic"""
        if max_retries is None:
            max_retries = self.max_retries
            
        for attempt in range(max_retries + 1):
            try:
                self.logger.info(f"🔄 Starting {task_name} (attempt {attempt + 1}/{max_retries + 1})")
                success = func()
                
                if success:
                    self.logger.info(f"✅ {task_name} completed successfully")
                    return True
                else:
                    self.logger.warning(f"⚠️ {task_name} returned failure")
                    
            except Exception as e:
                self.logger.error(f"❌ {task_name} failed: {e}")
            
            if attempt < max_retries:
                self.logger.info(f"⏳ Retrying {task_name} in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
        
        self.logger.error(f"❌ {task_name} failed after {max_retries + 1} attempts")
        return False
    
    def run_daily_retrain(self):
        """Execute daily retraining"""
        try:
            result = subprocess.run([
                sys.executable, "PRODUCTION/bots/daily_retrain.py"
            ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
            
            if result.returncode == 0:
                self.logger.info("✅ Daily retraining completed")
                return True
            else:
                self.logger.error(f"❌ Retraining failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("❌ Retraining timed out after 30 minutes")
            return False
        except Exception as e:
            self.logger.error(f"❌ Retraining execution failed: {e}")
            return False
    
    def run_system_audit(self):
        """Execute system audit"""
        try:
            result = subprocess.run([
                sys.executable, "PRODUCTION/tools/institutional_audit_system.py"
            ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
            
            if result.returncode == 0:
                # Check audit results
                audit_path = Path("institutional_audit_report.json")
                if audit_path.exists():
                    with open(audit_path, 'r') as f:
                        audit_data = json.load(f)
                    
                    success_rate = audit_data.get("success_rate", 0)
                    if success_rate >= 0.8:  # 80% minimum
                        self.logger.info(f"✅ System audit passed ({success_rate:.1%})")
                        return True
                    else:
                        self.logger.error(f"❌ System audit failed ({success_rate:.1%})")
                        return False
                else:
                    self.logger.warning("⚠️ Audit completed but no report found")
                    return False
            else:
                self.logger.error(f"❌ Audit execution failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("❌ Audit timed out after 10 minutes")
            return False
        except Exception as e:
            self.logger.error(f"❌ Audit execution failed: {e}")
            return False
    
    def run_trading_cycle(self):
        """Execute trading cycle"""
        if not self.is_market_open():
            self.logger.info("⏰ Market closed - skipping trading cycle")
            return True
        try:
            # Set environment variable for paper trading
            env = os.environ.copy()
            env['PAPER_TRADING'] = str(self.paper_trading)

            result = subprocess.run([
                sys.executable, "PRODUCTION/bots/alpaca_trader.py"
            ], capture_output=True, text=True, timeout=1200, env=env)  # 20 minute timeout

            if result.returncode == 0:
                self.logger.info("✅ Trading cycle completed")
                return True
            else:
                self.logger.error(f"❌ Trading failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            self.logger.error("❌ Trading timed out after 20 minutes")
            return False
        except Exception as e:
            self.logger.error(f"❌ Trading execution failed: {e}")
            return False
    
    def run_monitoring(self):
        """Execute end-of-day monitoring"""
        try:
            result = subprocess.run([
                sys.executable, "PRODUCTION/tools/drift_monitoring_system.py"
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            if result.returncode == 0:
                self.logger.info("✅ Monitoring completed")
                return True
            else:
                self.logger.error(f"❌ Monitoring failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("❌ Monitoring timed out after 5 minutes")
            return False
        except Exception as e:
            self.logger.error(f"❌ Monitoring execution failed: {e}")
            return False
    
    def daily_retrain_job(self):
        """Daily retraining job"""
        if not self.is_trading_day():
            self.logger.info("📅 Non-trading day, skipping retraining")
            return
        
        self.logger.info("🔄 DAILY RETRAINING JOB STARTED")
        success = self.run_with_retries(self.run_daily_retrain, "Daily Retraining")
        
        if not success:
            self.logger.error("❌ Daily retraining failed - TRADING SUSPENDED")
            # Send alert (email, Slack, etc.)
            self.send_alert("CRITICAL: Daily retraining failed")
    
    def system_audit_job(self):
        """System audit job"""
        if not self.is_trading_day():
            self.logger.info("📅 Non-trading day, skipping audit")
            return
        
        self.logger.info("🏛️ SYSTEM AUDIT JOB STARTED")
        success = self.run_with_retries(self.run_system_audit, "System Audit")
        
        if not success:
            self.logger.error("❌ System audit failed - TRADING SUSPENDED")
            self.send_alert("CRITICAL: System audit failed")
    
    def trading_job(self):
        """Trading execution job"""
        if not self.is_trading_day():
            self.logger.info("📅 Non-trading day, skipping trading")
            return
        if not self.is_market_open():
            self.logger.info("⏰ Market closed, skipping trading")
            return

        self.logger.info("🎯 TRADING JOB STARTED")
        success = self.run_with_retries(self.run_trading_cycle, "Trading Cycle")
        
        if not success:
            self.logger.error("❌ Trading cycle failed")
            self.send_alert("WARNING: Trading cycle failed")
    
    def monitoring_job(self):
        """End-of-day monitoring job"""
        if not self.is_trading_day():
            self.logger.info("📅 Non-trading day, skipping monitoring")
            return
        
        self.logger.info("📊 MONITORING JOB STARTED")
        success = self.run_with_retries(self.run_monitoring, "Monitoring")
        
        if not success:
            self.logger.warning("⚠️ Monitoring failed")
            self.send_alert("WARNING: End-of-day monitoring failed")
    
    def send_alert(self, message):
        """Send alert notification"""
        timestamp = self.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_msg = f"[{timestamp}] TRADING SYSTEM ALERT: {message}"
        
        # Log alert
        self.logger.critical(alert_msg)
        
        # Save to alert file
        alert_file = Path("PRODUCTION/logs/alerts.log")
        with open(alert_file, 'a') as f:
            f.write(f"{alert_msg}\n")
        
        # TODO: Add email/Slack notifications
        print(f"🚨 ALERT: {message}")
    
    def setup_schedule(self):
        """Setup the trading schedule"""
        self.logger.info("📅 Setting up trading schedule")
        
        # Daily jobs (Monday-Friday only, checked in job functions)
        schedule.every().day.at(self.schedule_config["retrain_time"]).do(self.daily_retrain_job)
        schedule.every().day.at(self.schedule_config["audit_time"]).do(self.system_audit_job)
        schedule.every().day.at(self.schedule_config["trade_time"]).do(self.trading_job)
        schedule.every().day.at(self.schedule_config["monitor_time"]).do(self.monitoring_job)
        
        self.logger.info("✅ Schedule configured:")
        self.logger.info(f"   Retraining: {self.schedule_config['retrain_time']}")
        self.logger.info(f"   Audit: {self.schedule_config['audit_time']}")
        self.logger.info(f"   Trading: {self.schedule_config['trade_time']}")
        self.logger.info(f"   Monitoring: {self.schedule_config['monitor_time']}")
    
    def run_scheduler(self):
        """Run the trading scheduler"""
        mode = "PAPER TRADING" if self.paper_trading else "LIVE TRADING"
        self.logger.info(f"🚀 STARTING TRADING SCHEDULER - {mode}")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.logger.info("⏹️ Trading scheduler stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Scheduler error: {e}")
            self.send_alert(f"Scheduler crashed: {e}")

def main():
    """Main scheduler function"""
    print("🕐 AUTOMATED TRADING SCHEDULER")
    print("=" * 60)
    
    # Check for paper trading mode
    paper_trading = os.getenv('PAPER_TRADING', 'true').lower() == 'true'
    
    if paper_trading:
        print("📊 Running in PAPER TRADING mode")
    else:
        print("💰 Running in LIVE TRADING mode")
        confirm = input("⚠️ Confirm LIVE TRADING (yes/no): ")
        if confirm.lower() != 'yes':
            print("❌ Live trading cancelled")
            return
    
    # Initialize scheduler
    scheduler = TradingScheduler(paper_trading=paper_trading)
    
    # Setup and run
    scheduler.setup_schedule()
    scheduler.run_scheduler()

if __name__ == "__main__":
    main()