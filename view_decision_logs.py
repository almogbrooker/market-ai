#!/usr/bin/env python3
"""
DECISION LOG VIEWER
==================
Monitor and analyze trading bot decisions in real-time
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import argparse

def view_latest_logs(log_type="both", lines=50):
    """View latest decision and trading logs"""
    log_dir = Path("../artifacts/logs/live_trading")
    
    if not log_dir.exists():
        print("❌ No log directory found. Run the trading bot first.")
        return
    
    today = datetime.now().strftime('%Y%m%d')
    
    # Decision log
    if log_type in ["both", "decisions"]:
        decision_log = log_dir / f"decision_log_{today}.log"
        if decision_log.exists():
            print("🧠 TRADING DECISION LOG")
            print("=" * 80)
            with open(decision_log) as f:
                content = f.read()
                print(content[-lines*200:] if len(content) > lines*200 else content)
        else:
            print(f"📊 No decision log found for today ({decision_log})")
    
    # Trading log
    if log_type in ["both", "trading"]:
        trading_log = log_dir / f"live_trading_{today}.log"
        if trading_log.exists():
            print("\n📈 TRADING EXECUTION LOG")
            print("=" * 80)
            with open(trading_log) as f:
                lines_list = f.readlines()
                recent_lines = lines_list[-lines:] if len(lines_list) > lines else lines_list
                print(''.join(recent_lines))
        else:
            print(f"📊 No trading log found for today ({trading_log})")

def monitor_logs():
    """Monitor logs in real-time (tail -f equivalent)"""
    log_dir = Path("../artifacts/logs/live_trading")
    today = datetime.now().strftime('%Y%m%d')
    
    decision_log = log_dir / f"decision_log_{today}.log"
    trading_log = log_dir / f"live_trading_{today}.log"
    
    print("📊 MONITORING TRADING DECISIONS (Press Ctrl+C to stop)")
    print("=" * 80)
    
    try:
        import time
        decision_pos = 0
        trading_pos = 0
        
        while True:
            # Check decision log
            if decision_log.exists():
                with open(decision_log) as f:
                    f.seek(decision_pos)
                    new_content = f.read()
                    if new_content:
                        print("🧠 NEW DECISIONS:")
                        print(new_content)
                        decision_pos = f.tell()
            
            # Check trading log  
            if trading_log.exists():
                with open(trading_log) as f:
                    f.seek(trading_pos)
                    new_content = f.read()
                    if new_content:
                        print("📈 TRADING ACTIVITY:")
                        print(new_content)
                        trading_pos = f.tell()
            
            time.sleep(2)  # Check every 2 seconds
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")

def analyze_decisions(date=None):
    """Analyze decisions for a specific date"""
    if date is None:
        date = datetime.now().strftime('%Y%m%d')
    
    log_dir = Path("../artifacts/logs/live_trading")
    decision_log = log_dir / f"decision_log_{date}.log"
    
    if not decision_log.exists():
        print(f"❌ No decision log found for {date}")
        return
    
    print(f"📊 DECISION ANALYSIS FOR {date}")
    print("=" * 80)
    
    with open(decision_log) as f:
        content = f.read()
    
    # Extract statistics
    buy_count = content.count("🟢 LONG POSITIONS")
    sell_count = content.count("🔴 SHORT POSITIONS")
    trade_executions = content.count("TRADE EXECUTION DECISIONS")
    
    print(f"📈 Daily Summary:")
    print(f"   Portfolio decisions: {buy_count}")
    print(f"   Trade executions: {trade_executions}")
    print(f"   Average positions: {(buy_count + sell_count) // max(1, buy_count) if buy_count > 0 else 0}")
    
    # Show recent decisions
    print(f"\n🎯 Recent Decision Summary:")
    lines = content.split('\n')
    for line in lines[-20:]:
        if any(keyword in line for keyword in ["PREDICTION:", "Key factors", "Trade rationale"]):
            print(f"   {line.strip()}")

def main():
    parser = argparse.ArgumentParser(description="Trading Bot Decision Log Viewer")
    parser.add_argument("--monitor", action="store_true", help="Monitor logs in real-time")
    parser.add_argument("--type", choices=["decisions", "trading", "both"], default="both", 
                       help="Type of logs to view")
    parser.add_argument("--lines", type=int, default=50, help="Number of lines to show")
    parser.add_argument("--analyze", help="Analyze decisions for specific date (YYYYMMDD)")
    
    args = parser.parse_args()
    
    if args.monitor:
        monitor_logs()
    elif args.analyze:
        analyze_decisions(args.analyze)
    else:
        view_latest_logs(args.type, args.lines)

if __name__ == "__main__":
    main()