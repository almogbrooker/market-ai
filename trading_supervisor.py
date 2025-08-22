#!/usr/bin/env python3
"""Supervisor for the automated trading system.

- References the exchange calendar when available to decide if today is a trading day.
- Ensures only a single instance of the supervisor runs via a lock file.
- Launches the main trading system script when appropriate.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:  # Optional dependency for real exchange calendars
    import pandas as pd
    import exchange_calendars as ec
except Exception:  # Fallback if package not installed
    ec = None  # type: ignore
    pd = None  # type: ignore

LOCK_FILE = Path("supervisor.lock")
TRADING_SCRIPT = Path("run_automated_trading.py")
EXCHANGE = "XNYS"  # New York Stock Exchange


def is_trading_day(day: datetime | None = None) -> bool:
    """Return True if *day* is a trading session for the target exchange.

    Uses ``exchange_calendars`` when available, otherwise falls back to a
    simple weekday check.
    """
    day = day or datetime.now()
    if ec and pd:
        cal = ec.get_calendar(EXCHANGE)
        return cal.is_session(pd.Timestamp(day.date()))
    return day.weekday() < 5  # Mon-Fri fallback


def main() -> None:
    if LOCK_FILE.exists():
        print("❌ Supervisor already running")
        return

    LOCK_FILE.write_text(f"Started: {datetime.now()}\nPID: {os.getpid()}\n")
    try:
        if not is_trading_day():
            print("📅 Not a trading day – exiting")
            return

        if not TRADING_SCRIPT.exists():
            print(f"❌ Trading script {TRADING_SCRIPT} not found")
            return

        print("🚀 Launching automated trading system")
        proc = subprocess.Popen([sys.executable, str(TRADING_SCRIPT)])
        proc.wait()
    finally:
        try:
            LOCK_FILE.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
