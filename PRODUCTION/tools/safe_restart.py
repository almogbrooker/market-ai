#!/usr/bin/env python3
"""Safe restart utility for the automated trading system.

Sends a soft interrupt to the running system, waits for a clean shutdown,
then relaunches it. Uses the `trading_system.lock` file to guarantee
exactly-once semantics.
"""

from pathlib import Path
import os
import signal
import subprocess
import sys
import time

LOCK_FILE = Path("trading_system.lock")
TARGET_SCRIPT = Path("run_automated_trading.py")

def _read_pid() -> int | None:
    if LOCK_FILE.exists():
        with open(LOCK_FILE) as f:
            for line in f:
                if line.startswith("PID"):
                    try:
                        return int(line.split(":", 1)[1].strip())
                    except ValueError:
                        return None
    return None

def stop_system():
    pid = _read_pid()
    if pid:
        print(f"Stopping trading system (PID {pid})...")
        try:
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            print("Process not found; removing stale lock.")
            if LOCK_FILE.exists():
                LOCK_FILE.unlink()
            return
        for _ in range(30):
            if not LOCK_FILE.exists():
                print("Trading system stopped.")
                return
            time.sleep(1)
        print("Timeout waiting for shutdown; removing lock file.")
        LOCK_FILE.unlink(missing_ok=True)
    else:
        print("No running trading system found.")

def start_system():
    if LOCK_FILE.exists():
        print("Lock file present; aborting start to avoid duplicate run.")
        return
    print("Starting trading system...")
    subprocess.Popen([sys.executable, str(TARGET_SCRIPT)])
    for _ in range(30):
        if LOCK_FILE.exists():
            print("Trading system started.")
            return
        time.sleep(1)
    print("Warning: trading system did not create lock file.")

def main():
    stop_system()
    start_system()

if __name__ == "__main__":
    main()
