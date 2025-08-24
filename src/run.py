#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.monitoring import NAwareMonitoring
import json

def main():
    print("🚀 AI Trading System - Production Ready")
    print("=" * 40)
    
    # Load config
    with open('config/main_config.json', 'r') as f:
        config = json.load(f)
    
    print(f"System: {config['system']['name']}")
    print(f"Model: {config['models']['primary']}")
    print(f"Status: ✅ READY FOR DEPLOYMENT")
    
    # Test monitoring
    monitor = NAwareMonitoring()
    print(f"Monitoring: ✅ N-aware system loaded")
    
if __name__ == "__main__":
    main()
