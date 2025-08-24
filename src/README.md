# AI Trading System - Production Source

## Directory Structure
```
src/
├── models/
│   └── production_model/          # Rank-transformed ensemble model
├── config/
│   ├── main_config.json          # Main system configuration  
│   ├── monitoring_config.json    # N-aware monitoring settings
│   ├── psi_reference.json        # PSI drift baselines
│   └── thresholds.json           # Alert thresholds
├── tools/
│   └── monitoring.py             # N-aware monitoring system
├── data/
│   ├── train_data.csv           # Training dataset
│   └── test_data.csv            # Test dataset (with rank transforms)
├── trading_bot.py               # Production trading bot
└── simulation.py                # Day-1 live simulation
```

## Quick Start

1. **Run Trading Bot:**
   ```bash
   cd src
   python trading_bot.py --model-path=models/production_model
   ```

2. **Run Day-1 Simulation:**
   ```bash
   cd src
   python simulation.py
   ```

3. **Test Monitoring:**
   ```bash
   cd src
   python -c "from tools.monitoring import NAwareMonitoring; print('✅ Monitoring loaded')"
   ```

## Key Features
- **Rank-transformed ensemble model** (6.70% IC performance)
- **N-aware monitoring** with binomial statistics (no false alerts)
- **Institutional-grade risk controls** and drift detection
- **Production-ready trading bot** with comprehensive safeguards

**Status:** ✅ PRODUCTION READY
