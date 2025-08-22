# 🚀 Market AI Trading System

**Institutional-grade AI trading system with real-time optimization and risk management**

## 🎯 Quick Start

```bash
# 1. Set up environment
cp config/example.env .env
# Edit .env with your API keys (optional - works with free tiers)

# 2. Start live trading
python start_live_trading.py

# 3. Or run components separately:
python production/alpaca_setup.py      # Setup Alpaca connection
python production/live_trading_system.py  # Start live trading
```

## 📊 System Overview

### **Performance**
- See [Methodology](reports/methodology.md) for detailed validation metrics and performance evaluation.

### **Scale**
- **Universe**: 260 stocks (Russell 1000 Extended)
- **Capacity**: $50M - $2B theoretical
- **Sectors**: Technology, Financial, Healthcare, Consumer, Energy, Industrial
- **Flexibility**: Scale from 12 to 260+ stocks with one command

### **Technology**
- **Models**: XGBoost optimized with Optuna + PurgedKFold
- **Data**: Real-time prices, fundamentals, news sentiment, macro indicators
- **Validation**: Temporal safeguards, conformal prediction, risk controls
- **Infrastructure**: Production-ready with automatic retraining

---

## 📁 Directory Structure

```
market-ai/
├── 🚀 CORE SYSTEM
│   ├── start_live_trading.py          # Main launcher
│   ├── config/
│   │   ├── live_trading_config.json   # Trading configuration  
│   │   ├── stock_universes.py         # Universe management
│   │   └── example.env                # API keys template
│   └── CLAUDE.md                      # System documentation
│
├── 📊 DATA PIPELINE
│   ├── pipelines/
│   │   ├── update_daily.py            # Daily data orchestrator
│   │   └── retrain_monthly.py         # Monthly optimization
│   ├── data_providers/
│   │   ├── price.py                   # Price data (Alpaca/yfinance)
│   │   ├── fundamentals.py            # Fundamental data (FMP/IEX)
│   │   └── news.py                    # News sentiment (NewsAPI)
│   └── utils/
│       └── pit.py                     # Point-in-time validation
│
├── 🧠 AI MODELS
│   ├── src/models/
│   │   ├── advanced_models.py         # PatchTST + iTransformer
│   │   ├── meta_ensemble.py           # Ensemble methods
│   │   └── super_ensemble.py          # Production ensemble
│   └── models/202508/                 # Trained models
│       ├── model.xgb                  # XGBoost model
│       ├── gate.json                  # Conformal prediction
│       └── scaler.joblib              # Feature scaler
│
├── 🏛️ TRADING ENGINE
│   ├── production/
│   │   ├── live_trading_system.py     # Complete live system
│   │   └── alpaca_setup.py            # Broker setup
│   └── src/trading/
│       └── paper_trader.py            # Risk-managed trader
│
├── 📈 RISK & VALIDATION
│   ├── src/evaluation/
│   │   ├── production_conformal_gate.py  # Uncertainty gates
│   │   ├── risk_management.py         # Risk controls
│   │   └── backtester.py              # Performance evaluation
│   └── tests/
│       └── test_institutional_guardrails.py  # Validation suite
│
└── 📋 ARTIFACTS
    ├── data/                          # Training data & cache
    ├── reports/                       # Training reports
    └── logs/                          # System logs
```

---

## 🔧 Configuration

### **Universe Selection**
```python
from config.stock_universes import StockUniverses

# Switch between universes instantly
StockUniverses.update_config_universe('sp100')        # 120 stocks
StockUniverses.update_config_universe('russell1000')  # 260 stocks  
StockUniverses.update_config_universe('mega_cap_tech') # 12 stocks
```

### **Risk Controls**
- **Position Sizing**: Auto-adjusted (3-10% per name based on universe)
- **Gross Exposure**: 60% maximum with kill switch
- **Daily Loss Limit**: 3% with automatic halt
- **Conformal Gates**: 85% coverage for signal filtering

### **API Keys (Optional)**
```bash
# Priority APIs (free tiers available)
ALPACA_API_KEY=your_alpaca_key          # Paper trading
FRED_API_KEY=your_fred_key              # Macro indicators

# Enhanced data (optional)  
FMP_API_KEY=your_fmp_key                # Fundamentals
NEWS_API_KEY=your_news_key              # News sentiment
```

---

## 🚀 Deployment

### **Daily Automation**
```bash
# Cron: Daily at 17:10 ET
10 17 * * 1-5 python -m pipelines.update_daily --update-training
```

### **Monthly Optimization**
```bash
# Cron: First business day 18:00 ET
0 18 1 * * python -m pipelines.retrain_monthly --horizon 5 --trials 100
```

### **Monitoring**
- **Kill Switches**: Watch for `trading_halted = True`
- **Performance**: Track IC and conformal gate metrics
- **Risk**: Monitor exposure and loss limits
- **Data**: Verify daily fetch success
- **Dashboard**: `streamlit run PRODUCTION/tools/trading_dashboard.py`

## 📘 Runbooks

Operational playbooks for common scenarios are available in [PRODUCTION/docs/runbooks](PRODUCTION/docs/runbooks):

- [Missing Data](PRODUCTION/docs/runbooks/missing_data.md)
- [Partial Fills](PRODUCTION/docs/runbooks/partial_fills.md)
- [Rate-Limit Errors](PRODUCTION/docs/runbooks/rate_limit_errors.md)
- [Off-Calendar Openings](PRODUCTION/docs/runbooks/off_calendar_openings.md)
- [Safe Shutdown & Resume](PRODUCTION/docs/runbooks/safe_shutdown_resume.md)

---

## 📊 Performance Validation

Detailed backtesting procedures, institutional guardrail results, transaction cost assumptions, and sensitivity analyses are documented in the [Methodology report](reports/methodology.md).

---

## 🎯 Key Features

### **🧠 Advanced AI**
- **Optimization**: Optuna hyperparameter tuning
- **Validation**: PurgedKFold + 5-day embargo
- **Objective**: Spearman IC (cross-sectional robust)
- **Models**: XGBoost with conformal prediction

### **📊 Multi-Modal Data**
- **Real-time Prices**: Alpaca → yfinance fallback
- **Fundamentals**: Cross-sectional rankings (PE, ROE, etc.)
- **News Sentiment**: Multi-source with deduplication
- **Macro Indicators**: FRED economic data

### **🛡️ Risk Management**
- **Kill Switches**: Automatic trading halts
- **Position Limits**: Dynamic sizing by universe
- **Conformal Gates**: Uncertainty-based filtering
- **Temporal Validation**: No look-ahead bias

### **⚡ Production Ready**
- **Real-time Data**: 5-minute update frequency
- **Adaptive Training**: Weekly model retraining
- **Scalable Architecture**: 12 to 260+ stocks
- **Institutional Controls**: Complete validation

---

## 🏆 System Status

**✅ Production Ready**
- All validation tests passed
- Risk controls active
- Kill switches verified
- Real-time data feeds operational

**✅ Scalable**
- 260 stock universe active
- $50M-$2B capacity
- One-command universe switching
- Institutional-grade validation

**✅ Optimized**
- Training and inference pipelines fully automated
- Conformal prediction gates calibrated
- Monthly retraining pipeline active
- Historical performance metrics available in the [Methodology report](reports/methodology.md)

**🚀 Ready for institutional deployment!**