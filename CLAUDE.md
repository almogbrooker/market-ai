# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is an institutional-grade AI trading system with three main layers:

1. **PRODUCTION/**: Live trading system with trained models, bots, tools, and monitoring
2. **src/**: Core libraries for features, models, evaluation, and training
3. **Data Pipeline**: Real-time data feeds, feature engineering, and model training

The system uses ML models (XGBoost, neural networks) with conformal prediction gates to trade equities with institutional risk controls.

## Common Commands

### Training and Model Management
```bash
# Train new model with leak-free validation
python leak_free_model_trainer.py

# Train advanced models with MLflow tracking
python src/training/train_advanced.py --experiment my_experiment --model financial_transformer_small

# Promote model to production
python src/training/promote_model.py

# Build training dataset
PYTHONPATH=. python src/data/data_builder.py
```

### Production Trading
```bash
# Main trading bot
python PRODUCTION/bots/main_trading_bot.py

# Trading dashboard
streamlit run PRODUCTION/tools/trading_dashboard.py

# Run with memory optimization
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python [script.py]
```

### System Monitoring & Validation
```bash
# Institutional audit system
python PRODUCTION/tools/institutional_audit_system.py

# Check for model drift
python PRODUCTION/tools/drift_monitoring_system.py

# Validate IC performance
python PRODUCTION/tools/ic_reality_check.py

# Fix conformal prediction gates
python PRODUCTION/tools/fix_conformal_gate.py

# Comprehensive system test
python comprehensive_system_test.py
```

### Code Quality
```bash
# Lint code
ruff check .
flake8 .

# Format code  
black .

# Type checking
mypy src/

# Run tests
pytest tests/
```

## Key Architecture Components

### Models & Training
- **src/models/**: Advanced models (PatchTST, iTransformer, ensembles)
- **src/training/**: Training pipelines with custom loss functions (Sharpe, RankIC)
- **MLflow integration**: All training runs tracked with experiment management
- **Conformal prediction**: Uncertainty quantification for trade filtering

### Feature Engineering
- **src/features/**: Multi-modal feature engineering (prices, fundamentals, sentiment, macro)
- **Cross-sectional ranking**: Prevents look-ahead bias in feature construction
- **Leakage-free scaling**: Point-in-time data preprocessing
- **Alternative data**: News sentiment, FRED macro indicators, SEC filings

### Risk Management & Evaluation  
- **src/evaluation/**: Backtesting, risk management, conformal gating
- **Production gates**: Real-time uncertainty filtering with 15% accept rate
- **Risk controls**: Position limits, exposure caps, stop-losses, daily limits
- **PurgedKFold validation**: Prevents temporal data leakage

### Trading Infrastructure
- **PRODUCTION/bots/**: Live trading execution, monitoring, order management
- **Alpaca integration**: Paper and live trading through Alpaca API
- **Smart rebalancing**: Intelligent position sizing and risk management
- **Decision logging**: Complete audit trail of all trading decisions

## Data Flow

1. **Data ingestion**: Real-time prices, fundamentals, news via data_providers/
2. **Feature engineering**: Multi-modal features with leakage prevention
3. **Model training**: ML models with conformal prediction calibration  
4. **Production deployment**: Trained models moved to PRODUCTION/models/
5. **Live trading**: Real-time inference with uncertainty filtering
6. **Monitoring**: Drift detection, performance tracking, risk monitoring

## Key Configuration Files

- **PRODUCTION/config/main_config.json**: System configuration and risk parameters
- **requirements.txt**: Core Python dependencies
- **data/training_data_enhanced_FIXED.csv**: Primary training dataset (46 features)
- **Model artifacts**: config.json, features.json, model files in PRODUCTION/models/

## Memory Management

Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` prefix for GPU-intensive scripts to prevent CUDA out-of-memory errors.

## Validation Standards

- All models must pass institutional audit system (8/8 checks)
- IC validation on leak-free datasets required
- Conformal prediction gates must maintain 10-30% accept rate
- No data leakage tolerance - use PurgedKFold with 5-day embargo
- Production deployment requires drift monitoring and emergency stops

## Important Notes

- **Never deploy without validation**: Run institutional_audit_system.py before any production deployment
- **Monitor drift actively**: PSI drift > 0.25 requires immediate model recalibration
- **Respect risk limits**: System has kill switches at 2% daily loss and 60% gross exposure
- **Use PYTHONPATH=.**: Required for proper module imports from project root
- **Leverage existing patterns**: Check neighboring files for coding conventions before implementing new features