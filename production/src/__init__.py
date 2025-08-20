"""
AI Trading System - Core Source Code
===================================

Organized modular architecture for production trading system:

📊 data/
    - data_builder.py      : Main data pipeline builder
    - alpha_loader.py      : Cross-sectional alpha data loading  
    - data_validation.py   : Data quality validation

🎯 features/
    - features.py          : Base technical indicators
    - features_enhanced.py : Advanced feature engineering
    - market_regime.py     : Market regime detection
    - data_integration.py  : Feature pipeline integration
    
    📈 sentiment/
        - llm_sentiment.py         : LLM-based sentiment analysis
        - advanced_sentiment.py    : Multi-model sentiment
        - multilingual_sentiment.py: Multi-language processing
        - llm_news_summarization.py: News summarization
        - sentiment.py             : Base sentiment analysis
    
    🌐 external_data/
        - fred.py                  : Federal Reserve Economic Data
        - edgar.py                 : SEC Edgar filings
        - gdelt.py                 : Global news and events
        - reddit.py                : Reddit sentiment data
        - stocktwits.py            : StockTwits social data
        - sec_edgar_fundamentals.py: Company fundamentals
        - alternative_data_sources.py: Backup data sources
        - streamlined_pipeline.py  : Optimized data pipeline

🧠 models/
    - model_trainer.py     : Enhanced trainer with purged CV
    - advanced_models.py   : PatchTST, iTransformer, GRU models
    - super_ensemble.py    : Meta-learning ensemble

📊 evaluation/
    - backtester.py        : Cost-aware backtesting
    - evaluate.py          : Model evaluation metrics
    - conformal_prediction.py: Uncertainty quantification
    - portfolio_optimizer.py : Portfolio construction
    - risk_management.py   : Risk controls

💼 trading/
    - paper_trader.py      : Alpaca paper trading integration

🏋️ training/
    - train_advanced.py    : Advanced training workflows
    - sharpe_loss.py       : Custom loss functions

🔧 utils/
    - config.py            : Configuration management
    - logger.py            : Logging utilities

This architecture supports the complete trading workflow:
Data → Features → Models → Evaluation → Trading
"""