# 🎯 NASDAQ Stock-Picker - Production Implementation

Clean implementation of the Mission Brief specifications for NASDAQ stock picking with beta-neutral long/short portfolios.

## 🏗️ **Core Components**

### **📊 Universe Builder**
- `nasdaq_stock_picker.py` - Downloads NASDAQ universe with proper filters
- Applies liquidity/quality screens (price ≥$3, ADV ≥$2M)  
- Creates 10 lagged momentum/quality features
- Saves versioned dataset with hash for reproducibility

### **🤖 Model Trainer**  
- `sleeve_c_trainer.py` - Trains Momentum + Quality baseline ranker
- Implements purged CV (purge=10d, embargo=3d)
- Enforces training IC ≤ 3% gate to prevent overfitting
- Evaluates cross-sectional Rank IC with proper validation

### **💼 Portfolio Constructor**
- `portfolio_constructor.py` - Beta-neutral long/short portfolio construction
- Long top 30%, short bottom 30% with 8% position limits
- Includes transaction costs (3.5bps fee + 8.5bps slippage + borrow)
- Risk management with beta neutralization and volatility targeting

## 📁 **Artifacts Structure**

```
artifacts/
├── nasdaq_picker/          # Universe data with versioning hash
│   ├── nasdaq_dataset_dc709706.csv
│   └── dataset_metadata_dc709706.json
├── sleeves/sleeve_c/       # Model artifacts  
│   ├── sleeve_c_fold_*_model.txt    # Trained models
│   ├── sleeve_c_oof_predictions.csv # Out-of-fold predictions
│   ├── sleeve_c_daily_IC.csv        # Daily IC time series
│   └── sleeve_c_metadata.json       # Training config & results
└── portfolio/              # Portfolio simulation results
    ├── daily_portfolios.csv
    ├── daily_performance.csv
    └── portfolio_summary.json
```

## 🚀 **Quick Start**

```bash
# 1. Build NASDAQ universe
python nasdaq_stock_picker.py

# 2. Train Sleeve C model  
python sleeve_c_trainer.py

# 3. Construct portfolios (if model passes gates)
python portfolio_constructor.py
```

## 📊 **Current Status**

- **✅ Infrastructure**: Production-ready, leak-proof, properly gated
- **✅ Universe**: 112,623 samples across 66 NASDAQ stocks (2018-2025)
- **❌ Signal**: Sleeve C Rank IC = -0.23% (Target: ≥0.8%)
- **🎯 Next**: Need enhanced features to generate predictive signal

## 🎯 **Mission Brief Compliance**

All specification requirements implemented:
- ✅ Proper universe filters and cost modeling
- ✅ Leak-proof feature engineering (all features lagged)
- ✅ Purged cross-validation with embargo periods  
- ✅ Beta-neutral portfolio construction with limits
- ✅ Complete artifacts and reporting pipeline
- ❌ Signal generation (current features insufficient)

See `MISSION_BRIEF_SUMMARY.md` for complete implementation details and next steps.

## 📋 **Dependencies**

```bash
pip install -r requirements.txt
```

Key libraries: pandas, numpy, lightgbm, yfinance, scipy, scikit-learn