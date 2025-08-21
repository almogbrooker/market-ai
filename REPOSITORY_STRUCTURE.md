# 📁 CLEAN REPOSITORY STRUCTURE

## 🚀 **CORE EXECUTION FILES**
```
/home/almog/market-ai/
├── oos_validation_2023_2025.py          # 🔒 FIXED OOS validation (no leakage)
├── comprehensive_oos_validation.py      # 🏛️ Institutional validation framework  
├── clean_validation_protocol.py         # ✅ Clean temporal validation
├── proper_walk_forward_validation.py    # 📊 Proper WFV implementation
├── enhance_features.py                  # 🌟 Academic feature enhancement
└── run_offline_validation.py            # 🔍 Offline validation runner
```

## 📊 **DATA & ARTIFACTS**
```
├── data/                                # 📈 Training datasets
│   ├── training_data_enhanced_with_fundamentals.csv  # Main enhanced dataset
│   ├── training_data_2020_2024_complete.csv         # Base dataset
│   └── [company_fundamentals.csv, news.parquet...]  # External data
├── artifacts/                           # 🗃️ Training artifacts
│   ├── ds_train.parquet                 # Training split (≤2022)
│   ├── ds_oos_2023H1.parquet          # OOS test periods
│   ├── ds_oos_2023H2_2024.parquet     
│   ├── ds_oos_2025YTD.parquet         
│   └── cv_report.json                  # Cross-validation results
└── reports/                            # 📋 Validation reports
    ├── oos_validation_2023_2025_FIXED.json     # 🔒 Fixed validation results  
    ├── oos_validation_2023_2025.json           # ❌ Original (leaked) results
    └── comprehensive_oos_validation.json       # 🏛️ Institutional validation
```

## 🧠 **SOURCE CODE STRUCTURE**
```
├── src/                                # 📦 Core library
│   ├── data/                          # 🔄 Data pipeline
│   │   ├── alpha_loader.py           # Main data loading
│   │   └── data_builder.py           # Dataset construction
│   ├── models/                       # 🤖 ML models
│   │   ├── advanced_models.py        # PatchTST + iTransformer
│   │   ├── meta_ensemble.py          # Ensemble system
│   │   └── tiered_system.py          # Tiered architecture
│   ├── features/                     # 🌐 Feature engineering
│   │   ├── external_data/            # External data sources
│   │   └── sentiment/                # Sentiment processing
│   ├── evaluation/                   # 📊 Validation & risk
│   │   ├── conformal_prediction.py   # Uncertainty quantification
│   │   └── risk_management.py        # Risk controls
│   └── training/                     # 🏋️ Training utilities
└── production/                       # 🚀 Production systems
    ├── nasdaq_stock_picker.py        # Stock selection
    └── portfolio_constructor.py      # Portfolio construction
```

## 📋 **DOCUMENTATION**
```
├── CLAUDE.md                         # 📖 Project instructions
├── CRITICAL_VALIDATION_ISSUES.md     # 🚨 Critical issues identified
├── CRITICAL_CODE_LOCATIONS.md        # 🔗 Raw code links for review
├── REPOSITORY_STRUCTURE.md           # 📁 This file
└── chat-g.txt                        # 📋 Original requirements
```

## 🗑️ **CLEANED UP (REMOVED)**
- ✅ Duplicate test files (`test_*.py`)
- ✅ Temporary fix files (`fix_*.py`, `implement_*.py`)
- ✅ Unused agent directories (`agents/`, `validation/`)
- ✅ Old validation reports in root
- ✅ Redundant utilities (`institutional_risk_guards.py`, `pit_data_enforcer.py`)

## 🎯 **KEY WORKFLOW FILES**

### **For Fixing Data Leakage**
1. `oos_validation_2023_2025.py` - Main validation with fixes applied
2. `CRITICAL_VALIDATION_ISSUES.md` - Issues documentation  
3. `CRITICAL_CODE_LOCATIONS.md` - Raw code review links

### **For Production Deployment**
1. `src/data/alpha_loader.py` - Core data pipeline (needs temporal fixes)
2. `src/models/meta_ensemble.py` - Ensemble system
3. `production/nasdaq_stock_picker.py` - Stock selection
4. `reports/oos_validation_2023_2025_FIXED.json` - Realistic performance

## 🔥 **CRITICAL NEXT STEPS**
1. **Fix temporal leakage** in `src/data/alpha_loader.py`
2. **Implement proper cost model** with realistic transaction costs
3. **Add regime detection** for 2025 adaptation
4. **Reduce position sizing** from 100% to 30-50% gross exposure
5. **Deploy with institutional risk controls**

**Repository is now clean and organized for production-ready development.**