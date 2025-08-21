# 🚨 CRITICAL CODE LOCATIONS - RAW HTTP LINKS

## Repository Information
- **Repository**: almogbrooker/market-ai
- **Branch**: master
- **Last Commit**: afb8292 (Critical validation issues documented)

## 🔴 MOST CRITICAL FILES - IMMEDIATE REVIEW REQUIRED

### **1. Portfolio Simulation with Severe Model Selection Leak**
```
https://raw.githubusercontent.com/almogbrooker/market-ai/master/comprehensive_oos_validation.py
```
**CRITICAL LINES 288-298**: Model selection during OOS testing = severe peek
**CRITICAL LINES 242-396**: Unrealistic cost model (0.1% flat fee)
**CRITICAL LINES 136-276**: Training ensemble with potential horizon selection leak

### **2. Clean Validation Protocol - Temporal Ordering Issues**
```
https://raw.githubusercontent.com/almogbrooker/market-ai/master/clean_validation_protocol.py
```
**CRITICAL LINES 59-98**: Time slicing logic - verify no same-day joins
**CRITICAL LINES 241-284**: Multi-horizon IC calculation - potential peek
**CRITICAL LINES 186-203**: Feature selection logic

### **3. Feature Enhancement - Point-in-Time Compliance**
```
https://raw.githubusercontent.com/almogbrooker/market-ai/master/enhance_features.py
```
**CRITICAL LINES 148-188**: Cross-sectional feature creation
**CRITICAL LINES 190-228**: Time-aligned sentiment features
**CRITICAL LINES 264-286**: Value-momentum combinations

### **4. OOS Validation 2023-2025 - Original Validation**
```
https://raw.githubusercontent.com/almogbrooker/market-ai/master/oos_validation_2023_2025.py
```
**CRITICAL LINES 128-354**: Complete OOS validation logic
**CRITICAL LINES 245-284**: Portfolio simulation with cost model
**CRITICAL LINES 58-126**: Ensemble model training

## 🏗️ CORE INFRASTRUCTURE FILES

### **5. Data Loading and Alpha Generation**
```
https://raw.githubusercontent.com/almogbrooker/market-ai/master/src/data/alpha_loader.py
```
**PURPOSE**: Core data pipeline for alpha signal generation
**CRITICAL AREAS**: Same-day joins, feature-target temporal alignment

### **6. Training Data Builder**
```
https://raw.githubusercontent.com/almogbrooker/market-ai/master/src/data/data_builder.py
```
**PURPOSE**: Dataset construction and preprocessing
**CRITICAL AREAS**: Target creation, lag enforcement, universe selection

### **7. Advanced Models Implementation**
```
https://raw.githubusercontent.com/almogbrooker/market-ai/master/src/models/advanced_models.py
```
**PURPOSE**: PatchTST and iTransformer implementations
**CRITICAL AREAS**: Sequence creation, feature engineering

### **8. Meta Ensemble System**
```
https://raw.githubusercontent.com/almogbrooker/market-ai/master/src/models/meta_ensemble.py
```
**PURPOSE**: Model combination and ensemble logic
**CRITICAL AREAS**: OOF prediction generation, model selection

## 🤖 TRADING BOTS AND PRODUCTION SYSTEMS

### **9. Ultimate Trading Bot**
```
https://raw.githubusercontent.com/almogbrooker/market-ai/master/practical_ultimate_bot.py
```
**PURPOSE**: Main production trading system
**CRITICAL AREAS**: Risk management, position sizing, execution logic

### **10. Production Data Engine**
```
https://raw.githubusercontent.com/almogbrooker/market-ai/master/production_data_engine.py
```
**PURPOSE**: 92-feature production pipeline
**CRITICAL AREAS**: Feature computation, real-time updates

### **11. Simple Backtest System**
```
https://raw.githubusercontent.com/almogbrooker/market-ai/master/simple_backtest_2025.py
```
**PURPOSE**: Performance validation and backtesting
**CRITICAL AREAS**: Return calculation, cost modeling

### **12. NASDAQ Stock Picker**
```
https://raw.githubusercontent.com/almogbrooker/market-ai/master/nasdaq_stock_picker.py
```
**PURPOSE**: Stock selection and ranking system
**CRITICAL AREAS**: Cross-sectional ranking, universe definition

## 📊 VALIDATION AND EVALUATION FILES

### **13. Walk-Forward Validation**
```
https://raw.githubusercontent.com/almogbrooker/market-ai/master/proper_walk_forward_validation.py
```
**CRITICAL AREAS**: Window definition, temporal splits, performance measurement

### **14. Offline Validation (Modified)**
```
https://raw.githubusercontent.com/almogbrooker/market-ai/master/run_offline_validation.py
```
**CRITICAL AREAS**: Model validation logic, acceptance gates

### **15. Training Status Tracker**
```
https://raw.githubusercontent.com/almogbrooker/market-ai/master/models/training_status.txt
```
**PURPOSE**: Model training progress and status

## 📋 CONFIGURATION AND SETUP FILES

### **16. Main Training Interface**
```
https://raw.githubusercontent.com/almogbrooker/market-ai/master/train.py
```
**PURPOSE**: Primary training script with multi-modal data
**CRITICAL AREAS**: Data loading, model selection, hyperparameters

### **17. Project Instructions (CLAUDE.md)**
```
https://raw.githubusercontent.com/almogbrooker/market-ai/master/CLAUDE.md
```
**PURPOSE**: Project documentation and implementation guide
**CRITICAL AREAS**: Feature specifications, performance targets

### **18. Requirements and Dependencies**
```
https://raw.githubusercontent.com/almogbrooker/market-ai/master/requirements.txt
```
**PURPOSE**: Python package dependencies

## 📈 RESULTS AND ARTIFACTS

### **19. Comprehensive OOS Results (JSON)**
```
https://raw.githubusercontent.com/almogbrooker/market-ai/master/reports/comprehensive_oos_validation.json
```
**CRITICAL DATA**: Impossible returns (+272% H1-2023), Sharpe 14.8
**RED FLAGS**: Performance magnitudes indicating severe leakage

### **20. OOS Validation Results (JSON)**
```
https://raw.githubusercontent.com/almogbrooker/market-ai/master/reports/oos_validation_2023_2025.json
```
**CRITICAL DATA**: IC values +18.9%, portfolio returns +365%
**RED FLAGS**: Institutional impossibility of these numbers

### **21. CV Report (Artifacts)**
```
https://raw.githubusercontent.com/almogbrooker/market-ai/master/artifacts/cv_report.json
```
**PURPOSE**: Cross-validation results and feature validation

## 🗃️ TRAINING DATA FILES

### **22. Enhanced Training Data**
```
https://raw.githubusercontent.com/almogbrooker/market-ai/master/data/training_data_enhanced_with_fundamentals.csv
```
**CRITICAL**: Main training dataset with 110 features
**VALIDATION NEEDED**: Verify no same-day feature-target joins

### **23. Base Training Data**
```
https://raw.githubusercontent.com/almogbrooker/market-ai/master/data/training_data_2020_2024_complete.csv
```
**PURPOSE**: Original dataset before enhancements

## 🔧 UTILITY AND SUPPORT FILES

### **24. Data Source Testing**
```
https://raw.githubusercontent.com/almogbrooker/market-ai/master/test_data_sources.py
```
**PURPOSE**: Validate external data feeds (GDELT, FRED, SEC EDGAR)

### **25. Model Building Interface**
```
https://raw.githubusercontent.com/almogbrooker/market-ai/master/build_dataset.py
```
**PURPOSE**: Dataset construction with multi-modal sources

### **26. Start Script (Interactive)**
```
https://raw.githubusercontent.com/almogbrooker/market-ai/master/start.py
```
**PURPOSE**: System launcher and interface

## 🚨 MOST URGENT REVIEW PRIORITIES

### **PRIORITY 1 (IMMEDIATE)**
1. `comprehensive_oos_validation.py` - Lines 288-298 (model selection leak)
2. `comprehensive_oos_validation.py` - Lines 242-396 (cost model)
3. `data/training_data_enhanced_with_fundamentals.csv` (target lag validation)

### **PRIORITY 2 (CRITICAL)**
1. `src/data/alpha_loader.py` (feature-target temporal alignment)
2. `clean_validation_protocol.py` (time slicing logic)
3. `enhance_features.py` (point-in-time compliance)

### **PRIORITY 3 (HIGH)**
1. `practical_ultimate_bot.py` (production trading logic)
2. `oos_validation_2023_2025.py` (validation methodology)
3. `reports/*.json` (result validation)

## 📝 VALIDATION CHECKLIST

For each file, verify:
- [ ] **No same-day feature-target joins**
- [ ] **Proper temporal lag enforcement (≥1 day)**
- [ ] **Point-in-time universe compliance**
- [ ] **Realistic cost modeling (10-25 bps + slippage)**
- [ ] **No model selection during OOS periods**
- [ ] **Proper cross-validation without peek**
- [ ] **Position and risk constraints enforcement**

## 🎯 EXPECTED FINDINGS

Based on impossible performance results, expect to find:
1. **Same-day joins** between features and targets
2. **Forward-looking universe selection**
3. **Model/horizon selection using OOS data**
4. **Severely underestimated trading costs**
5. **Missing survivorship bias controls**
6. **Inadequate risk management constraints**

**CONCLUSION**: These raw code links provide direct access to all critical implementations requiring immediate review and fixes before any production deployment.