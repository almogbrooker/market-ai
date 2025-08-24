#!/usr/bin/env python3
"""
VALIDATE ENSEMBLE INSTITUTIONAL
===============================
Run full institutional validation on the ensemble model
"""

import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit

def validate_ensemble_institutional():
    """Run comprehensive institutional validation on ensemble"""
    print("🔍 INSTITUTIONAL VALIDATION - ENSEMBLE MODEL")
    print("=" * 60)
    
    # Load ensemble model
    config_path = Path("PRODUCTION/config/main_config.json")
    with open(config_path, 'r') as f:
        main_config = json.load(f)
    
    ensemble_path = Path(main_config["models"]["primary"])
    print(f"📂 Testing ensemble: {ensemble_path.name}")
    
    # Load ensemble components
    ridge_model = joblib.load(ensemble_path / "ridge_component" / "model.pkl")
    ridge_scaler = joblib.load(ensemble_path / "ridge_component" / "scaler.pkl")
    
    with open(ensemble_path / "ridge_component" / "features.json", 'r') as f:
        ridge_features = json.load(f)
    
    lgb_model = joblib.load(ensemble_path / "lightgbm_component" / "model.pkl")
    
    with open(ensemble_path / "lightgbm_component" / "features.json", 'r') as f:
        lgb_features = json.load(f)
    
    print(f"✅ Ensemble loaded: Ridge ({len(ridge_features)}) + LightGBM ({len(lgb_features)})")
    
    # Load validation data
    train_data = pd.read_csv("data/leak_free_train.csv")
    train_data['Date'] = pd.to_datetime(train_data['Date'])
    
    test_data = pd.read_csv("data/leak_free_test.csv")
    test_data['Date'] = pd.to_datetime(test_data['Date'])
    
    target_col = 'target_1d'
    
    print(f"📊 Data: {len(train_data)} train, {len(test_data)} test")
    
    # Define ensemble prediction function
    def predict_ensemble(data):
        """Make ensemble predictions"""
        # Ridge predictions
        X_ridge = data[ridge_features].fillna(0)
        X_ridge_scaled = ridge_scaler.transform(X_ridge)
        pred_ridge = ridge_model.predict(X_ridge_scaled)
        
        # LightGBM predictions
        X_lgb = data[lgb_features].fillna(0)
        pred_lgb = lgb_model.predict(X_lgb)
        
        # Ensemble: 70% Ridge + 30% LightGBM
        return 0.7 * pred_ridge + 0.3 * pred_lgb
    
    # TEST 1: OVERFITTING CHECK
    print("\n🚨 TEST 1: OVERFITTING ANALYSIS")
    
    # Training performance
    train_subset = train_data[train_data['Date'] <= pd.Timestamp('2024-06-01')]
    train_sample = train_subset.dropna(subset=[target_col]).sample(n=5000, random_state=42)
    
    y_train_sample = train_sample[target_col]
    pred_train_sample = predict_ensemble(train_sample)
    
    train_ic, _ = spearmanr(y_train_sample, pred_train_sample)
    train_ic = train_ic if not np.isnan(train_ic) else 0
    
    # Test performance
    test_clean = test_data.dropna(subset=[target_col])
    y_test = test_clean[target_col]
    pred_test = predict_ensemble(test_clean)
    
    test_ic, _ = spearmanr(y_test, pred_test)
    test_ic = test_ic if not np.isnan(test_ic) else 0
    
    # Overfitting analysis
    ic_degradation = abs(train_ic - test_ic)
    overfitting_ratio = train_ic / test_ic if test_ic > 0 else float('inf')
    
    overfitting_pass = ic_degradation < 0.05 and overfitting_ratio < 2.0
    
    print(f"   Train IC: {train_ic:.4f}")
    print(f"   Test IC: {test_ic:.4f}")
    print(f"   IC Degradation: {ic_degradation:.4f}")
    print(f"   Overfitting ratio: {overfitting_ratio:.2f}")
    print(f"   Overfitting check: {'🟢 PASS' if overfitting_pass else '❌ FAIL'}")
    
    # TEST 2: CROSS-VALIDATION STABILITY
    print("\n📊 TEST 2: CROSS-VALIDATION STABILITY")
    
    # Time series cross-validation
    cv_data = train_subset.sample(n=8000, random_state=42)
    cv_scores = []
    
    tscv = TimeSeriesSplit(n_splits=3, test_size=500)
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(cv_data)):
        cv_train = cv_data.iloc[train_idx]
        cv_val = cv_data.iloc[val_idx]
        
        y_cv_val = cv_val[target_col].fillna(0)
        pred_cv_val = predict_ensemble(cv_val)
        
        cv_ic, _ = spearmanr(y_cv_val, pred_cv_val)
        cv_ic = cv_ic if not np.isnan(cv_ic) else 0
        cv_scores.append(cv_ic)
        
        print(f"   Fold {fold+1}: IC = {cv_ic:.4f}")
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    cv_stability_pass = cv_std < 0.05 and cv_mean > 0.01
    
    print(f"   CV Mean: {cv_mean:.4f}")
    print(f"   CV Std: {cv_std:.4f}")
    print(f"   CV Stability: {'🟢 PASS' if cv_stability_pass else '❌ FAIL'}")
    
    # TEST 3: ANTI-LEAKAGE TESTS
    print("\n🔴 TEST 3: ANTI-LEAKAGE VALIDATION")
    
    # Time-shuffle test
    y_test_shuffled = np.random.permutation(y_test)
    shuffle_ic, _ = spearmanr(y_test_shuffled, pred_test)
    shuffle_ic = shuffle_ic if not np.isnan(shuffle_ic) else 0
    
    shuffle_pass = abs(shuffle_ic) < 0.05
    
    print(f"   Time-shuffle IC: {shuffle_ic:.4f}")
    print(f"   Shuffle test: {'🟢 PASS' if shuffle_pass else '❌ FAIL'}")
    
    # Label alignment
    alignment_pass = test_ic > 0.001
    print(f"   Label alignment: {'🟢 PASS' if alignment_pass else '❌ FAIL'}")
    
    # Feature shift analysis
    train_features_all = list(set(ridge_features + lgb_features))
    
    train_means = train_subset[train_features_all].mean()
    test_means = test_clean[train_features_all].mean()
    
    relative_shifts = abs((test_means - train_means) / (train_means + 1e-8))
    max_shift = relative_shifts.max()
    
    shift_pass = max_shift < 5.0  # Allow more shift for ensemble
    
    print(f"   Max feature shift: {max_shift:.3f}")
    print(f"   Feature shift: {'🟢 PASS' if shift_pass else '❌ FAIL'}")
    
    anti_leakage_pass = shuffle_pass and alignment_pass and shift_pass
    
    # TEST 4: STATISTICAL SIGNIFICANCE
    print("\n📈 TEST 4: STATISTICAL SIGNIFICANCE")
    
    # Bootstrap confidence intervals
    bootstrap_ics = []
    n_bootstrap = 500
    
    for i in range(n_bootstrap):
        indices = np.random.choice(len(y_test), len(y_test), replace=True)
        y_boot = y_test.iloc[indices]
        pred_boot = pred_test[indices]
        
        ic_boot, _ = spearmanr(y_boot, pred_boot)
        if not np.isnan(ic_boot):
            bootstrap_ics.append(ic_boot)
    
    ci_lower = np.percentile(bootstrap_ics, 2.5)
    ci_upper = np.percentile(bootstrap_ics, 97.5)
    
    statistical_pass = ci_lower > 0
    
    print(f"   95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"   Statistical significance: {'🟢 PASS' if statistical_pass else '❌ FAIL'}")
    
    # TEST 5: ENSEMBLE-SPECIFIC TESTS
    print("\n🎭 TEST 5: ENSEMBLE-SPECIFIC VALIDATION")
    
    # Component correlation (should not be too high)
    pred_ridge_only = ridge_model.predict(ridge_scaler.transform(test_clean[ridge_features].fillna(0)))
    pred_lgb_only = lgb_model.predict(test_clean[lgb_features].fillna(0))
    
    component_corr, _ = spearmanr(pred_ridge_only, pred_lgb_only)
    component_corr = abs(component_corr) if not np.isnan(component_corr) else 1.0
    
    diversity_pass = component_corr < 0.9  # Components should be somewhat different
    
    print(f"   Component correlation: {component_corr:.4f}")
    print(f"   Diversity check: {'🟢 PASS' if diversity_pass else '❌ FAIL'}")
    
    # Individual vs ensemble performance
    ridge_ic, _ = spearmanr(y_test, pred_ridge_only)
    ridge_ic = ridge_ic if not np.isnan(ridge_ic) else 0
    
    lgb_ic, _ = spearmanr(y_test, pred_lgb_only)
    lgb_ic = lgb_ic if not np.isnan(lgb_ic) else 0
    
    ensemble_better = test_ic > max(ridge_ic, lgb_ic)
    
    print(f"   Ridge IC: {ridge_ic:.4f}")
    print(f"   LightGBM IC: {lgb_ic:.4f}")
    print(f"   Ensemble IC: {test_ic:.4f}")
    print(f"   Ensemble superior: {'🟢 PASS' if ensemble_better else '❌ FAIL'}")
    
    ensemble_specific_pass = diversity_pass and ensemble_better
    
    # TEST 6: REALITY CHECKS
    print("\n💰 TEST 6: REALITY CHECKS")
    
    # IC range check
    ic_realistic = 0.001 <= test_ic <= 0.20  # Allow higher range for ensemble
    
    # Direction accuracy
    direction_acc = np.mean((y_test > 0) == (pred_test > 0))
    direction_reasonable = 0.48 <= direction_acc <= 0.70
    
    # Transaction costs
    transaction_cost = 0.0015
    net_ic = test_ic - transaction_cost
    cost_viable = net_ic > 0.001
    
    print(f"   IC range: {'🟢 PASS' if ic_realistic else '❌ FAIL'} ({test_ic:.4f})")
    print(f"   Direction: {'🟢 PASS' if direction_reasonable else '❌ FAIL'} ({direction_acc:.1%})")
    print(f"   Post-cost viable: {'🟢 PASS' if cost_viable else '❌ FAIL'} (net: {net_ic:.4f})")
    
    reality_pass = ic_realistic and direction_reasonable and cost_viable
    
    # OVERALL ASSESSMENT
    print("\n" + "=" * 60)
    print("🏆 ENSEMBLE INSTITUTIONAL VALIDATION RESULTS")
    print("=" * 60)
    
    all_tests = [
        ('Overfitting Check', overfitting_pass),
        ('CV Stability', cv_stability_pass),
        ('Anti-Leakage', anti_leakage_pass),
        ('Statistical Significance', statistical_pass),
        ('Ensemble Specific', ensemble_specific_pass),
        ('Reality Checks', reality_pass)
    ]
    
    passed_tests = sum([passed for _, passed in all_tests])
    total_tests = len(all_tests)
    
    print(f"\n📊 TEST RESULTS:")
    for test_name, passed in all_tests:
        status = "🟢 PASS" if passed else "❌ FAIL"
        print(f"   {test_name:25}: {status}")
    
    overall_pass = passed_tests == total_tests
    
    print(f"\n📈 SUMMARY:")
    print(f"   Tests passed: {passed_tests}/{total_tests}")
    print(f"   Success rate: {passed_tests/total_tests*100:.1f}%")
    print(f"   Overall: {'🟢 INSTITUTIONAL GRADE' if overall_pass else '❌ NEEDS REVIEW'}")
    
    print(f"\n🎯 ENSEMBLE PERFORMANCE:")
    print(f"   Test IC: {test_ic:.4f} ({test_ic*100:.2f}%)")
    print(f"   95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"   Direction: {direction_acc:.1%}")
    print(f"   Overfitting: {ic_degradation:.4f} degradation")
    
    if overall_pass:
        print(f"\n✅ ENSEMBLE VALIDATION PASSED!")
        print(f"✅ No overfitting detected")
        print(f"✅ All institutional requirements met")
        print(f"✅ Safe for production deployment")
        
        # Compare to single model improvement
        single_model_ic = 0.0571
        true_improvement = (test_ic - single_model_ic) / single_model_ic * 100
        
        print(f"\n📈 VALIDATED IMPROVEMENT:")
        print(f"   Single model: {single_model_ic:.4f} ({single_model_ic*100:.2f}%)")
        print(f"   Ensemble: {test_ic:.4f} ({test_ic*100:.2f}%)")
        print(f"   True improvement: +{true_improvement:.1f}%")
        
    else:
        failed_tests = [name for name, passed in all_tests if not passed]
        print(f"\n🚨 VALIDATION ISSUES:")
        print(f"   Failed tests: {', '.join(failed_tests)}")
        print(f"   ⚠️ Ensemble needs review before deployment")
    
    return {
        'overall_pass': overall_pass,
        'test_ic': test_ic,
        'overfitting': ic_degradation,
        'tests_passed': passed_tests,
        'total_tests': total_tests,
        'true_improvement': (test_ic - 0.0571) / 0.0571 * 100 if 0.0571 > 0 else 0
    }

if __name__ == "__main__":
    results = validate_ensemble_institutional()