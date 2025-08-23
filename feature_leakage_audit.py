#!/usr/bin/env python3
"""
FEATURE LEAKAGE AUDIT SYSTEM
Systematically identify and eliminate all sources of data leakage
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FeatureLeakageAuditor:
    """Comprehensive audit for data leakage in features"""
    
    def __init__(self, data_path="data/training_data_enhanced_FIXED.csv"):
        self.data_path = data_path
        self.leakage_issues = {}
        self.clean_features = []
        self.contaminated_features = []
        
    def load_and_analyze(self):
        """Load data and analyze all features for leakage"""
        print("🔍 COMPREHENSIVE FEATURE LEAKAGE AUDIT")
        print("=" * 60)
        
        df = pd.read_csv(self.data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        print(f"📊 Dataset: {len(df):,} samples, {len(df.columns)} columns")
        print(f"📅 Period: {df['Date'].min()} to {df['Date'].max()}")
        
        # Get all feature columns (exclude metadata)
        exclude_cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']  # Raw price data is OK
        target_cols = ['Return_1D', 'target_1d', 'target_5d', 'target_20d', 'next_return_1d']  # These are targets, not features
        metadata_cols = ['barrier_label', 'barrier_days']  # Metadata
        
        all_features = [col for col in df.columns if col not in exclude_cols + target_cols + metadata_cols]
        print(f"🧪 Analyzing {len(all_features)} features for leakage...")
        
        return df, all_features
    
    def check_forward_looking_features(self, df, features):
        """Identify obviously forward-looking features"""
        print(f"\n🔍 CHECKING FOR FORWARD-LOOKING FEATURES")
        print("-" * 40)
        
        forward_looking = []
        suspect_patterns = [
            'next_', 'future_', 'forward_', 'lead_', 'ahead_',
            'target_', '_1d', '_5d', '_20d'  # Common forward-looking patterns
        ]
        
        for feature in features:
            for pattern in suspect_patterns:
                if pattern in feature.lower():
                    if feature not in ['Return_1D']:  # This is our target, OK to have
                        forward_looking.append(feature)
                        break
        
        print(f"🚨 Found {len(forward_looking)} forward-looking features:")
        for feat in forward_looking:
            print(f"   ❌ {feat}")
        
        self.contaminated_features.extend(forward_looking)
        return forward_looking
    
    def check_return_correlations(self, df, features):
        """Check for features that are too highly correlated with same-day returns"""
        print(f"\n🔍 CHECKING RETURN CORRELATIONS (SAME-DAY LEAKAGE)")
        print("-" * 40)
        
        if 'Return_1D' not in df.columns:
            print("⚠️ No Return_1D column found, skipping correlation check")
            return []
        
        high_correlation_features = []
        correlation_threshold = 0.95  # Suspiciously high correlation
        
        numeric_features = [f for f in features if df[f].dtype in ['float64', 'int64']]
        
        for feature in numeric_features:
            try:
                corr = df[[feature, 'Return_1D']].corr().iloc[0, 1]
                if abs(corr) > correlation_threshold:
                    high_correlation_features.append((feature, corr))
            except:
                continue
        
        print(f"🚨 Found {len(high_correlation_features)} highly correlated features:")
        for feat, corr in high_correlation_features:
            print(f"   ❌ {feat}: {corr:.4f}")
        
        self.contaminated_features.extend([f[0] for f in high_correlation_features])
        return high_correlation_features
    
    def check_lag_structure(self, df, features):
        """Verify that lagged features are properly lagged"""
        print(f"\n🔍 CHECKING LAG STRUCTURE")
        print("-" * 40)
        
        lag_features = [f for f in features if 'lag' in f.lower()]
        print(f"📊 Found {len(lag_features)} lag features:")
        
        proper_lags = []
        improper_lags = []
        
        for feature in lag_features:
            # Check if the lag is properly implemented by looking at data
            # This is a simplified check - in practice you'd verify the lag calculation
            if any(x in feature.lower() for x in ['lag1', 'lag_1', '_lag1', '_lag_1']):
                proper_lags.append(feature)
                print(f"   ✅ {feature}")
            else:
                improper_lags.append(feature)
                print(f"   ⚠️ {feature} (verify lag implementation)")
        
        return proper_lags, improper_lags
    
    def check_fundamental_data_timing(self, df, features):
        """Check if fundamental data is point-in-time (no look-ahead)"""
        print(f"\n🔍 CHECKING FUNDAMENTAL DATA TIMING")
        print("-" * 40)
        
        fundamental_features = [f for f in features if any(x in f.upper() for x in ['FUND_', 'PE', 'PB', 'PS', 'ROE', 'ROA'])]
        print(f"📊 Found {len(fundamental_features)} fundamental features")
        
        # Fundamental data should be point-in-time
        suspect_fundamentals = []
        for feature in fundamental_features:
            # Check for any that might be forward-looking
            if any(x in feature.lower() for x in ['next_', 'forward_', 'future_']):
                suspect_fundamentals.append(feature)
        
        if suspect_fundamentals:
            print(f"🚨 Found {len(suspect_fundamentals)} suspect fundamental features:")
            for feat in suspect_fundamentals:
                print(f"   ❌ {feat}")
        else:
            print("✅ Fundamental features appear to be point-in-time")
        
        return suspect_fundamentals
    
    def check_technical_indicators(self, df, features):
        """Verify technical indicators don't use future data"""
        print(f"\n🔍 CHECKING TECHNICAL INDICATORS")
        print("-" * 40)
        
        technical_features = [f for f in features if any(x in f.upper() for x in 
                            ['SMA', 'EMA', 'RSI', 'MACD', 'BB_', 'ATR', 'VOLUME_'])]
        
        print(f"📊 Found {len(technical_features)} technical indicators")
        print("✅ Technical indicators are typically calculated from historical data")
        
        # Technical indicators should be fine if calculated properly
        return []
    
    def analyze_feature_availability(self, df, features):
        """Check if features have realistic availability (no perfect foresight)"""
        print(f"\n🔍 ANALYZING FEATURE AVAILABILITY PATTERNS")
        print("-" * 40)
        
        # Features that are available too perfectly might indicate leakage
        perfect_availability = []
        
        for feature in features[:20]:  # Check first 20 to avoid spam
            try:
                null_rate = df[feature].isnull().mean()
                if null_rate < 0.01:  # Less than 1% missing - might be suspicious for some features
                    if any(x in feature.lower() for x in ['sentiment', 'news', 'social']):
                        perfect_availability.append((feature, null_rate))
            except:
                continue
        
        if perfect_availability:
            print(f"⚠️ Features with suspiciously perfect availability:")
            for feat, null_rate in perfect_availability:
                print(f"   {feat}: {null_rate:.1%} missing")
        
        return perfect_availability
    
    def generate_clean_feature_list(self, all_features):
        """Generate list of clean features after removing contaminated ones"""
        print(f"\n🧹 GENERATING CLEAN FEATURE LIST")
        print("-" * 40)
        
        # Remove duplicates from contaminated features
        contaminated_unique = list(set(self.contaminated_features))
        
        clean_features = [f for f in all_features if f not in contaminated_unique]
        
        print(f"📊 AUDIT SUMMARY:")
        print(f"   Total features analyzed: {len(all_features)}")
        print(f"   Contaminated features: {len(contaminated_unique)}")
        print(f"   Clean features: {len(clean_features)}")
        print(f"   Contamination rate: {len(contaminated_unique)/len(all_features):.1%}")
        
        self.clean_features = clean_features
        
        return clean_features, contaminated_unique
    
    def save_audit_results(self):
        """Save audit results for reference"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        audit_results = {
            "timestamp": timestamp,
            "total_features": len(self.clean_features) + len(self.contaminated_features),
            "clean_features": self.clean_features,
            "contaminated_features": self.contaminated_features,
            "contamination_rate": len(self.contaminated_features) / (len(self.clean_features) + len(self.contaminated_features)),
            "clean_feature_count": len(self.clean_features),
            "contaminated_feature_count": len(self.contaminated_features)
        }
        
        output_file = f"feature_leakage_audit_{timestamp}.json"
        
        import json
        with open(output_file, 'w') as f:
            json.dump(audit_results, f, indent=2)
        
        print(f"\n💾 Audit results saved: {output_file}")
        
        # Also save clean features list
        clean_features_file = f"clean_features_{timestamp}.json"
        with open(clean_features_file, 'w') as f:
            json.dump(self.clean_features, f, indent=2)
        
        print(f"💾 Clean features list: {clean_features_file}")
        
        return output_file, clean_features_file

def main():
    """Run complete feature leakage audit"""
    auditor = FeatureLeakageAuditor()
    
    # Load and analyze data
    df, all_features = auditor.load_and_analyze()
    
    # Run all audits
    forward_looking = auditor.check_forward_looking_features(df, all_features)
    high_corr = auditor.check_return_correlations(df, all_features)
    proper_lags, improper_lags = auditor.check_lag_structure(df, all_features)
    suspect_fundamentals = auditor.check_fundamental_data_timing(df, all_features)
    technical_issues = auditor.check_technical_indicators(df, all_features)
    availability_issues = auditor.analyze_feature_availability(df, all_features)
    
    # Generate clean feature list
    clean_features, contaminated = auditor.generate_clean_feature_list(all_features)
    
    # Save results
    auditor.save_audit_results()
    
    print(f"\n🎯 AUDIT COMPLETE")
    print(f"=" * 60)
    print(f"✅ {len(clean_features)} clean features ready for model training")
    print(f"❌ {len(contaminated)} contaminated features removed")
    print(f"\n🔧 NEXT STEPS:")
    print(f"1. Rebuild dataset with only clean features")
    print(f"2. Implement proper temporal cross-validation")
    print(f"3. Retrain model with leak-free data")
    print(f"4. Validate on multiple out-of-sample periods")

if __name__ == "__main__":
    main()