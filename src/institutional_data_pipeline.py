#!/usr/bin/env python3
"""
INSTITUTIONAL DATA PIPELINE
===========================
Fresh start - Clean, institutional-grade data preprocessing pipeline
for 24-ticker daily equity alpha model
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class InstitutionalDataPipeline:
    """Institutional-grade data preprocessing pipeline"""
    
    def __init__(self):
        print("🏛️ INSTITUTIONAL DATA PIPELINE")
        print("=" * 50)
        
        # Data specifications
        self.target_universe_size = 24
        self.target_column = 'target_1d'  # Focus on 1-day returns
        self.min_history_days = 500  # Minimum history per ticker
        
        # Feature categories for institutional model
        self.core_features = {
            # Price momentum features (most reliable)
            'momentum': [
                'return_5d_lag1', 'return_20d_lag1', 'vol_20d_lag1'
            ],
            
            # Market microstructure
            'microstructure': [
                'Volume_Ratio', 'Volume_ZScore'
            ],
            
            # Technical indicators (conservative selection)
            'technical': [
                'RSI_14', 'MACD_Signal'
            ],
            
            # Fundamental ranks (when available)
            'fundamental': [
                'RANK_PE', 'RANK_PB', 'RANK_ROE'
            ],
            
            # Market regime
            'market_regime': [
                'VIX', 'VIX_Spike', 'Treasury_10Y'
            ],
            
            # Alpha signals
            'alpha_signals': [
                'ml_pos', 'ml_neg', 'alpha_1d'
            ]
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            'max_missing_pct': 20.0,  # Max 20% missing for core features
            'min_observations': 1000,  # Min observations per feature
            'max_extreme_pct': 5.0,   # Max 5% extreme values
            'min_variance': 1e-8      # Minimum variance threshold
        }
        
    def load_and_validate_data(self) -> pd.DataFrame:
        """Load and perform basic validation"""
        print("\n📊 LOADING & VALIDATING RAW DATA")
        print("-" * 40)
        
        # Load training data
        df = pd.read_parquet("../artifacts/ds_train.parquet")
        df['Date'] = pd.to_datetime(df['Date'])
        
        print(f"   📄 Raw data: {len(df):,} rows, {len(df.columns)} columns")
        print(f"   📅 Period: {df['Date'].min()} to {df['Date'].max()}")
        print(f"   🏢 Tickers: {df['Ticker'].nunique()}")
        
        # Validate universe size
        actual_tickers = df['Ticker'].nunique()
        if actual_tickers != self.target_universe_size:
            print(f"   ⚠️ Universe size mismatch: {actual_tickers} vs expected {self.target_universe_size}")
        else:
            print(f"   ✅ Universe size correct: {actual_tickers} tickers")
        
        # Validate target column
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found")
        
        target_stats = df[self.target_column].describe()
        print(f"   🎯 Target stats: mean={target_stats['mean']:.6f}, std={target_stats['std']:.4f}")
        
        # Check for realistic daily returns
        if 0.01 <= target_stats['std'] <= 0.10:  # 1% to 10% daily volatility
            print(f"   ✅ Target volatility realistic for daily returns")
        else:
            print(f"   ⚠️ Unusual target volatility: {target_stats['std']:.4f}")
        
        return df
    
    def select_institutional_features(self, df: pd.DataFrame) -> tuple:
        """Select and validate features for institutional model"""
        print("\n🎯 INSTITUTIONAL FEATURE SELECTION")
        print("-" * 40)
        
        selected_features = []
        feature_quality_report = {}
        
        # Process each feature category
        for category, feature_list in self.core_features.items():
            print(f"\n   📊 {category.upper()} FEATURES:")
            category_features = []
            
            for feature in feature_list:
                if feature in df.columns:
                    quality_score = self._assess_feature_quality(df, feature)
                    feature_quality_report[feature] = quality_score
                    
                    if quality_score['passes_institutional_standards']:
                        category_features.append(feature)
                        selected_features.append(feature)
                        print(f"     ✅ {feature}: APPROVED")
                        print(f"        Missing: {quality_score['missing_pct']:.1f}%")
                        print(f"        Observations: {quality_score['valid_observations']:,}")
                    else:
                        print(f"     ❌ {feature}: REJECTED")
                        for issue in quality_score['issues']:
                            print(f"        Issue: {issue}")
                else:
                    print(f"     ❌ {feature}: NOT FOUND")
            
            print(f"   📋 {category} approved: {len(category_features)}/{len(feature_list)}")
        
        print(f"\n🎯 FEATURE SELECTION SUMMARY:")
        print(f"   Total approved: {len(selected_features)}")
        print(f"   Categories with features: {sum(1 for cat, feats in self.core_features.items() if any(f in selected_features for f in feats))}")
        
        if len(selected_features) < 10:
            print(f"   ⚠️ Limited feature set: {len(selected_features)} features")
        elif len(selected_features) > 30:
            print(f"   ⚠️ Large feature set: {len(selected_features)} features")
        else:
            print(f"   ✅ Appropriate feature set: {len(selected_features)} features")
        
        return selected_features, feature_quality_report
    
    def _assess_feature_quality(self, df: pd.DataFrame, feature: str) -> dict:
        """Assess feature quality against institutional standards"""
        feature_data = df[feature]
        
        # Basic statistics
        missing_pct = (feature_data.isnull().sum() / len(feature_data)) * 100
        valid_data = feature_data.dropna()
        valid_observations = len(valid_data)
        
        quality_issues = []
        
        # Check missing data
        if missing_pct > self.quality_thresholds['max_missing_pct']:
            quality_issues.append(f"High missing data: {missing_pct:.1f}%")
        
        # Check sufficient observations
        if valid_observations < self.quality_thresholds['min_observations']:
            quality_issues.append(f"Insufficient observations: {valid_observations}")
        
        # Check variance (not constant)
        if len(valid_data) > 0:
            if valid_data.var() < self.quality_thresholds['min_variance']:
                quality_issues.append("Near-constant feature (low variance)")
            
            # Check for extreme values
            if len(valid_data) > 100:
                q1, q3 = valid_data.quantile([0.25, 0.75])
                iqr = q3 - q1
                
                if iqr > 0:
                    extreme_mask = (valid_data < q1 - 5*iqr) | (valid_data > q3 + 5*iqr)
                    extreme_pct = extreme_mask.sum() / len(valid_data) * 100
                    
                    if extreme_pct > self.quality_thresholds['max_extreme_pct']:
                        quality_issues.append(f"High extreme values: {extreme_pct:.1f}%")
        
        passes_standards = len(quality_issues) == 0
        
        return {
            'missing_pct': missing_pct,
            'valid_observations': valid_observations,
            'issues': quality_issues,
            'passes_institutional_standards': passes_standards
        }
    
    def create_leak_free_alignment(self, df: pd.DataFrame, selected_features: list) -> pd.DataFrame:
        """Create leak-free temporal alignment"""
        print("\n⏰ LEAK-FREE TEMPORAL ALIGNMENT")
        print("-" * 40)
        
        # Sort by ticker and date
        df = df.sort_values(['Ticker', 'Date']).copy()
        
        print(f"   🔧 Creating T-1 → T+1 alignment...")
        
        # Create lagged features (T-1)
        aligned_features = []
        for feature in selected_features:
            if feature in df.columns:
                lagged_feature = f"{feature}_t1"
                df[lagged_feature] = df.groupby('Ticker')[feature].shift(1)
                aligned_features.append(lagged_feature)
        
        # Create forward target (T+1)
        df['target_forward'] = df.groupby('Ticker')[self.target_column].shift(-1)
        
        # Keep only aligned data
        alignment_columns = ['Date', 'Ticker'] + aligned_features + ['target_forward']
        df_aligned = df[alignment_columns].dropna()
        
        alignment_retention = len(df_aligned) / len(df) * 100
        print(f"   📊 Alignment retention: {alignment_retention:.1f}% ({len(df_aligned):,} rows)")
        print(f"   🎯 Features aligned: {len(aligned_features)}")
        print(f"   ⏰ Time separation: T-1 features → T+1 target (2-day gap)")
        
        # Validation: Check alignment integrity
        if alignment_retention < 30:
            print(f"   ⚠️ Low retention rate: {alignment_retention:.1f}%")
        else:
            print(f"   ✅ Acceptable retention rate")
        
        return df_aligned
    
    def split_train_validation_test(self, df_aligned: pd.DataFrame) -> tuple:
        """Create institutional train/validation/test splits"""
        print("\n📊 INSTITUTIONAL DATA SPLITS")
        print("-" * 40)
        
        # Chronological splits with embargoes
        total_samples = len(df_aligned)
        
        # 60% training, 20% validation, 20% test
        train_end_idx = int(total_samples * 0.60)
        val_end_idx = int(total_samples * 0.80)
        
        # Apply 5-day embargo between splits
        embargo_days = 5
        
        # Training set
        train_df = df_aligned.iloc[:train_end_idx]
        train_end_date = train_df['Date'].max()
        
        # Validation set with embargo
        val_start_date = train_end_date + timedelta(days=embargo_days)
        val_df = df_aligned[(df_aligned['Date'] >= val_start_date)].iloc[:val_end_idx-train_end_idx]
        val_end_date = val_df['Date'].max() if len(val_df) > 0 else train_end_date
        
        # Test set with embargo
        test_start_date = val_end_date + timedelta(days=embargo_days)
        test_df = df_aligned[df_aligned['Date'] >= test_start_date]
        
        print(f"   📈 Training: {len(train_df):,} samples")
        print(f"      Period: {train_df['Date'].min()} to {train_df['Date'].max()}")
        
        print(f"   🔍 Validation: {len(val_df):,} samples")
        if len(val_df) > 0:
            print(f"      Period: {val_df['Date'].min()} to {val_df['Date'].max()}")
        
        print(f"   🧪 Test: {len(test_df):,} samples")
        if len(test_df) > 0:
            print(f"      Period: {test_df['Date'].min()} to {test_df['Date'].max()}")
        
        print(f"   🚧 Embargo: {embargo_days} days between splits")
        
        # Validate splits
        min_samples_per_split = 1000
        if len(train_df) < min_samples_per_split:
            print(f"   ⚠️ Training set too small: {len(train_df)}")
        if len(val_df) < min_samples_per_split * 0.5:
            print(f"   ⚠️ Validation set too small: {len(val_df)}")
        if len(test_df) < min_samples_per_split * 0.5:
            print(f"   ⚠️ Test set too small: {len(test_df)}")
        
        if len(train_df) >= min_samples_per_split and len(val_df) >= 500 and len(test_df) >= 500:
            print(f"   ✅ All splits have adequate sample sizes")
        
        return train_df, val_df, test_df
    
    def validate_data_integrity(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        """Validate data integrity across splits"""
        print("\n🔍 DATA INTEGRITY VALIDATION")
        print("-" * 40)
        
        integrity_results = {}
        
        # Check for data leakage between splits
        train_max_date = train_df['Date'].max()
        val_min_date = val_df['Date'].min() if len(val_df) > 0 else train_max_date
        test_min_date = test_df['Date'].min() if len(test_df) > 0 else val_min_date
        
        train_val_gap = (val_min_date - train_max_date).days if len(val_df) > 0 else 0
        val_test_gap = (test_min_date - val_min_date).days if len(test_df) > 0 and len(val_df) > 0 else 0
        
        print(f"   📅 Train → Val gap: {train_val_gap} days")
        print(f"   📅 Val → Test gap: {val_test_gap} days")
        
        if train_val_gap >= 5 and val_test_gap >= 5:
            print(f"   ✅ Adequate temporal separation")
            integrity_results['temporal_separation'] = True
        else:
            print(f"   ⚠️ Insufficient temporal separation")
            integrity_results['temporal_separation'] = False
        
        # Check target distributions
        target_col = 'target_forward'
        
        train_target = train_df[target_col]
        val_target = val_df[target_col] if len(val_df) > 0 else pd.Series()
        test_target = test_df[target_col] if len(test_df) > 0 else pd.Series()
        
        print(f"   🎯 Target distribution analysis:")
        print(f"      Training: mean={train_target.mean():.6f}, std={train_target.std():.4f}")
        
        if len(val_target) > 0:
            print(f"      Validation: mean={val_target.mean():.6f}, std={val_target.std():.4f}")
        
        if len(test_target) > 0:
            print(f"      Test: mean={test_target.mean():.6f}, std={test_target.std():.4f}")
        
        # Check for distribution shifts
        target_distributions_stable = True
        
        if len(val_target) > 100:
            std_ratio = val_target.std() / train_target.std()
            if not (0.5 <= std_ratio <= 2.0):  # Allow 2x variation
                print(f"   ⚠️ Validation std deviation ratio: {std_ratio:.2f}")
                target_distributions_stable = False
        
        if len(test_target) > 100:
            std_ratio = test_target.std() / train_target.std()
            if not (0.5 <= std_ratio <= 2.0):
                print(f"   ⚠️ Test std deviation ratio: {std_ratio:.2f}")
                target_distributions_stable = False
        
        if target_distributions_stable:
            print(f"   ✅ Target distributions stable across splits")
        
        integrity_results['target_distributions_stable'] = target_distributions_stable
        
        # Check for duplicates across splits
        train_tickers_dates = set(zip(train_df['Ticker'], train_df['Date']))
        val_tickers_dates = set(zip(val_df['Ticker'], val_df['Date'])) if len(val_df) > 0 else set()
        test_tickers_dates = set(zip(test_df['Ticker'], test_df['Date'])) if len(test_df) > 0 else set()
        
        overlaps = []
        if train_tickers_dates & val_tickers_dates:
            overlaps.append("train-val")
        if train_tickers_dates & test_tickers_dates:
            overlaps.append("train-test")
        if val_tickers_dates & test_tickers_dates:
            overlaps.append("val-test")
        
        if overlaps:
            print(f"   ⚠️ Data overlaps found: {overlaps}")
            integrity_results['no_overlaps'] = False
        else:
            print(f"   ✅ No data overlaps between splits")
            integrity_results['no_overlaps'] = True
        
        # Overall integrity assessment
        integrity_results['overall_pass'] = all([
            integrity_results['temporal_separation'],
            integrity_results['target_distributions_stable'],
            integrity_results['no_overlaps']
        ])
        
        return integrity_results
    
    def save_processed_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                           test_df: pd.DataFrame, selected_features: list) -> None:
        """Save processed data for model training"""
        print("\n💾 SAVING PROCESSED DATA")
        print("-" * 40)
        
        # Create processed data directory
        processed_dir = Path("../artifacts/processed")
        processed_dir.mkdir(exist_ok=True)
        
        # Save splits
        train_df.to_parquet(processed_dir / "train_institutional.parquet", index=False)
        if len(val_df) > 0:
            val_df.to_parquet(processed_dir / "validation_institutional.parquet", index=False)
        if len(test_df) > 0:
            test_df.to_parquet(processed_dir / "test_institutional.parquet", index=False)
        
        # Save feature list
        feature_config = {
            'selected_features': selected_features,
            'target_column': 'target_forward',
            'processing_date': datetime.now().isoformat(),
            'data_specs': {
                'universe_size': self.target_universe_size,
                'min_history_days': self.min_history_days,
                'target_column_original': self.target_column
            }
        }
        
        import json
        with open(processed_dir / "feature_config.json", 'w') as f:
            json.dump(feature_config, f, indent=2)
        
        print(f"   ✅ Training data: {len(train_df):,} rows")
        print(f"   ✅ Validation data: {len(val_df):,} rows")
        print(f"   ✅ Test data: {len(test_df):,} rows")
        print(f"   ✅ Feature config saved")
        print(f"   📁 Location: {processed_dir}")
    
    def run_complete_pipeline(self) -> dict:
        """Run the complete institutional data pipeline"""
        print("🚀 RUNNING INSTITUTIONAL DATA PIPELINE")
        print("=" * 60)
        
        try:
            # 1. Load and validate raw data
            df_raw = self.load_and_validate_data()
            
            # 2. Select institutional features
            selected_features, feature_quality = self.select_institutional_features(df_raw)
            
            if len(selected_features) < 5:
                raise ValueError(f"Insufficient quality features: {len(selected_features)}")
            
            # 3. Create leak-free alignment
            df_aligned = self.create_leak_free_alignment(df_raw, selected_features)
            
            # 4. Create institutional splits
            train_df, val_df, test_df = self.split_train_validation_test(df_aligned)
            
            # 5. Validate data integrity
            integrity_results = self.validate_data_integrity(train_df, val_df, test_df)
            
            if not integrity_results['overall_pass']:
                print(f"   ⚠️ Data integrity issues detected")
            
            # 6. Save processed data
            self.save_processed_data(train_df, val_df, test_df, selected_features)
            
            # 7. Generate pipeline summary
            pipeline_summary = {
                'success': True,
                'raw_data_shape': df_raw.shape,
                'aligned_data_shape': df_aligned.shape,
                'selected_features_count': len(selected_features),
                'splits': {
                    'train_samples': len(train_df),
                    'validation_samples': len(val_df),
                    'test_samples': len(test_df)
                },
                'data_integrity': integrity_results,
                'feature_quality': feature_quality
            }
            
            print(f"\n🎉 PIPELINE COMPLETION SUMMARY")
            print("=" * 50)
            print(f"✅ Raw data processed: {df_raw.shape[0]:,} rows")
            print(f"✅ Features selected: {len(selected_features)}")
            print(f"✅ Data aligned: {df_aligned.shape[0]:,} rows")
            print(f"✅ Institutional splits created")
            print(f"✅ Data integrity: {'PASS' if integrity_results['overall_pass'] else 'WARNING'}")
            print(f"📊 Ready for institutional model training")
            
            return pipeline_summary
            
        except Exception as e:
            print(f"❌ Pipeline failed: {str(e)}")
            return {'success': False, 'error': str(e)}

def main():
    """Main pipeline execution"""
    pipeline = InstitutionalDataPipeline()
    results = pipeline.run_complete_pipeline()
    
    if results['success']:
        print(f"\n🎯 INSTITUTIONAL DATA PIPELINE: ✅ SUCCESS")
    else:
        print(f"\n🚨 INSTITUTIONAL DATA PIPELINE: ❌ FAILED")
    
    return results

if __name__ == "__main__":
    results = main()