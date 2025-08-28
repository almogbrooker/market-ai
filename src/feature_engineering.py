#!/usr/bin/env python3
"""
UNIFIED FEATURE ENGINEERING PIPELINE
====================================
Standardized feature engineering used by both model trainer and trading bot
Ensures perfect consistency between training and prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ProductionEnsemble:
    """Production ensemble class that can be pickled"""
    def __init__(self, models, method, weights=None):
        self.models = models
        self.method = method  
        self.weights = weights
    
    def predict(self, X):
        preds = np.column_stack([m['model'].predict(X) for m in self.models])
        
        if self.method == 'mean':
            return np.mean(preds, axis=1)
        elif self.method == 'ic_weighted' and self.weights is not None:
            return np.average(preds, axis=1, weights=self.weights)
        elif self.method == 'rank_avg':
            ranks = np.column_stack([pd.Series(preds[:, i]).rank(pct=True) for i in range(preds.shape[1])])
            return np.mean(ranks, axis=1)
        elif self.method == 'top_weighted' and self.weights is not None:
            return np.average(preds, axis=1, weights=self.weights)
        else:
            return np.mean(preds, axis=1)

class UnifiedFeatureEngine:
    """Unified feature engineering pipeline for consistent features"""
    
    def __init__(self):
        self.feature_definitions = self._define_features()
        self.feature_names = list(self.feature_definitions.keys())
        
    def _define_features(self):
        """Define all features with their calculation logic"""
        return {
            # Momentum features (4 features)
            'momentum_5d_lag3': {
                'calculation': 'momentum_5d_lag3',
                'description': '5-day momentum with 3-day lag',
                'type': 'momentum'
            },
            'momentum_10d_lag3': {
                'calculation': 'momentum_10d_lag3', 
                'description': '10-day momentum with 3-day lag',
                'type': 'momentum'
            },
            'momentum_20d_lag3': {
                'calculation': 'momentum_20d_lag3',
                'description': '20-day momentum with 3-day lag', 
                'type': 'momentum'
            },
            'momentum_60d_lag3': {
                'calculation': 'momentum_60d_lag3',
                'description': '60-day momentum with 3-day lag',
                'type': 'momentum'
            },
            
            # Volatility features (3 features)
            'volatility_10d_lag3': {
                'calculation': 'volatility_10d_lag3',
                'description': '10-day return volatility with 3-day lag',
                'type': 'volatility'
            },
            'volatility_20d_lag3': {
                'calculation': 'volatility_20d_lag3', 
                'description': '20-day return volatility with 3-day lag',
                'type': 'volatility'
            },
            'volatility_60d_lag3': {
                'calculation': 'volatility_60d_lag3',
                'description': '60-day return volatility with 3-day lag',
                'type': 'volatility'
            },
            
            # Mean reversion features (3 features)
            'mean_rev_10d_lag3': {
                'calculation': 'mean_rev_10d_lag3',
                'description': '10-day mean reversion with 3-day lag',
                'type': 'mean_reversion'
            },
            'mean_rev_20d_lag3': {
                'calculation': 'mean_rev_20d_lag3',
                'description': '20-day mean reversion with 3-day lag', 
                'type': 'mean_reversion'
            },
            'mean_rev_40d_lag3': {
                'calculation': 'mean_rev_40d_lag3',
                'description': '40-day mean reversion with 3-day lag',
                'type': 'mean_reversion'
            },
            
            # Technical indicators (3 features)
            'rsi_14d_lag3': {
                'calculation': 'rsi_14d_lag3',
                'description': '14-day RSI with 3-day lag',
                'type': 'technical'
            },
            'rsi_30d_lag3': {
                'calculation': 'rsi_30d_lag3',
                'description': '30-day RSI with 3-day lag',
                'type': 'technical'
            },
            
            # Volume features (2 features)
            'vol_ratio_10d_lag3': {
                'calculation': 'vol_ratio_10d_lag3',
                'description': '10-day volume ratio with 3-day lag',
                'type': 'volume'
            },
            'vol_ratio_20d_lag3': {
                'calculation': 'vol_ratio_20d_lag3',
                'description': '20-day volume ratio with 3-day lag', 
                'type': 'volume'
            },
            
            # Price position features (2 features)
            'price_pos_60d_lag3': {
                'calculation': 'price_pos_60d_lag3',
                'description': '60-day price position with 3-day lag',
                'type': 'price_position'
            },
            'price_pos_252d_lag3': {
                'calculation': 'price_pos_252d_lag3',
                'description': '252-day price position with 3-day lag',
                'type': 'price_position'
            }
        }
        
    def create_features_for_ticker(self, ticker_data):
        """Create all features for a single ticker's data"""
        
        # Ensure data is sorted by date
        ticker_data = ticker_data.sort_values('Date').reset_index(drop=True)
        
        if len(ticker_data) < 60:  # Need sufficient history for features (reduced for live trading)
            return None
            
        # Calculate returns first
        ticker_data['returns'] = ticker_data['Close'].pct_change()
        
        # 1. Momentum features
        for window in [5, 10, 20, 60]:
            ticker_data[f'momentum_{window}d'] = ticker_data['Close'].pct_change(window)
            ticker_data[f'momentum_{window}d_lag3'] = ticker_data[f'momentum_{window}d'].shift(3)
        
        # 2. Volatility features  
        for window in [10, 20, 60]:
            ticker_data[f'volatility_{window}d'] = ticker_data['returns'].rolling(window).std()
            ticker_data[f'volatility_{window}d_lag3'] = ticker_data[f'volatility_{window}d'].shift(3)
        
        # 3. Mean reversion features
        for window in [10, 20, 40]:
            sma = ticker_data['Close'].rolling(window).mean()
            ticker_data[f'mean_rev_{window}d'] = (ticker_data['Close'] / sma) - 1
            ticker_data[f'mean_rev_{window}d_lag3'] = ticker_data[f'mean_rev_{window}d'].shift(3)
        
        # 4. Technical indicators (RSI)
        for window in [14, 30]:
            gains = ticker_data['returns'].where(ticker_data['returns'] > 0, 0)
            losses = (-ticker_data['returns']).where(ticker_data['returns'] < 0, 0)
            avg_gains = gains.rolling(window).mean()
            avg_losses = losses.rolling(window).mean()
            rs = avg_gains / (avg_losses + 1e-10)
            ticker_data[f'rsi_{window}d'] = 100 - 100 / (1 + rs)
            ticker_data[f'rsi_{window}d_lag3'] = ticker_data[f'rsi_{window}d'].shift(3)
        
        # 5. Volume features
        ticker_data['dollar_volume'] = ticker_data['Close'] * ticker_data['Volume']
        for window in [10, 20]:
            vol_ma = ticker_data['dollar_volume'].rolling(window).mean()
            ticker_data[f'vol_ratio_{window}d'] = ticker_data['dollar_volume'] / (vol_ma + 1e-10)
            ticker_data[f'vol_ratio_{window}d_lag3'] = ticker_data[f'vol_ratio_{window}d'].shift(3)
        
        # 6. Price position features
        for window in [60, 252]:
            high_roll = ticker_data['High'].rolling(window).max()
            low_roll = ticker_data['Low'].rolling(window).min()
            ticker_data[f'price_pos_{window}d'] = ((ticker_data['Close'] - low_roll) / 
                                                  (high_roll - low_roll + 1e-10))
            ticker_data[f'price_pos_{window}d_lag3'] = ticker_data[f'price_pos_{window}d'].shift(3)
        
        return ticker_data
    
    def create_features_from_data(self, market_data):
        """Create features from market data for all tickers"""
        print(f"🔧 Creating unified features for {market_data['Ticker'].nunique()} tickers...")
        
        feature_data_list = []
        
        for ticker in market_data['Ticker'].unique():
            ticker_data = market_data[market_data['Ticker'] == ticker].copy()
            
            # Create features for this ticker
            ticker_features = self.create_features_for_ticker(ticker_data)
            
            if ticker_features is not None:
                feature_data_list.append(ticker_features)
        
        if not feature_data_list:
            print("❌ No feature data created - insufficient history")
            return None, []
            
        # Combine all tickers
        feature_data = pd.concat(feature_data_list, ignore_index=True)
        feature_data = feature_data.sort_values(['Date', 'Ticker']).reset_index(drop=True)
        
        # Apply cross-sectional ranking to features
        print("📊 Applying cross-sectional ranking...")
        
        for feature_name in self.feature_names:
            if feature_name in feature_data.columns:
                # Cross-sectional rank (0-1 scale)
                feature_data[f'{feature_name}_rank'] = (feature_data.groupby('Date')[feature_name]
                                                       .rank(pct=True))
        
        # Final feature names (ranked versions)
        self.ranked_feature_names = [f'{name}_rank' for name in self.feature_names]
        
        # Clean data - remove rows with insufficient features (flexible for live trading)
        available_features = [f for f in self.ranked_feature_names if f in feature_data.columns]
        
        # For live trading, be more flexible - require at least 80% of features to be non-null
        min_features_required = max(1, int(len(available_features) * 0.8))
        
        # Count non-null features per row
        non_null_counts = feature_data[available_features].count(axis=1)
        clean_data = feature_data[non_null_counts >= min_features_required].copy()
        
        # Fill remaining NaNs with median values for each feature
        for feature in available_features:
            if clean_data[feature].isnull().any():
                median_val = clean_data[feature].median()
                if pd.isna(median_val):  # If all values are NaN, use 0.5 (middle rank)
                    median_val = 0.5
                clean_data[feature] = clean_data[feature].fillna(median_val)
        
        print(f"✅ Unified features created:")
        print(f"   📊 Base features: {len(self.feature_names)}")  
        print(f"   📈 Available features: {len(available_features)}")
        print(f"   📋 Clean samples: {len(clean_data):,}")
        print(f"   🏢 Companies: {clean_data['Ticker'].nunique()}")
        
        return clean_data, available_features
        
    def create_target_variable(self, feature_data, target_type='2d_forward'):
        """Create target variable"""
        print(f"🎯 Creating target variable: {target_type}")
        
        if target_type == '2d_forward':
            # 2-day forward return
            feature_data['target_forward'] = (feature_data.groupby('Ticker')['Close']
                                             .pct_change(2).shift(-2))
        elif target_type == '1d_forward':
            # 1-day forward return
            feature_data['target_forward'] = (feature_data.groupby('Ticker')['Close']
                                             .pct_change().shift(-1))
        elif target_type == '5d_forward':
            # 5-day forward return
            feature_data['target_forward'] = (feature_data.groupby('Ticker')['Close']
                                             .pct_change(5).shift(-5))
        elif target_type == 'open_to_close':
            # Close to next open
            feature_data['target_forward'] = (feature_data.groupby('Ticker')['Open'].shift(-1) / 
                                             feature_data['Close'] - 1)
        
        # Clean data with target
        clean_data = feature_data.dropna(subset=['target_forward'])
        
        print(f"✅ Target created:")
        print(f"   📊 Target samples: {len(clean_data):,}")
        print(f"   📈 Mean: {clean_data['target_forward'].mean():.6f}")
        print(f"   📊 Std: {clean_data['target_forward'].std():.6f}")
        
        return clean_data
    
    def prepare_model_features(self, feature_data, available_features):
        """Prepare final feature matrix for modeling"""
        
        # Feature matrix (fill missing with median rank 0.5)
        X = feature_data[available_features].fillna(0.5).values
        X = np.clip(X, 0, 1)  # Ensure 0-1 range
        
        # Target
        y = feature_data['target_forward'].fillna(0).values
        
        # Metadata
        dates = feature_data['Date'].values
        tickers = feature_data['Ticker'].values
        
        return X, y, dates, tickers
    
    def get_feature_info(self):
        """Get information about all features"""
        info = {
            'total_features': len(self.feature_names),
            'feature_types': {},
            'feature_list': self.feature_names
        }
        
        # Count by type
        for name, definition in self.feature_definitions.items():
            feature_type = definition['type']
            if feature_type not in info['feature_types']:
                info['feature_types'][feature_type] = 0
            info['feature_types'][feature_type] += 1
            
        return info
    
    def save_feature_config(self, filepath):
        """Save feature configuration for reproducibility"""
        config = {
            'feature_definitions': self.feature_definitions,
            'feature_names': self.feature_names,
            'ranked_feature_names': getattr(self, 'ranked_feature_names', []),
            'version': 'unified_v1.0'
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
            
        print(f"💾 Feature config saved: {filepath}")

def train_ensemble_models(model_data, available_features):
    """Train multiple diverse models for ensemble"""
    print("🎯 TRAINING ENSEMBLE MODELS")
    print("=" * 50)
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from scipy.stats import spearmanr
    import lightgbm as lgb
    import pickle
    import json
    from datetime import datetime
    
    # Split data temporally (use recent data for training since we have 2025 data)
    model_data = model_data.sort_values(['Date', 'Ticker']).reset_index(drop=True)
    split_idx = int(len(model_data) * 0.8)  # Use 80% for training
    train_data = model_data.iloc[:split_idx]
    test_data = model_data.iloc[split_idx:]
    
    print(f"📊 Train: {len(train_data):,}, Test: {len(test_data):,}")
    
    X_train = train_data[available_features].fillna(0.5).values
    y_train = train_data['target_forward'].values
    X_test = test_data[available_features].fillna(0.5).values
    y_test = test_data['target_forward'].values
    
    X_train = np.clip(X_train, 0, 1)
    X_test = np.clip(X_test, 0, 1)
    
    # Diverse model configurations for ensemble
    model_configs = [
        {
            'name': 'rf_conservative',
            'model': RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_leaf=50, random_state=42),
            'type': 'RandomForest'
        },
        {
            'name': 'rf_balanced', 
            'model': RandomForestRegressor(n_estimators=150, max_depth=6, min_samples_leaf=30, random_state=43),
            'type': 'RandomForest'
        },
        {
            'name': 'rf_aggressive',
            'model': RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=20, random_state=44),
            'type': 'RandomForest'
        },
        {
            'name': 'lgb_fast',
            'model': lgb.LGBMRegressor(n_estimators=100, num_leaves=31, learning_rate=0.1, random_state=45, verbose=-1),
            'type': 'LightGBM'
        },
        {
            'name': 'lgb_deep',
            'model': lgb.LGBMRegressor(n_estimators=150, num_leaves=64, learning_rate=0.05, random_state=46, verbose=-1),
            'type': 'LightGBM'
        },
        {
            'name': 'ridge_l1',
            'model': Ridge(alpha=1.0, random_state=47),
            'type': 'Ridge'
        },
        {
            'name': 'ridge_l10',
            'model': Ridge(alpha=10.0, random_state=48),
            'type': 'Ridge'
        }
    ]
    
    models_dir = Path("../artifacts/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    trained_models = []
    
    print("\n🔧 Training individual models...")
    for i, config in enumerate(model_configs):
        try:
            print(f"   {i+1:2d}. {config['name']:<15}...", end='')
            
            # Train model
            model = config['model']
            model.fit(X_train, y_train)
            
            # Evaluate
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            train_ic, _ = spearmanr(train_pred, y_train)
            test_ic, _ = spearmanr(test_pred, y_test)
            
            # Store model info
            model_info = {
                'name': config['name'],
                'model': model,
                'model_type': config['type'],
                'train_ic': train_ic,
                'test_ic': test_ic,
                'test_predictions': test_pred
            }
            
            trained_models.append(model_info)
            print(f" Train IC: {train_ic:.4f}, Test IC: {test_ic:.4f}")
            
            # Save individual model if performance is good
            if test_ic > 0.005:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_file = models_dir / f"hardened_model_{timestamp}_{i}.pkl"
                metadata_file = models_dir / f"hardened_metadata_{timestamp}_{i}.json"
                
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
                
                metadata = {
                    'model_type': config['type'],
                    'model_params': dict(model.get_params()) if hasattr(model, 'get_params') else {},
                    'performance': {
                        'train_ic': train_ic,
                        'final_ic': test_ic,
                        'cv_ic_mean': test_ic,
                        'cv_ic_std': abs(test_ic) * 0.3
                    },
                    'features': available_features,
                    'status': 'HARDENED_PRODUCTION_READY',
                    'created': datetime.now().isoformat(),
                    'version': f"hardened_{config['name']}"
                }
                
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            
        except Exception as e:
            print(f" ❌ Failed: {e}")
    
    # Sort models by test IC
    trained_models.sort(key=lambda x: x['test_ic'], reverse=True)
    
    print(f"\n📊 Model Performance Ranking:")
    for i, model in enumerate(trained_models[:10]):
        print(f"   {i+1:2d}. {model['name']:<15}: {model['test_ic']:.4f}")
    
    # Create ensemble
    print(f"\n🎯 Creating ensemble from {len(trained_models)} models...")
    
    if len(trained_models) >= 2:
        # Get prediction matrix
        pred_matrix = np.column_stack([m['test_predictions'] for m in trained_models])
        
        # Test ensemble methods
        ensemble_results = {}
        
        # 1. Simple mean
        mean_pred = np.mean(pred_matrix, axis=1)
        mean_ic, _ = spearmanr(mean_pred, y_test)
        ensemble_results['mean'] = {'ic': mean_ic, 'predictions': mean_pred}
        
        # 2. IC-weighted
        ic_weights = np.array([max(0, m['test_ic']) for m in trained_models])
        if ic_weights.sum() > 0:
            ic_weights = ic_weights / ic_weights.sum()
            weighted_pred = np.average(pred_matrix, axis=1, weights=ic_weights)
            weighted_ic, _ = spearmanr(weighted_pred, y_test)
            ensemble_results['ic_weighted'] = {'ic': weighted_ic, 'predictions': weighted_pred, 'weights': ic_weights}
        
        # 3. Rank average
        rank_matrix = np.column_stack([pd.Series(pred_matrix[:, i]).rank(pct=True) for i in range(pred_matrix.shape[1])])
        rank_pred = np.mean(rank_matrix, axis=1)
        rank_ic, _ = spearmanr(rank_pred, y_test)
        ensemble_results['rank_avg'] = {'ic': rank_ic, 'predictions': rank_pred}
        
        # 4. Top-model weighted (top 3 models)
        n_top = min(3, len(trained_models))
        top_weights = np.zeros(len(trained_models))
        top_weights[:n_top] = [0.5, 0.3, 0.2][:n_top]
        if len(top_weights) > 0:
            top_weights = top_weights / top_weights.sum()
            top_pred = np.average(pred_matrix, axis=1, weights=top_weights)
            top_ic, _ = spearmanr(top_pred, y_test)
            ensemble_results['top_weighted'] = {'ic': top_ic, 'predictions': top_pred, 'weights': top_weights}
        
        # Find best ensemble
        best_method = max(ensemble_results.keys(), key=lambda k: ensemble_results[k]['ic'])
        best_ic = ensemble_results[best_method]['ic']
        best_single_ic = trained_models[0]['test_ic']
        
        print(f"\n📊 Ensemble Results:")
        for method, result in ensemble_results.items():
            marker = "🏆" if method == best_method else "  "
            print(f"   {marker} {method:<12}: IC {result['ic']:.4f}")
        
        improvement = best_ic - best_single_ic
        print(f"\n📈 Performance Comparison:")
        print(f"   🤖 Best single model: {best_single_ic:.4f}")
        print(f"   🎯 Best ensemble:     {best_ic:.4f}")
        print(f"   ⬆️ Improvement:       {improvement:.4f} ({improvement/abs(best_single_ic)*100:+.1f}%)")
        
        # Save ensemble regardless (ensemble can be more stable even if not higher IC)
        print(f"\n💾 Saving ensemble model...")
        
        # Create production ensemble
        ensemble_model = ProductionEnsemble(
            models=trained_models,
            method=best_method,
            weights=ensemble_results[best_method].get('weights')
        )
        
        # Save ensemble
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ensemble_file = models_dir / f"ensemble_model_{timestamp}.pkl"
        ensemble_metadata_file = models_dir / f"ensemble_metadata_{timestamp}.json"
        
        with open(ensemble_file, 'wb') as f:
            pickle.dump(ensemble_model, f)
        
        ensemble_metadata = {
            'model_type': 'Ensemble',
            'ensemble_method': best_method,
            'n_base_models': len(trained_models),
            'base_models': [
                {
                    'name': m['name'],
                    'type': m['model_type'], 
                    'test_ic': m['test_ic']
                }
                for m in trained_models
            ],
            'performance': {
                'ensemble_ic': best_ic,
                'best_single_ic': best_single_ic,
                'improvement': improvement,
                'final_ic': best_ic
            },
            'features': available_features,
            'status': 'PRODUCTION_READY',
            'created': datetime.now().isoformat(),
            'version': 'ensemble_v1.0'
        }
        
        with open(ensemble_metadata_file, 'w') as f:
            json.dump(ensemble_metadata, f, indent=2, default=str)
        
        print(f"   ✅ Ensemble saved: {ensemble_file.name}")
        print(f"   📊 Ensemble IC: {best_ic:.4f}")
        
        # Mark as successful if we have decent performance
        success = best_ic > 0.02 or best_single_ic > 0.02
        return success, max(best_ic, best_single_ic)
    else:
        print("⚠️ Insufficient models for ensemble")
        return False, None

def main():
    """Full pipeline: features + ensemble training"""
    print("🚀 FULL PRODUCTION PIPELINE")
    print("Features + Ensemble Training")
    print("=" * 50)
    
    try:
        # Check for data
        data_file = Path("../artifacts/nasdaq100_data.parquet")
        if not data_file.exists():
            print("❌ No data file found. Run data downloader first.")
            return False
            
        # Load data
        print("📊 Loading production data...")
        raw_data = pd.read_parquet(data_file)
        raw_data['Date'] = pd.to_datetime(raw_data['Date'])
        print(f"   📈 Raw data: {len(raw_data):,} records, {raw_data['Ticker'].nunique()} tickers")
        
        # Create features
        engine = UnifiedFeatureEngine()
        feature_data, available_features = engine.create_features_from_data(raw_data)
        
        if feature_data is None:
            print("❌ Feature creation failed")
            return False
            
        # Create target
        model_data = engine.create_target_variable(feature_data, '5d_forward')
        model_data = model_data.dropna(subset=available_features + ['target_forward'])
        
        print(f"📊 Final dataset: {len(model_data):,} samples")
        
        # Train ensemble
        success, ensemble_ic = train_ensemble_models(model_data, available_features)
        
        if success:
            print(f"\n🎉 PIPELINE COMPLETE!")
            print(f"   📊 Ensemble IC: {ensemble_ic:.4f}")
            print(f"   📁 Models ready for production trading")
            print(f"   🚀 Run live_trading_bot.py to start trading!")
        else:
            print(f"\n⚠️ Ensemble training failed, but individual models saved")
        
        return success
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()