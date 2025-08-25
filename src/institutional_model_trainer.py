#!/usr/bin/env python3
"""
INSTITUTIONAL MODEL TRAINER
===========================
Clean, institutional-grade model training system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class InstitutionalModelTrainer:
    """Institutional-grade model training system"""
    
    def __init__(self):
        print("🏛️ INSTITUTIONAL MODEL TRAINER")
        print("=" * 50)
        
        # Load processed data configuration
        self.processed_dir = Path("../artifacts/processed")
        self.models_dir = Path("../artifacts/models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Model configuration
        self.institutional_thresholds = {
            'min_ic': 0.005,     # Minimum 0.5% IC
            'max_ic': 0.080,     # Maximum 8% IC (avoid overfitting)
            'min_samples': 1000,  # Minimum training samples
            'max_drawdown': 0.15, # Maximum 15% drawdown tolerance
            'min_sharpe': 0.5     # Minimum annualized Sharpe ratio
        }
        
        # Conservative model parameters
        self.model_configs = {
            'ridge': {
                'alphas': [0.1, 1.0, 10.0, 100.0, 1000.0],
                'cv_folds': 5,
                'random_state': 42
            },
            'lasso': {
                'alphas': [0.001, 0.01, 0.1, 1.0],
                'cv_folds': 5,
                'max_iter': 2000,
                'random_state': 42
            },
            'lightgbm': {
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': 15,        # Conservative
                'learning_rate': 0.03,   # Slow learning
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_data_in_leaf': 100, # Prevent overfitting
                'lambda_l1': 2.0,
                'lambda_l2': 2.0,
                'verbose': -1,
                'random_state': 42,
                'num_boost_round': 100,  # Limited boosting
                'early_stopping_rounds': 20
            }
        }
    
    def load_processed_data(self) -> tuple:
        """Load processed data from pipeline"""
        print("\n📊 LOADING PROCESSED DATA")
        print("-" * 40)
        
        # Load feature configuration
        with open(self.processed_dir / "feature_config.json", 'r') as f:
            feature_config = json.load(f)
        
        selected_features = feature_config['selected_features']
        target_column = feature_config['target_column']
        
        print(f"   🎯 Target: {target_column}")
        print(f"   📊 Features: {len(selected_features)}")
        
        # Load training data
        train_df = pd.read_parquet(self.processed_dir / "train_institutional.parquet")
        print(f"   📈 Training data: {len(train_df):,} samples")
        
        # Create validation split from training data (since original val/test were empty)
        # Use last 30% for validation to ensure enough data
        val_split_idx = int(len(train_df) * 0.7)
        
        actual_train_df = train_df.iloc[:val_split_idx].copy()
        actual_val_df = train_df.iloc[val_split_idx:].copy()
        
        print(f"   📊 Adjusted training: {len(actual_train_df):,} samples")
        print(f"   📊 Adjusted validation: {len(actual_val_df):,} samples")
        
        # Get feature columns (those ending with _t1)
        feature_columns = [col for col in train_df.columns if col.endswith('_t1')]
        
        print(f"   🔧 Available features: {len(feature_columns)}")
        print(f"   📅 Training period: {actual_train_df['Date'].min()} to {actual_train_df['Date'].max()}")
        print(f"   📅 Validation period: {actual_val_df['Date'].min()} to {actual_val_df['Date'].max()}")
        
        return actual_train_df, actual_val_df, feature_columns, target_column
    
    def prepare_model_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                          feature_columns: list, target_column: str) -> tuple:
        """Prepare data for model training"""
        print("\n🔧 PREPARING MODEL DATA")
        print("-" * 40)
        
        # Extract features and targets
        X_train = train_df[feature_columns].copy()
        y_train = train_df[target_column].copy()
        X_val = val_df[feature_columns].copy()
        y_val = val_df[target_column].copy()
        
        print(f"   📊 Training features shape: {X_train.shape}")
        print(f"   📊 Validation features shape: {X_val.shape}")
        
        # Handle missing values (conservative approach)
        missing_counts = X_train.isnull().sum()
        high_missing_features = missing_counts[missing_counts > len(X_train) * 0.1].index
        
        if len(high_missing_features) > 0:
            print(f"   ⚠️ Removing {len(high_missing_features)} features with >10% missing")
            feature_columns = [col for col in feature_columns if col not in high_missing_features]
            X_train = X_train[feature_columns]
            X_val = X_val[feature_columns]
        
        # Fill remaining missing with median (robust)
        for col in feature_columns:
            if X_train[col].isnull().sum() > 0:
                median_val = X_train[col].median()
                X_train[col].fillna(median_val, inplace=True)
                X_val[col].fillna(median_val, inplace=True)
        
        # Remove features with zero variance
        feature_std = X_train.std()
        zero_var_features = feature_std[feature_std < 1e-8].index
        
        if len(zero_var_features) > 0:
            print(f"   ⚠️ Removing {len(zero_var_features)} zero-variance features")
            feature_columns = [col for col in feature_columns if col not in zero_var_features]
            X_train = X_train[feature_columns]
            X_val = X_val[feature_columns]
        
        print(f"   ✅ Final features: {len(feature_columns)}")
        
        # Validate data quality
        if X_train.isnull().any().any() or X_val.isnull().any().any():
            print(f"   ❌ Still have missing values after preprocessing")
            
        if len(feature_columns) < 5:
            print(f"   ⚠️ Very few features remaining: {len(feature_columns)}")
        
        return X_train, X_val, y_train, y_val, feature_columns
    
    def train_ridge_model(self, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                         y_train: pd.Series, y_val: pd.Series) -> dict:
        """Train institutional Ridge regression model"""
        print("\n🎯 TRAINING RIDGE MODEL")
        print("-" * 30)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Cross-validation for alpha selection
        config = self.model_configs['ridge']
        best_alpha = None
        best_cv_score = -np.inf
        
        tscv = TimeSeriesSplit(n_splits=config['cv_folds'])
        
        print(f"   🔍 Cross-validating alpha parameter...")
        
        for alpha in config['alphas']:
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X_train_scaled):
                ridge = Ridge(alpha=alpha, random_state=config['random_state'])
                ridge.fit(X_train_scaled[train_idx], y_train.iloc[train_idx])
                
                pred = ridge.predict(X_train_scaled[val_idx])
                ic, _ = spearmanr(y_train.iloc[val_idx], pred)
                cv_scores.append(ic if not np.isnan(ic) else 0)
            
            mean_cv_score = np.mean(cv_scores)
            print(f"      Alpha {alpha}: CV IC = {mean_cv_score:.4f}")
            
            if mean_cv_score > best_cv_score:
                best_cv_score = mean_cv_score
                best_alpha = alpha
        
        print(f"   🎯 Best alpha: {best_alpha} (CV IC: {best_cv_score:.4f})")
        
        # Train final model
        ridge_final = Ridge(alpha=best_alpha, random_state=config['random_state'])
        ridge_final.fit(X_train_scaled, y_train)
        
        # Evaluate on validation
        val_pred = ridge_final.predict(X_val_scaled)
        val_ic, _ = spearmanr(y_val, val_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        
        print(f"   📊 Validation IC: {val_ic:.4f}")
        print(f"   📊 Validation MSE: {val_mse:.6f}")
        
        # Institutional assessment
        ic_assessment = self._assess_ic_quality(val_ic, 'Ridge')
        
        return {
            'model': ridge_final,
            'scaler': scaler,
            'validation_ic': val_ic,
            'validation_mse': val_mse,
            'best_alpha': best_alpha,
            'cv_score': best_cv_score,
            'ic_assessment': ic_assessment,
            'model_type': 'ridge'
        }
    
    def train_lasso_model(self, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                         y_train: pd.Series, y_val: pd.Series) -> dict:
        """Train institutional Lasso regression model"""
        print("\n🎯 TRAINING LASSO MODEL")
        print("-" * 30)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Cross-validation for alpha selection
        config = self.model_configs['lasso']
        best_alpha = None
        best_cv_score = -np.inf
        
        tscv = TimeSeriesSplit(n_splits=config['cv_folds'])
        
        print(f"   🔍 Cross-validating alpha parameter...")
        
        for alpha in config['alphas']:
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X_train_scaled):
                lasso = Lasso(alpha=alpha, max_iter=config['max_iter'], 
                             random_state=config['random_state'])
                lasso.fit(X_train_scaled[train_idx], y_train.iloc[train_idx])
                
                pred = lasso.predict(X_train_scaled[val_idx])
                ic, _ = spearmanr(y_train.iloc[val_idx], pred)
                cv_scores.append(ic if not np.isnan(ic) else 0)
            
            mean_cv_score = np.mean(cv_scores)
            print(f"      Alpha {alpha}: CV IC = {mean_cv_score:.4f}")
            
            if mean_cv_score > best_cv_score:
                best_cv_score = mean_cv_score
                best_alpha = alpha
        
        print(f"   🎯 Best alpha: {best_alpha} (CV IC: {best_cv_score:.4f})")
        
        # Train final model
        lasso_final = Lasso(alpha=best_alpha, max_iter=config['max_iter'],
                           random_state=config['random_state'])
        lasso_final.fit(X_train_scaled, y_train)
        
        # Feature selection analysis
        selected_features = np.sum(np.abs(lasso_final.coef_) > 1e-6)
        print(f"   📊 Features selected: {selected_features}/{len(X_train.columns)}")
        
        # Evaluate on validation
        val_pred = lasso_final.predict(X_val_scaled)
        val_ic, _ = spearmanr(y_val, val_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        
        print(f"   📊 Validation IC: {val_ic:.4f}")
        print(f"   📊 Validation MSE: {val_mse:.6f}")
        
        # Institutional assessment
        ic_assessment = self._assess_ic_quality(val_ic, 'Lasso')
        
        return {
            'model': lasso_final,
            'scaler': scaler,
            'validation_ic': val_ic,
            'validation_mse': val_mse,
            'best_alpha': best_alpha,
            'cv_score': best_cv_score,
            'selected_features': selected_features,
            'ic_assessment': ic_assessment,
            'model_type': 'lasso'
        }
    
    def train_lightgbm_model(self, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                            y_train: pd.Series, y_val: pd.Series) -> dict:
        """Train institutional LightGBM model"""
        print("\n🎯 TRAINING LIGHTGBM MODEL")
        print("-" * 30)
        
        config = self.model_configs['lightgbm']
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train with early stopping
        print(f"   🌟 Training with conservative parameters...")
        
        model = lgb.train(
            config,
            train_data,
            valid_sets=[val_data],
            num_boost_round=config['num_boost_round'],
            callbacks=[
                lgb.early_stopping(config['early_stopping_rounds']),
                lgb.log_evaluation(0)
            ]
        )
        
        # Evaluate on validation
        val_pred = model.predict(X_val)
        val_ic, _ = spearmanr(y_val, val_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        
        print(f"   📊 Validation IC: {val_ic:.4f}")
        print(f"   📊 Validation MSE: {val_mse:.6f}")
        print(f"   🎯 Best iteration: {model.best_iteration}")
        
        # Feature importance analysis
        importance = model.feature_importance(importance_type='gain')
        top_features = np.argsort(importance)[-5:][::-1]
        
        print(f"   📊 Top 5 features:")
        for i, idx in enumerate(top_features):
            feature_name = X_train.columns[idx]
            print(f"      {i+1}. {feature_name}: {importance[idx]:.1f}")
        
        # Institutional assessment
        ic_assessment = self._assess_ic_quality(val_ic, 'LightGBM')
        
        return {
            'model': model,
            'validation_ic': val_ic,
            'validation_mse': val_mse,
            'best_iteration': model.best_iteration,
            'feature_importance': importance,
            'ic_assessment': ic_assessment,
            'model_type': 'lightgbm'
        }
    
    def _assess_ic_quality(self, ic: float, model_name: str) -> dict:
        """Assess IC quality against institutional standards"""
        abs_ic = abs(ic)
        
        # Quality assessment
        if abs_ic >= self.institutional_thresholds['max_ic']:
            quality = 'SUSPICIOUS'
            message = f"IC too high ({ic:.4f}) - potential overfitting"
            institutional_approved = False
        elif abs_ic >= self.institutional_thresholds['min_ic']:
            quality = 'GOOD'
            message = f"IC within institutional range ({ic:.4f})"
            institutional_approved = True
        else:
            quality = 'WEAK'
            message = f"IC below minimum threshold ({ic:.4f})"
            institutional_approved = False
        
        print(f"   🏛️ {model_name} Assessment: {quality} - {message}")
        
        return {
            'quality': quality,
            'message': message,
            'institutional_approved': institutional_approved,
            'ic_value': ic
        }
    
    def select_best_model(self, ridge_results: dict, lasso_results: dict, 
                         lgb_results: dict) -> dict:
        """Select best model using institutional criteria"""
        print("\n🏆 MODEL SELECTION")
        print("-" * 30)
        
        models = [ridge_results, lasso_results, lgb_results]
        
        print(f"   📊 Model Performance Comparison:")
        for model in models:
            ic = model['validation_ic']
            quality = model['ic_assessment']['quality']
            approved = model['ic_assessment']['institutional_approved']
            print(f"      {model['model_type'].title()}: IC={ic:.4f}, Quality={quality}, Approved={'✅' if approved else '❌'}")
        
        # First filter: Only institutionally approved models
        approved_models = [m for m in models if m['ic_assessment']['institutional_approved']]
        
        if not approved_models:
            print(f"   ⚠️ No models meet institutional standards")
            # Fallback: select best IC regardless
            best_model = max(models, key=lambda x: abs(x['validation_ic']))
            print(f"   🔄 Fallback selection: {best_model['model_type'].title()}")
        else:
            # Select best approved model
            # Prefer Ridge for stability, then Lasso, then LightGBM
            model_preference = {'ridge': 3, 'lasso': 2, 'lightgbm': 1}
            
            best_model = max(approved_models, key=lambda x: (
                abs(x['validation_ic']),  # Primary: IC magnitude
                model_preference.get(x['model_type'], 0)  # Secondary: Model preference
            ))
            
            print(f"   ✅ Selected: {best_model['model_type'].title()}")
        
        print(f"   🎯 Final model IC: {best_model['validation_ic']:.4f}")
        
        return best_model
    
    def save_institutional_model(self, best_model: dict, feature_columns: list) -> str:
        """Save model with institutional documentation"""
        print("\n💾 SAVING INSTITUTIONAL MODEL")
        print("-" * 40)
        
        # Create timestamped model directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"institutional_model_{timestamp}"
        model_dir = self.models_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Save model components
        if best_model['model_type'] in ['ridge', 'lasso']:
            # Save sklearn models
            joblib.dump(best_model['model'], model_dir / "model.pkl")
            joblib.dump(best_model['scaler'], model_dir / "scaler.pkl")
        else:
            # Save LightGBM model
            best_model['model'].save_model(str(model_dir / "model.txt"))
        
        # Save feature list
        with open(model_dir / "features.json", 'w') as f:
            json.dump(feature_columns, f, indent=2)
        
        # Create institutional model card
        model_card = {
            'model_info': {
                'model_type': best_model['model_type'],
                'training_date': datetime.now().isoformat(),
                'model_name': model_name,
                'institutional_approved': best_model['ic_assessment']['institutional_approved']
            },
            'performance': {
                'validation_ic': best_model['validation_ic'],
                'validation_mse': best_model['validation_mse'],
                'ic_quality': best_model['ic_assessment']['quality'],
                'assessment_message': best_model['ic_assessment']['message']
            },
            'features': {
                'count': len(feature_columns),
                'names': feature_columns
            },
            'institutional_standards': {
                'min_ic_threshold': self.institutional_thresholds['min_ic'],
                'max_ic_threshold': self.institutional_thresholds['max_ic'],
                'meets_standards': best_model['ic_assessment']['institutional_approved']
            },
            'deployment_ready': best_model['ic_assessment']['institutional_approved']
        }
        
        # Add model-specific information
        if best_model['model_type'] == 'ridge':
            model_card['model_params'] = {
                'best_alpha': best_model['best_alpha'],
                'cv_score': best_model['cv_score']
            }
        elif best_model['model_type'] == 'lasso':
            model_card['model_params'] = {
                'best_alpha': best_model['best_alpha'],
                'cv_score': best_model['cv_score'],
                'selected_features': best_model['selected_features']
            }
        elif best_model['model_type'] == 'lightgbm':
            model_card['model_params'] = {
                'best_iteration': best_model['best_iteration'],
                'feature_importance_available': True
            }
        
        # Save model card
        with open(model_dir / "model_card.json", 'w') as f:
            json.dump(model_card, f, indent=2, default=str)
        
        print(f"   ✅ Model saved: {model_name}")
        print(f"   📁 Location: {model_dir}")
        print(f"   📄 Model card created with institutional documentation")
        print(f"   🏛️ Institutional approved: {'✅' if model_card['deployment_ready'] else '❌'}")
        
        return model_name
    
    def run_institutional_training(self) -> dict:
        """Run complete institutional model training"""
        print("🚀 RUNNING INSTITUTIONAL MODEL TRAINING")
        print("=" * 60)
        
        try:
            # 1. Load processed data
            train_df, val_df, feature_columns, target_column = self.load_processed_data()
            
            # 2. Prepare model data
            X_train, X_val, y_train, y_val, final_features = self.prepare_model_data(
                train_df, val_df, feature_columns, target_column
            )
            
            if len(final_features) < 5:
                raise ValueError(f"Insufficient features for training: {len(final_features)}")
            
            # 3. Train multiple models
            ridge_results = self.train_ridge_model(X_train, X_val, y_train, y_val)
            lasso_results = self.train_lasso_model(X_train, X_val, y_train, y_val)
            lgb_results = self.train_lightgbm_model(X_train, X_val, y_train, y_val)
            
            # 4. Select best model
            best_model = self.select_best_model(ridge_results, lasso_results, lgb_results)
            
            # 5. Save institutional model
            model_name = self.save_institutional_model(best_model, final_features)
            
            # 6. Training summary
            training_summary = {
                'success': True,
                'model_name': model_name,
                'best_model_type': best_model['model_type'],
                'validation_ic': best_model['validation_ic'],
                'institutional_approved': best_model['ic_assessment']['institutional_approved'],
                'features_used': len(final_features),
                'training_samples': len(X_train),
                'validation_samples': len(X_val)
            }
            
            print(f"\n🎉 TRAINING COMPLETION SUMMARY")
            print("=" * 50)
            print(f"✅ Best model: {best_model['model_type'].title()}")
            print(f"✅ Validation IC: {best_model['validation_ic']:.4f}")
            print(f"✅ Features used: {len(final_features)}")
            print(f"✅ Training samples: {len(X_train):,}")
            print(f"🏛️ Institutional approved: {'✅' if training_summary['institutional_approved'] else '❌'}")
            print(f"📊 Ready for validation and deployment")
            
            return training_summary
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

def main():
    """Main training execution"""
    trainer = InstitutionalModelTrainer()
    results = trainer.run_institutional_training()
    
    if results['success']:
        print(f"\n🎯 INSTITUTIONAL MODEL TRAINING: ✅ SUCCESS")
    else:
        print(f"\n🚨 INSTITUTIONAL MODEL TRAINING: ❌ FAILED")
    
    return results

if __name__ == "__main__":
    results = main()