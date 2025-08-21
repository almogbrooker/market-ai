#!/usr/bin/env python3
"""
COMPLETE TIERED ARCHITECTURE SYSTEM
Integrates Models A-D: LightGBM + LSTM + Regime + Meta-Ensemble
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our models
from .temporal_lstm import TemporalLSTMTrainer
from .regime_classifier import RegimeClassifier
from .meta_ensemble import MetaEnsemble
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TieredAlphaSystem:
    """
    Complete Tiered Architecture System
    
    Models:
    A) LightGBM Ranker (baseline cross-sectional)
    B) Compact LSTM (temporal specialist) 
    C) Regime Classifier (gating)
    D) Meta-Ensemble (combiner + uncertainty)
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        
        # Model components
        self.lgbm_models = []  # Ensemble of LightGBM models
        self.lgbm_scalers = []
        self.lstm_trainer = None
        self.regime_classifier = None
        self.meta_ensemble = None
        
        # System state
        self.is_trained = False
        self.feature_names = []
        self.training_results = {}
        
        logger.info("🏗️ Tiered Alpha System initialized")
    
    def _default_config(self) -> Dict:
        """Default system configuration"""
        return {
            # Model A: LightGBM
            'lgbm': {
                'n_models': 3,  # Ensemble size
                'objective': 'regression',
                'num_leaves': 31,
                'max_depth': 7,
                'learning_rate': 0.03,
                'subsample': 0.8,
                'colsample_bytree': 0.7,
                'min_child_samples': 200,
                'num_boost_round': 1000,
                'early_stopping_rounds': 50
            },
            
            # Model B: LSTM  
            'lstm': {
                'enabled': True,
                'hidden_dim': 64,
                'num_layers': 2,
                'dropout': 0.2,
                'seq_len': 30,
                'batch_size': 256,
                'learning_rate': 1e-3,
                'max_epochs': 60,
                'early_stop_patience': 8
            },
            
            # Model C: Regime
            'regime': {
                'enabled': True,
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_leaf': 20
            },
            
            # Model D: Meta-ensemble
            'meta': {
                'combiner_type': 'ridge',  # 'ridge' or 'lgbm'
                'conformal_alpha': 0.1     # 90% prediction intervals
            },
            
            # System settings
            'validation_split': 0.2,
            'random_state': 42
        }
    
    def load_existing_lgbm_models(self, model_dir: Path) -> bool:
        """Load existing LightGBM models from sleeve_c"""
        
        logger.info("📂 Loading existing LightGBM models...")
        
        try:
            # Look for sleeve_c models
            sleeve_dir = model_dir / "sleeves" / "sleeve_c"
            
            if not sleeve_dir.exists():
                logger.warning("No sleeve_c directory found")
                return False
            
            # Load models and scalers
            for fold_id in range(3):
                model_path = sleeve_dir / f"sleeve_c_fold_{fold_id}_model.txt"
                scaler_path = sleeve_dir / f"sleeve_c_fold_{fold_id}_scaler.pkl"
                
                if model_path.exists() and scaler_path.exists():
                    # Load LightGBM model
                    model = lgb.Booster(model_file=str(model_path))
                    
                    # Load scaler
                    scaler = joblib.load(scaler_path)
                    
                    self.lgbm_models.append(model)
                    self.lgbm_scalers.append(scaler)
                    
                    logger.info(f"   ✅ Loaded fold {fold_id}: {model_path.name}")
                else:
                    logger.warning(f"   ❌ Missing fold {fold_id} files")
            
            if self.lgbm_models:
                # Load feature names from metadata
                metadata_path = sleeve_dir / "sleeve_c_metadata.json"
                if metadata_path.exists():
                    import json
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    self.feature_names = metadata.get('features', [])
                
                logger.info(f"✅ Loaded {len(self.lgbm_models)} LightGBM models")
                return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load existing models: {e}")
        
        return False
    
    def train_system(self, 
                    train_data: pd.DataFrame,
                    val_data: pd.DataFrame = None) -> Dict:
        """Train the complete tiered system"""
        
        logger.info("🚀 Training Complete Tiered Alpha System...")
        
        results = {}
        
        # Step 1: Load/Train Model A (LightGBM)
        model_dir = Path(__file__).parent.parent.parent / "artifacts"
        
        if self.load_existing_lgbm_models(model_dir):
            logger.info("✅ Using existing LightGBM models")
            results['lgbm'] = {'status': 'loaded_existing', 'n_models': len(self.lgbm_models)}
        else:
            logger.info("🏋️ Training new LightGBM models...")
            results['lgbm'] = self._train_lgbm_ensemble(train_data, val_data)
        
        # Step 2: Train Model B (LSTM)
        if self.config['lstm']['enabled']:
            logger.info("🏋️ Training LSTM model...")
            results['lstm'] = self._train_lstm(train_data, val_data)
        else:
            logger.info("⏸️ LSTM disabled")
            results['lstm'] = {'status': 'disabled'}
        
        # Step 3: Train Model C (Regime)
        if self.config['regime']['enabled']:
            logger.info("🏋️ Training Regime Classifier...")
            results['regime'] = self._train_regime_classifier()
        else:
            logger.info("⏸️ Regime classifier disabled")
            results['regime'] = {'status': 'disabled'}
        
        # Step 4: Train Model D (Meta-ensemble)
        logger.info("🏋️ Training Meta-Ensemble...")
        results['meta'] = self._train_meta_ensemble(train_data, val_data)
        
        self.is_trained = True
        self.training_results = results
        
        logger.info("🏆 Complete Tiered System Training Finished!")
        self._log_training_summary(results)
        
        return results
    
    def _train_lgbm_ensemble(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict:
        """Train LightGBM ensemble"""
        
        # Feature columns
        feature_cols = [col for col in train_data.columns if col.endswith('_lag1')]
        if not feature_cols:
            feature_cols = [
                'return_5d_lag1', 'return_20d_lag1', 'return_60d_lag1', 'return_12m_ex_1m_lag1',
                'vol_20d_lag1', 'vol_60d_lag1', 'volume_ratio_lag1', 'dollar_volume_ratio_lag1',
                'log_price_lag1', 'price_volume_trend_lag1'
            ]
            feature_cols = [col for col in feature_cols if col in train_data.columns]
        
        self.feature_names = feature_cols
        
        # Prepare data
        X_train = train_data[feature_cols].fillna(0)
        y_train = train_data['next_return_1d'].fillna(0)
        
        if val_data is not None:
            X_val = val_data[feature_cols].fillna(0)
            y_val = val_data['next_return_1d'].fillna(0)
        else:
            # Split training data
            n_val = int(len(X_train) * 0.2)
            X_val = X_train.iloc[-n_val:]
            y_val = y_train.iloc[-n_val:]
            X_train = X_train.iloc[:-n_val]
            y_train = y_train.iloc[:-n_val]
        
        # Train ensemble
        models = []
        scalers = []
        
        for i in range(self.config['lgbm']['n_models']):
            logger.info(f"   Training LightGBM model {i+1}/{self.config['lgbm']['n_models']}")
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train model
            train_data_lgb = lgb.Dataset(X_train_scaled, label=y_train)
            val_data_lgb = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data_lgb)
            
            params = self.config['lgbm'].copy()
            params.pop('n_models')
            params['random_state'] = self.config['random_state'] + i
            params['verbose'] = -1
            
            model = lgb.train(
                params,
                train_data_lgb,
                valid_sets=[val_data_lgb],
                callbacks=[lgb.early_stopping(params.pop('early_stopping_rounds', 50)),
                          lgb.log_evaluation(0)]
            )
            
            models.append(model)
            scalers.append(scaler)
        
        self.lgbm_models = models
        self.lgbm_scalers = scalers
        
        return {
            'status': 'trained',
            'n_models': len(models),
            'feature_names': self.feature_names
        }
    
    def _train_lstm(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict:
        """Train LSTM model"""
        
        self.lstm_trainer = TemporalLSTMTrainer(self.config['lstm'])
        results = self.lstm_trainer.train_model(train_data, val_data)
        
        return {
            'status': 'trained',
            **results
        }
    
    def _train_regime_classifier(self) -> Dict:
        """Train regime classifier"""
        
        self.regime_classifier = RegimeClassifier(self.config['regime'])
        results = self.regime_classifier.train()
        
        return {
            'status': 'trained',
            **results
        }
    
    def _train_meta_ensemble(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict:
        """Train meta-ensemble"""
        
        # Get predictions from base models
        lgbm_scores = self._get_lgbm_predictions(train_data)
        
        lstm_scores = None
        if self.lstm_trainer:
            try:
                lstm_scores = self.lstm_trainer.predict(train_data)
                
                # Handle shape mismatch between LGBM and LSTM
                if len(lstm_scores) != len(lgbm_scores):
                    logger.warning(f"Shape mismatch: LGBM {len(lgbm_scores)}, LSTM {len(lstm_scores)}")
                    # Pad LSTM scores with zeros or truncate LGBM scores
                    if len(lstm_scores) < len(lgbm_scores):
                        # Pad LSTM scores
                        padded_scores = np.zeros(len(lgbm_scores))
                        padded_scores[:len(lstm_scores)] = lstm_scores
                        lstm_scores = padded_scores
                    else:
                        # Truncate LSTM scores
                        lstm_scores = lstm_scores[:len(lgbm_scores)]
                        
            except Exception as e:
                logger.warning(f"Failed to get LSTM predictions: {e}")
                lstm_scores = None
        
        # Dummy regime probabilities for training
        regime_probs = {'bull': 0.3, 'neutral': 0.4, 'bear': 0.2, 'high_vol': 0.1}
        market_features = {'vix_level': 20, 'realized_vol_20d': 0.2}
        
        # Initialize meta-ensemble
        self.meta_ensemble = MetaEnsemble(self.config['meta'])
        
        # Prepare meta-features
        meta_features = self.meta_ensemble.prepare_meta_features(
            lgbm_scores, lstm_scores, regime_probs, market_features
        )
        
        # Train
        targets = train_data['next_return_1d'].fillna(0).values
        results = self.meta_ensemble.train_combiner(meta_features, targets)
        
        return {
            'status': 'trained',
            **results
        }
    
    def _get_lgbm_predictions(self, data: pd.DataFrame) -> np.ndarray:
        """Get ensemble LightGBM predictions"""
        
        if not self.lgbm_models:
            return np.zeros(len(data))
        
        X = data[self.feature_names].fillna(0)
        
        predictions = []
        for model, scaler in zip(self.lgbm_models, self.lgbm_scalers):
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)
            predictions.append(pred)
        
        # Average ensemble predictions
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
    
    def predict_alpha(self, data: pd.DataFrame, current_date: str = None) -> Dict:
        """Generate alpha predictions with uncertainty"""
        
        if not self.is_trained:
            raise ValueError("System not trained")
        
        logger.info("🔮 Generating alpha predictions...")
        
        # Step 1: Get base model predictions
        lgbm_scores = self._get_lgbm_predictions(data)
        
        lstm_scores = None
        if self.lstm_trainer:
            try:
                lstm_scores = self.lstm_trainer.predict(data)
            except Exception as e:
                logger.warning(f"LSTM prediction failed: {e}")
        
        # Step 2: Get regime prediction
        regime_result = None
        if self.regime_classifier:
            try:
                regime_result = self.regime_classifier.predict_regime(current_date)
                regime = regime_result['regime']
                regime_probs = regime_result['probabilities']
                market_features = regime_result['features']
            except Exception as e:
                logger.warning(f"Regime prediction failed: {e}")
                regime = 'neutral'
                regime_probs = {'bull': 0.25, 'neutral': 0.5, 'bear': 0.25, 'high_vol': 0.0}
                market_features = {}
        else:
            regime = 'neutral'
            regime_probs = {'bull': 0.25, 'neutral': 0.5, 'bear': 0.25, 'high_vol': 0.0}
            market_features = {}
        
        # Step 3: Meta-ensemble prediction with uncertainty
        if self.meta_ensemble:
            meta_result = self.meta_ensemble.predict_with_uncertainty(
                lgbm_scores, lstm_scores, regime_probs, market_features, regime
            )
        else:
            meta_result = {
                'final_scores': lgbm_scores,
                'prob_up': (lgbm_scores > 0).astype(float),
                'trade_filter': np.ones(len(lgbm_scores), dtype=bool),
                'signal_strength': np.abs(lgbm_scores)
            }
        
        # Step 4: Position sizing
        if self.regime_classifier and regime_result is not None:
            sizing_info = self.regime_classifier.get_position_sizing_multiplier(
                regime, regime_result.get('confidence', 0.5)
            )
            regime_multiplier = sizing_info['multiplier']
        else:
            regime_multiplier = 1.0
        
        if self.meta_ensemble:
            position_sizes = self.meta_ensemble.get_position_sizes(
                meta_result['final_scores'],
                meta_result['prob_up'],
                meta_result['interval_width'],
                meta_result['trade_filter'],
                regime_multiplier
            )
        else:
            position_sizes = np.tanh(meta_result['final_scores']) * regime_multiplier
            position_sizes = np.clip(position_sizes, -0.15, 0.15)
        
        # Compile results
        predictions = {
            'final_scores': meta_result['final_scores'],
            'position_sizes': position_sizes,
            'trade_filter': meta_result['trade_filter'],
            'prob_up': meta_result['prob_up'],
            'regime': regime,
            'regime_multiplier': regime_multiplier,
            'signal_strength': meta_result['signal_strength'],
            'n_tradeable': meta_result['trade_filter'].sum(),
            'base_predictions': {
                'lgbm_scores': lgbm_scores,
                'lstm_scores': lstm_scores
            },
            'market_context': {
                'regime': regime,
                'regime_probs': regime_probs,
                'market_features': market_features
            }
        }
        
        logger.info(f"✅ Alpha predictions generated:")
        logger.info(f"   Regime: {regime} (multiplier: {regime_multiplier:.2f})")
        logger.info(f"   Tradeable positions: {predictions['n_tradeable']}/{len(data)}")
        logger.info(f"   Score range: [{predictions['final_scores'].min():.4f}, {predictions['final_scores'].max():.4f}]")
        
        return predictions
    
    def _log_training_summary(self, results: Dict):
        """Log training summary"""
        
        logger.info("📊 TIERED SYSTEM TRAINING SUMMARY:")
        logger.info("=" * 50)
        
        for model_name, model_results in results.items():
            status = model_results.get('status', 'unknown')
            logger.info(f"   {model_name.upper()}: {status}")
            
            if 'val_ic' in model_results:
                logger.info(f"      Val IC: {model_results['val_ic']:.4f}")
            if 'accuracy' in model_results:
                logger.info(f"      Accuracy: {model_results['accuracy']:.3f}")
    
    def save_system(self, save_dir: Path):
        """Save complete system"""
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save LightGBM models
        lgbm_dir = save_dir / "lgbm"
        lgbm_dir.mkdir(exist_ok=True)
        
        for i, (model, scaler) in enumerate(zip(self.lgbm_models, self.lgbm_scalers)):
            model.save_model(str(lgbm_dir / f"model_{i}.txt"))
            joblib.dump(scaler, lgbm_dir / f"scaler_{i}.pkl")
        
        # Save LSTM
        if self.lstm_trainer:
            self.lstm_trainer.save_model(save_dir / "lstm_model.pth")
        
        # Save regime classifier
        if self.regime_classifier:
            self.regime_classifier.save_model(save_dir / "regime_model.pkl")
        
        # Save meta-ensemble
        if self.meta_ensemble:
            self.meta_ensemble.save_ensemble(save_dir / "meta_ensemble.pkl")
        
        # Save system config and metadata
        import json
        system_metadata = {
            'config': self.config,
            'feature_names': self.feature_names,
            'training_results': self.training_results,
            'is_trained': self.is_trained,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(save_dir / "system_metadata.json", 'w') as f:
            json.dump(system_metadata, f, indent=2, default=str)
        
        logger.info(f"✅ Complete system saved: {save_dir}")

def main():
    """Test the complete tiered system"""
    
    # Generate synthetic test data
    np.random.seed(42)
    
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'META']
    
    data = []
    for ticker in tickers:
        for date in dates:
            row = {
                'Date': date,
                'Ticker': ticker,
                'next_return_1d': np.random.normal(0, 0.02),
                **{f'return_{d}d_lag1': np.random.normal(0, 0.01) for d in [5, 20, 60]},
                'return_12m_ex_1m_lag1': np.random.normal(0, 0.05),
                **{f'vol_{d}d_lag1': np.random.uniform(0.1, 0.5) for d in [20, 60]},
                'volume_ratio_lag1': np.random.uniform(0.5, 2.0),
                'dollar_volume_ratio_lag1': np.random.uniform(0.5, 2.0),
                'log_price_lag1': np.log(np.random.uniform(50, 300)),
                'price_volume_trend_lag1': np.random.normal(0, 0.01)
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    
    # Split data
    split_date = '2023-01-01'
    train_data = df[df['Date'] < split_date]
    val_data = df[df['Date'] >= split_date]
    
    # Initialize and train system
    system = TieredAlphaSystem()
    
    # Train
    results = system.train_system(train_data, val_data)
    
    # Test predictions
    test_data = val_data.head(100)
    predictions = system.predict_alpha(test_data)
    
    print(f"🎉 System test completed!")
    print(f"Training results: {results}")
    print(f"Predictions: {len(predictions['final_scores'])} scores generated")

if __name__ == "__main__":
    main()