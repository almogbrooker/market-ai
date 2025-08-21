#!/usr/bin/env python3
"""
Model C: Regime Classifier (Gating, not alpha)
Gates & resizes positions, sets thresholds, chooses which alpha(s) to trust
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RegimeFeatureEngine:
    """Engineer regime classification features"""
    
    def __init__(self):
        self.feature_names = [
            'vix_level', 'vix_change_5d', 'vix_change_20d',
            'qqq_momentum_20d', 'qqq_momentum_50d', 'qqq_ema_slope',
            'realized_vol_20d', 'realized_vol_60d',
            'breadth_above_50dma', 'breadth_above_200dma',
            'index_vs_ma20', 'index_vs_ma50', 'index_vs_ma200'
        ]
    
    def fetch_market_data(self, start_date: str = '2018-01-01', end_date: str = None) -> pd.DataFrame:
        """Fetch market data for regime classification"""
        
        logger.info("📊 Fetching market data for regime classification...")
        
        if end_date is None:
            end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        
        # Fetch key market indicators
        symbols = {
            'QQQ': 'NASDAQ-100 ETF',
            '^VIX': 'VIX Volatility Index', 
            'SPY': 'S&P 500 ETF',
            'IWM': 'Russell 2000 ETF'
        }
        
        market_data = {}
        
        for symbol, description in symbols.items():
            try:
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    market_data[symbol] = data
                    logger.info(f"   ✅ {symbol} ({description}): {len(data)} days")
                else:
                    logger.warning(f"   ❌ {symbol}: No data")
            except Exception as e:
                logger.warning(f"   ❌ {symbol}: {e}")
        
        if not market_data:
            logger.warning("No market data fetched - using fallback")
            # Return minimal fallback data structure
            return {
                'FALLBACK': pd.DataFrame({
                    'Close': [100.0] * 30,  # 30 days of dummy data
                    'Volume': [1000000] * 30
                }, index=pd.date_range(start='2023-01-01', periods=30))
            }
        
        return market_data
    
    def engineer_regime_features(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Engineer regime classification features"""
        
        logger.info("🔧 Engineering regime features...")
        
        # Use any available symbol as base, with fallback
        available_symbols = list(market_data.keys())
        if not available_symbols:
            raise ValueError("No market data available")
        
        # Prefer QQQ, then SPY, then any available symbol
        if 'QQQ' in market_data:
            base_symbol = 'QQQ'
        elif 'SPY' in market_data:
            base_symbol = 'SPY'
        else:
            base_symbol = available_symbols[0]
        
        logger.info(f"   Using {base_symbol} as base index")
        
        qqq = market_data[base_symbol].copy()
        qqq['returns'] = qqq['Close'].pct_change()
        
        features = pd.DataFrame(index=qqq.index)
        
        # VIX features (if available)
        if '^VIX' in market_data:
            vix = market_data['^VIX']['Close']
            features['vix_level'] = vix
            features['vix_change_5d'] = vix.pct_change(5)
            features['vix_change_20d'] = vix.pct_change(20)
        else:
            # Estimate VIX from QQQ volatility
            vol_proxy = qqq['returns'].rolling(20).std() * np.sqrt(252) * 100
            features['vix_level'] = vol_proxy
            features['vix_change_5d'] = vol_proxy.pct_change(5)
            features['vix_change_20d'] = vol_proxy.pct_change(20)
        
        # QQQ momentum features
        features['qqq_momentum_20d'] = qqq['Close'].pct_change(20)
        features['qqq_momentum_50d'] = qqq['Close'].pct_change(50)
        
        # EMA slope (trend strength)
        qqq['ema_20'] = qqq['Close'].ewm(span=20).mean()
        qqq['ema_50'] = qqq['Close'].ewm(span=50).mean()
        features['qqq_ema_slope'] = (qqq['ema_20'] / qqq['ema_50'] - 1) * 100
        
        # Realized volatility
        features['realized_vol_20d'] = qqq['returns'].rolling(20).std() * np.sqrt(252)
        features['realized_vol_60d'] = qqq['returns'].rolling(60).std() * np.sqrt(252)
        
        # Market breadth (simplified - using multiple ETFs as proxy)
        breadth_above_50 = []
        breadth_above_200 = []
        
        for symbol in ['QQQ', 'SPY', 'IWM']:
            if symbol in market_data:
                data = market_data[symbol]
                ma_50 = data['Close'].rolling(50).mean()
                ma_200 = data['Close'].rolling(200).mean()
                
                breadth_above_50.append((data['Close'] > ma_50).astype(int))
                breadth_above_200.append((data['Close'] > ma_200).astype(int))
        
        if breadth_above_50:
            features['breadth_above_50dma'] = np.mean(breadth_above_50, axis=0)
            features['breadth_above_200dma'] = np.mean(breadth_above_200, axis=0)
        else:
            features['breadth_above_50dma'] = 0.5
            features['breadth_above_200dma'] = 0.5
        
        # Index vs moving averages
        qqq_ma_20 = qqq['Close'].rolling(20).mean()
        qqq_ma_50 = qqq['Close'].rolling(50).mean()
        qqq_ma_200 = qqq['Close'].rolling(200).mean()
        
        features['index_vs_ma20'] = (qqq['Close'] / qqq_ma_20 - 1) * 100
        features['index_vs_ma50'] = (qqq['Close'] / qqq_ma_50 - 1) * 100  
        features['index_vs_ma200'] = (qqq['Close'] / qqq_ma_200 - 1) * 100
        
        # Fill missing values
        features = features.fillna(method='ffill').fillna(0)
        
        logger.info(f"✅ Regime features engineered: {len(features)} samples, {len(features.columns)} features")
        
        return features
    
    def create_regime_labels(self, features: pd.DataFrame, qqq_data: pd.DataFrame) -> pd.Series:
        """Create regime labels based on market conditions"""
        
        logger.info("🏷️ Creating regime labels...")
        
        # Calculate forward returns for labeling
        qqq_returns = qqq_data['Close'].pct_change(20)  # 20-day forward returns
        realized_vol = features['realized_vol_20d']
        vix_level = features['vix_level']
        
        regimes = []
        
        for i in range(len(features)):
            vol = realized_vol.iloc[i] if not pd.isna(realized_vol.iloc[i]) else 0.2
            vix = vix_level.iloc[i] if not pd.isna(vix_level.iloc[i]) else 20
            
            # High volatility regime
            if vol > 0.3 or vix > 30:
                regimes.append('high_vol')
            # Bull regime (low vol + positive momentum)
            elif vol < 0.15 and features['qqq_momentum_20d'].iloc[i] > 0.02:
                regimes.append('bull')
            # Bear regime (negative momentum + elevated vol)
            elif features['qqq_momentum_20d'].iloc[i] < -0.05:
                regimes.append('bear')
            # Neutral regime
            else:
                regimes.append('neutral')
        
        regime_series = pd.Series(regimes, index=features.index)
        
        # Log regime distribution
        regime_counts = regime_series.value_counts()
        logger.info("📊 Regime distribution:")
        for regime, count in regime_counts.items():
            pct = count / len(regime_series) * 100
            logger.info(f"   {regime}: {count} days ({pct:.1f}%)")
        
        return regime_series

class RegimeClassifier:
    """Lightweight regime classification model"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.model = None
        self.scaler = None
        self.feature_names = []
        
        # Model parameters
        self.n_estimators = self.config.get('n_estimators', 100)
        self.max_depth = self.config.get('max_depth', 10)
        self.min_samples_leaf = self.config.get('min_samples_leaf', 20)
        
        self.feature_engine = RegimeFeatureEngine()
    
    def train(self, start_date: str = '2018-01-01', end_date: str = '2023-12-31') -> Dict:
        """Train the regime classifier"""
        
        logger.info("🏋️ Training Regime Classifier...")
        
        # Fetch market data
        market_data = self.feature_engine.fetch_market_data(start_date, end_date)
        
        # Engineer features
        features_df = self.feature_engine.engineer_regime_features(market_data)
        
        # Create labels
        labels = self.feature_engine.create_regime_labels(features_df, market_data['QQQ'])
        
        # Align features and labels
        common_index = features_df.index.intersection(labels.index)
        X = features_df.loc[common_index]
        y = labels.loc[common_index]
        
        # Remove rows with missing values
        valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 100:
            raise ValueError("Insufficient data for training")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.feature_names = list(X.columns)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_scaled, y)
        
        # Evaluate
        predictions = self.model.predict(X_scaled)
        accuracy = accuracy_score(y, predictions)
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        results = {
            'accuracy': accuracy,
            'n_samples': len(X),
            'feature_importance': importance,
            'regime_distribution': y.value_counts().to_dict()
        }
        
        logger.info(f"✅ Regime Classifier trained:")
        logger.info(f"   Accuracy: {accuracy:.3f}")
        logger.info(f"   Samples: {len(X):,}")
        logger.info(f"   Top features: {importance.head(3)['feature'].tolist()}")
        
        return results
    
    def predict_regime(self, current_date: str = None) -> Dict:
        """Predict current market regime"""
        
        if self.model is None:
            raise ValueError("Model not trained")
        
        if current_date is None:
            current_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        
        # Fetch recent data
        end_date = current_date
        start_date = (pd.Timestamp(current_date) - pd.Timedelta(days=300)).strftime('%Y-%m-%d')
        
        market_data = self.feature_engine.fetch_market_data(start_date, end_date)
        features_df = self.feature_engine.engineer_regime_features(market_data)
        
        # Get latest features
        latest_features = features_df.iloc[-1:][self.feature_names]
        X_scaled = self.scaler.transform(latest_features)
        
        # Predict
        regime_pred = self.model.predict(X_scaled)[0]
        regime_proba = self.model.predict_proba(X_scaled)[0]
        
        # Get class names
        classes = self.model.classes_
        proba_dict = dict(zip(classes, regime_proba))
        
        # Confidence
        confidence = max(regime_proba)
        
        result = {
            'regime': regime_pred,
            'confidence': confidence,
            'probabilities': proba_dict,
            'date': current_date,
            'features': latest_features.iloc[0].to_dict()
        }
        
        logger.info(f"🎯 Current regime: {regime_pred} (confidence: {confidence:.3f})")
        
        return result
    
    def get_position_sizing_multiplier(self, regime: str, confidence: float) -> Dict:
        """Get position sizing multiplier based on regime"""
        
        # Regime-based position sizing
        regime_multipliers = {
            'bull': 1.2,      # Increase exposure in bull markets
            'neutral': 1.0,   # Normal exposure
            'bear': 0.8,      # Reduce exposure in bear markets
            'high_vol': 0.6   # Significantly reduce in high vol
        }
        
        base_multiplier = regime_multipliers.get(regime, 1.0)
        
        # Adjust by confidence
        confidence_adjustment = 0.5 + 0.5 * confidence  # Scale from 0.5 to 1.0
        final_multiplier = base_multiplier * confidence_adjustment
        
        # Hard limits
        final_multiplier = max(0.3, min(1.5, final_multiplier))
        
        return {
            'multiplier': final_multiplier,
            'regime': regime,
            'confidence': confidence,
            'base_multiplier': base_multiplier,
            'confidence_adjustment': confidence_adjustment
        }
    
    def save_model(self, path: Path):
        """Save model and scaler"""
        import joblib
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'config': self.config
        }
        
        joblib.dump(model_data, path)
        logger.info(f"✅ Regime classifier saved: {path}")
    
    def load_model(self, path: Path):
        """Load model and scaler"""
        import joblib
        
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.config = model_data['config']
        
        logger.info(f"✅ Regime classifier loaded: {path}")

def main():
    """Test the regime classifier"""
    
    config = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_leaf': 20
    }
    
    classifier = RegimeClassifier(config)
    
    # Train
    results = classifier.train('2020-01-01', '2024-12-31')
    print(f"Training results: {results}")
    
    # Predict current regime
    current_regime = classifier.predict_regime()
    print(f"Current regime: {current_regime}")
    
    # Get position sizing
    sizing = classifier.get_position_sizing_multiplier(
        current_regime['regime'], 
        current_regime['confidence']
    )
    print(f"Position sizing: {sizing}")

if __name__ == "__main__":
    main()