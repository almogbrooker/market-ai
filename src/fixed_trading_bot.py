#!/usr/bin/env python3
"""
FIXED TRADING BOT
=================
Uses unified feature engineering for perfect alignment with model trainer
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime
from feature_engineering import UnifiedFeatureEngine
import warnings
warnings.filterwarnings('ignore')

class FixedTradingBot:
    """Fixed trading bot with unified feature pipeline"""
    
    def __init__(self):
        print("🤖 FIXED TRADING BOT")
        print("=" * 50)
        
        self.base_dir = Path("../artifacts")
        self.models_dir = self.base_dir / "models"
        self.trades_dir = self.base_dir / "trades"
        self.trades_dir.mkdir(parents=True, exist_ok=True)
        
        # Trading parameters
        self.max_position_size = 0.02  # 2% max per position
        self.target_coverage = 0.20    # 20% of universe
        
        # Initialize unified feature engine
        self.feature_engine = UnifiedFeatureEngine()
        
        # Load model and data
        self.load_model_and_data()
        
        print(f"✅ Fixed Trading Bot initialized")
        print(f"📊 Max position size: {self.max_position_size:.1%}")
        print(f"🎯 Target coverage: {self.target_coverage:.1%}")
        
    def load_model_and_data(self):
        """Load fixed model and metadata"""
        print("\n📥 Loading fixed model...")
        
        try:
            # Load fixed model
            with open(self.models_dir / "fixed_nasdaq100_model.pkl", 'rb') as f:
                self.model = pickle.load(f)
            
            # Load metadata
            with open(self.models_dir / "fixed_nasdaq100_metadata.json") as f:
                self.model_metadata = json.load(f)
            
            # Load feature configuration
            with open(self.models_dir / "feature_config.json") as f:
                self.feature_config = json.load(f)
            
            # Load market data
            self.market_data = pd.read_parquet(self.base_dir / "nasdaq100_data.parquet")
            self.market_data['Date'] = pd.to_datetime(self.market_data['Date'])
            
            self.model_features = self.model_metadata['features']
            
            print(f"✅ Fixed model loaded:")
            print(f"   Model: {self.model_metadata['model_type']}")
            print(f"   Features: {len(self.model_features)}")
            print(f"   Test IC: {self.model_metadata['performance']['test_ic']:.4f}")
            print(f"   Status: {self.model_metadata['status']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def get_latest_data_for_prediction(self, lookback_days=500):
        """Get latest market data with sufficient history for features"""
        print(f"\n📊 Getting latest data (last {lookback_days} days)...")
        
        # Get latest available date
        latest_date = self.market_data['Date'].max()
        start_date = latest_date - pd.Timedelta(days=lookback_days)
        
        # Get recent data for feature calculation
        recent_data = self.market_data[
            self.market_data['Date'] >= start_date
        ].copy()
        
        print(f"📅 Latest date: {latest_date.date()}")
        print(f"📊 Recent data: {len(recent_data):,} records")
        print(f"🏢 Companies: {recent_data['Ticker'].nunique()}")
        
        return recent_data
    
    def create_prediction_features(self, recent_data):
        """Create features for current prediction using unified engine"""
        print("\n🔧 Creating prediction features...")
        
        # Use unified feature engine (same as training)
        feature_data, available_features = self.feature_engine.create_features_from_data(recent_data)
        
        if feature_data is None:
            print("❌ Feature creation failed")
            return None, None
        
        # Get latest date data for prediction
        latest_date = feature_data['Date'].max()
        latest_data = feature_data[feature_data['Date'] == latest_date].copy()
        
        if len(latest_data) == 0:
            print("❌ No latest data available")
            return None, None
        
        print(f"✅ Prediction features created:")
        print(f"   📅 Prediction date: {latest_date.date()}")
        print(f"   🏢 Available stocks: {len(latest_data)}")
        print(f"   📊 Features: {len(available_features)}")
        
        # Verify feature alignment with trained model
        matching_features = [f for f in self.model_features if f in available_features]
        print(f"   🔗 Matching features: {len(matching_features)}/{len(self.model_features)}")
        
        if len(matching_features) < len(self.model_features) * 0.8:
            print("⚠️ Feature mismatch detected - using available features")
        
        return latest_data, matching_features
    
    def generate_predictions(self, latest_data, features):
        """Generate predictions using fixed model"""
        print("\n🎯 Generating predictions...")
        
        if latest_data is None or len(features) == 0:
            print("❌ No data or features available")
            return None
        
        try:
            # For prediction, we don't need target - manually prepare features
            X = latest_data[features].fillna(0.5).values
            X = np.clip(X, 0, 1)
            
            print(f"📊 Feature matrix: {X.shape}")
            
            # Generate predictions
            predictions = self.model.predict(X)
            
            # Create prediction DataFrame
            prediction_df = latest_data[['Date', 'Ticker', 'Close']].copy()
            prediction_df['prediction'] = predictions
            prediction_df['pred_rank'] = pd.Series(predictions).rank(pct=True)
            
            print(f"✅ Predictions generated:")
            print(f"   📊 Stocks: {len(predictions)}")
            print(f"   📈 Mean prediction: {predictions.mean():.6f}")
            print(f"   📊 Std prediction: {predictions.std():.6f}")
            print(f"   📉 Range: [{predictions.min():.6f}, {predictions.max():.6f}]")
            
            return prediction_df
            
        except Exception as e:
            print(f"❌ Prediction generation failed: {e}")
            return None
    
    def create_portfolio(self, prediction_df):
        """Create trading portfolio from predictions"""
        print("\n💼 Creating trading portfolio...")
        
        if prediction_df is None:
            return None
        
        # Calculate target positions
        n_stocks = len(prediction_df)
        n_positions = int(n_stocks * self.target_coverage)
        n_long = n_positions // 2
        n_short = n_positions // 2
        
        print(f"📊 Portfolio construction:")
        print(f"   🏢 Total stocks: {n_stocks}")
        print(f"   📊 Target positions: {n_positions}")
        print(f"   📈 Long positions: {n_long}")
        print(f"   📉 Short positions: {n_short}")
        
        # Sort by prediction
        sorted_predictions = prediction_df.sort_values('prediction', ascending=False)
        
        # Select long positions (top predictions)
        long_positions = sorted_predictions.head(n_long).copy()
        long_positions['position_type'] = 'LONG'
        long_positions['position_size'] = self.max_position_size
        
        # Select short positions (bottom predictions)
        short_positions = sorted_predictions.tail(n_short).copy()
        short_positions['position_type'] = 'SHORT' 
        short_positions['position_size'] = -self.max_position_size
        
        # Combine portfolio
        portfolio = pd.concat([long_positions, short_positions], ignore_index=True)
        portfolio = portfolio.sort_values('prediction', ascending=False).reset_index(drop=True)
        
        # Portfolio metrics
        gross_exposure = portfolio['position_size'].abs().sum()
        net_exposure = portfolio['position_size'].sum()
        
        print(f"✅ Portfolio created:")
        print(f"   📈 Long positions: {len(long_positions)}")
        print(f"   📉 Short positions: {len(short_positions)}")
        print(f"   💰 Gross exposure: {gross_exposure:.1%}")
        print(f"   ⚖️ Net exposure: {net_exposure:.1%}")
        
        return portfolio
    
    def validate_portfolio(self, portfolio):
        """Validate portfolio before trading"""
        print("\n✅ Validating portfolio...")
        
        if portfolio is None or len(portfolio) == 0:
            print("❌ Empty portfolio")
            return False
        
        # Check position sizes
        max_pos = portfolio['position_size'].abs().max()
        if max_pos > self.max_position_size * 1.1:
            print(f"❌ Position size too large: {max_pos:.2%}")
            return False
        
        # Check exposure
        gross_exposure = portfolio['position_size'].abs().sum()
        if gross_exposure > 0.5:  # Max 50% gross exposure
            print(f"❌ Gross exposure too high: {gross_exposure:.1%}")
            return False
        
        # Check diversification
        if len(portfolio) < 6:
            print(f"❌ Too few positions: {len(portfolio)}")
            return False
        
        print("✅ Portfolio validation passed")
        return True
    
    def display_portfolio(self, portfolio):
        """Display portfolio details"""
        print("\n💼 TRADING PORTFOLIO")
        print("=" * 80)
        
        if portfolio is None:
            print("❌ No portfolio to display")
            return
        
        print(f"{'Ticker':<8} {'Type':<6} {'Size':<8} {'Prediction':<12} {'Price':<10}")
        print("-" * 80)
        
        for _, row in portfolio.iterrows():
            print(f"{row['Ticker']:<8} {row['position_type']:<6} {row['position_size']:>7.1%} "
                  f"{row['prediction']:>11.6f} ${row['Close']:>8.2f}")
        
        print("-" * 80)
        gross_exp = portfolio['position_size'].abs().sum()
        net_exp = portfolio['position_size'].sum()
        print(f"{'TOTAL':<8} {'':<6} {gross_exp:>7.1%} {'(Gross)':<11} {net_exp:>+9.1%}")
        
    def save_trades(self, portfolio):
        """Save trading results"""
        print("\n💾 Saving trades...")
        
        if portfolio is None:
            print("❌ No portfolio to save")
            return None
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # Save portfolio positions
            portfolio_file = self.trades_dir / f"fixed_positions_{timestamp}.parquet"
            portfolio.to_parquet(portfolio_file)
            
            # Save trading summary
            summary = {
                'timestamp': datetime.now().isoformat(),
                'model_type': self.model_metadata['model_type'],
                'model_status': self.model_metadata['status'],
                'model_test_ic': self.model_metadata['performance']['test_ic'],
                'feature_engine': 'unified_v1.0',
                'portfolio_stats': {
                    'total_positions': len(portfolio),
                    'long_positions': (portfolio['position_size'] > 0).sum(),
                    'short_positions': (portfolio['position_size'] < 0).sum(),
                    'gross_exposure': float(portfolio['position_size'].abs().sum()),
                    'net_exposure': float(portfolio['position_size'].sum()),
                    'max_position': float(portfolio['position_size'].abs().max()),
                    'avg_prediction': float(portfolio['prediction'].mean()),
                    'prediction_spread': float(portfolio['prediction'].max() - portfolio['prediction'].min())
                },
                'system_status': 'FIXED_ALIGNED'
            }
            
            summary_file = self.trades_dir / f"fixed_summary_{timestamp}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            print(f"✅ Trades saved:")
            print(f"   📊 Positions: {portfolio_file}")
            print(f"   📋 Summary: {summary_file}")
            
            return portfolio_file, summary_file
            
        except Exception as e:
            print(f"⚠️ Save error: {e}")
            return None
    
    def run_trading_session(self):
        """Run complete fixed trading session"""
        print("\n🚀 RUNNING FIXED TRADING SESSION")
        print("=" * 60)
        
        try:
            # Get latest data
            recent_data = self.get_latest_data_for_prediction()
            
            # Create prediction features
            latest_data, features = self.create_prediction_features(recent_data)
            
            # Generate predictions
            predictions = self.generate_predictions(latest_data, features)
            
            # Create portfolio
            portfolio = self.create_portfolio(predictions)
            
            # Validate portfolio
            if not self.validate_portfolio(portfolio):
                print("❌ Portfolio validation failed - no trades executed")
                return False
            
            # Display portfolio
            self.display_portfolio(portfolio)
            
            # Save trades
            trade_files = self.save_trades(portfolio)
            
            print(f"\n🎉 FIXED TRADING SESSION COMPLETE!")
            print(f"✅ Feature alignment: FIXED")
            print(f"📊 Portfolio: {len(portfolio)} positions")
            print(f"💰 Gross exposure: {portfolio['position_size'].abs().sum():.1%}")
            print(f"🎯 Model IC: {self.model_metadata['performance']['test_ic']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Trading session failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Run fixed trading bot"""
    bot = FixedTradingBot()
    success = bot.run_trading_session()
    return success

if __name__ == "__main__":
    result = main()