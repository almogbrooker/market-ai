#!/usr/bin/env python3
"""
Create Synthetic OOF Predictions for Conformal Calibration Demo
Since we don't have pre-trained models, create realistic synthetic OOF data
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
import pandas_ta as ta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_synthetic_oof(output_file="oof_predictions_for_conformal.csv", lookback_days=180):
    """Create synthetic but realistic OOF predictions"""
    
    logger.info("🚀 Creating synthetic OOF dataset for conformal calibration")
    
    # Stock universe - liquid NASDAQ names
    stocks = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD',
        'INTC', 'QCOM', 'AVGO', 'TXN', 'ORCL', 'CRM', 'ADBE', 'NOW',
        'PYPL', 'NFLX', 'CMCSA', 'PEP', 'COST', 'TMUS', 'UBER', 'ABNB'
    ]
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days + 50)  # Extra buffer
    
    all_data = []
    
    for symbol in tqdm(stocks, desc="Processing stocks"):
        try:
            # Download data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            
            if len(hist) < 50:
                continue
            
            df = hist.copy()
            df['symbol'] = symbol
            df['date'] = df.index
            
            # Technical indicators
            df['rsi'] = ta.rsi(df['Close'], length=14)
            df['macd'] = ta.macd(df['Close'])['MACD_12_26_9']
            df['sma_5'] = ta.sma(df['Close'], length=5)
            df['sma_20'] = ta.sma(df['Close'], length=20)
            df['ema_12'] = ta.ema(df['Close'], length=12)
            
            # Returns
            df['return_1d'] = df['Close'].pct_change(1)
            df['return_5d'] = df['Close'].pct_change(5)
            df['return_20d'] = df['Close'].pct_change(20)
            
            # Volume indicators
            df['volume_sma'] = df['Volume'].rolling(20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma']
            
            # Volatility
            df['volatility'] = df['return_1d'].rolling(20).std() * np.sqrt(252)
            
            # Target (next day return)
            df['target'] = df['return_1d'].shift(-1)
            
            # Market regime detection
            df['vix'] = 20.0  # Simplified VIX proxy
            df['regime'] = 'neutral'
            
            # Bull: positive 20-day returns + low volatility
            bull_mask = (df['return_20d'] > 0.03) & (df['volatility'] < 0.30)
            df.loc[bull_mask, 'regime'] = 'bull'
            
            # Bear: negative returns + high volatility  
            bear_mask = (df['return_20d'] < -0.03) & (df['volatility'] > 0.35)
            df.loc[bear_mask, 'regime'] = 'bear'
            
            # Filter to desired date range and drop NaN
            df = df[df['date'] >= end_date - timedelta(days=lookback_days)]
            df = df.dropna()
            
            if len(df) < 20:
                continue
                
            # Generate realistic synthetic predictions
            # Model prediction = f(technical indicators) + noise
            
            # Simple momentum signal
            momentum_signal = (df['return_5d'] * 0.3 + 
                             (df['Close'] / df['sma_20'] - 1) * 0.4 +
                             (50 - df['rsi']) / 100 * 0.2 +
                             df['macd'] * 0.1)
            
            # Add regime-specific adjustments
            regime_mult = df['regime'].map({'bull': 1.2, 'bear': 0.8, 'neutral': 1.0})
            momentum_signal *= regime_mult
            
            # Add realistic noise and bias
            signal_noise = np.random.normal(0, 0.01, len(df))
            prediction_bias = np.random.normal(0, 0.005, len(df))  # Slight model bias
            
            df['prediction'] = momentum_signal + signal_noise + prediction_bias
            
            # Clip extreme predictions
            df['prediction'] = np.clip(df['prediction'], -0.1, 0.1)
            
            # Add model confidence (inversely related to volatility)
            df['confidence'] = np.clip(1.0 / (df['volatility'] + 0.1), 0.3, 1.0)
            
            # Add some realistic prediction errors
            # Good models have IC ~0.05-0.15 in practice
            target_ic = 0.08  # Realistic information coefficient
            
            # Adjust predictions to achieve target IC
            actual_ic = df['prediction'].corr(df['target'])
            if not np.isnan(actual_ic) and abs(actual_ic) > 0.01:
                adjustment_factor = target_ic / actual_ic
                df['prediction'] *= adjustment_factor
            
            # Add metadata
            df['abs_prediction'] = np.abs(df['prediction'])
            df['abs_target'] = np.abs(df['target'])
            
            # Select final columns
            final_cols = ['symbol', 'date', 'prediction', 'target', 'regime', 
                         'confidence', 'abs_prediction', 'abs_target', 'volatility']
            df = df[final_cols]
            
            all_data.append(df)
            
        except Exception as e:
            logger.warning(f"Error processing {symbol}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No data generated")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.dropna()
    
    # Save to file
    combined_df.to_csv(output_file, index=False)
    
    # Statistics
    ic_by_regime = combined_df.groupby('regime').apply(
        lambda x: x['prediction'].corr(x['target'])
    )
    
    logger.info(f"✅ Created synthetic OOF dataset: {len(combined_df)} predictions")
    logger.info(f"📊 Dataset statistics:")
    logger.info(f"   Symbols: {combined_df['symbol'].nunique()}")
    logger.info(f"   Date range: {combined_df['date'].min().date()} to {combined_df['date'].max().date()}")
    logger.info(f"   Regimes: {combined_df['regime'].value_counts().to_dict()}")
    logger.info(f"   Overall IC: {combined_df['prediction'].corr(combined_df['target']):.3f}")
    logger.info(f"   IC by regime: {ic_by_regime.to_dict()}")
    logger.info(f"   Mean |prediction|: {combined_df['abs_prediction'].mean():.4f}")
    logger.info(f"   Mean |target|: {combined_df['abs_target'].mean():.4f}")
    
    return combined_df

if __name__ == "__main__":
    # Create synthetic OOF dataset
    oof_df = create_synthetic_oof()
    
    print("\n" + "="*60)
    print("🎯 SYNTHETIC OOF GENERATION COMPLETED")
    print("="*60)
    print(f"📁 Output file: oof_predictions_for_conformal.csv")
    print(f"📊 Total samples: {len(oof_df):,}")
    print(f"📈 Ready for conformal calibration!")
    print("="*60)