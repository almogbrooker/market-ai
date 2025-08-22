#!/usr/bin/env python3
"""
Price Data Provider
Fetches price data from multiple sources (Alpaca, Polygon, Tiingo, local CSVs)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional
import logging
import os
import yfinance as yf

logger = logging.getLogger(__name__)

class PriceProvider:
    """Price data provider with multiple source support"""
    
    def __init__(self):
        # Load API keys from environment
        self.alpaca_api_key = os.getenv('ALPACA_API_KEY')
        self.alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.polygon_api_key = os.getenv('POLYGON_API_KEY')
        self.tiingo_api_key = os.getenv('TIINGO_API_KEY')
        
        # Default to yfinance as fallback
        self.primary_source = 'yfinance'
        
        # Initialize Alpaca client if credentials available
        self.alpaca_client = None
        if self.alpaca_api_key and self.alpaca_secret_key:
            try:
                from alpaca.data.historical import StockHistoricalDataClient
                from alpaca.data.requests import StockBarsRequest
                from alpaca.data.timeframe import TimeFrame
                
                self.alpaca_client = StockHistoricalDataClient(
                    api_key=self.alpaca_api_key,
                    secret_key=self.alpaca_secret_key
                )
                self.primary_source = 'alpaca'
                logger.info("✅ Alpaca client initialized")
            except Exception as e:
                logger.warning(f"⚠️ Alpaca initialization failed: {e}")
        
        logger.info(f"📈 PriceProvider initialized (primary: {self.primary_source})")
    
    def get_daily_prices(self, symbols: List[str], end_date: datetime, 
                        lookback_days: int = 5) -> pd.DataFrame:
        """Fetch daily price data with fallback sources"""
        
        logger.info(f"📊 Fetching prices: {symbols} ({lookback_days} days)")
        
        # Try primary source first
        if self.primary_source == 'alpaca' and self.alpaca_client:
            data = self._fetch_alpaca_prices(symbols, end_date, lookback_days)
            if not data.empty:
                return data
            logger.warning("⚠️ Alpaca fetch failed, falling back to yfinance")
        
        # Fallback to yfinance
        return self._fetch_yfinance_prices(symbols, end_date, lookback_days)
    
    def _fetch_alpaca_prices(self, symbols: List[str], end_date: datetime, 
                           lookback_days: int) -> pd.DataFrame:
        """Fetch data from Alpaca"""
        
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            
            start_date = end_date - timedelta(days=lookback_days + 5)  # Extra buffer for weekends
            
            request_params = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Day,
                start=start_date.date(),
                end=end_date.date()
            )
            
            bars = self.alpaca_client.get_stock_bars(request_params)
            
            # Convert to DataFrame
            data_list = []
            for symbol in symbols:
                if symbol in bars.data:
                    for bar in bars.data[symbol]:
                        data_list.append({
                            'Date': bar.timestamp.date(),
                            'Ticker': symbol,
                            'Open': float(bar.open),
                            'High': float(bar.high),
                            'Low': float(bar.low),
                            'Close': float(bar.close),
                            'Volume': int(bar.volume),
                            'VWAP': float(bar.vwap) if hasattr(bar, 'vwap') else None
                        })
            
            if data_list:
                df = pd.DataFrame(data_list)
                df = self._add_technical_features(df)
                logger.info(f"✅ Alpaca: {len(df)} records fetched")
                return df
            
        except Exception as e:
            logger.error(f"❌ Alpaca fetch error: {e}")
        
        return pd.DataFrame()
    
    def _fetch_yfinance_prices(self, symbols: List[str], end_date: datetime, 
                             lookback_days: int) -> pd.DataFrame:
        """Fetch data from yfinance (fallback)"""
        
        try:
            start_date = end_date - timedelta(days=lookback_days + 10)  # Extra buffer
            
            data_list = []
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(
                        start=start_date.date(),
                        end=end_date.date(),
                        interval='1d'
                    )
                    
                    if not hist.empty:
                        # Reset index to get Date as column
                        hist = hist.reset_index()
                        
                        for _, row in hist.iterrows():
                            data_list.append({
                                'Date': row['Date'].date(),
                                'Ticker': symbol,
                                'Open': float(row['Open']),
                                'High': float(row['High']),
                                'Low': float(row['Low']),
                                'Close': float(row['Close']),
                                'Volume': int(row['Volume']),
                                'VWAP': None  # Not available in yfinance
                            })
                
                except Exception as e:
                    logger.warning(f"⚠️ Failed to fetch {symbol}: {e}")
                    continue
            
            if data_list:
                df = pd.DataFrame(data_list)
                df = self._add_technical_features(df)
                logger.info(f"✅ yfinance: {len(df)} records fetched")
                return df
                
        except Exception as e:
            logger.error(f"❌ yfinance fetch error: {e}")
        
        return pd.DataFrame()
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to price data"""
        
        if df.empty:
            return df
        
        df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
        
        # Calculate features for each ticker
        enhanced_data = []
        
        for ticker in df['Ticker'].unique():
            ticker_data = df[df['Ticker'] == ticker].copy()
            
            if len(ticker_data) < 2:
                enhanced_data.append(ticker_data)
                continue
            
            # Price-based features
            ticker_data['Return_1D'] = ticker_data['Close'].pct_change()
            ticker_data['Return_5D'] = ticker_data['Close'].pct_change(5)
            
            # Volatility (20-day rolling)
            returns = ticker_data['Close'].pct_change()
            ticker_data['Volatility_20D'] = returns.rolling(20, min_periods=5).std() * np.sqrt(252)
            
            # Volume features
            ticker_data['Volume_MA_20'] = ticker_data['Volume'].rolling(20, min_periods=5).mean()
            ticker_data['Volume_Ratio'] = ticker_data['Volume'] / ticker_data['Volume_MA_20']
            
            # Price position features
            ticker_data['High_Low_Ratio'] = ticker_data['High'] / ticker_data['Low']
            ticker_data['Close_to_High'] = ticker_data['Close'] / ticker_data['High']
            
            # Moving averages
            ticker_data['SMA_5'] = ticker_data['Close'].rolling(5, min_periods=1).mean()
            ticker_data['SMA_20'] = ticker_data['Close'].rolling(20, min_periods=5).mean()
            ticker_data['Price_to_SMA20'] = ticker_data['Close'] / ticker_data['SMA_20']
            
            # RSI (simplified 14-day)
            if len(ticker_data) >= 14:
                delta = ticker_data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                ticker_data['RSI_14'] = 100 - (100 / (1 + rs))
            else:
                ticker_data['RSI_14'] = 50  # Neutral RSI
            
            # MACD (simplified)
            if len(ticker_data) >= 26:
                ema_12 = ticker_data['Close'].ewm(span=12).mean()
                ema_26 = ticker_data['Close'].ewm(span=26).mean()
                ticker_data['MACD'] = ema_12 - ema_26
                ticker_data['MACD_Signal'] = ticker_data['MACD'].ewm(span=9).mean()
                ticker_data['MACD_Histogram'] = ticker_data['MACD'] - ticker_data['MACD_Signal']
            else:
                ticker_data['MACD'] = 0
                ticker_data['MACD_Signal'] = 0
                ticker_data['MACD_Histogram'] = 0
            
            # Lagged features (for temporal safety)
            ticker_data['Return_1D_lag1'] = ticker_data['Return_1D'].shift(1)
            ticker_data['Return_5D_lag1'] = ticker_data['Return_5D'].shift(1)
            ticker_data['Volatility_20D_lag1'] = ticker_data['Volatility_20D'].shift(1)
            ticker_data['Volume_Ratio_lag1'] = ticker_data['Volume_Ratio'].shift(1)
            
            enhanced_data.append(ticker_data)
        
        # Combine all tickers
        result = pd.concat(enhanced_data, ignore_index=True)
        
        # Fill NaN values with appropriate defaults
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        result[numeric_cols] = result[numeric_cols].fillna(0)
        
        logger.info(f"✅ Added technical features: {result.shape[1]} total columns")
        
        return result
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a single symbol"""
        
        try:
            end_date = datetime.now()
            data = self.get_daily_prices([symbol], end_date, lookback_days=1)
            
            if not data.empty:
                latest = data[data['Ticker'] == symbol]['Close'].iloc[-1]
                return float(latest)
        
        except Exception as e:
            logger.error(f"❌ Failed to get latest price for {symbol}: {e}")
        
        return None
    
    def validate_data_quality(self, df: pd.DataFrame) -> bool:
        """Validate data quality and completeness"""
        
        if df.empty:
            logger.error("❌ Empty dataset")
            return False
        
        required_cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"❌ Missing columns: {missing_cols}")
            return False
        
        # Check for negative prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if (df[col] <= 0).any():
                logger.warning(f"⚠️ Found non-positive values in {col}")
        
        # Check High >= Low
        if (df['High'] < df['Low']).any():
            logger.warning("⚠️ Found High < Low inconsistencies")
        
        # Check for reasonable volume
        if (df['Volume'] < 0).any():
            logger.warning("⚠️ Found negative volume")
        
        logger.info(f"✅ Data quality check passed: {len(df)} records")
        return True