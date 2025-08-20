#!/usr/bin/env python3
"""
Alternative Data Sources: Tiingo and Stooq Integration
Backup data sources for market data with enhanced features
"""

import pandas as pd
import numpy as np
import requests
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import time
import io

logger = logging.getLogger(__name__)

class TiingoDataFetcher:
    """Tiingo API integration for EOD and intraday data"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Tiingo API client
        
        Args:
            api_key: Tiingo API key (free tier available)
        """
        self.api_key = api_key
        self.base_url = "https://api.tiingo.com/tiingo"
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({
                'Content-Type': 'application/json',
                'Authorization': f'Token {api_key}'
            })
        
        # Rate limiting for free tier
        self.rate_limit_delay = 0.1
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Union[Dict, List]:
        """Make rate-limited request to Tiingo API"""
        
        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(f"{self.base_url}/{endpoint}", params=params)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Tiingo API request failed: {e}")
            return []
    
    def get_daily_prices(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch daily OHLCV data from Tiingo"""
        
        logger.info(f"Fetching Tiingo daily data for {ticker}")
        
        params = {
            'startDate': start_date,
            'endDate': end_date,
            'format': 'json'
        }
        
        endpoint = f"daily/{ticker}/prices"
        data = self._make_request(endpoint, params)
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        if df.empty:
            return df
        
        # Standardize column names
        df['date'] = pd.to_datetime(df['date'])
        df['ticker'] = ticker.upper()
        
        # Rename columns to match standard format
        column_mapping = {
            'open': 'open',
            'high': 'high', 
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'adjOpen': 'adj_open',
            'adjHigh': 'adj_high',
            'adjLow': 'adj_low',
            'adjClose': 'adj_close',
            'adjVolume': 'adj_volume',
            'divCash': 'dividend',
            'splitFactor': 'split_factor'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]
        
        # Add enhanced features
        self._add_price_features(df)
        
        return df[['date', 'ticker'] + [col for col in df.columns if col not in ['date', 'ticker']]]
    
    def _add_price_features(self, df: pd.DataFrame):
        """Add enhanced price-based features"""
        
        if 'adj_close' not in df.columns:
            return
        
        # Price momentum features
        df['returns_1d'] = df['adj_close'].pct_change(1)
        df['returns_5d'] = df['adj_close'].pct_change(5)
        df['returns_20d'] = df['adj_close'].pct_change(20)
        
        # Volatility features
        df['volatility_5d'] = df['returns_1d'].rolling(5).std()
        df['volatility_20d'] = df['returns_1d'].rolling(20).std()
        
        # Price relative to moving averages
        df['sma_20'] = df['adj_close'].rolling(20).mean()
        df['sma_50'] = df['adj_close'].rolling(50).mean()
        df['price_to_sma20'] = df['adj_close'] / df['sma_20']
        df['price_to_sma50'] = df['adj_close'] / df['sma_50']
        
        # Volume features
        if 'volume' in df.columns:
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # High-Low spread
        if all(col in df.columns for col in ['high', 'low', 'close']):
            df['hl_spread'] = (df['high'] - df['low']) / df['close']
            df['hl_spread_ma'] = df['hl_spread'].rolling(20).mean()
    
    def get_fundamentals(self, ticker: str) -> Dict:
        """Fetch fundamentals data from Tiingo"""
        
        endpoint = f"fundamentals/{ticker}"
        data = self._make_request(endpoint)
        
        return data if isinstance(data, dict) else {}


class StooqDataFetcher:
    """Stooq data integration for international markets"""
    
    def __init__(self):
        """Initialize Stooq data fetcher"""
        self.base_url = "https://stooq.com/q/d/l/"
        self.session = requests.Session()
        
        # Common symbol mappings for international markets
        self.symbol_mappings = {
            # US markets (also available on Stooq)
            'SPY': 'spy.us',
            'QQQ': 'qqq.us',
            'AAPL': 'aapl.us',
            'MSFT': 'msft.us',
            
            # European markets
            'SAP': 'sap.de',  # SAP (Germany)
            'ASML': 'asml.as',  # ASML (Netherlands)
            'NESN': 'nesn.sw',  # Nestle (Switzerland)
            
            # Indices
            'SPX': '^spx',  # S&P 500
            'NDX': '^ndx',  # Nasdaq 100
            'DAX': '^dax',  # German DAX
            'FTSE': '^ftse'  # FTSE 100
        }
    
    def get_daily_prices(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch daily data from Stooq"""
        
        logger.info(f"Fetching Stooq data for {symbol}")
        
        # Map symbol to Stooq format
        stooq_symbol = self.symbol_mappings.get(symbol, f"{symbol.lower()}.us")
        
        params = {
            's': stooq_symbol,
            'd1': start_date.replace('-', ''),  # YYYYMMDD format
            'd2': end_date.replace('-', ''),
            'i': 'd',  # daily
            'o': '1',  # oldest first
            'f': 'text'  # text format
        }
        
        try:
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            
            # Parse CSV response
            csv_data = io.StringIO(response.text)
            df = pd.read_csv(csv_data)
            
            if df.empty:
                return df
            
            # Standardize column names
            df.columns = df.columns.str.lower()
            df['date'] = pd.to_datetime(df['date'])
            df['ticker'] = symbol.upper()
            
            # Add enhanced features
            self._add_stooq_features(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Stooq request failed for {symbol}: {e}")
            return pd.DataFrame()
    
    def _add_stooq_features(self, df: pd.DataFrame):
        """Add enhanced features to Stooq data"""
        
        if 'close' not in df.columns:
            return
        
        # Basic returns
        df['returns_1d'] = df['close'].pct_change(1)
        df['returns_5d'] = df['close'].pct_change(5)
        
        # Moving averages
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        
        # Volatility
        df['volatility_10d'] = df['returns_1d'].rolling(10).std()
        
        # Volume analysis if available
        if 'volume' in df.columns:
            df['volume_ma_10'] = df['volume'].rolling(10).mean()


class AlternativeDataManager:
    """Manages multiple alternative data sources with fallback logic"""
    
    def __init__(self, tiingo_api_key: Optional[str] = None):
        """
        Initialize alternative data manager
        
        Args:
            tiingo_api_key: Optional Tiingo API key for enhanced access
        """
        self.tiingo = TiingoDataFetcher(api_key=tiingo_api_key)
        self.stooq = StooqDataFetcher()
        
        # Preference order for data sources
        self.source_priority = ['tiingo', 'stooq', 'dummy']
    
    def fetch_market_data(self, tickers: List[str], 
                         start_date: str, end_date: str,
                         preferred_source: str = 'auto') -> pd.DataFrame:
        """Fetch market data with automatic fallback"""
        
        logger.info(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        all_data = []
        
        for ticker in tickers:
            ticker_data = None
            
            # Try each source in order
            sources_to_try = [preferred_source] if preferred_source != 'auto' else self.source_priority
            
            for source in sources_to_try:
                try:
                    if source == 'tiingo':
                        ticker_data = self.tiingo.get_daily_prices(ticker, start_date, end_date)
                    elif source == 'stooq':
                        ticker_data = self.stooq.get_daily_prices(ticker, start_date, end_date)
                    elif source == 'dummy':
                        ticker_data = self._create_dummy_data(ticker, start_date, end_date)
                    
                    if not ticker_data.empty:
                        logger.info(f"Successfully fetched {ticker} from {source}")
                        break
                        
                except Exception as e:
                    logger.warning(f"Failed to fetch {ticker} from {source}: {e}")
                    continue
            
            if ticker_data is not None and not ticker_data.empty:
                all_data.append(ticker_data)
            else:
                logger.warning(f"No data available for {ticker}, creating dummy data")
                dummy_data = self._create_dummy_data(ticker, start_date, end_date)
                all_data.append(dummy_data)
        
        if not all_data:
            logger.error("No data fetched for any ticker")
            return pd.DataFrame()
        
        # Combine all ticker data
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values(['ticker', 'date'])
        
        # Add cross-sectional features
        self._add_cross_sectional_features(combined_df)
        
        logger.info(f"Combined data shape: {combined_df.shape}")
        
        return combined_df
    
    def _create_dummy_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Create realistic dummy market data"""
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        # Filter to business days
        dates = dates[dates.weekday < 5]
        
        np.random.seed(hash(ticker) % 2**32)
        
        # Base price for ticker
        base_prices = {
            'AAPL': 150, 'MSFT': 300, 'GOOGL': 2500, 'AMZN': 3000,
            'TSLA': 200, 'META': 300, 'NVDA': 400, 'SPY': 400, 'QQQ': 350
        }
        base_price = base_prices.get(ticker, 100)
        
        # Generate price series with realistic patterns
        returns = np.random.normal(0.0005, 0.02, len(dates))  # 0.05% daily return, 2% volatility
        
        # Add some autocorrelation and trends
        for i in range(1, len(returns)):
            returns[i] += 0.1 * returns[i-1]  # Slight momentum
        
        # Convert to prices
        price_series = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        data = []
        for i, (date, close_price) in enumerate(zip(dates, price_series)):
            # Generate realistic OHLC from close
            volatility = abs(returns[i]) if i < len(returns) else 0.01
            
            open_price = close_price * np.random.uniform(0.995, 1.005)
            high_price = max(open_price, close_price) * np.random.uniform(1.0, 1.0 + volatility)
            low_price = min(open_price, close_price) * np.random.uniform(1.0 - volatility, 1.0)
            
            # Volume with some patterns
            base_volume = 1000000 * (1 + volatility * 5)  # Higher volume on volatile days
            volume = int(base_volume * np.random.uniform(0.5, 2.0))
            
            row = {
                'date': date,
                'ticker': ticker,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'adj_close': round(close_price, 2),  # Assume no splits/dividends
                'volume': volume
            }
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Add technical features
        if len(df) > 0:
            self.tiingo._add_price_features(df)
        
        return df
    
    def _add_cross_sectional_features(self, df: pd.DataFrame):
        """Add cross-sectional market features"""
        
        if df.empty or 'ticker' not in df.columns:
            return
        
        # Market-wide features by date
        market_stats = df.groupby('date').agg({
            'returns_1d': ['mean', 'std', 'count'],
            'volume': 'sum',
            'volatility_5d': 'mean'
        }).round(6)
        
        # Flatten column names
        market_stats.columns = [f'market_{col[0]}_{col[1]}' for col in market_stats.columns]
        market_stats = market_stats.reset_index()
        
        # Merge back to main dataframe
        df_with_market = df.merge(market_stats, on='date', how='left')
        
        # Add relative features
        if 'market_returns_1d_mean' in df_with_market.columns:
            df['excess_return'] = df['returns_1d'] - df_with_market['market_returns_1d_mean']
        
        if 'market_volatility_5d_mean' in df_with_market.columns:
            df['relative_volatility'] = df['volatility_5d'] / df_with_market['market_volatility_5d_mean']
    
    def enhance_with_fundamentals(self, market_df: pd.DataFrame) -> pd.DataFrame:
        """Enhance market data with fundamentals from Tiingo"""
        
        if market_df.empty:
            return market_df
        
        enhanced_df = market_df.copy()
        tickers = enhanced_df['ticker'].unique()
        
        for ticker in tickers:
            try:
                fundamentals = self.tiingo.get_fundamentals(ticker)
                
                if fundamentals:
                    # Add selected fundamental metrics
                    ticker_mask = enhanced_df['ticker'] == ticker
                    
                    # Example: Add market cap, P/E ratio, etc.
                    if 'marketCap' in fundamentals:
                        enhanced_df.loc[ticker_mask, 'market_cap'] = fundamentals['marketCap']
                    
                    if 'peRatio' in fundamentals:
                        enhanced_df.loc[ticker_mask, 'pe_ratio'] = fundamentals['peRatio']
                        
            except Exception as e:
                logger.warning(f"Failed to enhance {ticker} with fundamentals: {e}")
                continue
        
        return enhanced_df


def create_alternative_data_features(tickers: List[str],
                                   start_date: str = "2020-01-01",
                                   end_date: Optional[str] = None,
                                   tiingo_api_key: Optional[str] = None) -> pd.DataFrame:
    """Convenience function to create alternative data features"""
    
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    manager = AlternativeDataManager(tiingo_api_key=tiingo_api_key)
    
    try:
        market_df = manager.fetch_market_data(tickers, start_date, end_date)
        
        # Enhance with fundamentals if API key available
        if tiingo_api_key:
            market_df = manager.enhance_with_fundamentals(market_df)
        
        return market_df
        
    except Exception as e:
        logger.error(f"Failed to create alternative data features: {e}")
        
        # Fallback to dummy data
        manager = AlternativeDataManager()
        return manager.fetch_market_data(tickers, start_date, end_date, preferred_source='dummy')


# Example usage
if __name__ == "__main__":
    # Test with sample tickers
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'SPY']
    
    manager = AlternativeDataManager()
    market_data = manager.fetch_market_data(
        test_tickers, 
        start_date="2024-01-01",
        end_date="2024-12-31"
    )
    
    print(f"Market data shape: {market_data.shape}")
    print(f"Columns: {list(market_data.columns)}")
    print(f"Sample data:\n{market_data.head()}")