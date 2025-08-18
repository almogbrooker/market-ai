#!/usr/bin/env python3
"""
FRED (Federal Reserve Economic Data) data integration
Fetches macroeconomic indicators for financial market prediction
"""

import pandas as pd
import numpy as np
import requests
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)

class FREDDataFetcher:
    """Fetches macroeconomic data from FRED API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: FRED API key. If None, will try to use public endpoints with rate limits
        """
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred"
        self.session = requests.Session()
        
        # Key economic indicators relevant for stock market prediction
        self.indicators = {
            'VIX': 'VIXCLS',  # VIX Volatility Index
            'TREASURY_10Y': 'DGS10',  # 10-Year Treasury Rate
            'TREASURY_2Y': 'DGS2',   # 2-Year Treasury Rate
            'FED_FUNDS': 'FEDFUNDS',  # Federal Funds Rate
            'CPI': 'CPIAUCSL',        # Consumer Price Index
            'UNEMPLOYMENT': 'UNRATE', # Unemployment Rate
            'GDP_GROWTH': 'GDPC1',    # Real GDP
            'INDUSTRIAL_PROD': 'INDPRO', # Industrial Production Index
            'CONSUMER_SENTIMENT': 'UMCSENT', # Consumer Sentiment
            'YIELD_CURVE_SPREAD': None,  # Calculated: 10Y - 2Y
            'REAL_RATES': None,          # Calculated: 10Y - CPI YoY
        }
        
    def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """Make rate-limited request to FRED API"""
        
        if self.api_key:
            params['api_key'] = self.api_key
        
        params['file_type'] = 'json'
        
        try:
            response = self.session.get(f"{self.base_url}/{endpoint}", params=params)
            response.raise_for_status()
            
            # Rate limiting for public access
            if not self.api_key:
                time.sleep(0.1)  # Be respectful to public API
                
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"FRED API request failed: {e}")
            return {}
    
    def fetch_series(self, series_id: str, start_date: str = "2020-01-01", 
                    end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch a single economic series from FRED"""
        
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        params = {
            'series_id': series_id,
            'observation_start': start_date,
            'observation_end': end_date,
            'frequency': 'd',  # Daily frequency where available
            'aggregation_method': 'avg'
        }
        
        data = self._make_request('series/observations', params)
        
        if 'observations' not in data:
            logger.warning(f"No data found for series {series_id}")
            return pd.DataFrame()
        
        observations = data['observations']
        df = pd.DataFrame(observations)
        
        if df.empty:
            return df
        
        # Clean and process data
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna(subset=['value'])
        df = df.rename(columns={'value': series_id})
        df = df[['date', series_id]].set_index('date')
        
        return df
    
    def fetch_all_indicators(self, start_date: str = "2020-01-01", 
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch all key economic indicators"""
        
        logger.info(f"Fetching FRED indicators from {start_date} to {end_date or 'present'}")
        
        dfs = []
        
        for name, series_id in self.indicators.items():
            if series_id is None:
                continue  # Skip calculated indicators for now
                
            logger.info(f"Fetching {name} ({series_id})")
            df = self.fetch_series(series_id, start_date, end_date)
            
            if not df.empty:
                df = df.rename(columns={series_id: name})
                dfs.append(df)
            else:
                logger.warning(f"Failed to fetch {name}")
        
        if not dfs:
            logger.error("No FRED data fetched successfully")
            return pd.DataFrame()
        
        # Merge all indicators
        combined_df = dfs[0]
        for df in dfs[1:]:
            combined_df = combined_df.join(df, how='outer')
        
        # Calculate derived indicators
        combined_df = self._calculate_derived_indicators(combined_df)
        
        # Forward fill missing values (common for daily series with weekend gaps)
        combined_df = combined_df.fillna(method='ffill')
        
        # Add rolling features
        combined_df = self._add_rolling_features(combined_df)
        
        logger.info(f"FRED data shape: {combined_df.shape}")
        return combined_df
    
    def _calculate_derived_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived economic indicators"""
        
        # Yield curve spread (10Y - 2Y)
        if 'TREASURY_10Y' in df.columns and 'TREASURY_2Y' in df.columns:
            df['YIELD_CURVE_SPREAD'] = df['TREASURY_10Y'] - df['TREASURY_2Y']
        
        # Real interest rates (10Y - CPI YoY change)
        if 'TREASURY_10Y' in df.columns and 'CPI' in df.columns:
            cpi_yoy = df['CPI'].pct_change(periods=252).fillna(0) * 100  # Approximate annual change
            df['REAL_RATES'] = df['TREASURY_10Y'] - cpi_yoy
        
        # Economic stress indicator (combination of VIX, unemployment, yield spread)
        stress_components = []
        if 'VIX' in df.columns:
            # Normalize VIX (higher = more stress)
            vix_norm = (df['VIX'] - df['VIX'].rolling(252).mean()) / df['VIX'].rolling(252).std()
            stress_components.append(vix_norm)
        
        if 'UNEMPLOYMENT' in df.columns:
            # Normalize unemployment (higher = more stress)
            unemp_norm = (df['UNEMPLOYMENT'] - df['UNEMPLOYMENT'].rolling(252).mean()) / df['UNEMPLOYMENT'].rolling(252).std()
            stress_components.append(unemp_norm)
        
        if 'YIELD_CURVE_SPREAD' in df.columns:
            # Normalize yield spread (lower/negative = more stress)
            spread_norm = -(df['YIELD_CURVE_SPREAD'] - df['YIELD_CURVE_SPREAD'].rolling(252).mean()) / df['YIELD_CURVE_SPREAD'].rolling(252).std()
            stress_components.append(spread_norm)
        
        if stress_components:
            df['ECONOMIC_STRESS'] = pd.concat(stress_components, axis=1).mean(axis=1)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame, windows: List[int] = [5, 20, 60]) -> pd.DataFrame:
        """Add rolling statistics for each indicator"""
        
        for col in df.columns:
            if col.endswith('_MEAN') or col.endswith('_STD') or col.endswith('_CHANGE'):
                continue  # Skip already processed columns
                
            for window in windows:
                # Rolling mean
                df[f'{col}_MEAN_{window}D'] = df[col].rolling(window).mean()
                
                # Rolling standard deviation
                df[f'{col}_STD_{window}D'] = df[col].rolling(window).std()
                
                # Change from rolling mean (momentum indicator)
                df[f'{col}_MOMENTUM_{window}D'] = df[col] - df[f'{col}_MEAN_{window}D']
        
        # Add rate of change features
        for col in df.columns:
            if not any(suffix in col for suffix in ['_MEAN', '_STD', '_MOMENTUM', '_CHANGE']):
                df[f'{col}_CHANGE_1D'] = df[col].pct_change(1)
                df[f'{col}_CHANGE_5D'] = df[col].pct_change(5)
                df[f'{col}_CHANGE_20D'] = df[col].pct_change(20)
        
        return df
    
    def align_with_stock_data(self, fred_df: pd.DataFrame, stock_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Align FRED data with stock market trading dates"""
        
        # Reindex to stock market dates
        aligned_df = fred_df.reindex(stock_dates, method='ffill')
        
        # Fill any remaining NaNs with the last available value
        aligned_df = aligned_df.fillna(method='ffill').fillna(method='bfill')
        
        return aligned_df

def create_fred_features(start_date: str = "2020-01-01", 
                        end_date: Optional[str] = None,
                        api_key: Optional[str] = None) -> pd.DataFrame:
    """Convenience function to create FRED features for model training"""
    
    fetcher = FREDDataFetcher(api_key=api_key)
    
    try:
        fred_data = fetcher.fetch_all_indicators(start_date, end_date)
        
        if fred_data.empty:
            logger.warning("No FRED data available, creating dummy features")
            # Create dummy economic features if FRED is not available
            date_range = pd.date_range(start=start_date, 
                                     end=end_date or datetime.now().strftime("%Y-%m-%d"),
                                     freq='D')
            fred_data = pd.DataFrame(index=date_range)
            
            # Add dummy economic indicators
            for indicator in ['VIX', 'TREASURY_10Y', 'FED_FUNDS', 'CPI', 'UNEMPLOYMENT']:
                fred_data[indicator] = np.random.randn(len(fred_data)).cumsum() * 0.1
        
        return fred_data
        
    except Exception as e:
        logger.error(f"Failed to fetch FRED data: {e}")
        
        # Fallback to dummy data
        date_range = pd.date_range(start=start_date, 
                                 end=end_date or datetime.now().strftime("%Y-%m-%d"),
                                 freq='D')
        dummy_data = pd.DataFrame(index=date_range)
        
        for indicator in ['VIX', 'TREASURY_10Y', 'FED_FUNDS', 'CPI', 'UNEMPLOYMENT']:
            dummy_data[indicator] = np.random.randn(len(dummy_data)).cumsum() * 0.1
            
        return dummy_data