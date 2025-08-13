import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    def __init__(self):
        self.validation_results = {}
        self.errors = []
        self.warnings = []
    
    def validate_price_data(self, df: pd.DataFrame, required_columns: List[str] = None) -> Dict[str, Any]:
        if required_columns is None:
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'missing_data_pct': {},
            'outliers': {},
            'data_quality_score': 0.0
        }
        
        # Check required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            results['errors'].append(f"Missing required columns: {missing_cols}")
            results['is_valid'] = False
        
        if not results['is_valid']:
            return results
        
        # Check for missing data
        for col in required_columns:
            missing_pct = df[col].isnull().sum() / len(df) * 100
            results['missing_data_pct'][col] = missing_pct
            
            if missing_pct > 10:
                results['errors'].append(f"Column {col} has {missing_pct:.1f}% missing data")
                results['is_valid'] = False
            elif missing_pct > 5:
                results['warnings'].append(f"Column {col} has {missing_pct:.1f}% missing data")
        
        # Validate price data logic
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            # High should be >= Open, Close, Low
            invalid_highs = (df['High'] < df[['Open', 'Close', 'Low']].max(axis=1)).sum()
            if invalid_highs > 0:
                results['errors'].append(f"{invalid_highs} rows where High < max(Open,Close,Low)")
                results['is_valid'] = False
            
            # Low should be <= Open, Close, High
            invalid_lows = (df['Low'] > df[['Open', 'Close', 'High']].min(axis=1)).sum()
            if invalid_lows > 0:
                results['errors'].append(f"{invalid_lows} rows where Low > min(Open,Close,High)")
                results['is_valid'] = False
        
        # Check for outliers (price changes > 50% in one day)
        if 'Close' in df.columns:
            price_changes = df['Close'].pct_change().abs()
            extreme_changes = price_changes > 0.5
            if extreme_changes.sum() > 0:
                results['warnings'].append(f"{extreme_changes.sum()} days with >50% price changes")
                results['outliers']['extreme_price_changes'] = df[extreme_changes].index.tolist()
        
        # Check date continuity
        if 'Date' in df.columns:
            df_sorted = df.sort_values('Date')
            date_gaps = df_sorted['Date'].diff().dt.days
            large_gaps = date_gaps > 7  # More than a week gap
            if large_gaps.sum() > 0:
                results['warnings'].append(f"{large_gaps.sum()} large date gaps found")
        
        # Calculate data quality score
        quality_score = 100.0
        quality_score -= sum(results['missing_data_pct'].values()) / len(required_columns)
        quality_score -= len(results['errors']) * 20
        quality_score -= len(results['warnings']) * 5
        results['data_quality_score'] = max(0, quality_score)
        
        return results
    
    def validate_news_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        required_columns = ['ticker', 'publishedAt', 'text']
        
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'missing_data_pct': {},
            'data_quality_score': 0.0
        }
        
        # Check required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            results['errors'].append(f"Missing required columns: {missing_cols}")
            results['is_valid'] = False
            return results
        
        # Check for missing/empty text
        empty_text = df['text'].isnull() | (df['text'].str.strip() == '')
        empty_text_pct = empty_text.sum() / len(df) * 100
        results['missing_data_pct']['text'] = empty_text_pct
        
        if empty_text_pct > 20:
            results['errors'].append(f"{empty_text_pct:.1f}% of news articles have empty text")
            results['is_valid'] = False
        elif empty_text_pct > 10:
            results['warnings'].append(f"{empty_text_pct:.1f}% of news articles have empty text")
        
        # Check text length distribution
        text_lengths = df['text'].fillna('').str.len()
        very_short = (text_lengths < 50).sum()
        very_long = (text_lengths > 5000).sum()
        
        if very_short / len(df) > 0.3:
            results['warnings'].append(f"{very_short} articles are very short (<50 chars)")
        
        if very_long > 0:
            results['warnings'].append(f"{very_long} articles are very long (>5000 chars)")
        
        # Check date format
        try:
            pd.to_datetime(df['publishedAt'])
        except:
            results['errors'].append("Invalid date format in publishedAt column")
            results['is_valid'] = False
        
        # Calculate quality score
        quality_score = 100.0
        quality_score -= empty_text_pct
        quality_score -= len(results['errors']) * 25
        quality_score -= len(results['warnings']) * 10
        results['data_quality_score'] = max(0, quality_score)
        
        return results
    
    def clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        
        # Remove rows where all price columns are null
        price_cols = [col for col in ['Open', 'High', 'Low', 'Close'] if col in df_clean.columns]
        if price_cols:
            df_clean = df_clean.dropna(subset=price_cols, how='all')
        
        # Forward fill missing prices (typical for financial data)
        for col in price_cols:
            df_clean[col] = df_clean[col].fillna(method='ffill')
        
        # Fix volume
        if 'Volume' in df_clean.columns:
            df_clean['Volume'] = df_clean['Volume'].fillna(0)
            df_clean.loc[df_clean['Volume'] < 0, 'Volume'] = 0
        
        # Remove extreme outliers (>3 standard deviations)
        if 'Close' in df_clean.columns:
            returns = df_clean['Close'].pct_change()
            mean_return = returns.mean()
            std_return = returns.std()
            
            outlier_mask = abs(returns - mean_return) > 3 * std_return
            if outlier_mask.sum() > 0:
                logger.warning(f"Removing {outlier_mask.sum()} outlier data points")
                # Replace outliers with interpolated values
                df_clean.loc[outlier_mask, 'Close'] = np.nan
                df_clean['Close'] = df_clean['Close'].interpolate()
        
        return df_clean
    
    def handle_market_holidays(self, df: pd.DataFrame, market: str = 'US') -> pd.DataFrame:
        """Remove or mark market holidays and non-trading days"""
        df_clean = df.copy()
        
        if 'Date' not in df_clean.columns:
            return df_clean
        
        df_clean['Date'] = pd.to_datetime(df_clean['Date'])
        df_clean['Weekday'] = df_clean['Date'].dt.dayofweek
        
        # Remove weekends (Saturday=5, Sunday=6)
        weekend_mask = df_clean['Weekday'].isin([5, 6])
        if weekend_mask.sum() > 0:
            logger.info(f"Removing {weekend_mask.sum()} weekend trading days")
            df_clean = df_clean[~weekend_mask]
        
        # Mark US holidays (basic list)
        if market == 'US':
            holidays = self._get_us_market_holidays(df_clean['Date'].min().year, 
                                                  df_clean['Date'].max().year)
            holiday_mask = df_clean['Date'].dt.date.isin(holidays)
            if holiday_mask.sum() > 0:
                logger.info(f"Removing {holiday_mask.sum()} holiday trading days")
                df_clean = df_clean[~holiday_mask]
        
        return df_clean.drop('Weekday', axis=1)
    
    def _get_us_market_holidays(self, start_year: int, end_year: int) -> List[datetime]:
        """Get basic US market holidays (simplified)"""
        holidays = []
        
        for year in range(start_year, end_year + 1):
            # New Year's Day
            holidays.append(datetime(year, 1, 1).date())
            
            # Independence Day
            holidays.append(datetime(year, 7, 4).date())
            
            # Christmas
            holidays.append(datetime(year, 12, 25).date())
            
            # Thanksgiving (4th Thursday in November)
            november_first = datetime(year, 11, 1)
            first_thursday = november_first + timedelta(days=(3 - november_first.weekday()) % 7)
            thanksgiving = first_thursday + timedelta(days=21)
            holidays.append(thanksgiving.date())
        
        return holidays

def validate_and_clean_data(price_df: pd.DataFrame, news_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any]]:
    validator = DataValidator()
    
    results = {'price_validation': None, 'news_validation': None}
    
    # Validate price data
    logger.info("Validating price data...")
    price_validation = validator.validate_price_data(price_df)
    results['price_validation'] = price_validation
    
    if not price_validation['is_valid']:
        logger.error("Price data validation failed:")
        for error in price_validation['errors']:
            logger.error(f"  - {error}")
        raise ValueError("Price data validation failed. Check logs for details.")
    
    # Clean price data
    price_df_clean = validator.clean_price_data(price_df)
    price_df_clean = validator.handle_market_holidays(price_df_clean)
    
    # Validate news data if provided
    news_df_clean = None
    if news_df is not None:
        logger.info("Validating news data...")
        news_validation = validator.validate_news_data(news_df)
        results['news_validation'] = news_validation
        
        if news_validation['is_valid']:
            news_df_clean = news_df.copy()
            # Basic news cleaning
            news_df_clean = news_df_clean.dropna(subset=['text'])
            news_df_clean['text'] = news_df_clean['text'].str.strip()
        else:
            logger.warning("News data validation failed, but continuing with price data only")
    
    logger.info(f"Data validation completed. Price data quality score: {price_validation['data_quality_score']:.1f}")
    
    return price_df_clean, news_df_clean, results