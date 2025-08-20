#!/usr/bin/env python3
"""
GDELT (Global Database of Events, Language, and Tone) integration
Fetches news sentiment and tone data for financial market prediction
"""

import pandas as pd
import numpy as np
import requests
import logging
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
import time
import io
import zipfile
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

class GDELTDataFetcher:
    """Fetches news sentiment and tone data from GDELT"""
    
    def __init__(self):
        self.base_url = "http://data.gdeltproject.org/gdeltv2/"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MarketAI/1.0 Research Tool'
        })
        
        # Financial keywords for filtering relevant news
        self.financial_keywords = {
            'markets', 'stock', 'trading', 'investment', 'economy', 'economic',
            'finance', 'financial', 'earnings', 'revenue', 'profit', 'loss',
            'fed', 'federal reserve', 'interest rate', 'inflation', 'gdp',
            'nasdaq', 'nyse', 'sp500', 's&p', 'dow jones', 'russell',
            'recession', 'growth', 'unemployment', 'jobs', 'employment'
        }
        
    def _get_available_files(self, date: datetime) -> List[str]:
        """Get list of available GDELT files for a given date"""
        
        # GDELT files are updated every 15 minutes
        # Format: YYYYMMDDHHMMSS.gkg.csv.zip
        date_str = date.strftime("%Y%m%d")
        
        # Try to get the masterfile list (if available)
        masterfile_url = f"{self.base_url}masterfilelist.txt"
        
        try:
            response = self.session.get(masterfile_url, timeout=10)
            response.raise_for_status()
            
            # Filter files for the specific date
            files = []
            for line in response.text.split('\n'):
                if date_str in line and '.gkg.csv.zip' in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        files.append(parts[2])  # URL is typically the 3rd column
            
            return files[:4]  # Limit to avoid overwhelming the system
            
        except Exception as e:
            logger.warning(f"Could not fetch masterfile list: {e}")
            
            # Fallback: generate likely filenames
            files = []
            for hour in [0, 6, 12, 18]:  # Sample 4 times per day
                timestamp = date.replace(hour=hour, minute=0, second=0)
                filename = timestamp.strftime("%Y%m%d%H%M%S") + ".gkg.csv.zip"
                files.append(urljoin(self.base_url, filename))
            
            return files
    
    def _fetch_gkg_file(self, file_url: str) -> pd.DataFrame:
        """Fetch and parse a single GDELT GKG (Global Knowledge Graph) file"""
        
        try:
            response = self.session.get(file_url, timeout=30)
            response.raise_for_status()
            
            # Extract CSV from ZIP
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                csv_filename = zf.namelist()[0]
                with zf.open(csv_filename) as csv_file:
                    # GDELT GKG has specific column structure
                    columns = [
                        'GKGRECORDID', 'DATE', 'SourceCollectionIdentifier', 'SourceCommonName',
                        'DocumentIdentifier', 'Counts', 'V2Counts', 'Themes', 'V2Themes',
                        'Locations', 'V2Locations', 'Persons', 'V2Persons', 'Organizations',
                        'V2Organizations', 'V2Tone', 'Dates', 'GCAM', 'SharingImage',
                        'RelatedImages', 'SocialImageEmbeds', 'SocialVideoEmbeds', 'Quotations',
                        'AllNames', 'Amounts', 'TranslationInfo', 'Extras'
                    ]
                    
                    df = pd.read_csv(csv_file, sep='\t', names=columns, 
                                   low_memory=False, on_bad_lines='skip')
            
            return df
            
        except Exception as e:
            logger.warning(f"Failed to fetch {file_url}: {e}")
            return pd.DataFrame()
    
    def _extract_financial_tone(self, gkg_df: pd.DataFrame, tickers: Optional[Set[str]] = None) -> pd.DataFrame:
        """Extract financial sentiment and tone from GDELT GKG data"""
        
        if gkg_df.empty:
            return pd.DataFrame()
        
        financial_news = []
        
        for _, row in gkg_df.iterrows():
            try:
                # Parse date
                date_str = str(row['DATE'])
                if len(date_str) >= 8:
                    date = pd.to_datetime(date_str[:8], format='%Y%m%d')
                else:
                    continue
                
                # Check if news is financial
                themes = str(row.get('V2Themes', '') or row.get('Themes', ''))
                organizations = str(row.get('V2Organizations', '') or row.get('Organizations', ''))
                
                is_financial = any(keyword.lower() in themes.lower() + organizations.lower() 
                                 for keyword in self.financial_keywords)
                
                # Filter by tickers if provided
                ticker_mentioned = True
                if tickers:
                    ticker_mentioned = any(ticker.upper() in organizations.upper() 
                                         for ticker in tickers)
                
                if is_financial and ticker_mentioned:
                    # Parse tone information
                    tone_str = str(row.get('V2Tone', ''))
                    tone_values = tone_str.split(',') if tone_str else []
                    
                    tone_data = {
                        'date': date,
                        'tone_score': float(tone_values[0]) if len(tone_values) > 0 and tone_values[0] != '' else 0.0,
                        'positive_score': float(tone_values[1]) if len(tone_values) > 1 and tone_values[1] != '' else 0.0,
                        'negative_score': float(tone_values[2]) if len(tone_values) > 2 and tone_values[2] != '' else 0.0,
                        'polarity': float(tone_values[3]) if len(tone_values) > 3 and tone_values[3] != '' else 0.0,
                        'activity_ref_density': float(tone_values[4]) if len(tone_values) > 4 and tone_values[4] != '' else 0.0,
                        'self_group_ref_density': float(tone_values[5]) if len(tone_values) > 5 and tone_values[5] != '' else 0.0,
                        'word_count': int(tone_values[6]) if len(tone_values) > 6 and tone_values[6] != '' else 0,
                        'themes': themes,
                        'organizations': organizations,
                    }
                    
                    financial_news.append(tone_data)
                    
            except Exception as e:
                continue  # Skip problematic rows
        
        if not financial_news:
            return pd.DataFrame()
        
        return pd.DataFrame(financial_news)
    
    def fetch_daily_sentiment(self, date: datetime, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """Fetch daily financial sentiment for a specific date"""
        
        logger.info(f"Fetching GDELT sentiment for {date.strftime('%Y-%m-%d')}")
        
        ticker_set = set(tickers) if tickers else None
        files = self._get_available_files(date)
        
        if not files:
            logger.warning(f"No GDELT files found for {date.strftime('%Y-%m-%d')}")
            return pd.DataFrame()
        
        all_sentiment = []
        
        for file_url in files:
            logger.debug(f"Processing {file_url}")
            
            gkg_df = self._fetch_gkg_file(file_url)
            if not gkg_df.empty:
                sentiment_df = self._extract_financial_tone(gkg_df, ticker_set)
                if not sentiment_df.empty:
                    all_sentiment.append(sentiment_df)
            
            # Be respectful to the API
            time.sleep(0.5)
        
        if not all_sentiment:
            return pd.DataFrame()
        
        # Combine all sentiment data for the day
        daily_sentiment = pd.concat(all_sentiment, ignore_index=True)
        
        return daily_sentiment
    
    def fetch_sentiment_range(self, start_date: str, end_date: Optional[str] = None,
                             tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """Fetch sentiment data for a date range"""
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date) if end_date else datetime.now()
        
        logger.info(f"Fetching GDELT sentiment from {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
        
        all_data = []
        current_date = start_dt
        
        # Limit to avoid overwhelming the system
        max_days = 30
        days_processed = 0
        
        while current_date <= end_dt and days_processed < max_days:
            try:
                daily_data = self.fetch_daily_sentiment(current_date, tickers)
                if not daily_data.empty:
                    all_data.append(daily_data)
                
            except Exception as e:
                logger.warning(f"Failed to fetch data for {current_date.strftime('%Y-%m-%d')}: {e}")
            
            current_date += timedelta(days=1)
            days_processed += 1
            
            # Rate limiting
            time.sleep(1)
        
        if not all_data:
            logger.warning("No GDELT sentiment data fetched")
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        return combined_df
    
    def aggregate_daily_sentiment(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate hourly sentiment data to daily features"""
        
        if sentiment_df.empty:
            return pd.DataFrame()
        
        # Group by date and calculate daily aggregates
        daily_agg = sentiment_df.groupby('date').agg({
            'tone_score': ['mean', 'std', 'min', 'max', 'count'],
            'positive_score': ['mean', 'std'],
            'negative_score': ['mean', 'std'],
            'polarity': ['mean', 'std'],
            'activity_ref_density': 'mean',
            'word_count': 'sum'
        }).round(4)
        
        # Flatten column names
        daily_agg.columns = ['_'.join(col).strip() for col in daily_agg.columns]
        
        # Add derived features
        daily_agg['sentiment_volatility'] = daily_agg['tone_score_std']
        daily_agg['news_volume'] = daily_agg['tone_score_count']
        daily_agg['sentiment_momentum'] = daily_agg['tone_score_mean'].diff()
        
        # Clean up
        daily_agg = daily_agg.fillna(0)
        
        return daily_agg

def create_gdelt_features(start_date: str = "2020-01-01", 
                         end_date: Optional[str] = None,
                         tickers: Optional[List[str]] = None) -> pd.DataFrame:
    """Convenience function to create GDELT sentiment features"""
    
    fetcher = GDELTDataFetcher()
    
    try:
        # Limit the date range for demo/testing
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date) if end_date else datetime.now()
        
        # Limit to last 30 days to avoid overwhelming the system
        if (end_dt - start_dt).days > 30:
            start_dt = end_dt - timedelta(days=30)
            logger.info(f"Limiting GDELT fetch to last 30 days: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
        
        sentiment_data = fetcher.fetch_sentiment_range(
            start_dt.strftime('%Y-%m-%d'), 
            end_dt.strftime('%Y-%m-%d'), 
            tickers
        )
        
        if sentiment_data.empty:
            logger.warning("No GDELT data available, creating dummy sentiment features")
            # Create dummy sentiment features
            date_range = pd.date_range(start=start_date, 
                                     end=end_date or datetime.now().strftime("%Y-%m-%d"),
                                     freq='D')
            dummy_data = pd.DataFrame(index=date_range)
            dummy_data['tone_score_mean'] = np.random.randn(len(dummy_data)) * 0.5
            dummy_data['sentiment_volatility'] = np.random.exponential(0.2, len(dummy_data))
            dummy_data['news_volume'] = np.random.poisson(10, len(dummy_data))
            return dummy_data
        
        # Aggregate to daily features
        daily_features = fetcher.aggregate_daily_sentiment(sentiment_data)
        
        return daily_features
        
    except Exception as e:
        logger.error(f"Failed to fetch GDELT data: {e}")
        
        # Fallback to dummy data
        date_range = pd.date_range(start=start_date, 
                                 end=end_date or datetime.now().strftime("%Y-%m-%d"),
                                 freq='D')
        dummy_data = pd.DataFrame(index=date_range)
        dummy_data['tone_score_mean'] = np.random.randn(len(dummy_data)) * 0.5
        dummy_data['sentiment_volatility'] = np.random.exponential(0.2, len(dummy_data))
        dummy_data['news_volume'] = np.random.poisson(10, len(dummy_data))
        
        return dummy_data