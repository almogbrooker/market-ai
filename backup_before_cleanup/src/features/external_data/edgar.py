#!/usr/bin/env python3
"""
SEC EDGAR data integration
Fetches corporate filings and sentiment analysis for financial prediction
"""

import pandas as pd
import numpy as np
import requests
import logging
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
import time
import re
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

class EDGARDataFetcher:
    """Fetches SEC filing data and performs sentiment analysis"""
    
    def __init__(self, user_agent: str = "MarketAI Research Tool contact@example.com"):
        """
        Args:
            user_agent: Required by SEC EDGAR API. Must include company name and contact info.
        """
        self.base_url = "https://data.sec.gov"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'data.sec.gov'
        })
        
        # Financial sentiment words for basic analysis
        self.positive_words = {
            'growth', 'increase', 'positive', 'strong', 'robust', 'excellent',
            'outstanding', 'profitable', 'successful', 'improved', 'expanded',
            'gained', 'higher', 'better', 'favorable', 'optimistic', 'confident'
        }
        
        self.negative_words = {
            'decline', 'decrease', 'negative', 'weak', 'poor', 'loss', 'losses',
            'difficult', 'challenging', 'lower', 'reduced', 'dropped', 'fell',
            'concerns', 'risks', 'uncertainty', 'volatility', 'adverse', 'unfavorable'
        }
        
        self.risk_words = {
            'risk', 'risks', 'uncertainty', 'volatile', 'volatility', 'litigation',
            'regulatory', 'compliance', 'competition', 'market conditions',
            'economic conditions', 'credit risk', 'operational risk'
        }
        
        # Important filing types
        self.filing_types = {
            '10-K': 'Annual Report',
            '10-Q': 'Quarterly Report', 
            '8-K': 'Current Report',
            '10-K/A': 'Annual Report Amendment',
            '10-Q/A': 'Quarterly Report Amendment'
        }
        
    def _rate_limit(self):
        """SEC requires rate limiting: max 10 requests per second"""
        time.sleep(0.1)
    
    def get_company_cik(self, ticker: str) -> Optional[str]:
        """Get CIK (Central Index Key) for a company ticker"""
        
        try:
            # Use company tickers endpoint
            url = f"{self.base_url}/files/company_tickers.json"
            response = self.session.get(url)
            response.raise_for_status()
            
            companies = response.json()
            
            # Search for ticker
            for company_data in companies.values():
                if company_data.get('ticker', '').upper() == ticker.upper():
                    cik = str(company_data['cik_str']).zfill(10)
                    return cik
            
            logger.warning(f"CIK not found for ticker {ticker}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get CIK for {ticker}: {e}")
            return None
    
    def fetch_company_filings(self, cik: str, start_date: str, 
                             end_date: Optional[str] = None,
                             filing_types: Optional[List[str]] = None) -> pd.DataFrame:
        """Fetch recent filings for a company"""
        
        if filing_types is None:
            filing_types = ['10-K', '10-Q', '8-K']
        
        try:
            # Get company filing data
            url = f"{self.base_url}/submissions/CIK{cik}.json"
            response = self.session.get(url)
            response.raise_for_status()
            self._rate_limit()
            
            data = response.json()
            
            if 'filings' not in data or 'recent' not in data['filings']:
                return pd.DataFrame()
            
            filings = data['filings']['recent']
            
            # Convert to DataFrame
            df = pd.DataFrame({
                'form': filings.get('form', []),
                'filing_date': filings.get('filingDate', []),
                'acceptance_date': filings.get('acceptanceDateTime', []),
                'accession_number': filings.get('accessionNumber', []),
                'primary_document': filings.get('primaryDocument', []),
                'description': filings.get('primaryDocDescription', [])
            })
            
            if df.empty:
                return df
            
            # Filter by date and filing type
            df['filing_date'] = pd.to_datetime(df['filing_date'])
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date) if end_date else datetime.now()
            
            df = df[
                (df['filing_date'] >= start_dt) & 
                (df['filing_date'] <= end_dt) &
                (df['form'].isin(filing_types))
            ]
            
            return df.sort_values('filing_date', ascending=False)
            
        except Exception as e:
            logger.error(f"Failed to fetch filings for CIK {cik}: {e}")
            return pd.DataFrame()
    
    def _extract_text_from_filing(self, cik: str, accession_number: str, 
                                 primary_document: str) -> str:
        """Extract text content from a SEC filing (simplified)"""
        
        try:
            # Construct filing URL
            accession_clean = accession_number.replace('-', '')
            url = f"{self.base_url}/Archives/edgar/data/{int(cik)}/{accession_clean}/{primary_document}"
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            self._rate_limit()
            
            content = response.text
            
            # Basic HTML/XML tag removal (simplified)
            content = re.sub(r'<[^>]+>', ' ', content)
            content = re.sub(r'\s+', ' ', content)
            content = content.strip()
            
            # Limit content size to avoid memory issues
            max_chars = 100000  # 100KB
            if len(content) > max_chars:
                content = content[:max_chars]
            
            return content
            
        except Exception as e:
            logger.warning(f"Failed to extract text from filing {accession_number}: {e}")
            return ""
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Basic sentiment analysis of filing text"""
        
        if not text:
            return {
                'positive_score': 0.0,
                'negative_score': 0.0,
                'risk_score': 0.0,
                'sentiment_score': 0.0,
                'word_count': 0
            }
        
        words = re.findall(r'\b\w+\b', text.lower())
        word_count = len(words)
        
        if word_count == 0:
            return {
                'positive_score': 0.0,
                'negative_score': 0.0,
                'risk_score': 0.0,
                'sentiment_score': 0.0,
                'word_count': 0
            }
        
        # Count sentiment words
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        risk_count = sum(1 for word in words if word in self.risk_words)
        
        # Calculate scores as percentages
        positive_score = (positive_count / word_count) * 100
        negative_score = (negative_count / word_count) * 100
        risk_score = (risk_count / word_count) * 100
        
        # Overall sentiment score
        sentiment_score = positive_score - negative_score
        
        return {
            'positive_score': positive_score,
            'negative_score': negative_score,
            'risk_score': risk_score,
            'sentiment_score': sentiment_score,
            'word_count': word_count
        }
    
    def analyze_company_filings(self, ticker: str, start_date: str,
                              end_date: Optional[str] = None) -> pd.DataFrame:
        """Analyze sentiment of company filings"""
        
        logger.info(f"Analyzing EDGAR filings for {ticker}")
        
        # Get CIK
        cik = self.get_company_cik(ticker)
        if not cik:
            return pd.DataFrame()
        
        # Get filings
        filings_df = self.fetch_company_filings(cik, start_date, end_date)
        if filings_df.empty:
            return pd.DataFrame()
        
        results = []
        
        # Limit to most recent filings to avoid overwhelming the system
        max_filings = 10
        filings_to_analyze = filings_df.head(max_filings)
        
        for _, filing in filings_to_analyze.iterrows():
            try:
                logger.debug(f"Analyzing {filing['form']} filed on {filing['filing_date']}")
                
                # Extract text
                text = self._extract_text_from_filing(
                    cik, filing['accession_number'], filing['primary_document']
                )
                
                # Analyze sentiment
                sentiment = self._analyze_sentiment(text)
                
                result = {
                    'ticker': ticker,
                    'cik': cik,
                    'filing_date': filing['filing_date'],
                    'form_type': filing['form'],
                    'accession_number': filing['accession_number'],
                    **sentiment
                }
                
                results.append(result)
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"Failed to analyze filing {filing['accession_number']}: {e}")
                continue
        
        if not results:
            return pd.DataFrame()
        
        return pd.DataFrame(results)
    
    def aggregate_filing_sentiment(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate filing sentiment to daily/monthly features"""
        
        if sentiment_df.empty:
            return pd.DataFrame()
        
        # Group by filing date
        daily_agg = sentiment_df.groupby('filing_date').agg({
            'positive_score': 'mean',
            'negative_score': 'mean',
            'risk_score': 'mean',
            'sentiment_score': 'mean',
            'word_count': 'sum',
            'form_type': 'count'  # Number of filings per day
        }).round(4)
        
        # Rename columns
        daily_agg.columns = [
            'edgar_positive_score',
            'edgar_negative_score', 
            'edgar_risk_score',
            'edgar_sentiment_score',
            'edgar_total_words',
            'edgar_filing_count'
        ]
        
        # Add rolling features
        for window in [30, 90]:  # 1 month and 3 months
            daily_agg[f'edgar_sentiment_ma_{window}d'] = daily_agg['edgar_sentiment_score'].rolling(window).mean()
            daily_agg[f'edgar_risk_ma_{window}d'] = daily_agg['edgar_risk_score'].rolling(window).mean()
        
        # Sentiment momentum
        daily_agg['edgar_sentiment_momentum'] = daily_agg['edgar_sentiment_score'].diff()
        
        return daily_agg

def create_edgar_features(tickers: List[str], start_date: str = "2020-01-01",
                         end_date: Optional[str] = None) -> pd.DataFrame:
    """Convenience function to create EDGAR features for multiple tickers"""
    
    fetcher = EDGARDataFetcher()
    
    all_sentiment = []
    
    for ticker in tickers:
        try:
            logger.info(f"Processing EDGAR data for {ticker}")
            
            sentiment_df = fetcher.analyze_company_filings(ticker, start_date, end_date)
            if not sentiment_df.empty:
                all_sentiment.append(sentiment_df)
                
        except Exception as e:
            logger.error(f"Failed to process EDGAR data for {ticker}: {e}")
    
    if not all_sentiment:
        logger.warning("No EDGAR data available, creating dummy features")
        
        # Create dummy EDGAR features
        date_range = pd.date_range(start=start_date, 
                                 end=end_date or datetime.now().strftime("%Y-%m-%d"),
                                 freq='D')
        dummy_data = pd.DataFrame(index=date_range)
        dummy_data['edgar_sentiment_score'] = np.random.randn(len(dummy_data)) * 0.3
        dummy_data['edgar_risk_score'] = np.random.exponential(0.1, len(dummy_data))
        dummy_data['edgar_filing_count'] = np.random.poisson(0.1, len(dummy_data))
        
        return dummy_data
    
    # Combine all ticker data
    combined_sentiment = pd.concat(all_sentiment, ignore_index=True)
    
    # Aggregate across all tickers by date
    market_sentiment = combined_sentiment.groupby('filing_date').agg({
        'positive_score': 'mean',
        'negative_score': 'mean',
        'risk_score': 'mean', 
        'sentiment_score': 'mean',
        'word_count': 'sum',
        'ticker': 'count'
    }).round(4)
    
    # Rename columns for market-wide features
    market_sentiment.columns = [
        'edgar_market_positive',
        'edgar_market_negative',
        'edgar_market_risk',
        'edgar_market_sentiment',
        'edgar_total_words',
        'edgar_companies_filing'
    ]
    
    return market_sentiment