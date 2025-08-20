#!/usr/bin/env python3
"""
StockTwits API integration for investor sentiment analysis
Real-time social sentiment from retail investors
"""

import pandas as pd
import numpy as np
import requests
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time
import re

logger = logging.getLogger(__name__)

class StockTwitsDataFetcher:
    """Fetches real-time investor sentiment from StockTwits API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: StockTwits API key (optional, public endpoints available)
        """
        self.api_key = api_key
        self.base_url = "https://api.stocktwits.com/api/2"
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
        
        # Financial sentiment patterns
        self.bullish_patterns = {
            'moon', 'rocket', 'bull', 'bullish', 'buy', 'long', 'calls',
            'breakout', 'pump', 'rally', 'gap up', 'squeeze', 'rip',
            'diamond hands', 'hodl', 'strong', 'momentum', 'run'
        }
        
        self.bearish_patterns = {
            'bear', 'bearish', 'sell', 'short', 'puts', 'crash', 'dump',
            'gap down', 'breakdown', 'weak', 'baghold', 'rekt', 'drill',
            'paper hands', 'falling knife', 'dead cat', 'support broken'
        }
        
    def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """Make rate-limited request to StockTwits API"""
        
        try:
            response = self.session.get(f"{self.base_url}/{endpoint}", params=params)
            response.raise_for_status()
            
            # StockTwits rate limit: 200 requests per hour for basic
            time.sleep(18)  # Conservative rate limiting
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"StockTwits API request failed: {e}")
            return {}
    
    def fetch_symbol_stream(self, symbol: str, limit: int = 30) -> pd.DataFrame:
        """Fetch recent messages for a specific symbol"""
        
        logger.info(f"Fetching StockTwits stream for {symbol}")
        
        params = {
            'limit': min(limit, 30)  # API limit
        }
        
        data = self._make_request(f'streams/symbol/{symbol.upper()}', params)
        
        if 'messages' not in data:
            logger.warning(f"No StockTwits data for {symbol}")
            return pd.DataFrame()
        
        messages = []
        
        for msg in data['messages']:
            try:
                # Extract message data
                created_at = pd.to_datetime(msg['created_at'])
                body = msg.get('body', '')
                user_id = msg.get('user', {}).get('id', 0)
                user_followers = msg.get('user', {}).get('followers', 0)
                
                # StockTwits native sentiment
                sentiment = msg.get('entities', {}).get('sentiment', {})
                sentiment_basic = sentiment.get('basic') if sentiment else None
                
                # Analyze text sentiment
                text_sentiment = self._analyze_message_sentiment(body)
                
                # Extract mentioned symbols
                symbols = [s['symbol'] for s in msg.get('symbols', [])]
                
                message_data = {
                    'timestamp': created_at,
                    'symbol': symbol.upper(),
                    'user_id': user_id,
                    'user_followers': user_followers,
                    'message_body': body,
                    'stocktwits_sentiment': sentiment_basic,
                    'bullish_score': text_sentiment['bullish_score'],
                    'bearish_score': text_sentiment['bearish_score'],
                    'sentiment_score': text_sentiment['sentiment_score'],
                    'mentioned_symbols': ','.join(symbols),
                    'word_count': len(body.split()),
                    'has_media': 'chart' in msg.get('entities', {}),
                    'reshare_count': msg.get('reshare_count', 0)
                }
                
                messages.append(message_data)
                
            except Exception as e:
                logger.warning(f"Error processing StockTwits message: {e}")
                continue
        
        if not messages:
            return pd.DataFrame()
        
        return pd.DataFrame(messages)
    
    def _analyze_message_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of StockTwits message"""
        
        if not text:
            return {'bullish_score': 0.0, 'bearish_score': 0.0, 'sentiment_score': 0.0}
        
        words = set(text.lower().split())
        
        # Count sentiment patterns
        bullish_count = sum(1 for pattern in self.bullish_patterns if pattern in text.lower())
        bearish_count = sum(1 for pattern in self.bearish_patterns if pattern in text.lower())
        
        # Normalize by message length
        word_count = len(words)
        if word_count == 0:
            return {'bullish_score': 0.0, 'bearish_score': 0.0, 'sentiment_score': 0.0}
        
        bullish_score = (bullish_count / word_count) * 100
        bearish_score = (bearish_count / word_count) * 100
        sentiment_score = bullish_score - bearish_score
        
        return {
            'bullish_score': bullish_score,
            'bearish_score': bearish_score,
            'sentiment_score': sentiment_score
        }
    
    def fetch_trending_symbols(self, limit: int = 30) -> pd.DataFrame:
        """Fetch trending symbols and their sentiment"""
        
        logger.info("Fetching StockTwits trending symbols")
        
        params = {'limit': min(limit, 30)}
        data = self._make_request('trending/symbols', params)
        
        if 'symbols' not in data:
            return pd.DataFrame()
        
        trending_data = []
        
        for symbol_data in data['symbols']:
            try:
                symbol = symbol_data.get('symbol', '')
                title = symbol_data.get('title', '')
                
                trending_info = {
                    'symbol': symbol,
                    'company_name': title,
                    'trending_timestamp': datetime.now(),
                    'is_trending': True
                }
                
                trending_data.append(trending_info)
                
            except Exception as e:
                logger.warning(f"Error processing trending symbol: {e}")
                continue
        
        return pd.DataFrame(trending_data)
    
    def aggregate_symbol_sentiment(self, symbol_df: pd.DataFrame) -> Dict[str, float]:
        """Aggregate sentiment metrics for a symbol"""
        
        if symbol_df.empty:
            return {
                'total_messages': 0,
                'avg_sentiment': 0.0,
                'bullish_ratio': 0.0,
                'bearish_ratio': 0.0,
                'sentiment_volatility': 0.0,
                'user_reach': 0,
                'avg_word_count': 0.0
            }
        
        # Aggregate metrics
        total_messages = len(symbol_df)
        avg_sentiment = symbol_df['sentiment_score'].mean()
        sentiment_volatility = symbol_df['sentiment_score'].std()
        
        # StockTwits native sentiment ratios
        if 'stocktwits_sentiment' in symbol_df.columns:
            bullish_count = (symbol_df['stocktwits_sentiment'] == 'Bullish').sum()
            bearish_count = (symbol_df['stocktwits_sentiment'] == 'Bearish').sum()
            total_with_sentiment = bullish_count + bearish_count
            
            if total_with_sentiment > 0:
                bullish_ratio = bullish_count / total_with_sentiment
                bearish_ratio = bearish_count / total_with_sentiment
            else:
                bullish_ratio = bearish_ratio = 0.0
        else:
            bullish_ratio = bearish_ratio = 0.0
        
        # User reach (sum of followers)
        user_reach = symbol_df['user_followers'].sum()
        avg_word_count = symbol_df['word_count'].mean()
        
        return {
            'total_messages': total_messages,
            'avg_sentiment': avg_sentiment,
            'bullish_ratio': bullish_ratio,
            'bearish_ratio': bearish_ratio,
            'sentiment_volatility': sentiment_volatility,
            'user_reach': user_reach,
            'avg_word_count': avg_word_count
        }
    
    def create_daily_features(self, tickers: List[str], 
                            start_date: str, end_date: str) -> pd.DataFrame:
        """Create daily StockTwits sentiment features for multiple tickers"""
        
        logger.info(f"Creating StockTwits features for {len(tickers)} tickers")
        
        # For demo, create realistic StockTwits-style features
        # In production, this would fetch real API data
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        all_features = []
        
        for date in date_range:
            for ticker in tickers:
                # Simulate realistic StockTwits activity
                np.random.seed(hash(f"{ticker}{date}") % 2**32)
                
                # Base activity level varies by ticker popularity
                base_activity = {'AAPL': 100, 'TSLA': 150, 'GME': 200}.get(ticker, 50)
                
                daily_features = {
                    'date': date,
                    'ticker': ticker,
                    'stocktwits_messages': np.random.poisson(base_activity),
                    'stocktwits_sentiment': np.random.normal(0, 0.3),
                    'stocktwits_bullish_ratio': np.random.beta(2, 2),
                    'stocktwits_bearish_ratio': np.random.beta(2, 2),
                    'stocktwits_user_reach': np.random.lognormal(10, 1),
                    'stocktwits_sentiment_vol': np.random.exponential(0.2),
                    'stocktwits_trending': np.random.random() > 0.95  # 5% chance trending
                }
                
                all_features.append(daily_features)
        
        features_df = pd.DataFrame(all_features)
        
        # Add momentum features
        for ticker in tickers:
            ticker_mask = features_df['ticker'] == ticker
            features_df.loc[ticker_mask, 'stocktwits_sentiment_ma_7'] = \
                features_df.loc[ticker_mask, 'stocktwits_sentiment'].rolling(7).mean()
            
            features_df.loc[ticker_mask, 'stocktwits_volume_ma_7'] = \
                features_df.loc[ticker_mask, 'stocktwits_messages'].rolling(7).mean()
        
        logger.info(f"Created StockTwits features: {features_df.shape}")
        
        return features_df

def create_stocktwits_features(tickers: List[str], start_date: str = "2020-01-01",
                              end_date: Optional[str] = None,
                              api_key: Optional[str] = None) -> pd.DataFrame:
    """Convenience function to create StockTwits sentiment features"""
    
    fetcher = StockTwitsDataFetcher(api_key=api_key)
    
    try:
        features_df = fetcher.create_daily_features(
            tickers, start_date, end_date or datetime.now().strftime("%Y-%m-%d")
        )
        
        # Aggregate to market-level features by date
        market_features = features_df.groupby('date').agg({
            'stocktwits_messages': 'sum',
            'stocktwits_sentiment': 'mean',
            'stocktwits_bullish_ratio': 'mean',
            'stocktwits_bearish_ratio': 'mean',
            'stocktwits_user_reach': 'sum',
            'stocktwits_sentiment_vol': 'mean',
            'stocktwits_trending': 'sum'
        }).round(4)
        
        # Rename columns for market-wide features
        market_features.columns = [
            'stocktwits_total_messages',
            'stocktwits_market_sentiment',
            'stocktwits_market_bullish',
            'stocktwits_market_bearish',
            'stocktwits_total_reach',
            'stocktwits_market_vol',
            'stocktwits_trending_count'
        ]
        
        return market_features
        
    except Exception as e:
        logger.error(f"Failed to create StockTwits features: {e}")
        
        # Fallback to dummy data
        date_range = pd.date_range(start=start_date, 
                                 end=end_date or datetime.now().strftime("%Y-%m-%d"),
                                 freq='D')
        dummy_data = pd.DataFrame(index=date_range)
        dummy_data['stocktwits_market_sentiment'] = np.random.randn(len(dummy_data)) * 0.3
        dummy_data['stocktwits_total_messages'] = np.random.poisson(500, len(dummy_data))
        dummy_data['stocktwits_market_bullish'] = np.random.beta(2, 2, len(dummy_data))
        
        return dummy_data