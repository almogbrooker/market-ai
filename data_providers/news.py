#!/usr/bin/env python3
"""
News Data Provider
Fetches news data with deduplication and sentiment analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Set
import logging
import os
import requests
import hashlib
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

class NewsProvider:
    """News data provider with deduplication and sentiment analysis"""
    
    def __init__(self):
        # Load API keys from environment
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.finnhub_api_key = os.getenv('FINNHUB_API_KEY')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        
        # Rate limiting tracking
        self.last_fetch_time = {}
        
        # Deduplication tracking
        self.seen_articles = set()
        self.content_hashes = set()
        
        logger.info("📰 NewsProvider initialized")
        if self.news_api_key:
            logger.info("   ✅ NewsAPI key available")
        if self.finnhub_api_key:
            logger.info("   ✅ Finnhub API key available")
        if self.alpha_vantage_key:
            logger.info("   ✅ Alpha Vantage key available")
    
    def get_daily_news(self, symbols: List[str], target_date: datetime) -> pd.DataFrame:
        """Get daily news data with deduplication"""
        
        logger.info(f"📰 Fetching news for {len(symbols)} symbols on {target_date.date()}")
        
        all_news = []
        
        for symbol in symbols:
            # Fetch from multiple sources
            symbol_news = []
            
            # NewsAPI
            if self.news_api_key:
                news_api_articles = self._fetch_newsapi_data(symbol, target_date)
                symbol_news.extend(news_api_articles)
            
            # Finnhub
            if self.finnhub_api_key:
                finnhub_articles = self._fetch_finnhub_data(symbol, target_date)
                symbol_news.extend(finnhub_articles)
            
            # Alpha Vantage News
            if self.alpha_vantage_key:
                av_articles = self._fetch_alpha_vantage_news(symbol, target_date)
                symbol_news.extend(av_articles)
            
            # If no real data, create skeleton
            if not symbol_news:
                symbol_news = self._create_news_skeleton(symbol, target_date)
            
            all_news.extend(symbol_news)
        
        if not all_news:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_news)
        
        # Deduplicate articles
        df = self._deduplicate_articles(df)
        
        # Add sentiment analysis
        df = self._add_sentiment_analysis(df)
        
        # Aggregate by symbol and date
        df = self._aggregate_by_symbol_date(df)
        
        logger.info(f"✅ News data processed: {len(df)} aggregated records")
        
        return df
    
    def _fetch_newsapi_data(self, symbol: str, target_date: datetime) -> List[Dict]:
        """Fetch from NewsAPI"""
        
        try:
            # Rate limiting
            if 'newsapi' in self.last_fetch_time:
                elapsed = (datetime.now() - self.last_fetch_time['newsapi']).total_seconds()
                if elapsed < 1.0:  # 1 second between requests
                    return []
            
            # Search for company news
            company_names = {
                'AAPL': 'Apple Inc',
                'MSFT': 'Microsoft',
                'GOOGL': 'Google Alphabet',
                'AMZN': 'Amazon',
                'TSLA': 'Tesla',
                'META': 'Meta Facebook',
                'NVDA': 'NVIDIA',
                'NFLX': 'Netflix'
            }
            
            query = f"{symbol} OR \"{company_names.get(symbol, symbol)}\""
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': query,
                'from': (target_date - timedelta(days=1)).strftime('%Y-%m-%d'),
                'to': target_date.strftime('%Y-%m-%d'),
                'sortBy': 'publishedAt',
                'language': 'en',
                'apiKey': self.news_api_key,
                'pageSize': 20
            }
            
            response = requests.get(url, params=params, timeout=10)
            self.last_fetch_time['newsapi'] = datetime.now()
            
            if response.status_code == 200:
                data = response.json()
                articles = []
                
                for article in data.get('articles', []):
                    articles.append({
                        'symbol': symbol,
                        'date': target_date.date(),
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'content': article.get('content', '')[:500],  # Truncate
                        'url': article.get('url', ''),
                        'published_at': article.get('publishedAt'),
                        'source': f"newsapi_{article.get('source', {}).get('name', 'unknown')}",
                        'language': 'en'
                    })
                
                return articles
        
        except Exception as e:
            logger.warning(f"⚠️ NewsAPI fetch failed for {symbol}: {e}")
        
        return []
    
    def _fetch_finnhub_data(self, symbol: str, target_date: datetime) -> List[Dict]:
        """Fetch from Finnhub"""
        
        try:
            # Rate limiting
            if 'finnhub' in self.last_fetch_time:
                elapsed = (datetime.now() - self.last_fetch_time['finnhub']).total_seconds()
                if elapsed < 1.0:
                    return []
            
            # Calculate date range (Finnhub uses timestamps)
            from_date = int((target_date - timedelta(days=1)).timestamp())
            to_date = int(target_date.timestamp())
            
            url = "https://finnhub.io/api/v1/company-news"
            params = {
                'symbol': symbol,
                'from': from_date,
                'to': to_date,
                'token': self.finnhub_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            self.last_fetch_time['finnhub'] = datetime.now()
            
            if response.status_code == 200:
                data = response.json()
                articles = []
                
                for article in data:
                    articles.append({
                        'symbol': symbol,
                        'date': target_date.date(),
                        'title': article.get('headline', ''),
                        'description': article.get('summary', ''),
                        'content': article.get('summary', '')[:500],
                        'url': article.get('url', ''),
                        'published_at': datetime.fromtimestamp(
                            article.get('datetime', 0)
                        ).isoformat() if article.get('datetime') else None,
                        'source': f"finnhub_{article.get('source', 'unknown')}",
                        'language': 'en'
                    })
                
                return articles
        
        except Exception as e:
            logger.warning(f"⚠️ Finnhub fetch failed for {symbol}: {e}")
        
        return []
    
    def _fetch_alpha_vantage_news(self, symbol: str, target_date: datetime) -> List[Dict]:
        """Fetch from Alpha Vantage News Intelligence"""
        
        try:
            # Alpha Vantage has very strict rate limits
            if 'alpha_vantage_news' in self.last_fetch_time:
                elapsed = (datetime.now() - self.last_fetch_time['alpha_vantage_news']).total_seconds()
                if elapsed < 15:  # 4 calls per minute
                    return []
            
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'apikey': self.alpha_vantage_key,
                'limit': 20
            }
            
            response = requests.get(url, params=params, timeout=15)
            self.last_fetch_time['alpha_vantage_news'] = datetime.now()
            
            if response.status_code == 200:
                data = response.json()
                articles = []
                
                for article in data.get('feed', []):
                    # Filter for target date
                    pub_date = article.get('time_published', '')
                    if pub_date and pub_date.startswith(target_date.strftime('%Y%m%d')):
                        
                        # Extract sentiment for this ticker
                        ticker_sentiment = None
                        for sentiment in article.get('ticker_sentiment', []):
                            if sentiment.get('ticker') == symbol:
                                ticker_sentiment = sentiment
                                break
                        
                        articles.append({
                            'symbol': symbol,
                            'date': target_date.date(),
                            'title': article.get('title', ''),
                            'description': article.get('summary', ''),
                            'content': article.get('summary', '')[:500],
                            'url': article.get('url', ''),
                            'published_at': self._parse_av_timestamp(pub_date),
                            'source': f"alpha_vantage_{article.get('source', 'unknown')}",
                            'language': 'en',
                            'sentiment_score': float(ticker_sentiment.get('ticker_sentiment_score', 0)) if ticker_sentiment else None,
                            'sentiment_label': ticker_sentiment.get('ticker_sentiment_label') if ticker_sentiment else None
                        })
                
                return articles
        
        except Exception as e:
            logger.warning(f"⚠️ Alpha Vantage News fetch failed for {symbol}: {e}")
        
        return []
    
    def _parse_av_timestamp(self, timestamp_str: str) -> Optional[str]:
        """Parse Alpha Vantage timestamp format"""
        try:
            if len(timestamp_str) >= 8:
                dt = datetime.strptime(timestamp_str[:8], '%Y%m%d')
                if len(timestamp_str) > 8:
                    time_part = timestamp_str[9:]
                    if len(time_part) >= 6:
                        dt = dt.replace(
                            hour=int(time_part[:2]),
                            minute=int(time_part[2:4]),
                            second=int(time_part[4:6])
                        )
                return dt.isoformat()
        except:
            pass
        return None
    
    def _create_news_skeleton(self, symbol: str, target_date: datetime) -> List[Dict]:
        """Create skeleton news data when no real data available"""
        
        # Generate 1-3 skeleton articles per symbol
        num_articles = np.random.randint(1, 4)
        articles = []
        
        sample_headlines = [
            f"{symbol} reports quarterly earnings",
            f"Analysts update {symbol} price target",
            f"{symbol} announces new product development",
            f"Market volatility affects {symbol} trading",
            f"{symbol} management discusses growth strategy"
        ]
        
        for i in range(num_articles):
            sentiment_score = np.random.normal(0.1, 0.3)  # Slightly positive bias
            sentiment_score = np.clip(sentiment_score, -1, 1)
            
            articles.append({
                'symbol': symbol,
                'date': target_date.date(),
                'title': np.random.choice(sample_headlines),
                'description': f"Market news related to {symbol}",
                'content': f"Financial news content for {symbol}",
                'url': f"https://example.com/news/{symbol}_{i}",
                'published_at': (target_date - timedelta(hours=np.random.randint(1, 24))).isoformat(),
                'source': 'skeleton_generator',
                'language': 'en',
                'sentiment_score': sentiment_score,
                'sentiment_label': 'neutral'
            })
        
        return articles
    
    def _deduplicate_articles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate articles using multiple methods"""
        
        initial_count = len(df)
        
        # Method 1: Remove exact URL duplicates
        df = df.drop_duplicates(subset=['url'], keep='first')
        
        # Method 2: Remove similar titles (fuzzy matching)
        df = self._remove_similar_titles(df)
        
        # Method 3: Remove articles with similar content hashes
        df = self._remove_similar_content(df)
        
        final_count = len(df)
        removed = initial_count - final_count
        
        if removed > 0:
            logger.info(f"   🧹 Deduplication: removed {removed} duplicates ({initial_count} → {final_count})")
        
        return df
    
    def _remove_similar_titles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove articles with very similar titles"""
        
        def normalize_title(title):
            if not title:
                return ""
            # Remove special characters, convert to lowercase, remove extra spaces
            normalized = re.sub(r'[^\w\s]', '', title.lower())
            return ' '.join(normalized.split())
        
        df['normalized_title'] = df['title'].apply(normalize_title)
        
        # Group by symbol and remove titles that are >80% similar
        keep_indices = []
        
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol]
            seen_titles = set()
            
            for idx, row in symbol_df.iterrows():
                title = row['normalized_title']
                
                # Check similarity with existing titles
                is_duplicate = False
                for seen_title in seen_titles:
                    if title and seen_title:
                        similarity = self._title_similarity(title, seen_title)
                        if similarity > 0.8:
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    seen_titles.add(title)
                    keep_indices.append(idx)
        
        df = df.loc[keep_indices].drop('normalized_title', axis=1)
        return df
    
    def _title_similarity(self, title1: str, title2: str) -> float:
        """Calculate simple title similarity"""
        if not title1 or not title2:
            return 0.0
        
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _remove_similar_content(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove articles with similar content using hashing"""
        
        def content_hash(content):
            if not content:
                return ""
            # Create hash of normalized content
            normalized = re.sub(r'\s+', ' ', content.lower().strip())
            return hashlib.md5(normalized.encode()).hexdigest()
        
        df['content_hash'] = df['content'].apply(content_hash)
        
        # Remove duplicates based on content hash
        df = df.drop_duplicates(subset=['symbol', 'content_hash'], keep='first')
        df = df.drop('content_hash', axis=1)
        
        return df
    
    def _add_sentiment_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment analysis to news articles"""
        
        # If sentiment already exists from API, don't overwrite
        has_sentiment = 'sentiment_score' in df.columns and df['sentiment_score'].notna().any()
        
        if not has_sentiment:
            # Simple VADER-like sentiment analysis
            df['sentiment_score'] = df.apply(self._calculate_sentiment, axis=1)
            df['sentiment_label'] = df['sentiment_score'].apply(self._sentiment_label)
        
        # Ensure sentiment is in valid range
        df['sentiment_score'] = df['sentiment_score'].clip(-1, 1)
        
        logger.info("   ✅ Sentiment analysis completed")
        
        return df
    
    def _calculate_sentiment(self, row) -> float:
        """Simple sentiment calculation"""
        
        text = f"{row.get('title', '')} {row.get('description', '')}"
        if not text.strip():
            return 0.0
        
        # Simple word-based sentiment
        positive_words = {
            'growth', 'profit', 'gain', 'rise', 'increase', 'beat', 'exceed', 
            'strong', 'robust', 'positive', 'bullish', 'upgrade', 'buy',
            'outperform', 'recommend', 'success', 'record', 'high'
        }
        
        negative_words = {
            'loss', 'decline', 'fall', 'drop', 'decrease', 'miss', 'weak',
            'negative', 'bearish', 'downgrade', 'sell', 'underperform',
            'concern', 'risk', 'low', 'poor', 'disappointing', 'cut'
        }
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_words = len(words)
        if total_words == 0:
            return 0.0
        
        sentiment = (positive_count - negative_count) / total_words
        return np.clip(sentiment * 5, -1, 1)  # Scale and clip
    
    def _sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def _aggregate_by_symbol_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate news by symbol and date"""
        
        if df.empty:
            return df
        
        # Group by symbol and date
        aggregated = []
        
        for (symbol, date), group in df.groupby(['symbol', 'date']):
            
            # Calculate aggregated metrics
            agg_record = {
                'Date': date,
                'Ticker': symbol,
                
                # Volume metrics
                'News_Volume': len(group),
                'Unique_Sources': group['source'].nunique(),
                
                # Sentiment metrics
                'Avg_Sentiment': group['sentiment_score'].mean(),
                'Sentiment_Std': group['sentiment_score'].std(),
                'Positive_News_Pct': (group['sentiment_score'] > 0.1).mean(),
                'Negative_News_Pct': (group['sentiment_score'] < -0.1).mean(),
                
                # Sentiment momentum (not available without history)
                'Sentiment_Momentum': 0.0,  # Would need historical data
                
                # News novelty score
                'News_Novelty': group['source'].nunique() / len(group),
                
                # Aggregate text features
                'Total_Headline_Length': group['title'].str.len().sum(),
                'Avg_Headline_Length': group['title'].str.len().mean(),
                
                # Temporal features
                'News_Spread_Hours': self._calculate_news_spread(group),
                
                # Quality indicators
                'Has_Content': (group['content'].str.len() > 50).mean(),
                'Source_Diversity': group['source'].nunique() / max(1, len(group))
            }
            
            aggregated.append(agg_record)
        
        result = pd.DataFrame(aggregated)
        
        # Add cross-sectional features
        if len(result) > 1:
            result = self._add_cross_sectional_news_features(result)
        
        return result
    
    def _calculate_news_spread(self, group: pd.DataFrame) -> float:
        """Calculate temporal spread of news articles"""
        
        try:
            timestamps = pd.to_datetime(group['published_at'].dropna())
            if len(timestamps) > 1:
                spread = (timestamps.max() - timestamps.min()).total_seconds() / 3600
                return min(spread, 24.0)  # Cap at 24 hours
        except:
            pass
        
        return 0.0
    
    def _add_cross_sectional_news_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-sectional news features"""
        
        # Z-scores for key metrics
        for col in ['News_Volume', 'Avg_Sentiment', 'Sentiment_Std']:
            if col in df.columns and df[col].std() > 0:
                df[f'ZSCORE_{col}'] = (df[col] - df[col].mean()) / df[col].std()
                df[f'RANK_{col}'] = df[col].rank(pct=True)
        
        # Abnormal news volume detection
        if 'News_Volume' in df.columns:
            volume_mean = df['News_Volume'].mean()
            volume_std = df['News_Volume'].std()
            df['Abnormal_News_Volume'] = (df['News_Volume'] > volume_mean + 2 * volume_std).astype(int)
        
        return df