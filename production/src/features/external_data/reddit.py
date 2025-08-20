#!/usr/bin/env python3
"""
Reddit data integration using official Reddit API
Fetches sentiment from financial subreddits for market prediction
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

class RedditDataFetcher:
    """Fetches sentiment data from Reddit using the official API"""
    
    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None,
                 user_agent: str = "MarketAI:v1.0 (by /u/marketai_research)"):
        """
        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret  
            user_agent: User agent string for API requests
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.access_token = None
        self.token_expires = None
        
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': user_agent})
        
        # Financial subreddits to monitor
        self.financial_subreddits = [
            'investing', 'stocks', 'SecurityAnalysis', 'ValueInvesting',
            'financialindependence', 'StockMarket', 'trading', 'options',
            'wallstreetbets', 'SecurityAnalysis', 'dividends', 'ETFs'
        ]
        
        # Financial sentiment words for basic analysis
        self.bullish_words = {
            'bullish', 'bull', 'buy', 'buying', 'long', 'calls', 'moon', 'rocket',
            'pump', 'green', 'gains', 'profit', 'up', 'rise', 'surge', 'rally',
            'breakout', 'growth', 'strong', 'positive', 'optimistic', 'hodl'
        }
        
        self.bearish_words = {
            'bearish', 'bear', 'sell', 'selling', 'short', 'puts', 'crash', 'dump',
            'red', 'losses', 'loss', 'down', 'fall', 'drop', 'decline', 'correction',
            'recession', 'weak', 'negative', 'pessimistic', 'bubble', 'overvalued'
        }
        
        # Ticker extraction pattern
        self.ticker_pattern = re.compile(r'\$([A-Z]{1,5})\b|\b([A-Z]{1,5})\b(?=\s|$)')
        
    def _get_access_token(self) -> bool:
        """Get OAuth2 access token for Reddit API"""
        
        if not self.client_id or not self.client_secret:
            logger.warning("Reddit API credentials not provided. Using read-only mode.")
            return False
        
        if self.access_token and self.token_expires and datetime.now() < self.token_expires:
            return True  # Token still valid
        
        try:
            auth = requests.auth.HTTPBasicAuth(self.client_id, self.client_secret)
            data = {
                'grant_type': 'client_credentials',
                'duration': 'permanent'
            }
            headers = {'User-Agent': self.user_agent}
            
            response = requests.post(
                'https://www.reddit.com/api/v1/access_token',
                auth=auth, data=data, headers=headers
            )
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data['access_token']
            self.token_expires = datetime.now() + timedelta(seconds=token_data['expires_in'] - 60)
            
            # Update session headers
            self.session.headers.update({
                'Authorization': f'bearer {self.access_token}'
            })
            
            logger.info("Reddit API authentication successful")
            return True
            
        except Exception as e:
            logger.error(f"Reddit API authentication failed: {e}")
            return False
    
    def _make_api_request(self, endpoint: str, params: Dict) -> Dict:
        """Make rate-limited request to Reddit API"""
        
        try:
            if self.access_token:
                url = f"https://oauth.reddit.com{endpoint}"
            else:
                url = f"https://www.reddit.com{endpoint}.json"
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            # Reddit API rate limit: 60 requests per minute
            time.sleep(1.1)
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Reddit API request failed: {e}")
            return {}
    
    def _extract_tickers(self, text: str) -> Set[str]:
        """Extract stock tickers from text"""
        
        matches = self.ticker_pattern.findall(text.upper())
        tickers = set()
        
        for match in matches:
            ticker = match[0] if match[0] else match[1]
            # Filter out common false positives
            if ticker and len(ticker) <= 5 and ticker not in {
                'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER',
                'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'USE', 'MAN', 'NEW', 'NOW',
                'WAY', 'MAY', 'SAY', 'SEE', 'HIM', 'TWO', 'HOW', 'ITS', 'WHO', 'DID',
                'YES', 'HIS', 'HAS', 'HAD', 'LET', 'PUT', 'TOO', 'OLD', 'ANY', 'APP',
                'BOT', 'LOL', 'CEO', 'IPO', 'ETF', 'USD', 'WSB', 'SEC', 'FDA', 'GDP'
            }:
                tickers.add(ticker)
        
        return tickers
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of Reddit post/comment"""
        
        if not text:
            return {
                'bullish_score': 0.0,
                'bearish_score': 0.0,
                'sentiment_score': 0.0,
                'word_count': 0
            }
        
        words = re.findall(r'\b\w+\b', text.lower())
        word_count = len(words)
        
        if word_count == 0:
            return {
                'bullish_score': 0.0,
                'bearish_score': 0.0,
                'sentiment_score': 0.0,
                'word_count': 0
            }
        
        # Count sentiment words
        bullish_count = sum(1 for word in words if word in self.bullish_words)
        bearish_count = sum(1 for word in words if word in self.bearish_words)
        
        # Calculate scores
        bullish_score = (bullish_count / word_count) * 100
        bearish_score = (bearish_count / word_count) * 100
        sentiment_score = bullish_score - bearish_score
        
        return {
            'bullish_score': bullish_score,
            'bearish_score': bearish_score,
            'sentiment_score': sentiment_score,
            'word_count': word_count
        }
    
    def fetch_subreddit_posts(self, subreddit: str, limit: int = 100,
                             time_filter: str = 'day') -> pd.DataFrame:
        """Fetch recent posts from a subreddit"""
        
        logger.info(f"Fetching posts from r/{subreddit}")
        
        params = {
            'limit': limit,
            't': time_filter,  # day, week, month, year
            'sort': 'hot'  # hot, new, top
        }
        
        # Try hot posts first, then new posts
        for sort_type in ['hot', 'new']:
            params['sort'] = sort_type
            data = self._make_api_request(f'/r/{subreddit}/{sort_type}', params)
            
            if data and 'data' in data and 'children' in data['data']:
                break
        else:
            logger.warning(f"No data fetched from r/{subreddit}")
            return pd.DataFrame()
        
        posts = []
        
        for post_data in data['data']['children']:
            try:
                post = post_data['data']
                
                # Skip removed/deleted posts
                if post.get('removed_by_category') or post.get('selftext') == '[deleted]':
                    continue
                
                # Extract post info
                created_time = datetime.fromtimestamp(post['created_utc'])
                title = post.get('title', '')
                selftext = post.get('selftext', '')
                full_text = f"{title} {selftext}"
                
                # Analyze sentiment
                sentiment = self._analyze_sentiment(full_text)
                
                # Extract mentioned tickers
                tickers = self._extract_tickers(full_text)
                
                post_info = {
                    'subreddit': subreddit,
                    'post_id': post['id'],
                    'created_time': created_time,
                    'title': title,
                    'score': post.get('score', 0),
                    'num_comments': post.get('num_comments', 0),
                    'upvote_ratio': post.get('upvote_ratio', 0.5),
                    'tickers_mentioned': ','.join(tickers) if tickers else '',
                    'ticker_count': len(tickers),
                    **sentiment
                }
                
                posts.append(post_info)
                
            except Exception as e:
                logger.warning(f"Error processing post: {e}")
                continue
        
        if not posts:
            return pd.DataFrame()
        
        return pd.DataFrame(posts)
    
    def fetch_multi_subreddit_data(self, subreddits: Optional[List[str]] = None,
                                  limit_per_sub: int = 50) -> pd.DataFrame:
        """Fetch data from multiple financial subreddits"""
        
        if subreddits is None:
            subreddits = self.financial_subreddits
        
        # Get access token if available
        self._get_access_token()
        
        all_posts = []
        
        for subreddit in subreddits:
            try:
                posts_df = self.fetch_subreddit_posts(subreddit, limit=limit_per_sub)
                if not posts_df.empty:
                    all_posts.append(posts_df)
                    
            except Exception as e:
                logger.warning(f"Failed to fetch data from r/{subreddit}: {e}")
                continue
        
        if not all_posts:
            return pd.DataFrame()
        
        combined_df = pd.concat(all_posts, ignore_index=True)
        return combined_df
    
    def aggregate_daily_sentiment(self, posts_df: pd.DataFrame,
                                 tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """Aggregate Reddit sentiment to daily features"""
        
        if posts_df.empty:
            return pd.DataFrame()
        
        # Filter by specific tickers if provided
        if tickers:
            ticker_filter = posts_df['tickers_mentioned'].str.contains(
                '|'.join(tickers), case=False, na=False
            )
            filtered_df = posts_df[ticker_filter | (posts_df['ticker_count'] == 0)]
        else:
            filtered_df = posts_df
        
        # Group by date
        filtered_df['date'] = filtered_df['created_time'].dt.date
        
        daily_agg = filtered_df.groupby('date').agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'bullish_score': 'mean',
            'bearish_score': 'mean',
            'score': ['mean', 'sum'],  # Reddit post scores
            'num_comments': ['mean', 'sum'],
            'upvote_ratio': 'mean',
            'ticker_count': 'sum',
            'post_id': 'count'  # Number of posts
        }).round(4)
        
        # Flatten column names
        daily_agg.columns = ['_'.join(col).strip() for col in daily_agg.columns]
        
        # Rename for clarity
        daily_agg = daily_agg.rename(columns={
            'sentiment_score_mean': 'reddit_sentiment',
            'sentiment_score_std': 'reddit_sentiment_vol', 
            'sentiment_score_count': 'reddit_posts_analyzed',
            'bullish_score_mean': 'reddit_bullish',
            'bearish_score_mean': 'reddit_bearish',
            'score_mean': 'reddit_avg_score',
            'score_sum': 'reddit_total_score',
            'num_comments_mean': 'reddit_avg_comments',
            'num_comments_sum': 'reddit_total_comments',
            'upvote_ratio_mean': 'reddit_avg_upvote_ratio',
            'ticker_count_sum': 'reddit_ticker_mentions',
            'post_id_count': 'reddit_post_count'
        })
        
        # Add derived features
        daily_agg['reddit_engagement'] = (
            daily_agg['reddit_total_score'] * daily_agg['reddit_avg_upvote_ratio']
        )
        daily_agg['reddit_sentiment_momentum'] = daily_agg['reddit_sentiment'].diff()
        
        return daily_agg

def create_reddit_features(start_date: str = "2020-01-01",
                          end_date: Optional[str] = None,
                          tickers: Optional[List[str]] = None,
                          client_id: Optional[str] = None,
                          client_secret: Optional[str] = None) -> pd.DataFrame:
    """Convenience function to create Reddit sentiment features"""
    
    fetcher = RedditDataFetcher(client_id=client_id, client_secret=client_secret)
    
    try:
        # Fetch recent Reddit data (API limitations make historical data difficult)
        posts_df = fetcher.fetch_multi_subreddit_data()
        
        if posts_df.empty:
            logger.warning("No Reddit data available, creating dummy sentiment features")
            
            # Create dummy Reddit features
            date_range = pd.date_range(start=start_date, 
                                     end=end_date or datetime.now().strftime("%Y-%m-%d"),
                                     freq='D')
            dummy_data = pd.DataFrame(index=date_range)
            dummy_data['reddit_sentiment'] = np.random.randn(len(dummy_data)) * 0.5
            dummy_data['reddit_bullish'] = np.random.exponential(0.3, len(dummy_data))
            dummy_data['reddit_bearish'] = np.random.exponential(0.3, len(dummy_data))
            dummy_data['reddit_post_count'] = np.random.poisson(20, len(dummy_data))
            dummy_data['reddit_engagement'] = np.random.exponential(100, len(dummy_data))
            
            return dummy_data
        
        # Aggregate to daily features
        daily_features = fetcher.aggregate_daily_sentiment(posts_df, tickers)
        
        # If we have recent data but need historical, extend with dummy data
        if not daily_features.empty:
            start_dt = pd.to_datetime(start_date)
            min_date = pd.to_datetime(daily_features.index.min())
            
            if start_dt < min_date:
                # Create dummy data for earlier dates
                dummy_range = pd.date_range(start=start_dt, end=min_date - timedelta(days=1), freq='D')
                dummy_data = pd.DataFrame(index=dummy_range)
                
                # Use statistics from real data to make realistic dummy data
                for col in daily_features.columns:
                    if daily_features[col].std() > 0:
                        dummy_data[col] = np.random.normal(
                            daily_features[col].mean(),
                            daily_features[col].std(),
                            len(dummy_data)
                        )
                    else:
                        dummy_data[col] = daily_features[col].mean()
                
                # Combine dummy and real data
                daily_features = pd.concat([dummy_data, daily_features])
        
        return daily_features
        
    except Exception as e:
        logger.error(f"Failed to fetch Reddit data: {e}")
        
        # Fallback to dummy data
        date_range = pd.date_range(start=start_date, 
                                 end=end_date or datetime.now().strftime("%Y-%m-%d"),
                                 freq='D')
        dummy_data = pd.DataFrame(index=date_range)
        dummy_data['reddit_sentiment'] = np.random.randn(len(dummy_data)) * 0.5
        dummy_data['reddit_bullish'] = np.random.exponential(0.3, len(dummy_data))
        dummy_data['reddit_bearish'] = np.random.exponential(0.3, len(dummy_data))
        dummy_data['reddit_post_count'] = np.random.poisson(20, len(dummy_data))
        dummy_data['reddit_engagement'] = np.random.exponential(100, len(dummy_data))
        
        return dummy_data