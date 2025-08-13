#!/usr/bin/env python3
"""
Twitter/X data fetcher for financial market sentiment analysis
"""

import tweepy
import pandas as pd
import datetime
import time
import logging
from textblob import TextBlob
import re
from typing import List, Dict
import os
import requests
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TwitterFinancialFetcher:
    """Fetch financial discussions from Twitter/X"""
    
    def __init__(self, bearer_token=None, api_key=None, api_secret=None, access_token=None, access_token_secret=None):
        """
        Initialize Twitter API client
        
        For Twitter API v2 credentials:
        1. Go to https://developer.twitter.com/
        2. Create a new app
        3. Get your Bearer Token (for API v2)
        4. Or get API keys for v1.1 (api_key, api_secret, access_token, access_token_secret)
        """
        self.bearer_token = bearer_token or os.getenv('TWITTER_BEARER_TOKEN')
        self.api_key = api_key or os.getenv('TWITTER_API_KEY')
        self.api_secret = api_secret or os.getenv('TWITTER_API_SECRET')
        self.access_token = access_token or os.getenv('TWITTER_ACCESS_TOKEN')
        self.access_token_secret = access_token_secret or os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        
        self.client = None
        self.api = None
        
        # Try to initialize Twitter API v2 (preferred)
        if self.bearer_token:
            try:
                self.client = tweepy.Client(bearer_token=self.bearer_token, wait_on_rate_limit=True)
                logger.info("Twitter API v2 initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Twitter API v2: {e}")
        
        # Try to initialize Twitter API v1.1 (fallback)
        if not self.client and all([self.api_key, self.api_secret, self.access_token, self.access_token_secret]):
            try:
                auth = tweepy.OAuthHandler(self.api_key, self.api_secret)
                auth.set_access_token(self.access_token, self.access_token_secret)
                self.api = tweepy.API(auth, wait_on_rate_limit=True)
                logger.info("Twitter API v1.1 initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Twitter API v1.1: {e}")
        
        if not self.client and not self.api:
            logger.warning("Twitter API not available. Some features will be limited.")
    
    def get_financial_keywords(self) -> List[str]:
        """Get financial keywords and hashtags to search for"""
        return [
            # Stock tickers with $
            '$AAPL', '$MSFT', '$GOOGL', '$AMZN', '$TSLA', 
            '$META', '$NVDA', '$AMD', '$INTC', '$QCOM',
            
            # Company names
            'Apple', 'Microsoft', 'Google', 'Amazon', 'Tesla',
            'Meta', 'NVIDIA', 'AMD', 'Intel', 'Qualcomm',
            
            # Financial hashtags
            '#stocks', '#investing', '#trading', '#stockmarket',
            '#finance', '#earnings', '#bull', '#bear',
            '#portfolio', '#dividend', '#options'
        ]
    
    def get_stock_mentions(self, text: str) -> List[str]:
        """Extract stock ticker mentions from text"""
        # Pattern for stock tickers: $AAPL, AAPL, etc.
        pattern = r'\b(?:\$)?([A-Z]{1,5})\b'
        matches = re.findall(pattern, text.upper())
        
        # Filter for known stock tickers
        known_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'INTC', 'QCOM']
        return [ticker for ticker in matches if ticker in known_tickers]
    
    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using TextBlob"""
        try:
            # Clean text
            clean_text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            clean_text = re.sub(r'@\w+|#\w+', '', clean_text)
            
            blob = TextBlob(clean_text)
            return blob.sentiment.polarity
        except:
            return 0.0
    
    def search_tweets_v2(self, query: str, max_results: int = 100) -> List[Dict]:
        """Search tweets using Twitter API v2"""
        if not self.client:
            return []
        
        tweets_data = []
        
        try:
            tweets = tweepy.Paginator(
                self.client.search_recent_tweets,
                query=query,
                tweet_fields=['created_at', 'public_metrics', 'author_id', 'context_annotations'],
                max_results=max_results
            ).flatten(limit=max_results)
            
            for tweet in tweets:
                # Extract stock mentions
                stock_mentions = self.get_stock_mentions(tweet.text)
                
                if stock_mentions:  # Only process tweets mentioning stocks
                    sentiment = self.analyze_sentiment(tweet.text)
                    
                    for ticker in stock_mentions:
                        tweets_data.append({
                            'timestamp': tweet.created_at,
                            'text': tweet.text,
                            'ticker': ticker,
                            'sentiment_score': sentiment,
                            'retweet_count': tweet.public_metrics['retweet_count'],
                            'like_count': tweet.public_metrics['like_count'],
                            'reply_count': tweet.public_metrics['reply_count'],
                            'quote_count': tweet.public_metrics['quote_count'],
                            'tweet_id': tweet.id,
                            'author_id': tweet.author_id,
                            'source': 'twitter_v2'
                        })
        
        except Exception as e:
            logger.error(f"Error searching tweets with query '{query}': {e}")
        
        return tweets_data
    
    def search_tweets_v1(self, query: str, count: int = 100) -> List[Dict]:
        """Search tweets using Twitter API v1.1"""
        if not self.api:
            return []
        
        tweets_data = []
        
        try:
            tweets = tweepy.Cursor(
                self.api.search_tweets,
                q=query,
                result_type='mixed',
                lang='en'
            ).items(count)
            
            for tweet in tweets:
                # Extract stock mentions
                stock_mentions = self.get_stock_mentions(tweet.text)
                
                if stock_mentions:  # Only process tweets mentioning stocks
                    sentiment = self.analyze_sentiment(tweet.text)
                    
                    for ticker in stock_mentions:
                        tweets_data.append({
                            'timestamp': tweet.created_at,
                            'text': tweet.text,
                            'ticker': ticker,
                            'sentiment_score': sentiment,
                            'retweet_count': tweet.retweet_count,
                            'like_count': tweet.favorite_count,
                            'reply_count': 0,  # Not available in v1.1
                            'quote_count': 0,  # Not available in v1.1
                            'tweet_id': tweet.id,
                            'author_id': tweet.user.id,
                            'source': 'twitter_v1'
                        })
        
        except Exception as e:
            logger.error(f"Error searching tweets with query '{query}': {e}")
        
        return tweets_data
    
    def fetch_financial_tweets(self, max_tweets_per_query: int = 100) -> pd.DataFrame:
        """Fetch financial tweets using all available keywords"""
        all_tweets = []
        keywords = self.get_financial_keywords()
        
        logger.info(f"Searching for tweets with {len(keywords)} different keywords...")
        
        for keyword in keywords:
            logger.info(f"Searching for: {keyword}")
            
            # Try API v2 first, then fallback to v1.1
            tweets = []
            if self.client:
                tweets = self.search_tweets_v2(keyword, max_tweets_per_query)
            elif self.api:
                tweets = self.search_tweets_v1(keyword, max_tweets_per_query)
            
            all_tweets.extend(tweets)
            
            # Rate limiting
            time.sleep(1)
        
        if not all_tweets:
            logger.warning("No tweets fetched. Check Twitter API credentials and rate limits.")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_tweets)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['tweet_id', 'ticker'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        logger.info(f"Fetched {len(df)} tweets mentioning stocks")
        
        return df
    
    def get_trending_tickers(self, df: pd.DataFrame, min_mentions: int = 5) -> pd.DataFrame:
        """Get trending stock tickers from Twitter data"""
        if df.empty:
            return pd.DataFrame()
        
        # Calculate engagement score
        df['engagement_score'] = (
            df['retweet_count'] * 2 + 
            df['like_count'] + 
            df['reply_count'] * 1.5 + 
            df['quote_count'] * 1.5
        )
        
        trending = df.groupby('ticker').agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'engagement_score': 'sum',
            'retweet_count': 'sum',
            'like_count': 'sum',
            'timestamp': ['min', 'max']
        }).round(4)
        
        trending.columns = ['avg_sentiment', 'sentiment_std', 'mention_count',
                           'total_engagement', 'total_retweets', 'total_likes',
                           'first_mention', 'last_mention']
        
        # Filter by minimum mentions
        trending = trending[trending['mention_count'] >= min_mentions]
        
        # Sort by engagement
        trending = trending.sort_values('total_engagement', ascending=False)
        
        return trending.reset_index()
    
    def save_twitter_data(self, df: pd.DataFrame, filename: str = 'data/twitter_financial.csv'):
        """Save Twitter data to CSV"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)
        logger.info(f"Twitter data saved to {filename}")

def main():
    """Main function to fetch Twitter financial data"""
    # Initialize fetcher
    fetcher = TwitterFinancialFetcher()
    
    if not fetcher.client and not fetcher.api:
        print("\n" + "="*60)
        print("TWITTER API SETUP REQUIRED")
        print("="*60)
        print("To fetch Twitter data, you need API credentials:")
        print("1. Go to https://developer.twitter.com/")
        print("2. Create a new app")
        print("3. Get your Bearer Token (for API v2) or API keys")
        print("4. Set environment variables:")
        print("   export TWITTER_BEARER_TOKEN='your_bearer_token'")
        print("   # OR for API v1.1:")
        print("   export TWITTER_API_KEY='your_api_key'")
        print("   export TWITTER_API_SECRET='your_api_secret'")
        print("   export TWITTER_ACCESS_TOKEN='your_access_token'")
        print("   export TWITTER_ACCESS_TOKEN_SECRET='your_access_token_secret'")
        print("5. Or modify this script with your credentials")
        print("="*60)
        return
    
    # Fetch data
    print("Fetching Twitter financial discussions...")
    df = fetcher.fetch_financial_tweets(max_tweets_per_query=50)
    
    if df.empty:
        print("No tweets fetched. Check your Twitter API credentials and rate limits.")
        return
    
    # Save data
    fetcher.save_twitter_data(df)
    
    # Show summary
    print(f"\nFetched {len(df)} tweets")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Tickers mentioned: {', '.join(df['ticker'].unique())}")
    
    # Trending analysis
    trending = fetcher.get_trending_tickers(df)
    if not trending.empty:
        print("\nTrending Tickers on Twitter:")
        print(trending[['ticker', 'mention_count', 'avg_sentiment', 'total_engagement']].to_string(index=False))
        
        # Save trending data
        trending.to_csv('data/twitter_trending.csv', index=False)
    
    print("\nData saved to:")
    print("- data/twitter_financial.csv")
    if not trending.empty:
        print("- data/twitter_trending.csv")

if __name__ == "__main__":
    main()