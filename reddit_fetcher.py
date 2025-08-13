#!/usr/bin/env python3
"""
Reddit data fetcher for financial discussions and sentiment analysis
"""

import praw
import pandas as pd
import datetime
import time
import logging
from textblob import TextBlob
import re
from typing import List, Dict
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RedditFinancialFetcher:
    """Fetch financial discussions from Reddit"""
    
    def __init__(self, client_id=None, client_secret=None, user_agent="FinancialAI/1.0"):
        """
        Initialize Reddit API client
        
        To get Reddit API credentials:
        1. Go to https://www.reddit.com/prefs/apps
        2. Click "Create App" or "Create Another App"
        3. Choose "script" type
        4. Get your client_id and client_secret
        """
        self.client_id = client_id or os.getenv('REDDIT_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('REDDIT_CLIENT_SECRET')
        self.user_agent = user_agent
        
        if not self.client_id or not self.client_secret:
            logger.warning("Reddit API credentials not provided. Using read-only mode with limited functionality.")
            self.reddit = None
        else:
            try:
                self.reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent
                )
                logger.info("Reddit API initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Reddit API: {e}")
                self.reddit = None
    
    def get_financial_subreddits(self) -> List[str]:
        """Get list of financial subreddits to monitor"""
        return [
            'stocks', 'investing', 'SecurityAnalysis', 'ValueInvesting',
            'StockMarket', 'financialindependence', 'personalfinance',
            'wallstreetbets', 'SecurityAnalysis', 'investing_discussion',
            'SecurityHolders', 'StockPickerDaily', 'ValueInvesting',
            'dividends', 'options', 'SecurityAnalysis'
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
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0
    
    def fetch_subreddit_posts(self, subreddit_name: str, limit: int = 100, time_filter: str = 'month') -> List[Dict]:
        """Fetch posts from a specific subreddit"""
        if not self.reddit:
            logger.warning("Reddit API not available. Cannot fetch posts.")
            return []
        
        posts_data = []
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Get hot posts
            for post in subreddit.hot(limit=limit//3):
                posts_data.extend(self._process_post(post, subreddit_name))
                time.sleep(0.1)  # Rate limiting
            
            # Get top posts
            for post in subreddit.top(time_filter=time_filter, limit=limit//3):
                posts_data.extend(self._process_post(post, subreddit_name))
                time.sleep(0.1)
            
            # Get new posts
            for post in subreddit.new(limit=limit//3):
                posts_data.extend(self._process_post(post, subreddit_name))
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error fetching from r/{subreddit_name}: {e}")
        
        return posts_data
    
    def _process_post(self, post, subreddit_name: str) -> List[Dict]:
        """Process a single Reddit post and extract relevant data"""
        post_data = []
        
        # Combine title and selftext
        full_text = f"{post.title} {post.selftext}".strip()
        
        # Extract stock mentions
        stock_mentions = self.get_stock_mentions(full_text)
        
        if stock_mentions:  # Only process posts mentioning stocks
            sentiment = self.analyze_sentiment(full_text)
            
            for ticker in stock_mentions:
                post_data.append({
                    'timestamp': datetime.datetime.fromtimestamp(post.created_utc),
                    'subreddit': subreddit_name,
                    'title': post.title,
                    'text': post.selftext,
                    'full_text': full_text,
                    'ticker': ticker,
                    'sentiment_score': sentiment,
                    'upvotes': post.score,
                    'num_comments': post.num_comments,
                    'url': post.url,
                    'post_id': post.id,
                    'author': str(post.author) if post.author else 'deleted'
                })
        
        return post_data
    
    def fetch_all_financial_data(self, posts_per_subreddit: int = 100) -> pd.DataFrame:
        """Fetch financial discussions from all relevant subreddits"""
        all_data = []
        subreddits = self.get_financial_subreddits()
        
        logger.info(f"Fetching data from {len(subreddits)} subreddits...")
        
        for subreddit in subreddits:
            logger.info(f"Fetching from r/{subreddit}...")
            posts = self.fetch_subreddit_posts(subreddit, limit=posts_per_subreddit)
            all_data.extend(posts)
            
            # Rate limiting between subreddits
            time.sleep(1)
        
        if not all_data:
            logger.warning("No data fetched. Check Reddit API credentials.")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['post_id', 'ticker'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        logger.info(f"Fetched {len(df)} Reddit posts mentioning stocks")
        
        return df
    
    def save_reddit_data(self, df: pd.DataFrame, filename: str = 'data/reddit_financial.csv'):
        """Save Reddit data to CSV"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)
        logger.info(f"Reddit data saved to {filename}")
    
    def get_ticker_sentiment_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get sentiment summary by ticker"""
        if df.empty:
            return pd.DataFrame()
        
        summary = df.groupby('ticker').agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'upvotes': 'sum',
            'num_comments': 'sum',
            'timestamp': ['min', 'max']
        }).round(4)
        
        summary.columns = ['avg_sentiment', 'sentiment_std', 'mention_count', 
                          'total_upvotes', 'total_comments', 'first_mention', 'last_mention']
        
        return summary.reset_index()

def main():
    """Main function to fetch Reddit financial data"""
    # Initialize fetcher
    fetcher = RedditFinancialFetcher()
    
    if not fetcher.reddit:
        print("\n" + "="*60)
        print("REDDIT API SETUP REQUIRED")
        print("="*60)
        print("To fetch Reddit data, you need API credentials:")
        print("1. Go to https://www.reddit.com/prefs/apps")
        print("2. Click 'Create App' or 'Create Another App'")
        print("3. Choose 'script' type")
        print("4. Get your client_id and client_secret")
        print("5. Set environment variables:")
        print("   export REDDIT_CLIENT_ID='your_client_id'")
        print("   export REDDIT_CLIENT_SECRET='your_client_secret'")
        print("6. Or modify this script with your credentials")
        print("="*60)
        return
    
    # Fetch data
    print("Fetching Reddit financial discussions...")
    df = fetcher.fetch_all_financial_data(posts_per_subreddit=50)
    
    if df.empty:
        print("No data fetched. Check your Reddit API credentials.")
        return
    
    # Save data
    fetcher.save_reddit_data(df)
    
    # Show summary
    print(f"\nFetched {len(df)} Reddit posts")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Subreddits: {', '.join(df['subreddit'].unique())}")
    print(f"Tickers mentioned: {', '.join(df['ticker'].unique())}")
    
    # Sentiment summary
    summary = fetcher.get_ticker_sentiment_summary(df)
    print("\nSentiment Summary by Ticker:")
    print(summary.to_string(index=False))
    
    # Save summary
    summary.to_csv('data/reddit_sentiment_summary.csv', index=False)
    print("\nData saved to:")
    print("- data/reddit_financial.csv")
    print("- data/reddit_sentiment_summary.csv")

if __name__ == "__main__":
    main()