#!/usr/bin/env python3
"""
Process downloaded social media datasets and prepare for training integration
"""

import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime, timedelta
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetProcessor:
    """Process and standardize downloaded social media datasets"""
    
    def __init__(self):
        self.datasets_dir = "data/social_datasets"
        self.output_dir = "data"
        
    def extract_tickers(self, text: str) -> list:
        """Extract stock tickers from text"""
        if not isinstance(text, str):
            return []
        
        # Find $TICKER patterns
        dollar_tickers = re.findall(r'\$([A-Z]{1,5})\b', text)
        
        # Find standalone tickers (be more conservative)
        standalone_tickers = re.findall(r'\b([A-Z]{2,5})\b', text)
        
        # Known stock tickers to validate against
        known_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'INTC', 'QCOM',
                        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'PFE', 'JNJ', 'UNH', 'MRNA',
                        'XOM', 'CVX', 'COP', 'SLB', 'OXY', 'BYND', 'GME', 'AMC', 'BB', 'NOK']
        
        # Combine and filter
        all_tickers = set(dollar_tickers + standalone_tickers)
        valid_tickers = [ticker for ticker in all_tickers if ticker in known_tickers]
        
        return valid_tickers if valid_tickers else ['UNKNOWN']
    
    def process_twitter_financial_news(self):
        """Process Twitter financial news sentiment dataset"""
        logger.info("Processing Twitter financial news dataset...")
        
        file_path = f"{self.datasets_dir}/twitter_financial_news_sentiment.csv"
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} Twitter financial news records")
        
        # Process the dataset
        processed_data = []
        
        for _, row in df.iterrows():
            text = row['text']
            label = row['label']  # 0=bearish, 1=neutral, 2=bullish
            
            # Extract tickers
            tickers = self.extract_tickers(text)
            
            # Convert label to sentiment score
            sentiment_map = {0: -0.5, 1: 0.0, 2: 0.5}  # bearish, neutral, bullish
            sentiment_score = sentiment_map.get(label, 0.0)
            
            # Create record for each ticker
            for ticker in tickers:
                processed_data.append({
                    'text': text,
                    'ticker': ticker,
                    'sentiment_score': sentiment_score,
                    'source': 'twitter_financial_news',
                    'label': label,
                    'timestamp': datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 365))
                })
        
        result_df = pd.DataFrame(processed_data)
        logger.info(f"Processed into {len(result_df)} ticker-specific records")
        
        return result_df
    
    def process_financial_tweets_sentiment(self):
        """Process financial tweets sentiment dataset"""
        logger.info("Processing financial tweets sentiment dataset...")
        
        file_path = f"{self.datasets_dir}/financial_tweets_sentiment.csv"
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} financial tweets records")
        
        # Process the dataset
        processed_data = []
        
        for _, row in df.iterrows():
            text = row['tweet']
            sentiment = row['sentiment']  # Assuming 0=negative, 1=neutral, 2=positive
            
            # Extract tickers
            tickers = self.extract_tickers(text)
            
            # Convert sentiment to score
            if sentiment == 0:
                sentiment_score = -0.6  # Negative
            elif sentiment == 1:
                sentiment_score = 0.0   # Neutral
            elif sentiment == 2:
                sentiment_score = 0.6   # Positive
            else:
                sentiment_score = 0.0   # Default
            
            # Create record for each ticker
            for ticker in tickers:
                processed_data.append({
                    'text': text,
                    'ticker': ticker,
                    'sentiment_score': sentiment_score,
                    'source': 'financial_tweets_sentiment',
                    'label': sentiment,
                    'timestamp': datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 365))
                })
        
        result_df = pd.DataFrame(processed_data)
        logger.info(f"Processed into {len(result_df)} ticker-specific records")
        
        return result_df
    
    def process_sample_datasets(self):
        """Process sample datasets created earlier"""
        logger.info("Processing sample datasets...")
        
        all_sample_data = []
        
        # Process sample tweets
        tweets_path = f"{self.datasets_dir}/sample_financial_tweets.csv"
        if os.path.exists(tweets_path):
            tweets_df = pd.read_csv(tweets_path)
            tweets_df['source'] = 'sample_tweets'
            tweets_df['sentiment_score'] = np.random.uniform(-0.5, 0.5, len(tweets_df))  # Random sentiment
            all_sample_data.append(tweets_df[['text', 'ticker', 'sentiment_score', 'source', 'timestamp']])
        
        # Process sample Reddit
        reddit_path = f"{self.datasets_dir}/sample_reddit_financial.csv"
        if os.path.exists(reddit_path):
            reddit_df = pd.read_csv(reddit_path)
            reddit_df['source'] = 'sample_reddit'
            reddit_df['text'] = reddit_df['title']  # Use title as text
            reddit_df['sentiment_score'] = np.random.uniform(-0.4, 0.4, len(reddit_df))  # Random sentiment
            all_sample_data.append(reddit_df[['text', 'ticker', 'sentiment_score', 'source', 'timestamp']])
        
        if all_sample_data:
            result_df = pd.concat(all_sample_data, ignore_index=True)
            logger.info(f"Processed {len(result_df)} sample records")
            return result_df
        
        return pd.DataFrame()
    
    def combine_all_datasets(self):
        """Combine all processed datasets"""
        logger.info("Combining all datasets...")
        
        datasets = []
        
        # Process real datasets
        twitter_news = self.process_twitter_financial_news()
        if not twitter_news.empty:
            datasets.append(twitter_news)
        
        financial_tweets = self.process_financial_tweets_sentiment()
        if not financial_tweets.empty:
            datasets.append(financial_tweets)
        
        sample_data = self.process_sample_datasets()
        if not sample_data.empty:
            datasets.append(sample_data)
        
        if not datasets:
            logger.error("No datasets to combine!")
            return pd.DataFrame()
        
        # Combine all datasets
        combined_df = pd.concat(datasets, ignore_index=True)
        
        # Remove UNKNOWN tickers
        combined_df = combined_df[combined_df['ticker'] != 'UNKNOWN']
        
        # Add date column
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        combined_df['date'] = combined_df['timestamp'].dt.date
        
        # Add engagement score (mock for now)
        combined_df['engagement_score'] = np.random.randint(1, 100, len(combined_df))
        
        logger.info(f"Combined dataset: {len(combined_df)} records")
        logger.info(f"Unique tickers: {combined_df['ticker'].nunique()}")
        logger.info(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        
        return combined_df
    
    def save_processed_datasets(self, combined_df):
        """Save processed datasets"""
        
        # Save combined social media data
        output_path = f"{self.output_dir}/social_media_financial.csv"
        combined_df.to_csv(output_path, index=False)
        logger.info(f"✅ Saved combined dataset: {output_path}")
        
        # Create summary by ticker
        summary = combined_df.groupby('ticker').agg({
            'sentiment_score': ['count', 'mean', 'std'],
            'engagement_score': 'sum',
            'date': ['min', 'max']
        }).round(4)
        
        summary.columns = ['post_count', 'avg_sentiment', 'sentiment_std', 
                          'total_engagement', 'first_post', 'last_post']
        summary = summary.reset_index()
        
        summary_path = f"{self.output_dir}/social_media_summary.csv"
        summary.to_csv(summary_path, index=False)
        logger.info(f"✅ Saved summary: {summary_path}")
        
        return output_path, summary_path

def main():
    """Main function to process downloaded datasets"""
    print("🔄 Processing Downloaded Social Media Datasets")
    print("="*55)
    
    processor = DatasetProcessor()
    
    # Combine all datasets
    combined_df = processor.combine_all_datasets()
    
    if combined_df.empty:
        print("❌ No datasets found to process")
        return
    
    # Save processed datasets
    output_path, summary_path = processor.save_processed_datasets(combined_df)
    
    # Display summary
    print("\n" + "="*60)
    print("📊 PROCESSED DATASET SUMMARY")
    print("="*60)
    
    print(f"✅ Total records: {len(combined_df):,}")
    print(f"📅 Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    print(f"🎯 Unique tickers: {combined_df['ticker'].nunique()}")
    print(f"📱 Data sources: {', '.join(combined_df['source'].unique())}")
    
    # Top tickers by mentions
    top_tickers = combined_df['ticker'].value_counts().head(10)
    print(f"\n📈 Top 10 Most Mentioned Tickers:")
    for ticker, count in top_tickers.items():
        avg_sentiment = combined_df[combined_df['ticker'] == ticker]['sentiment_score'].mean()
        print(f"  {ticker}: {count:,} mentions (avg sentiment: {avg_sentiment:.3f})")
    
    # Sentiment distribution
    bullish = (combined_df['sentiment_score'] > 0.1).sum()
    bearish = (combined_df['sentiment_score'] < -0.1).sum()
    neutral = len(combined_df) - bullish - bearish
    
    print(f"\n📊 Sentiment Distribution:")
    print(f"  🐂 Bullish: {bullish:,} ({bullish/len(combined_df)*100:.1f}%)")
    print(f"  🐻 Bearish: {bearish:,} ({bearish/len(combined_df)*100:.1f}%)")
    print(f"  😐 Neutral: {neutral:,} ({neutral/len(combined_df)*100:.1f}%)")
    
    print(f"\n💾 Files created:")
    print(f"  📁 {output_path}")
    print(f"  📁 {summary_path}")
    
    print(f"\n🚀 Next Steps:")
    print(f"1. Integrate with training data:")
    print(f"   python social_data_integration.py")
    print(f"2. Train models with social features:")
    print(f"   python train_advanced.py --experiment social_enhanced")

if __name__ == "__main__":
    main()