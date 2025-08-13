#!/usr/bin/env python3
"""
Download free Reddit and Twitter financial datasets from public sources
"""

import requests
import pandas as pd
import os
import logging
import zipfile
from datetime import datetime
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SocialDatasetDownloader:
    """Download free social media datasets for financial analysis"""
    
    def __init__(self):
        self.data_dir = "data/social_datasets"
        os.makedirs(self.data_dir, exist_ok=True)
    
    def download_figshare_wallstreetbets(self):
        """Download WallStreetBets dataset from Figshare (2020-2022)"""
        logger.info("Downloading WallStreetBets dataset from Figshare...")
        
        # Figshare direct download URL
        figshare_url = "https://figshare.com/ndownloader/files/38702053"
        output_path = f"{self.data_dir}/wallstreetbets_2020_2022.zip"
        
        try:
            response = requests.get(figshare_url, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"✅ Downloaded WallStreetBets dataset to {output_path}")
            
            # Extract the zip file
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(f"{self.data_dir}/wallstreetbets_figshare")
            
            logger.info("✅ Extracted WallStreetBets dataset")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download Figshare dataset: {e}")
            return False
    
    def download_sample_financial_tweets(self):
        """Download sample financial tweets dataset"""
        logger.info("Creating sample financial tweets dataset...")
        
        # Since Twitter API requires authentication, we'll create a sample dataset
        # based on common patterns found in financial Twitter data
        
        sample_tweets = [
            {
                "text": "$AAPL breaking out! Strong buy signal 🚀",
                "ticker": "AAPL",
                "timestamp": "2024-01-15 10:30:00",
                "likes": 125,
                "retweets": 45,
                "replies": 12
            },
            {
                "text": "TSLA earnings miss expectations. Bearish outlook 📉",
                "ticker": "TSLA",
                "timestamp": "2024-01-16 14:22:00",
                "likes": 89,
                "retweets": 23,
                "replies": 67
            },
            {
                "text": "NVDA AI revolution continues. Long term bullish 💎",
                "ticker": "NVDA",
                "timestamp": "2024-01-17 09:15:00",
                "likes": 234,
                "retweets": 78,
                "replies": 34
            },
            # Add more realistic sample data...
        ]
        
        # Generate more sample data
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "AMD", "INTC", "QCOM"]
        bullish_phrases = [
            "to the moon", "strong buy", "bullish", "breakout", "rally",
            "pumping", "moon mission", "diamond hands", "hodl"
        ]
        bearish_phrases = [
            "crash incoming", "sell off", "bearish", "dump", "falling knife",
            "overvalued", "bubble", "short squeeze", "put options"
        ]
        
        import random
        from datetime import timedelta
        
        base_date = datetime(2024, 1, 1)
        
        for i in range(500):  # Generate 500 sample tweets
            ticker = random.choice(tickers)
            is_bullish = random.choice([True, False])
            
            if is_bullish:
                phrase = random.choice(bullish_phrases)
                sentiment_emoji = random.choice(["🚀", "📈", "💎", "🔥", "💰"])
            else:
                phrase = random.choice(bearish_phrases)
                sentiment_emoji = random.choice(["📉", "💸", "😭", "💀", "🔻"])
            
            text = f"${ticker} {phrase} {sentiment_emoji}"
            timestamp = base_date + timedelta(days=random.randint(0, 365), 
                                            hours=random.randint(0, 23), 
                                            minutes=random.randint(0, 59))
            
            sample_tweets.append({
                "text": text,
                "ticker": ticker,
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "likes": random.randint(1, 500),
                "retweets": random.randint(0, 100),
                "replies": random.randint(0, 50)
            })
        
        # Save sample dataset
        df = pd.DataFrame(sample_tweets)
        output_path = f"{self.data_dir}/sample_financial_tweets.csv"
        df.to_csv(output_path, index=False)
        
        logger.info(f"✅ Created sample financial tweets dataset: {output_path}")
        logger.info(f"📊 Generated {len(df)} sample tweets")
        return True
    
    def download_reddit_sample_data(self):
        """Create sample Reddit financial data"""
        logger.info("Creating sample Reddit financial discussions...")
        
        sample_posts = []
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "AMD", "INTC", "QCOM"]
        subreddits = ["wallstreetbets", "stocks", "investing", "SecurityAnalysis"]
        
        post_templates = [
            "DD: {ticker} is undervalued at current price",
            "{ticker} earnings play - what's your strategy?", 
            "Why {ticker} is my top pick for 2024",
            "{ticker} technical analysis - bullish breakout incoming",
            "Bag holding {ticker}, should I cut losses?",
            "{ticker} options play for next week",
            "Why I'm loading up on {ticker} shares",
            "{ticker} bear case - change my mind"
        ]
        
        import random
        from datetime import timedelta
        
        base_date = datetime(2024, 1, 1)
        
        for i in range(1000):  # Generate 1000 sample posts
            ticker = random.choice(tickers)
            subreddit = random.choice(subreddits)
            title = random.choice(post_templates).format(ticker=ticker)
            
            timestamp = base_date + timedelta(days=random.randint(0, 365),
                                            hours=random.randint(0, 23),
                                            minutes=random.randint(0, 59))
            
            sample_posts.append({
                "title": title,
                "subreddit": subreddit,
                "ticker": ticker,
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "upvotes": random.randint(1, 1000),
                "comments": random.randint(0, 200),
                "author": f"user_{random.randint(1000, 9999)}"
            })
        
        # Save sample dataset
        df = pd.DataFrame(sample_posts)
        output_path = f"{self.data_dir}/sample_reddit_financial.csv"
        df.to_csv(output_path, index=False)
        
        logger.info(f"✅ Created sample Reddit dataset: {output_path}")
        logger.info(f"📊 Generated {len(df)} sample posts")
        return True
    
    def download_huggingface_datasets(self):
        """Download financial datasets from Hugging Face"""
        try:
            from datasets import load_dataset
            logger.info("Downloading financial sentiment datasets from Hugging Face...")
            
            # Download Twitter financial news sentiment dataset
            try:
                dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
                
                # Convert to pandas and save
                train_df = pd.DataFrame(dataset['train'])
                
                output_path = f"{self.data_dir}/twitter_financial_news_sentiment.csv"
                train_df.to_csv(output_path, index=False)
                
                logger.info(f"✅ Downloaded Twitter financial news sentiment: {output_path}")
                logger.info(f"📊 {len(train_df)} records")
                
            except Exception as e:
                logger.warning(f"Could not download Twitter sentiment dataset: {e}")
            
            # Download another financial tweets dataset
            try:
                dataset = load_dataset("TimKoornstra/financial-tweets-sentiment")
                
                # Convert to pandas and save
                train_df = pd.DataFrame(dataset['train'])
                
                output_path = f"{self.data_dir}/financial_tweets_sentiment.csv"
                train_df.to_csv(output_path, index=False)
                
                logger.info(f"✅ Downloaded financial tweets sentiment: {output_path}")
                logger.info(f"📊 {len(train_df)} records")
                
            except Exception as e:
                logger.warning(f"Could not download financial tweets dataset: {e}")
                
            return True
            
        except ImportError:
            logger.warning("Hugging Face datasets library not installed. Skipping HF downloads.")
            logger.info("Install with: pip install datasets")
            return False
    
    def create_dataset_info(self):
        """Create information file about downloaded datasets"""
        info = {
            "download_date": datetime.now().isoformat(),
            "datasets": {
                "wallstreetbets_figshare": {
                    "source": "Figshare",
                    "period": "2020-2022",
                    "description": "Academic WallStreetBets dataset",
                    "url": "https://figshare.com/articles/dataset/Wallstreetbets_Reddit_Data_10_2020_-_04_2022_/22010699"
                },
                "sample_financial_tweets": {
                    "source": "Generated",
                    "description": "Sample financial Twitter data for testing",
                    "records": "500+"
                },
                "sample_reddit_financial": {
                    "source": "Generated", 
                    "description": "Sample Reddit financial discussions",
                    "records": "1000+"
                },
                "twitter_financial_news_sentiment": {
                    "source": "Hugging Face",
                    "description": "Twitter financial news with sentiment labels"
                },
                "financial_tweets_sentiment": {
                    "source": "Hugging Face",
                    "description": "Financial tweets with sentiment analysis"
                }
            },
            "usage_instructions": [
                "Load datasets with pandas: pd.read_csv('path/to/dataset.csv')",
                "Use with social_media_analyzer.py for sentiment analysis",
                "Integrate with training data using social_data_integration.py"
            ]
        }
        
        info_path = f"{self.data_dir}/dataset_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"📋 Created dataset info: {info_path}")

def main():
    """Main function to download social media datasets"""
    print("🌐 Social Media Dataset Downloader")
    print("="*50)
    
    downloader = SocialDatasetDownloader()
    
    print("📁 Creating data directory...")
    
    results = {}
    
    # Download Figshare WallStreetBets dataset
    print("\n🔍 Downloading WallStreetBets dataset from Figshare...")
    results['figshare'] = downloader.download_figshare_wallstreetbets()
    
    # Create sample datasets
    print("\n📝 Creating sample financial tweets...")
    results['sample_tweets'] = downloader.download_sample_financial_tweets()
    
    print("\n📝 Creating sample Reddit data...")
    results['sample_reddit'] = downloader.download_reddit_sample_data()
    
    # Download from Hugging Face
    print("\n🤗 Downloading from Hugging Face...")
    results['huggingface'] = downloader.download_huggingface_datasets()
    
    # Create info file
    print("\n📋 Creating dataset information...")
    downloader.create_dataset_info()
    
    # Summary
    print("\n" + "="*60)
    print("📊 DATASET DOWNLOAD SUMMARY")
    print("="*60)
    
    successful = sum(results.values())
    total = len(results)
    
    print(f"✅ Successfully downloaded: {successful}/{total} datasets")
    
    for dataset, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {dataset}: {status}")
    
    print(f"\n📁 All datasets saved to: {downloader.data_dir}")
    print(f"📋 Dataset info: {downloader.data_dir}/dataset_info.json")
    
    print(f"\n🚀 Next steps:")
    print(f"1. Explore datasets: ls -la {downloader.data_dir}")
    print(f"2. Analyze sentiment: python social_media_analyzer.py")
    print(f"3. Integrate with training: python social_data_integration.py")

if __name__ == "__main__":
    main()