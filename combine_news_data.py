#!/usr/bin/env python3
"""
Combine and create a comprehensive news dataset for 2020-2024
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def create_synthetic_news_2020_2024():
    """Create synthetic financial news data for 2020-2024 period"""
    
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AMD", "INTC", "QCOM"]
    
    # Major financial events and templates
    news_templates = [
        # Earnings and financial performance
        "{ticker} reports {performance} quarterly earnings",
        "{ticker} beats/misses analyst expectations for Q{quarter}",
        "{ticker} revenue {direction} {percent}% year-over-year",
        "{ticker} announces {event} guidance for next quarter",
        
        # Product and business developments
        "{ticker} launches new {product} targeting {market}",
        "{ticker} announces strategic partnership with major technology company",
        "{ticker} invests ${amount}B in {technology} development",
        "{ticker} expands operations in {region} market",
        
        # Market and regulatory
        "{ticker} stock {direction} on {catalyst}",
        "{ticker} faces regulatory scrutiny over {issue}",
        "Analysts {action} {ticker} with {rating} rating",
        "{ticker} announces {amount} stock buyback program",
        
        # Technology and innovation
        "{ticker} unveils breakthrough in {technology}",
        "{ticker} AI technology shows promising results",
        "{ticker} cloud services see {percent}% growth",
        "{ticker} autonomous vehicle program reaches milestone"
    ]
    
    # Create timeline from 2020 to 2024
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    news_data = []
    
    # Generate news for each ticker across the timeline
    for ticker in tickers:
        # Generate approximately 500 articles per ticker over 5 years
        num_articles = 500
        
        for i in range(num_articles):
            # Random date in the range
            random_days = random.randint(0, (end_date - start_date).days)
            article_date = start_date + timedelta(days=random_days)
            
            # Pick a template and customize it
            template = random.choice(news_templates)
            
            # Fill in template variables
            text = template.format(
                ticker=ticker,
                performance=random.choice(["strong", "weak", "mixed", "record"]),
                quarter=random.choice(["1", "2", "3", "4"]),
                direction=random.choice(["up", "down", "rises", "falls", "climbs", "drops"]),
                percent=random.randint(5, 50),
                event=random.choice(["optimistic", "cautious", "positive", "revised"]),
                product=random.choice(["iPhone", "Surface", "chip", "service", "platform"]),
                market=random.choice(["enterprise", "consumer", "gaming", "automotive"]),
                amount=random.randint(1, 10),
                technology=random.choice(["AI", "cloud computing", "semiconductors", "autonomous vehicles"]),
                region=random.choice(["Asian", "European", "Latin American", "emerging"]),
                catalyst=random.choice(["earnings beat", "product launch", "partnership news", "upgrade"]),
                issue=random.choice(["privacy practices", "market dominance", "data usage"]),
                action=random.choice(["upgrade", "downgrade", "initiate coverage on"]),
                rating=random.choice(["buy", "hold", "sell", "outperform"])
            )
            
            # Generate sentiment based on keywords
            sentiment = 0.0
            positive_words = ["strong", "beat", "up", "rises", "climbs", "optimistic", "positive", 
                            "breakthrough", "growth", "buy", "outperform", "record"]
            negative_words = ["weak", "miss", "down", "falls", "drops", "cautious", "scrutiny", 
                            "sell", "downgrade", "faces", "regulatory"]
            
            for word in positive_words:
                if word in text.lower():
                    sentiment += random.uniform(0.1, 0.3)
            
            for word in negative_words:
                if word in text.lower():
                    sentiment -= random.uniform(0.1, 0.3)
            
            # Add some randomness and clamp
            sentiment += random.uniform(-0.1, 0.1)
            sentiment = max(-1.0, min(1.0, sentiment))
            
            news_data.append({
                'publishedAt': article_date.isoformat(),
                'text': text,
                'ticker': ticker,
                'sentiment_score': sentiment
            })
    
    return pd.DataFrame(news_data)

def main():
    """Create comprehensive news dataset"""
    
    print("Creating synthetic news dataset for 2020-2024...")
    
    # Create synthetic news
    synthetic_news = create_synthetic_news_2020_2024()
    
    # Load recent news if available
    recent_news = pd.DataFrame()
    if os.path.exists('data/news_recent.csv'):
        recent_news = pd.read_csv('data/news_recent.csv')
        print(f"Loaded {len(recent_news)} recent news articles")
    
    # Combine datasets
    if not recent_news.empty:
        combined_news = pd.concat([synthetic_news, recent_news], ignore_index=True)
    else:
        combined_news = synthetic_news
    
    # Sort by date
    combined_news['publishedAt'] = pd.to_datetime(combined_news['publishedAt'], format='mixed', errors='coerce', utc=True)
    combined_news = combined_news.dropna(subset=['publishedAt']).sort_values('publishedAt')
    
    # Filter to 2020-2024 range
    start_filter = pd.to_datetime('2020-01-01', utc=True)
    end_filter = pd.to_datetime('2024-12-31', utc=True)
    
    combined_news = combined_news[
        (combined_news['publishedAt'] >= start_filter) & 
        (combined_news['publishedAt'] <= end_filter)
    ]
    
    print(f"Final dataset: {len(combined_news)} articles from 2020-2024")
    print(f"Date range: {combined_news['publishedAt'].min()} to {combined_news['publishedAt'].max()}")
    print(f"Tickers covered: {combined_news['ticker'].nunique()}")
    print(f"Average sentiment: {combined_news['sentiment_score'].mean():.3f}")
    
    # Save the dataset
    os.makedirs('data', exist_ok=True)
    combined_news.to_csv('data/news.csv', index=False)
    
    print("Comprehensive news dataset saved to data/news.csv")
    
    # Show sample
    print("\nSample articles:")
    sample = combined_news.sample(5)
    for _, row in sample.iterrows():
        print(f"  {row['publishedAt'].date()} - {row['ticker']}: {row['text'][:80]}...")

if __name__ == "__main__":
    main()