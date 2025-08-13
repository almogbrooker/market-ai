#!/usr/bin/env python3
"""
Advanced social media sentiment analysis for financial markets
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SocialMediaAnalyzer:
    """Advanced sentiment analysis for social media financial discussions"""
    
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Financial-specific positive/negative words
        self.positive_finance_words = {
            'bullish', 'moon', 'rocket', 'pump', 'surge', 'breakout', 'rally',
            'strong', 'growth', 'profit', 'gains', 'outperform', 'beat',
            'upgrade', 'buy', 'long', 'hodl', 'diamond', 'hands'
        }
        
        self.negative_finance_words = {
            'bearish', 'crash', 'dump', 'drop', 'fall', 'decline', 'sell',
            'short', 'bear', 'recession', 'loss', 'miss', 'downgrade',
            'overvalued', 'bubble', 'panic', 'fear', 'weak'
        }
        
        # Emoji sentiment mapping
        self.emoji_sentiment = {
            '🚀': 0.8, '📈': 0.6, '💎': 0.5, '🔥': 0.4, '💰': 0.3,
            '📉': -0.6, '💸': -0.4, '😭': -0.5, '😱': -0.6, '💀': -0.7
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for sentiment analysis"""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags for cleaner sentiment (but keep the text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)  # Keep hashtag content
        
        # Replace multiple spaces/newlines
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_financial_keywords(self, text: str) -> Dict[str, int]:
        """Extract and count financial keywords"""
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.positive_finance_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_finance_words if word in text_lower)
        
        return {
            'positive_keywords': positive_count,
            'negative_keywords': negative_count,
            'keyword_ratio': positive_count - negative_count
        }
    
    def analyze_emoji_sentiment(self, text: str) -> float:
        """Analyze sentiment based on emojis"""
        emoji_scores = [self.emoji_sentiment.get(char, 0) for char in text]
        return np.mean([score for score in emoji_scores if score != 0]) if any(emoji_scores) else 0
    
    def textblob_sentiment(self, text: str) -> Dict[str, float]:
        """Get TextBlob sentiment analysis"""
        try:
            blob = TextBlob(text)
            return {
                'textblob_polarity': blob.sentiment.polarity,
                'textblob_subjectivity': blob.sentiment.subjectivity
            }
        except:
            return {'textblob_polarity': 0.0, 'textblob_subjectivity': 0.0}
    
    def vader_sentiment(self, text: str) -> Dict[str, float]:
        """Get VADER sentiment analysis"""
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            return {
                'vader_positive': scores['pos'],
                'vader_negative': scores['neg'],
                'vader_neutral': scores['neu'],
                'vader_compound': scores['compound']
            }
        except:
            return {
                'vader_positive': 0.0, 'vader_negative': 0.0,
                'vader_neutral': 1.0, 'vader_compound': 0.0
            }
    
    def financial_sentiment_score(self, text: str) -> float:
        """Calculate custom financial sentiment score"""
        clean_text = self.clean_text(text)
        
        # Get individual sentiment scores
        textblob_scores = self.textblob_sentiment(clean_text)
        vader_scores = self.vader_sentiment(clean_text)
        keyword_scores = self.extract_financial_keywords(clean_text)
        emoji_score = self.analyze_emoji_sentiment(text)
        
        # Weighted combination
        financial_sentiment = (
            textblob_scores['textblob_polarity'] * 0.3 +
            vader_scores['vader_compound'] * 0.4 +
            (keyword_scores['keyword_ratio'] * 0.1) +  # Normalize keyword ratio
            emoji_score * 0.2
        )
        
        # Clamp to [-1, 1]
        return max(-1, min(1, financial_sentiment))
    
    def analyze_social_media_data(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """Comprehensive sentiment analysis for social media data"""
        if df.empty:
            logger.warning("Empty dataframe provided")
            return df
        
        logger.info(f"Analyzing sentiment for {len(df)} social media posts...")
        
        # Clean text
        df['clean_text'] = df[text_column].apply(self.clean_text)
        
        # Apply all sentiment analyses
        sentiment_results = []
        for _, row in df.iterrows():
            text = row['clean_text']
            original_text = row[text_column]
            
            # Get all sentiment scores
            textblob_scores = self.textblob_sentiment(text)
            vader_scores = self.vader_sentiment(text)
            keyword_scores = self.extract_financial_keywords(text)
            emoji_score = self.analyze_emoji_sentiment(original_text)
            financial_score = self.financial_sentiment_score(original_text)
            
            result = {
                **textblob_scores,
                **vader_scores,
                **keyword_scores,
                'emoji_sentiment': emoji_score,
                'financial_sentiment': financial_score
            }
            
            sentiment_results.append(result)
        
        # Add sentiment columns to dataframe
        sentiment_df = pd.DataFrame(sentiment_results)
        result_df = pd.concat([df, sentiment_df], axis=1)
        
        logger.info("Sentiment analysis completed")
        return result_df
    
    def calculate_market_sentiment_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate market-wide sentiment indicators"""
        if df.empty:
            return {}
        
        # Volume-weighted sentiment (if engagement metrics available)
        if 'engagement_score' in df.columns:
            weighted_sentiment = np.average(df['financial_sentiment'], weights=df['engagement_score'])
        else:
            weighted_sentiment = df['financial_sentiment'].mean()
        
        # Sentiment distribution
        bullish_ratio = (df['financial_sentiment'] > 0.1).mean()
        bearish_ratio = (df['financial_sentiment'] < -0.1).mean()
        neutral_ratio = 1 - bullish_ratio - bearish_ratio
        
        # Sentiment volatility
        sentiment_volatility = df['financial_sentiment'].std()
        
        return {
            'overall_sentiment': weighted_sentiment,
            'bullish_ratio': bullish_ratio,
            'bearish_ratio': bearish_ratio,
            'neutral_ratio': neutral_ratio,
            'sentiment_volatility': sentiment_volatility,
            'total_posts': len(df)
        }
    
    def get_ticker_sentiment_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get detailed sentiment summary by ticker"""
        if df.empty or 'ticker' not in df.columns:
            return pd.DataFrame()
        
        summary_data = []
        
        for ticker in df['ticker'].unique():
            ticker_df = df[df['ticker'] == ticker]
            
            # Basic stats
            basic_stats = {
                'ticker': ticker,
                'post_count': len(ticker_df),
                'avg_financial_sentiment': ticker_df['financial_sentiment'].mean(),
                'sentiment_std': ticker_df['financial_sentiment'].std(),
                'median_sentiment': ticker_df['financial_sentiment'].median()
            }
            
            # Sentiment ratios
            sentiment_ratios = {
                'bullish_ratio': (ticker_df['financial_sentiment'] > 0.1).mean(),
                'bearish_ratio': (ticker_df['financial_sentiment'] < -0.1).mean(),
                'neutral_ratio': ((ticker_df['financial_sentiment'] >= -0.1) & 
                                (ticker_df['financial_sentiment'] <= 0.1)).mean()
            }
            
            # Engagement metrics (if available)
            engagement_metrics = {}
            if 'engagement_score' in ticker_df.columns:
                engagement_metrics = {
                    'total_engagement': ticker_df['engagement_score'].sum(),
                    'avg_engagement': ticker_df['engagement_score'].mean()
                }
            
            # Time-based metrics
            if 'timestamp' in ticker_df.columns:
                time_metrics = {
                    'first_mention': ticker_df['timestamp'].min(),
                    'last_mention': ticker_df['timestamp'].max(),
                    'mention_span_hours': (ticker_df['timestamp'].max() - 
                                         ticker_df['timestamp'].min()).total_seconds() / 3600
                }
            else:
                time_metrics = {}
            
            # Combine all metrics
            summary = {**basic_stats, **sentiment_ratios, **engagement_metrics, **time_metrics}
            summary_data.append(summary)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by post count
        summary_df = summary_df.sort_values('post_count', ascending=False)
        
        return summary_df
    
    def generate_sentiment_report(self, reddit_df: pd.DataFrame = None, 
                                twitter_df: pd.DataFrame = None) -> Dict:
        """Generate comprehensive sentiment report"""
        report = {
            'timestamp': datetime.now(),
            'reddit_analysis': {},
            'twitter_analysis': {},
            'combined_analysis': {}
        }
        
        # Analyze Reddit data
        if reddit_df is not None and not reddit_df.empty:
            reddit_analyzed = self.analyze_social_media_data(reddit_df)
            report['reddit_analysis'] = {
                'market_indicators': self.calculate_market_sentiment_indicators(reddit_analyzed),
                'ticker_summary': self.get_ticker_sentiment_summary(reddit_analyzed)
            }
        
        # Analyze Twitter data
        if twitter_df is not None and not twitter_df.empty:
            twitter_analyzed = self.analyze_social_media_data(twitter_df)
            report['twitter_analysis'] = {
                'market_indicators': self.calculate_market_sentiment_indicators(twitter_analyzed),
                'ticker_summary': self.get_ticker_sentiment_summary(twitter_analyzed)
            }
        
        # Combined analysis
        if (reddit_df is not None and not reddit_df.empty and 
            twitter_df is not None and not twitter_df.empty):
            
            # Combine datasets
            reddit_analyzed['source'] = 'reddit'
            twitter_analyzed['source'] = 'twitter'
            combined_df = pd.concat([reddit_analyzed, twitter_analyzed], ignore_index=True)
            
            report['combined_analysis'] = {
                'market_indicators': self.calculate_market_sentiment_indicators(combined_df),
                'ticker_summary': self.get_ticker_sentiment_summary(combined_df)
            }
        
        return report

def main():
    """Main function to demonstrate social media sentiment analysis"""
    analyzer = SocialMediaAnalyzer()
    
    # Try to load existing social media data
    reddit_df = None
    twitter_df = None
    
    if os.path.exists('data/reddit_financial.csv'):
        try:
            reddit_df = pd.read_csv('data/reddit_financial.csv')
            logger.info(f"Loaded {len(reddit_df)} Reddit posts")
        except Exception as e:
            logger.error(f"Error loading Reddit data: {e}")
    
    if os.path.exists('data/twitter_financial.csv'):
        try:
            twitter_df = pd.read_csv('data/twitter_financial.csv')
            logger.info(f"Loaded {len(twitter_df)} Twitter posts")
        except Exception as e:
            logger.error(f"Error loading Twitter data: {e}")
    
    if reddit_df is None and twitter_df is None:
        print("No social media data found. Please run:")
        print("- python reddit_fetcher.py")
        print("- python twitter_fetcher.py")
        return
    
    # Generate sentiment report
    print("\nGenerating comprehensive sentiment analysis...")
    report = analyzer.generate_sentiment_report(reddit_df, twitter_df)
    
    # Display results
    print("\n" + "="*60)
    print("SOCIAL MEDIA SENTIMENT ANALYSIS REPORT")
    print("="*60)
    
    if 'reddit_analysis' in report and report['reddit_analysis']:
        print(f"\n📱 REDDIT ANALYSIS")
        print("-" * 30)
        reddit_indicators = report['reddit_analysis']['market_indicators']
        print(f"Overall Sentiment: {reddit_indicators.get('overall_sentiment', 0):.3f}")
        print(f"Bullish Ratio: {reddit_indicators.get('bullish_ratio', 0):.1%}")
        print(f"Bearish Ratio: {reddit_indicators.get('bearish_ratio', 0):.1%}")
        print(f"Total Posts: {reddit_indicators.get('total_posts', 0)}")
    
    if 'twitter_analysis' in report and report['twitter_analysis']:
        print(f"\n🐦 TWITTER ANALYSIS")
        print("-" * 30)
        twitter_indicators = report['twitter_analysis']['market_indicators']
        print(f"Overall Sentiment: {twitter_indicators.get('overall_sentiment', 0):.3f}")
        print(f"Bullish Ratio: {twitter_indicators.get('bullish_ratio', 0):.1%}")
        print(f"Bearish Ratio: {twitter_indicators.get('bearish_ratio', 0):.1%}")
        print(f"Total Posts: {twitter_indicators.get('total_posts', 0)}")
    
    if 'combined_analysis' in report and report['combined_analysis']:
        print(f"\n🔄 COMBINED ANALYSIS")
        print("-" * 30)
        combined_indicators = report['combined_analysis']['market_indicators']
        print(f"Overall Sentiment: {combined_indicators.get('overall_sentiment', 0):.3f}")
        print(f"Bullish Ratio: {combined_indicators.get('bullish_ratio', 0):.1%}")
        print(f"Bearish Ratio: {combined_indicators.get('bearish_ratio', 0):.1%}")
        print(f"Total Posts: {combined_indicators.get('total_posts', 0)}")
        
        # Show top tickers by sentiment
        ticker_summary = report['combined_analysis']['ticker_summary']
        if not ticker_summary.empty:
            print(f"\n📊 TOP TICKERS BY MENTION COUNT:")
            top_tickers = ticker_summary.head(5)[['ticker', 'post_count', 'avg_financial_sentiment', 'bullish_ratio']]
            print(top_tickers.to_string(index=False, float_format='%.3f'))
    
    # Save enhanced data
    if reddit_df is not None:
        reddit_enhanced = analyzer.analyze_social_media_data(reddit_df)
        reddit_enhanced.to_csv('data/reddit_financial_enhanced.csv', index=False)
        print(f"\n✅ Enhanced Reddit data saved to data/reddit_financial_enhanced.csv")
    
    if twitter_df is not None:
        twitter_enhanced = analyzer.analyze_social_media_data(twitter_df)
        twitter_enhanced.to_csv('data/twitter_financial_enhanced.csv', index=False)
        print(f"✅ Enhanced Twitter data saved to data/twitter_financial_enhanced.csv")

if __name__ == "__main__":
    main()