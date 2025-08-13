#!/usr/bin/env python3
"""
Integration of social media data into the training pipeline
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SocialDataIntegrator:
    """Integrate social media data into financial training dataset"""
    
    def __init__(self):
        self.social_features = [
            'daily_sentiment_avg', 'daily_sentiment_std', 'daily_post_count',
            'bullish_ratio', 'bearish_ratio', 'sentiment_momentum',
            'reddit_sentiment', 'twitter_sentiment', 'social_volume',
            'engagement_score', 'sentiment_volatility'
        ]
    
    def load_social_data(self) -> Dict[str, pd.DataFrame]:
        """Load all available social media data"""
        social_data = {}
        
        # Load Reddit data
        reddit_files = [
            'data/reddit_financial.csv',
            'data/reddit_financial_enhanced.csv'
        ]
        
        for file_path in reddit_files:
            if os.path.exists(file_path):
                try:
                    reddit_df = pd.read_csv(file_path)
                    social_data['reddit'] = reddit_df
                    logger.info(f"Loaded Reddit data: {len(reddit_df)} posts")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
        
        # Load Twitter data
        twitter_files = [
            'data/twitter_financial.csv',
            'data/twitter_financial_enhanced.csv'
        ]
        
        for file_path in twitter_files:
            if os.path.exists(file_path):
                try:
                    twitter_df = pd.read_csv(file_path)
                    social_data['twitter'] = twitter_df
                    logger.info(f"Loaded Twitter data: {len(twitter_df)} posts")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
        
        return social_data
    
    def preprocess_social_data(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Preprocess social media data for integration"""
        if df.empty:
            return df
        
        # Ensure timestamp column exists and is datetime
        timestamp_cols = ['timestamp', 'publishedAt', 'created_at', 'date']
        timestamp_col = None
        
        for col in timestamp_cols:
            if col in df.columns:
                timestamp_col = col
                break
        
        if timestamp_col is None:
            logger.warning(f"No timestamp column found in {source} data")
            return pd.DataFrame()
        
        # Convert to datetime
        try:
            df['timestamp'] = pd.to_datetime(df[timestamp_col])
        except Exception as e:
            logger.error(f"Failed to parse timestamps in {source} data: {e}")
            return pd.DataFrame()
        
        # Ensure required columns exist
        required_cols = ['ticker', 'timestamp']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing required columns in {source} data: {missing_cols}")
            return pd.DataFrame()
        
        # Use appropriate sentiment column
        sentiment_cols = ['financial_sentiment', 'sentiment_score', 'vader_compound']
        sentiment_col = None
        
        for col in sentiment_cols:
            if col in df.columns:
                sentiment_col = col
                break
        
        if sentiment_col is None:
            logger.warning(f"No sentiment column found in {source} data")
            df['sentiment'] = 0.0
        else:
            df['sentiment'] = df[sentiment_col].fillna(0.0)
        
        # Add engagement score if not present
        if 'engagement_score' not in df.columns:
            engagement_components = []
            if 'upvotes' in df.columns:
                engagement_components.append(df['upvotes'].fillna(0))
            if 'num_comments' in df.columns:
                engagement_components.append(df['num_comments'].fillna(0))
            if 'like_count' in df.columns:
                engagement_components.append(df['like_count'].fillna(0))
            if 'retweet_count' in df.columns:
                engagement_components.append(df['retweet_count'].fillna(0) * 2)  # Weight retweets more
            
            if engagement_components:
                df['engagement_score'] = sum(engagement_components)
            else:
                df['engagement_score'] = 1.0  # Default engagement
        
        # Extract date for daily aggregation
        df['date'] = df['timestamp'].dt.date
        
        return df[['ticker', 'date', 'timestamp', 'sentiment', 'engagement_score', 'text'] 
                 if 'text' in df.columns else ['ticker', 'date', 'timestamp', 'sentiment', 'engagement_score']]
    
    def aggregate_daily_social_features(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Aggregate social media data to daily features"""
        if df.empty:
            return pd.DataFrame()
        
        # Group by ticker and date
        daily_features = df.groupby(['ticker', 'date']).agg({
            'sentiment': ['mean', 'std', 'count'],
            'engagement_score': ['sum', 'mean'],
            'timestamp': ['min', 'max']
        }).round(4)
        
        # Flatten column names
        daily_features.columns = [
            f'{source}_sentiment_avg', f'{source}_sentiment_std', f'{source}_post_count',
            f'{source}_engagement_total', f'{source}_engagement_avg',
            f'{source}_first_post', f'{source}_last_post'
        ]
        
        # Calculate additional features
        daily_features[f'{source}_bullish_ratio'] = (df.groupby(['ticker', 'date'])['sentiment'] > 0.1).mean()
        daily_features[f'{source}_bearish_ratio'] = (df.groupby(['ticker', 'date'])['sentiment'] < -0.1).mean()
        daily_features[f'{source}_neutral_ratio'] = (
            (df.groupby(['ticker', 'date'])['sentiment'] >= -0.1) & 
            (df.groupby(['ticker', 'date'])['sentiment'] <= 0.1)
        ).mean()
        
        # Fill NaN standard deviations (when only one post per day)
        daily_features[f'{source}_sentiment_std'] = daily_features[f'{source}_sentiment_std'].fillna(0.0)
        
        # Reset index to make ticker and date regular columns
        daily_features = daily_features.reset_index()
        
        return daily_features
    
    def calculate_sentiment_momentum(self, df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
        """Calculate sentiment momentum features"""
        if df.empty or len(df) < window:
            return df
        
        # Sort by ticker and date
        df = df.sort_values(['ticker', 'date'])
        
        for source in ['reddit', 'twitter']:
            sentiment_col = f'{source}_sentiment_avg'
            if sentiment_col in df.columns:
                # Calculate rolling averages
                df[f'{source}_sentiment_ma{window}'] = (
                    df.groupby('ticker')[sentiment_col]
                    .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
                )
                
                # Calculate momentum (current vs moving average)
                df[f'{source}_sentiment_momentum'] = (
                    df[sentiment_col] - df[f'{source}_sentiment_ma{window}']
                )
                
                # Calculate sentiment change
                df[f'{source}_sentiment_change'] = (
                    df.groupby('ticker')[sentiment_col]
                    .transform(lambda x: x.diff().fillna(0))
                )
        
        return df
    
    def combine_social_features(self, reddit_daily: pd.DataFrame, 
                              twitter_daily: pd.DataFrame) -> pd.DataFrame:
        """Combine Reddit and Twitter daily features"""
        
        if reddit_daily.empty and twitter_daily.empty:
            return pd.DataFrame()
        elif reddit_daily.empty:
            combined = twitter_daily.copy()
        elif twitter_daily.empty:
            combined = reddit_daily.copy()
        else:
            # Merge on ticker and date
            combined = pd.merge(reddit_daily, twitter_daily, 
                              on=['ticker', 'date'], how='outer')
        
        # Calculate combined features
        sentiment_cols = []
        post_count_cols = []
        engagement_cols = []
        
        for source in ['reddit', 'twitter']:
            if f'{source}_sentiment_avg' in combined.columns:
                sentiment_cols.append(f'{source}_sentiment_avg')
            if f'{source}_post_count' in combined.columns:
                post_count_cols.append(f'{source}_post_count')
            if f'{source}_engagement_total' in combined.columns:
                engagement_cols.append(f'{source}_engagement_total')
        
        # Combined sentiment (weighted by post count)
        if len(sentiment_cols) > 1 and len(post_count_cols) > 1:
            weights = combined[post_count_cols].fillna(0)
            sentiments = combined[sentiment_cols].fillna(0)
            
            total_weights = weights.sum(axis=1)
            weighted_sentiment = (sentiments * weights).sum(axis=1) / total_weights.replace(0, 1)
            
            combined['combined_sentiment_avg'] = weighted_sentiment
        elif len(sentiment_cols) == 1:
            combined['combined_sentiment_avg'] = combined[sentiment_cols[0]].fillna(0)
        else:
            combined['combined_sentiment_avg'] = 0.0
        
        # Combined post count
        if post_count_cols:
            combined['combined_post_count'] = combined[post_count_cols].fillna(0).sum(axis=1)
        else:
            combined['combined_post_count'] = 0
        
        # Combined engagement
        if engagement_cols:
            combined['combined_engagement'] = combined[engagement_cols].fillna(0).sum(axis=1)
        else:
            combined['combined_engagement'] = 0
        
        # Fill missing values
        combined = combined.fillna(0)
        
        return combined
    
    def integrate_with_training_data(self, training_data_path: str = 'data/training_data.csv',
                                   output_path: str = 'data/training_data_with_social.csv') -> pd.DataFrame:
        """Integrate social media features with existing training data"""
        
        # Load existing training data
        try:
            training_df = pd.read_csv(training_data_path)
            logger.info(f"Loaded training data: {len(training_df)} rows")
        except Exception as e:
            logger.error(f"Failed to load training data from {training_data_path}: {e}")
            return pd.DataFrame()
        
        # Ensure Date column exists and is datetime
        if 'Date' not in training_df.columns:
            logger.error("Training data must have a 'Date' column")
            return pd.DataFrame()
        
        training_df['Date'] = pd.to_datetime(training_df['Date'])
        training_df['date'] = training_df['Date'].dt.date
        
        # Load and process social media data
        social_data = self.load_social_data()
        
        if not social_data:
            logger.warning("No social media data found. Returning original training data.")
            return training_df
        
        # Process each social media source
        reddit_daily = pd.DataFrame()
        twitter_daily = pd.DataFrame()
        
        if 'reddit' in social_data:
            reddit_processed = self.preprocess_social_data(social_data['reddit'], 'reddit')
            if not reddit_processed.empty:
                reddit_daily = self.aggregate_daily_social_features(reddit_processed, 'reddit')
        
        if 'twitter' in social_data:
            twitter_processed = self.preprocess_social_data(social_data['twitter'], 'twitter')
            if not twitter_processed.empty:
                twitter_daily = self.aggregate_daily_social_features(twitter_processed, 'twitter')
        
        # Combine social features
        social_features_df = self.combine_social_features(reddit_daily, twitter_daily)
        
        if social_features_df.empty:
            logger.warning("No social features generated. Returning original training data.")
            return training_df
        
        # Calculate momentum features
        social_features_df = self.calculate_sentiment_momentum(social_features_df)
        
        # Merge with training data
        enhanced_df = pd.merge(training_df, social_features_df, 
                             on=['Ticker', 'date'], how='left')
        
        # Fill missing social features with neutral values
        social_columns = [col for col in enhanced_df.columns 
                         if any(social_term in col.lower() 
                               for social_term in ['reddit', 'twitter', 'combined', 'sentiment', 'post', 'engagement'])]
        
        for col in social_columns:
            if 'sentiment' in col and 'count' not in col and 'post' not in col:
                enhanced_df[col] = enhanced_df[col].fillna(0.0)  # Neutral sentiment
            else:
                enhanced_df[col] = enhanced_df[col].fillna(0)    # Zero for counts/engagement
        
        # Save enhanced training data
        try:
            enhanced_df.to_csv(output_path, index=False)
            logger.info(f"Enhanced training data saved to {output_path}")
            logger.info(f"Added {len(social_columns)} social media features")
        except Exception as e:
            logger.error(f"Failed to save enhanced data: {e}")
        
        return enhanced_df
    
    def generate_social_features_summary(self, df: pd.DataFrame) -> Dict:
        """Generate summary of social media features"""
        if df.empty:
            return {}
        
        social_columns = [col for col in df.columns 
                         if any(social_term in col.lower() 
                               for social_term in ['reddit', 'twitter', 'combined', 'sentiment', 'post', 'engagement'])]
        
        summary = {
            'total_social_features': len(social_columns),
            'social_feature_names': social_columns,
            'date_range': {
                'start': df['Date'].min() if 'Date' in df.columns else None,
                'end': df['Date'].max() if 'Date' in df.columns else None
            },
            'coverage_stats': {}
        }
        
        # Calculate coverage statistics
        for col in social_columns:
            if col in df.columns:
                non_zero_count = (df[col] != 0).sum()
                coverage_pct = non_zero_count / len(df) * 100
                summary['coverage_stats'][col] = {
                    'non_zero_count': int(non_zero_count),
                    'coverage_percentage': round(coverage_pct, 2),
                    'mean_value': round(df[col].mean(), 4),
                    'std_value': round(df[col].std(), 4)
                }
        
        return summary

def main():
    """Main function to integrate social media data"""
    integrator = SocialDataIntegrator()
    
    print("Integrating social media data with training dataset...")
    
    # Integrate social data
    enhanced_df = integrator.integrate_with_training_data()
    
    if enhanced_df.empty:
        print("❌ Failed to integrate social media data")
        return
    
    # Generate summary
    summary = integrator.generate_social_features_summary(enhanced_df)
    
    print("\n" + "="*60)
    print("SOCIAL MEDIA INTEGRATION SUMMARY")
    print("="*60)
    
    print(f"✅ Successfully integrated social media data")
    print(f"📊 Original training data: {len(enhanced_df)} rows")
    print(f"🔗 Added social features: {summary.get('total_social_features', 0)}")
    
    if summary.get('date_range'):
        print(f"📅 Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    
    # Show top social features by coverage
    if summary.get('coverage_stats'):
        print(f"\n📈 Top Social Features by Coverage:")
        coverage_items = [(k, v['coverage_percentage']) for k, v in summary['coverage_stats'].items()]
        coverage_items.sort(key=lambda x: x[1], reverse=True)
        
        for feature, coverage in coverage_items[:10]:
            print(f"  {feature}: {coverage:.1f}% coverage")
    
    print(f"\n💾 Enhanced dataset saved to: data/training_data_with_social.csv")
    print(f"\n🚀 You can now train models with social media features:")
    print(f"   python train_advanced.py --experiment social_enhanced_models")

if __name__ == "__main__":
    main()