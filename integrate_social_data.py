#!/usr/bin/env python3
"""
Integrate social media data with training dataset
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def integrate_social_data():
    """Integrate social media data with training dataset"""
    
    print('🔄 Integrating 2020-2023 social media data with training dataset...')
    
    # Load the new social media data
    social_df = pd.read_csv('data/social_media_financial_2020_2023.csv')
    social_df['timestamp'] = pd.to_datetime(social_df['timestamp'])
    
    print(f'Social media data: {len(social_df)} records from {social_df["timestamp"].min().date()} to {social_df["timestamp"].max().date()}')
    
    # Load training data
    training_df = pd.read_csv('data/training_data.csv')
    training_df['Date'] = pd.to_datetime(training_df['Date'])
    
    print(f'Training data: {len(training_df)} records from {training_df["Date"].min().date()} to {training_df["Date"].max().date()}')
    
    # Aggregate social data by ticker and date
    daily_social = social_df.groupby(['ticker', 'date']).agg({
        'sentiment_score': ['mean', 'std', 'count'],
        'engagement_score': ['sum', 'mean']
    }).round(4)
    
    # Flatten column names
    daily_social.columns = [
        'social_sentiment_avg', 'social_sentiment_std', 'social_post_count',
        'social_engagement_total', 'social_engagement_avg'
    ]
    
    # Fill NaN std values
    daily_social['social_sentiment_std'] = daily_social['social_sentiment_std'].fillna(0.0)
    
    # Reset index
    daily_social = daily_social.reset_index()
    daily_social['date'] = pd.to_datetime(daily_social['date'])
    
    print(f'Daily social features: {len(daily_social)} ticker-date combinations')
    
    # Merge with training data
    training_df['date'] = training_df['Date'].dt.date
    daily_social['date'] = daily_social['date'].dt.date
    
    # Merge
    enhanced_df = pd.merge(
        training_df, 
        daily_social, 
        left_on=['Ticker', 'date'], 
        right_on=['ticker', 'date'], 
        how='left'
    )
    
    # Drop duplicate ticker column
    enhanced_df = enhanced_df.drop('ticker', axis=1)
    
    # Fill missing social features with neutral values
    social_cols = ['social_sentiment_avg', 'social_sentiment_std', 'social_post_count', 
                   'social_engagement_total', 'social_engagement_avg']
    
    for col in social_cols:
        if 'sentiment' in col and 'count' not in col and 'post' not in col:
            enhanced_df[col] = enhanced_df[col].fillna(0.0)  # Neutral sentiment
        else:
            enhanced_df[col] = enhanced_df[col].fillna(0)    # Zero for counts/engagement
    
    # Save enhanced dataset
    enhanced_df.to_csv('data/training_data_with_social.csv', index=False)
    
    print('✅ Enhanced training dataset created!')
    print(f'📊 Original features: {len(training_df.columns)}')
    print(f'🔗 Social features added: {len(social_cols)}')
    print(f'📈 Total features: {len(enhanced_df.columns)}')
    
    # Show coverage statistics
    coverage_stats = {}
    for col in social_cols:
        non_zero = (enhanced_df[col] != 0).sum()
        coverage_pct = non_zero / len(enhanced_df) * 100
        coverage_stats[col] = coverage_pct
    
    print('📈 Social Feature Coverage:')
    for col, coverage in coverage_stats.items():
        print(f'  {col}: {coverage:.1f}%')
    
    print('💾 Enhanced dataset saved to: data/training_data_with_social.csv')
    print('🚀 Ready to train with social features!')
    
    return enhanced_df

if __name__ == "__main__":
    integrate_social_data()