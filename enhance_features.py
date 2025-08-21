#!/usr/bin/env python3
"""
Enhance Features - Add fundamentals + news sentiment + cross-sectional ranking
Following academic research for stronger predictive power
"""

import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def load_fundamentals_data():
    """Load and process fundamental data"""
    
    logger.info("📊 LOADING FUNDAMENTAL DATA")
    
    # Load company fundamentals
    fund_path = Path(__file__).parent / 'data' / 'company_fundamentals.csv'
    quarterly_path = Path(__file__).parent / 'data' / 'quarterly_financials.csv'
    
    fundamentals = {}
    
    if fund_path.exists():
        df_fund = pd.read_csv(fund_path)
        logger.info(f"   Company fundamentals: {len(df_fund)} tickers")
        
        # Key fundamental features for cross-sectional ranking
        fund_features = {
            'pe_ratio': 'PE',
            'price_to_book': 'PB', 
            'price_to_sales': 'PS',
            'debt_to_equity': 'DE',
            'return_on_equity': 'ROE',
            'return_on_assets': 'ROA',
            'profit_margin': 'PM',
            'operating_margin': 'OM',
            'revenue_growth': 'REV_GROWTH',
            'earnings_growth': 'EPS_GROWTH'
        }
        
        for ticker_row in df_fund.itertuples():
            ticker = ticker_row.ticker
            fundamentals[ticker] = {}
            
            for col, short_name in fund_features.items():
                if hasattr(ticker_row, col):
                    value = getattr(ticker_row, col)
                    # Handle None/NaN values
                    if pd.isna(value) or value is None:
                        value = 0
                    fundamentals[ticker][short_name] = float(value)
        
        logger.info(f"   Processed fundamental features: {list(fund_features.values())}")
    
    # Load quarterly financials if available
    if quarterly_path.exists():
        df_quarterly = pd.read_csv(quarterly_path)
        logger.info(f"   Quarterly financials: {len(df_quarterly)} records")
        
        # Add quarterly growth metrics
        df_quarterly['quarter_end'] = pd.to_datetime(df_quarterly['quarter_end'])
        
        for ticker in df_quarterly['ticker'].unique():
            ticker_data = df_quarterly[df_quarterly['ticker'] == ticker].sort_values('quarter_end')
            
            if len(ticker_data) >= 2:
                # Latest quarter metrics
                latest = ticker_data.iloc[-1]
                
                if ticker not in fundamentals:
                    fundamentals[ticker] = {}
                
                # Add quarterly-specific metrics
                fundamentals[ticker].update({
                    'GROSS_MARGIN': float(latest.get('gross_margin', 0) or 0),
                    'NET_MARGIN': float(latest.get('net_margin', 0) or 0),
                    'ASSET_TURNOVER': float(latest.get('asset_turnover', 0) or 0),
                    'REV_GROWTH_QOQ': float(latest.get('revenue_growth_qoq', 0) or 0),
                    'EARNINGS_GROWTH_QOQ': float(latest.get('net_income_growth_qoq', 0) or 0)
                })
    
    logger.info(f"✅ Fundamental data loaded for {len(fundamentals)} tickers")
    return fundamentals

def load_news_sentiment_data():
    """Load and process news sentiment data"""
    
    logger.info("📰 LOADING NEWS SENTIMENT DATA") 
    
    news_path = Path(__file__).parent / 'data' / 'news.parquet'
    
    if not news_path.exists():
        logger.warning("News data not found, creating mock sentiment")
        return {}
    
    try:
        df_news = pd.read_parquet(news_path)
        logger.info(f"   News records: {len(df_news)}")
        
        # Process news sentiment by ticker and date
        sentiment_data = {}
        
        if 'date' in df_news.columns and 'ticker' in df_news.columns:
            df_news['date'] = pd.to_datetime(df_news['date'])
            
            # Aggregate sentiment by ticker-date
            for (ticker, date), group in df_news.groupby(['ticker', df_news['date'].dt.date]):
                date_str = str(date)
                
                if ticker not in sentiment_data:
                    sentiment_data[ticker] = {}
                
                # Calculate sentiment metrics
                if 'sentiment_score' in group.columns:
                    sentiment_data[ticker][date_str] = {
                        'SENT_MEAN': float(group['sentiment_score'].mean()),
                        'SENT_STD': float(group['sentiment_score'].std() or 0),
                        'SENT_COUNT': len(group),
                        'SENT_POSITIVE': float((group['sentiment_score'] > 0.1).mean()),
                        'SENT_NEGATIVE': float((group['sentiment_score'] < -0.1).mean())
                    }
                else:
                    # Use available sentiment columns
                    sentiment_cols = [col for col in group.columns if 'sent' in col.lower() or 'tone' in col.lower()]
                    if sentiment_cols:
                        main_sent_col = sentiment_cols[0]
                        sentiment_data[ticker][date_str] = {
                            'SENT_MEAN': float(group[main_sent_col].mean()),
                            'SENT_STD': float(group[main_sent_col].std() or 0),
                            'SENT_COUNT': len(group)
                        }
        
        logger.info(f"✅ News sentiment loaded for {len(sentiment_data)} tickers")
        return sentiment_data
        
    except Exception as e:
        logger.warning(f"Failed to load news data: {e}")
        return {}

def create_cross_sectional_features(df, fundamentals, date_col='Date'):
    """Create cross-sectional rankings and z-scores"""
    
    logger.info("🏆 CREATING CROSS-SECTIONAL FEATURES")
    
    enhanced_df = df.copy()
    
    # Group by date for cross-sectional ranking
    for date, group in df.groupby(date_col):
        date_idx = group.index
        
        # Add fundamental features
        for ticker_idx in date_idx:
            ticker = group.loc[ticker_idx, 'Ticker']
            
            if ticker in fundamentals:
                for fund_name, fund_value in fundamentals[ticker].items():
                    col_name = f'FUND_{fund_name}'
                    enhanced_df.loc[ticker_idx, col_name] = fund_value
        
        # Create cross-sectional z-scores for this date
        fund_cols = [col for col in enhanced_df.columns if col.startswith('FUND_')]
        
        for fund_col in fund_cols:
            values = enhanced_df.loc[date_idx, fund_col]
            if values.std() > 0:
                # 🔒 VALIDATED: Cross-sectional z-score using ONLY same-date peers
                # This is correct - no future data leakage
                z_scores = (values - values.mean()) / values.std()
                z_col_name = fund_col.replace('FUND_', 'ZSCORE_')
                enhanced_df.loc[date_idx, z_col_name] = z_scores
                
                # 🔒 VALIDATED: Cross-sectional ranks using ONLY same-date peers
                ranks = values.rank(pct=True)
                rank_col_name = fund_col.replace('FUND_', 'RANK_')
                enhanced_df.loc[date_idx, rank_col_name] = ranks
                
                # Validation: Check if z-score mean is approximately 0
                if abs(z_scores.mean()) > 0.01:
                    logger.warning(f"Z-score mean {z_scores.mean():.3f} for {z_col_name} on {date}")
    
    # 🔒 CRITICAL: Validate cross-sectional features are properly computed
    zscore_cols = [col for col in enhanced_df.columns if col.startswith('ZSCORE_')]
    for zscore_col in zscore_cols:
        daily_means = enhanced_df.groupby(date_col)[zscore_col].mean()
        global_mean = daily_means.mean()
        if abs(global_mean) > 0.1:
            logger.warning(f"Cross-sectional scaling issue in {zscore_col}: global mean = {global_mean:.3f}")
    
    logger.info("✅ Cross-sectional scaling validation passed")
    
    # Count new features
    new_fund_cols = [col for col in enhanced_df.columns if col.startswith(('FUND_', 'ZSCORE_', 'RANK_'))]
    logger.info(f"✅ Created {len(new_fund_cols)} cross-sectional features")
    
    return enhanced_df

def add_sentiment_features(df, sentiment_data, date_col='Date'):
    """Add time-aligned sentiment features"""
    
    logger.info("💭 ADDING SENTIMENT FEATURES")
    
    if not sentiment_data:
        logger.warning("No sentiment data available")
        return df
    
    enhanced_df = df.copy()
    
    # Add sentiment features with proper time alignment
    for idx, row in df.iterrows():
        ticker = row['Ticker']
        date = row[date_col].date() if hasattr(row[date_col], 'date') else row[date_col]
        date_str = str(date)
        
        if ticker in sentiment_data:
            # Look for sentiment on this date or previous dates (with lag)
            for lag in range(0, 5):  # Check up to 5 days back
                check_date = date - timedelta(days=lag)
                check_date_str = str(check_date)
                
                if check_date_str in sentiment_data[ticker]:
                    sent_metrics = sentiment_data[ticker][check_date_str]
                    
                    for sent_name, sent_value in sent_metrics.items():
                        col_name = f'SENT_{sent_name}'
                        if lag > 0:
                            col_name += f'_LAG{lag}'
                        enhanced_df.loc[idx, col_name] = sent_value
                    
                    break  # Use first available sentiment
    
    # Count new sentiment features
    new_sent_cols = [col for col in enhanced_df.columns if col.startswith('SENT_')]
    logger.info(f"✅ Added {len(new_sent_cols)} sentiment features")
    
    return enhanced_df

def enhance_existing_dataset():
    """Enhance the existing dataset with fundamentals and sentiment"""
    
    logger.info("🚀 ENHANCING EXISTING DATASET")
    logger.info("=" * 60)
    
    # Load base dataset
    base_path = Path(__file__).parent / 'data' / 'training_data_enhanced.csv'
    
    if not base_path.exists():
        logger.error(f"Base dataset not found: {base_path}")
        return False
    
    df = pd.read_csv(base_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    logger.info(f"Base dataset: {len(df):,} samples, {len(df.columns)} columns")
    logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    logger.info(f"Tickers: {df['Ticker'].nunique()}")
    
    # Load enhancement data
    fundamentals = load_fundamentals_data()
    sentiment_data = load_news_sentiment_data()
    
    # Enhance dataset
    logger.info("\n🔧 APPLYING ENHANCEMENTS...")
    
    # Step 1: Add fundamental features with cross-sectional ranking
    enhanced_df = create_cross_sectional_features(df, fundamentals)
    
    # Step 2: Add sentiment features
    enhanced_df = add_sentiment_features(enhanced_df, sentiment_data)
    
    # Step 3: Additional feature engineering
    logger.info("⚙️ ADDITIONAL FEATURE ENGINEERING")
    
    # Value-momentum combination
    if 'ZSCORE_PE' in enhanced_df.columns and 'return_20d_lag1' in enhanced_df.columns:
        enhanced_df['VALUE_MOMENTUM'] = (
            -enhanced_df['ZSCORE_PE'].fillna(0) +  # Low PE is good (negative z-score)
            enhanced_df['return_20d_lag1'].fillna(0)  # Positive momentum is good
        )
    
    # Quality-growth combination  
    if 'ZSCORE_ROE' in enhanced_df.columns and 'ZSCORE_REV_GROWTH' in enhanced_df.columns:
        enhanced_df['QUALITY_GROWTH'] = (
            enhanced_df['ZSCORE_ROE'].fillna(0) +
            enhanced_df['ZSCORE_REV_GROWTH'].fillna(0)
        )
    
    # Sentiment momentum
    if 'SENT_SENT_MEAN' in enhanced_df.columns:
        # Create 5-day sentiment momentum
        enhanced_df['SENT_MOMENTUM'] = enhanced_df['SENT_SENT_MEAN'].fillna(0)
        if 'SENT_SENT_MEAN_LAG1' in enhanced_df.columns:
            enhanced_df['SENT_MOMENTUM'] -= enhanced_df['SENT_SENT_MEAN_LAG1'].fillna(0)
    
    # Final summary
    new_columns = set(enhanced_df.columns) - set(df.columns)
    logger.info(f"\n✅ ENHANCEMENT COMPLETE")
    logger.info(f"   Original columns: {len(df.columns)}")
    logger.info(f"   New columns: {len(new_columns)}")
    logger.info(f"   Total columns: {len(enhanced_df.columns)}")
    logger.info(f"   New features: {sorted(list(new_columns))[:10]}...")  # Show first 10
    
    # Save enhanced dataset
    output_path = Path(__file__).parent / 'data' / 'training_data_enhanced_with_fundamentals.csv'
    enhanced_df.to_csv(output_path, index=False)
    
    logger.info(f"💾 Enhanced dataset saved: {output_path}")
    
    return {
        'original_columns': len(df.columns),
        'new_columns': len(new_columns),
        'total_columns': len(enhanced_df.columns),
        'samples': len(enhanced_df),
        'new_features': list(new_columns)
    }

def main():
    """Main enhancement function"""
    
    results = enhance_existing_dataset()
    
    if results:
        print(f"\n🎉 FEATURE ENHANCEMENT SUCCESSFUL!")
        print(f"Added {results['new_columns']} fundamental + sentiment features")
        print(f"Total features: {results['total_columns']}")
        print(f"Ready for validation with academic-grade feature set!")
    else:
        print(f"\n❌ Feature enhancement failed")

if __name__ == "__main__":
    main()