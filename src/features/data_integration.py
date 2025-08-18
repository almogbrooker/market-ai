#!/usr/bin/env python3
"""
Unified data integration module for combining all external data sources
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from .fred import create_fred_features
from .gdelt import create_gdelt_features
from .edgar import create_edgar_features
from .reddit import create_reddit_features

logger = logging.getLogger(__name__)

class UnifiedDataPipeline:
    """Unified pipeline for integrating all external data sources"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: Configuration dict with API keys and settings
        """
        self.config = config or {}
        
        # API credentials
        self.fred_api_key = self.config.get('fred_api_key')
        self.reddit_client_id = self.config.get('reddit_client_id')
        self.reddit_client_secret = self.config.get('reddit_client_secret')
        
    def build_comprehensive_features(self, 
                                   start_date: str = "2020-01-01",
                                   end_date: Optional[str] = None,
                                   tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """Build comprehensive feature set from all data sources"""
        
        logger.info(f"Building comprehensive features from {start_date} to {end_date or 'present'}")
        
        all_features = []
        feature_sources = []
        
        # 1. FRED Economic Data
        try:
            logger.info("Fetching FRED economic indicators...")
            fred_features = create_fred_features(
                start_date=start_date,
                end_date=end_date,
                api_key=self.fred_api_key
            )
            
            if not fred_features.empty:
                all_features.append(fred_features)
                feature_sources.append("FRED")
                logger.info(f"FRED features: {fred_features.shape}")
            
        except Exception as e:
            logger.error(f"Failed to fetch FRED data: {e}")
        
        # 2. GDELT News Sentiment
        try:
            logger.info("Fetching GDELT news sentiment...")
            gdelt_features = create_gdelt_features(
                start_date=start_date,
                end_date=end_date,
                tickers=tickers
            )
            
            if not gdelt_features.empty:
                all_features.append(gdelt_features)
                feature_sources.append("GDELT")
                logger.info(f"GDELT features: {gdelt_features.shape}")
                
        except Exception as e:
            logger.error(f"Failed to fetch GDELT data: {e}")
        
        # 3. SEC EDGAR Filings
        if tickers:
            try:
                logger.info("Fetching SEC EDGAR filing sentiment...")
                edgar_features = create_edgar_features(
                    tickers=tickers,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not edgar_features.empty:
                    all_features.append(edgar_features)
                    feature_sources.append("EDGAR")
                    logger.info(f"EDGAR features: {edgar_features.shape}")
                    
            except Exception as e:
                logger.error(f"Failed to fetch EDGAR data: {e}")
        
        # 4. Reddit Social Sentiment
        try:
            logger.info("Fetching Reddit sentiment...")
            reddit_features = create_reddit_features(
                start_date=start_date,
                end_date=end_date,
                tickers=tickers,
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret
            )
            
            if not reddit_features.empty:
                all_features.append(reddit_features)
                feature_sources.append("Reddit")
                logger.info(f"Reddit features: {reddit_features.shape}")
                
        except Exception as e:
            logger.error(f"Failed to fetch Reddit data: {e}")
        
        # Combine all features
        if not all_features:
            logger.warning("No external features fetched successfully")
            # Return empty DataFrame with date index
            date_range = pd.date_range(
                start=start_date,
                end=end_date or datetime.now().strftime("%Y-%m-%d"),
                freq='D'
            )
            return pd.DataFrame(index=date_range)
        
        # Merge all feature sets
        logger.info(f"Combining features from sources: {feature_sources}")
        
        # Start with the first feature set
        combined_features = all_features[0]
        
        # Join remaining feature sets
        for features in all_features[1:]:
            combined_features = combined_features.join(features, how='outer')
        
        # Forward fill missing values
        combined_features = combined_features.fillna(method='ffill')
        
        # Fill any remaining NaNs with 0 (for beginning of series)
        combined_features = combined_features.fillna(0)
        
        logger.info(f"Final combined features shape: {combined_features.shape}")
        logger.info(f"Feature columns: {list(combined_features.columns)}")
        
        return combined_features
    
    def align_with_stock_data(self, 
                            features_df: pd.DataFrame, 
                            stock_df: pd.DataFrame) -> pd.DataFrame:
        """Align external features with stock market data"""
        
        # Get stock market trading dates
        stock_dates = pd.to_datetime(stock_df['Date'].unique())
        
        # Reindex features to stock market dates
        aligned_features = features_df.reindex(stock_dates, method='ffill')
        
        # Fill any remaining NaNs
        aligned_features = aligned_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        logger.info(f"Aligned features shape: {aligned_features.shape}")
        
        return aligned_features
    
    def create_feature_summary(self, features_df: pd.DataFrame) -> Dict:
        """Create summary statistics of features"""
        
        summary = {
            'total_features': len(features_df.columns),
            'date_range': {
                'start': features_df.index.min().strftime('%Y-%m-%d'),
                'end': features_df.index.max().strftime('%Y-%m-%d'),
                'days': len(features_df)
            },
            'feature_categories': {},
            'missing_data_pct': features_df.isnull().sum().sum() / (len(features_df) * len(features_df.columns)) * 100
        }
        
        # Categorize features by source
        for col in features_df.columns:
            if any(prefix in col.lower() for prefix in ['vix', 'treasury', 'fed', 'cpi', 'unemployment']):
                category = 'Economic (FRED)'
            elif any(prefix in col.lower() for prefix in ['tone', 'sentiment', 'news', 'gdelt']):
                category = 'News Sentiment (GDELT)'
            elif any(prefix in col.lower() for prefix in ['edgar', 'filing']):
                category = 'Corporate Filings (EDGAR)'
            elif any(prefix in col.lower() for prefix in ['reddit', 'bullish', 'bearish']):
                category = 'Social Media (Reddit)'
            else:
                category = 'Other'
            
            if category not in summary['feature_categories']:
                summary['feature_categories'][category] = 0
            summary['feature_categories'][category] += 1
        
        return summary

def build_enhanced_dataset(stock_data_path: str,
                         start_date: str = "2020-01-01", 
                         end_date: Optional[str] = None,
                         tickers: Optional[List[str]] = None,
                         config: Optional[Dict] = None,
                         output_path: Optional[str] = None) -> pd.DataFrame:
    """Build enhanced dataset with external features integrated"""
    
    logger.info("Building enhanced dataset with external data sources...")
    
    # Load stock data
    stock_df = pd.read_csv(stock_data_path)
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    
    if tickers is None and 'Ticker' in stock_df.columns:
        tickers = stock_df['Ticker'].unique().tolist()[:10]  # Limit to top 10 for performance
    
    # Build external features
    pipeline = UnifiedDataPipeline(config)
    external_features = pipeline.build_comprehensive_features(
        start_date=start_date,
        end_date=end_date, 
        tickers=tickers
    )
    
    # Align features with stock data
    if not external_features.empty:
        aligned_features = pipeline.align_with_stock_data(external_features, stock_df)
        
        # Merge with stock data
        stock_df_enhanced = stock_df.copy()
        
        # Add external features for each date
        feature_dict = aligned_features.to_dict('index')
        
        for col in aligned_features.columns:
            stock_df_enhanced[col] = stock_df_enhanced['Date'].map(
                lambda x: feature_dict.get(x, {}).get(col, 0)
            )
    else:
        stock_df_enhanced = stock_df.copy()
        logger.warning("No external features available - using stock data only")
    
    # Create feature summary
    if not external_features.empty:
        summary = pipeline.create_feature_summary(external_features)
        logger.info(f"Feature summary: {summary}")
    
    # Save enhanced dataset
    if output_path:
        stock_df_enhanced.to_csv(output_path, index=False)
        logger.info(f"Enhanced dataset saved to {output_path}")
    
    logger.info(f"Final enhanced dataset shape: {stock_df_enhanced.shape}")
    
    return stock_df_enhanced