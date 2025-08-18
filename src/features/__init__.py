#!/usr/bin/env python3
"""
Features package - Feature engineering for enhanced market prediction
Organized into sentiment analysis and external data sources
"""

# Import from reorganized subdirectories
from .external_data.fred import FREDDataFetcher, create_fred_features
from .external_data.gdelt import GDELTDataFetcher, create_gdelt_features  
from .external_data.edgar import EDGARDataFetcher, create_edgar_features
from .external_data.reddit import RedditDataFetcher, create_reddit_features
from .sentiment.llm_sentiment import LLMSentimentEngine

__all__ = [
    'FREDDataFetcher', 'create_fred_features',
    'GDELTDataFetcher', 'create_gdelt_features',
    'EDGARDataFetcher', 'create_edgar_features', 
    'RedditDataFetcher', 'create_reddit_features',
    'LLMSentimentEngine'
]