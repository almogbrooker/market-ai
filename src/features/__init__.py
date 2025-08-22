#!/usr/bin/env python3
"""
Features package - Feature engineering for enhanced market prediction
Organized into sentiment analysis and external data sources
"""

# Import from reorganized subdirectories
from .external_data.fred import FREDDataFetcher, create_fred_features
from .external_data.gdelt import GDELTDataFetcher, create_gdelt_features  
from .external_data.edgar import EDGARDataFetcher, create_edgar_features
# Reddit module removed - not needed for production
# LLM sentiment module removed - using simplified sentiment

__all__ = [
    'FREDDataFetcher', 'create_fred_features',
    'GDELTDataFetcher', 'create_gdelt_features',
    'EDGARDataFetcher', 'create_edgar_features'
]