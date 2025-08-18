#!/usr/bin/env python3
"""
Advanced Sentiment Features with Anomaly Detection
Implements abnormal news volume, sentiment shocks, and novelty features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class AdvancedSentimentAnalyzer:
    """Advanced sentiment feature engineering with anomaly detection"""
    
    def __init__(self, window_size: int = 30, zscore_threshold: float = 2.0):
        """
        Args:
            window_size: Rolling window for statistical calculations
            zscore_threshold: Threshold for anomaly detection
        """
        self.window_size = window_size
        self.zscore_threshold = zscore_threshold
        
    def calculate_abnormal_news_volume(self, df: pd.DataFrame, 
                                     news_col: str = 'news_volume',
                                     date_col: str = 'date') -> pd.DataFrame:
        """Calculate abnormal news volume using z-score analysis"""
        
        df = df.copy()
        
        # Calculate rolling statistics
        df['news_volume_ma'] = df[news_col].rolling(self.window_size).mean()
        df['news_volume_std'] = df[news_col].rolling(self.window_size).std()
        
        # Z-score for abnormal volume detection
        df['news_volume_zscore'] = (df[news_col] - df['news_volume_ma']) / (df['news_volume_std'] + 1e-6)
        
        # Binary abnormal volume indicator
        df['abnormal_news_volume'] = (np.abs(df['news_volume_zscore']) > self.zscore_threshold).astype(int)
        
        # Extreme volume events (3+ sigma)
        df['extreme_news_volume'] = (np.abs(df['news_volume_zscore']) > 3.0).astype(int)
        
        logger.info(f"Abnormal news volume events: {df['abnormal_news_volume'].sum()}")
        
        return df
    
    def calculate_sentiment_shock(self, df: pd.DataFrame,
                                sentiment_col: str = 'sentiment_tone',
                                date_col: str = 'date') -> pd.DataFrame:
        """Calculate sentiment shock (Δtone) momentum features"""
        
        df = df.copy()
        
        # Simple momentum (1-day change)
        df['sentiment_shock_1d'] = df[sentiment_col].diff(1)
        
        # Multi-period momentum
        df['sentiment_shock_3d'] = df[sentiment_col].diff(3)
        df['sentiment_shock_7d'] = df[sentiment_col].diff(7)
        
        # Volatility of sentiment changes
        df['sentiment_shock_vol'] = df['sentiment_shock_1d'].rolling(self.window_size).std()
        
        # Z-score of sentiment changes (unusual sentiment shifts)
        shock_ma = df['sentiment_shock_1d'].rolling(self.window_size).mean()
        shock_std = df['sentiment_shock_1d'].rolling(self.window_size).std()
        df['sentiment_shock_zscore'] = (df['sentiment_shock_1d'] - shock_ma) / (shock_std + 1e-6)
        
        # Binary indicators for large sentiment shifts
        df['positive_sentiment_shock'] = (df['sentiment_shock_zscore'] > self.zscore_threshold).astype(int)
        df['negative_sentiment_shock'] = (df['sentiment_shock_zscore'] < -self.zscore_threshold).astype(int)
        
        # Sentiment reversal detection
        df['sentiment_reversal'] = self._detect_sentiment_reversals(df[sentiment_col])
        
        logger.info(f"Sentiment shock events: {(df['positive_sentiment_shock'] | df['negative_sentiment_shock']).sum()}")
        
        return df
    
    def _detect_sentiment_reversals(self, sentiment_series: pd.Series, 
                                  min_magnitude: float = 0.1) -> pd.Series:
        """Detect sentiment reversals (sentiment direction changes)"""
        
        # Calculate sentiment direction (sign of change)
        sentiment_direction = np.sign(sentiment_series.diff())
        
        # Detect direction changes
        direction_changes = (sentiment_direction.diff() != 0) & (sentiment_direction != 0)
        
        # Only consider reversals with sufficient magnitude
        magnitude_filter = np.abs(sentiment_series.diff()) > min_magnitude
        
        return (direction_changes & magnitude_filter).astype(int)
    
    def calculate_novelty_features(self, df: pd.DataFrame,
                                 source_col: str = 'news_sources',
                                 date_col: str = 'date') -> pd.DataFrame:
        """Calculate novelty features based on unique sources and content"""
        
        df = df.copy()
        
        # If sources column doesn't exist, create dummy data
        if source_col not in df.columns:
            # Simulate diverse news sources
            np.random.seed(42)
            sources_list = ['reuters', 'bloomberg', 'wsj', 'cnbc', 'marketwatch', 'ft', 'yahoo', 'seeking_alpha']
            df[source_col] = [
                ','.join(np.random.choice(sources_list, size=np.random.randint(1, 4), replace=False))
                for _ in range(len(df))
            ]
        
        # Count unique sources per day
        df['unique_sources_count'] = df[source_col].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
        
        # Rolling average of unique sources
        df['unique_sources_ma'] = df['unique_sources_count'].rolling(self.window_size).mean()
        
        # Novelty score: unusual number of unique sources
        df['source_novelty_zscore'] = (
            (df['unique_sources_count'] - df['unique_sources_ma']) / 
            (df['unique_sources_count'].rolling(self.window_size).std() + 1e-6)
        )
        
        # High novelty indicator
        df['high_source_novelty'] = (df['source_novelty_zscore'] > self.zscore_threshold).astype(int)
        
        # Source diversity (entropy-based measure)
        df['source_diversity'] = df[source_col].apply(self._calculate_source_entropy)
        
        # Rolling average of source diversity
        df['source_diversity_ma'] = df['source_diversity'].rolling(self.window_size).mean()
        
        logger.info(f"High novelty events: {df['high_source_novelty'].sum()}")
        
        return df
    
    def _calculate_source_entropy(self, sources_str: str) -> float:
        """Calculate entropy of news sources distribution"""
        
        if pd.isna(sources_str) or not sources_str:
            return 0.0
        
        sources = str(sources_str).split(',')
        if len(sources) <= 1:
            return 0.0
        
        # Count occurrences
        from collections import Counter
        source_counts = Counter(sources)
        
        # Calculate entropy
        total = len(sources)
        entropy = 0.0
        for count in source_counts.values():
            prob = count / total
            entropy -= prob * np.log2(prob + 1e-6)
        
        return entropy
    
    def calculate_cross_sectional_anomalies(self, df: pd.DataFrame,
                                          ticker_col: str = 'ticker',
                                          date_col: str = 'date',
                                          features: List[str] = None) -> pd.DataFrame:
        """Calculate cross-sectional anomalies (relative to market)"""
        
        if features is None:
            features = ['news_volume', 'sentiment_tone', 'unique_sources_count']
        
        df = df.copy()
        
        # Group by date for cross-sectional analysis
        for feature in features:
            if feature in df.columns:
                # Calculate market average for each date
                market_avg = df.groupby(date_col)[feature].transform('mean')
                market_std = df.groupby(date_col)[feature].transform('std')
                
                # Relative to market z-score
                df[f'{feature}_relative_zscore'] = (df[feature] - market_avg) / (market_std + 1e-6)
                
                # Binary indicator for outliers
                df[f'{feature}_market_outlier'] = (
                    np.abs(df[f'{feature}_relative_zscore']) > self.zscore_threshold
                ).astype(int)
        
        return df
    
    def create_composite_sentiment_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite sentiment signals combining multiple indicators"""
        
        df = df.copy()
        
        # Sentiment stress index (combination of volume and sentiment anomalies)
        sentiment_components = []
        
        if 'abnormal_news_volume' in df.columns:
            sentiment_components.append(df['abnormal_news_volume'])
        
        if 'positive_sentiment_shock' in df.columns:
            sentiment_components.append(df['positive_sentiment_shock'])
        
        if 'negative_sentiment_shock' in df.columns:
            sentiment_components.append(df['negative_sentiment_shock'])
        
        if 'high_source_novelty' in df.columns:
            sentiment_components.append(df['high_source_novelty'])
        
        if sentiment_components:
            df['sentiment_stress_index'] = sum(sentiment_components) / len(sentiment_components)
        else:
            df['sentiment_stress_index'] = 0.0
        
        # Market attention score (volume + novelty + volatility)
        attention_components = []
        
        if 'news_volume_zscore' in df.columns:
            attention_components.append(np.clip(df['news_volume_zscore'], -3, 3) / 3)
        
        if 'source_novelty_zscore' in df.columns:
            attention_components.append(np.clip(df['source_novelty_zscore'], -3, 3) / 3)
        
        if 'sentiment_shock_zscore' in df.columns:
            attention_components.append(np.abs(np.clip(df['sentiment_shock_zscore'], -3, 3)) / 3)
        
        if attention_components:
            df['market_attention_score'] = sum(attention_components) / len(attention_components)
        else:
            df['market_attention_score'] = 0.0
        
        # Information flow quality (diversity + volume + sentiment consistency)
        if 'source_diversity' in df.columns and 'news_volume' in df.columns:
            normalized_diversity = (df['source_diversity'] - df['source_diversity'].min()) / (
                df['source_diversity'].max() - df['source_diversity'].min() + 1e-6
            )
            normalized_volume = (df['news_volume'] - df['news_volume'].min()) / (
                df['news_volume'].max() - df['news_volume'].min() + 1e-6
            )
            
            df['information_quality_score'] = (normalized_diversity + normalized_volume) / 2
        else:
            df['information_quality_score'] = 0.5
        
        return df
    
    def create_advanced_features(self, df: pd.DataFrame,
                               news_volume_col: str = 'news_volume',
                               sentiment_col: str = 'sentiment_tone',
                               sources_col: str = 'news_sources',
                               date_col: str = 'date',
                               ticker_col: str = 'ticker') -> pd.DataFrame:
        """Create all advanced sentiment features"""
        
        logger.info("Creating advanced sentiment features...")
        
        # Step 1: Abnormal news volume
        if news_volume_col in df.columns:
            df = self.calculate_abnormal_news_volume(df, news_volume_col, date_col)
        
        # Step 2: Sentiment shocks
        if sentiment_col in df.columns:
            df = self.calculate_sentiment_shock(df, sentiment_col, date_col)
        
        # Step 3: Novelty features
        df = self.calculate_novelty_features(df, sources_col, date_col)
        
        # Step 4: Cross-sectional anomalies (if ticker column exists)
        if ticker_col in df.columns:
            df = self.calculate_cross_sectional_anomalies(df, ticker_col, date_col)
        
        # Step 5: Composite signals
        df = self.create_composite_sentiment_signals(df)
        
        logger.info(f"Advanced sentiment features created: {df.shape[1]} total columns")
        
        return df

def enhance_sentiment_data(df: pd.DataFrame, 
                         config: Optional[Dict] = None) -> pd.DataFrame:
    """Convenience function to enhance sentiment data with advanced features"""
    
    config = config or {}
    
    analyzer = AdvancedSentimentAnalyzer(
        window_size=config.get('window_size', 30),
        zscore_threshold=config.get('zscore_threshold', 2.0)
    )
    
    enhanced_df = analyzer.create_advanced_features(df)
    
    return enhanced_df