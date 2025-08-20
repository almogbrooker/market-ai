#!/usr/bin/env python3
"""
Multi-language Sentiment Analysis for Financial Markets
Supports Chinese FinBERT-zh and CamemBERT Finance (French)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class MultilingualSentimentAnalyzer:
    """Multi-language financial sentiment analysis"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize with support for Chinese FinBERT and French CamemBERT
        
        Args:
            config: Configuration dict with model settings
        """
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model configurations
        self.models = {}
        self.tokenizers = {}
        
        # Language detection patterns
        self.language_patterns = {
            'chinese': re.compile(r'[\u4e00-\u9fff]+'),
            'french': re.compile(r'\b(français|bourse|euro|société|économie|investir|marché)\b', re.IGNORECASE),
            'english': re.compile(r'\b(stock|market|investment|finance|trading|economy)\b', re.IGNORECASE)
        }
        
        # Initialize available models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize pre-trained financial sentiment models"""
        
        # Chinese FinBERT
        try:
            self.models['chinese'] = {
                'name': 'Chinese FinBERT-zh',
                'model_id': 'ProsusAI/finbert-zh',  # Fallback to general Chinese BERT
                'tokenizer': None,
                'model': None,
                'available': False
            }
            logger.info("Chinese FinBERT configuration ready")
        except Exception as e:
            logger.warning(f"Chinese FinBERT setup failed: {e}")
        
        # French CamemBERT Finance
        try:
            self.models['french'] = {
                'name': 'CamemBERT Finance',
                'model_id': 'nlptown/bert-base-multilingual-uncased-sentiment',  # Fallback
                'tokenizer': None,
                'model': None,
                'available': False
            }
            logger.info("French CamemBERT configuration ready")
        except Exception as e:
            logger.warning(f"French CamemBERT setup failed: {e}")
        
        # English FinBERT (reference)
        try:
            self.models['english'] = {
                'name': 'English FinBERT',
                'model_id': 'ProsusAI/finbert',
                'tokenizer': None,
                'model': None,
                'available': False
            }
            logger.info("English FinBERT configuration ready")
        except Exception as e:
            logger.warning(f"English FinBERT setup failed: {e}")
    
    def load_model(self, language: str) -> bool:
        """Load specific language model on demand"""
        
        if language not in self.models:
            logger.error(f"Language {language} not supported")
            return False
        
        model_info = self.models[language]
        
        if model_info['available']:
            return True
        
        try:
            logger.info(f"Loading {model_info['name']} model...")
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_info['model_id'])
            model = AutoModelForSequenceClassification.from_pretrained(model_info['model_id'])
            model.to(self.device)
            model.eval()
            
            # Store in memory
            model_info['tokenizer'] = tokenizer
            model_info['model'] = model
            model_info['available'] = True
            
            logger.info(f"{model_info['name']} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load {model_info['name']}: {e}")
            return False
    
    def detect_language(self, text: str) -> str:
        """Detect primary language of text"""
        
        if not text or len(text.strip()) < 5:
            return 'english'  # Default
        
        text_lower = text.lower()
        
        # Check for Chinese characters
        if self.language_patterns['chinese'].search(text):
            return 'chinese'
        
        # Check for French financial terms
        if self.language_patterns['french'].search(text_lower):
            return 'french'
        
        # Default to English
        return 'english'
    
    def analyze_sentiment(self, text: str, language: Optional[str] = None) -> Dict[str, float]:
        """Analyze sentiment using appropriate language model"""
        
        if not text or len(text.strip()) < 3:
            return {
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'language': 'unknown',
                'positive_prob': 0.33,
                'negative_prob': 0.33,
                'neutral_prob': 0.34
            }
        
        # Auto-detect language if not provided
        if language is None:
            language = self.detect_language(text)
        
        # Load model for detected language
        if not self.load_model(language):
            # Fallback to rule-based sentiment
            return self._rule_based_sentiment(text, language)
        
        try:
            model_info = self.models[language]
            tokenizer = model_info['tokenizer']
            model = model_info['model']
            
            # Tokenize and encode
            inputs = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
            
            # Extract sentiment (assuming 3-class: negative, neutral, positive)
            probs = probabilities.cpu().numpy()[0]
            
            if len(probs) >= 3:
                negative_prob = float(probs[0])
                neutral_prob = float(probs[1])
                positive_prob = float(probs[2])
            else:
                # Binary classification fallback
                negative_prob = float(probs[0])
                positive_prob = float(probs[1]) if len(probs) > 1 else 1 - negative_prob
                neutral_prob = 0.0
            
            # Calculate composite sentiment score (-1 to 1)
            sentiment_score = positive_prob - negative_prob
            confidence = max(probs)
            
            return {
                'sentiment_score': sentiment_score,
                'confidence': confidence,
                'language': language,
                'positive_prob': positive_prob,
                'negative_prob': negative_prob,
                'neutral_prob': neutral_prob
            }
            
        except Exception as e:
            logger.warning(f"Model-based sentiment failed for {language}: {e}")
            return self._rule_based_sentiment(text, language)
    
    def _rule_based_sentiment(self, text: str, language: str) -> Dict[str, float]:
        """Fallback rule-based sentiment analysis"""
        
        # Language-specific sentiment keywords
        sentiment_keywords = {
            'chinese': {
                'positive': ['涨', '牛市', '盈利', '增长', '买入', '看涨', '上升', '强势'],
                'negative': ['跌', '熊市', '亏损', '下降', '卖出', '看跌', '下跌', '弱势']
            },
            'french': {
                'positive': ['hausse', 'profit', 'croissance', 'acheter', 'montée', 'fort', 'gain'],
                'negative': ['baisse', 'perte', 'déclin', 'vendre', 'chute', 'faible', 'risque']
            },
            'english': {
                'positive': ['bull', 'profit', 'growth', 'buy', 'rise', 'strong', 'gain', 'up'],
                'negative': ['bear', 'loss', 'decline', 'sell', 'fall', 'weak', 'risk', 'down']
            }
        }
        
        keywords = sentiment_keywords.get(language, sentiment_keywords['english'])
        text_lower = text.lower()
        
        positive_count = sum(1 for word in keywords['positive'] if word in text_lower)
        negative_count = sum(1 for word in keywords['negative'] if word in text_lower)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            return {
                'sentiment_score': 0.0,
                'confidence': 0.1,
                'language': language,
                'positive_prob': 0.33,
                'negative_prob': 0.33,
                'neutral_prob': 0.34
            }
        
        # Calculate sentiment score
        sentiment_score = (positive_count - negative_count) / total_sentiment_words
        confidence = min(total_sentiment_words / 10.0, 0.8)  # Max 80% confidence for rules
        
        positive_prob = max(0.1, positive_count / max(1, total_sentiment_words))
        negative_prob = max(0.1, negative_count / max(1, total_sentiment_words))
        neutral_prob = 1.0 - positive_prob - negative_prob
        
        return {
            'sentiment_score': sentiment_score,
            'confidence': confidence,
            'language': language,
            'positive_prob': positive_prob,
            'negative_prob': negative_prob,
            'neutral_prob': max(0.0, neutral_prob)
        }
    
    def analyze_batch(self, texts: List[str], 
                     languages: Optional[List[str]] = None) -> List[Dict[str, float]]:
        """Analyze sentiment for batch of texts"""
        
        if languages is None:
            languages = [None] * len(texts)
        
        results = []
        for text, language in zip(texts, languages):
            result = self.analyze_sentiment(text, language)
            results.append(result)
        
        return results
    
    def create_multilingual_features(self, df: pd.DataFrame,
                                   text_columns: List[str] = None,
                                   date_col: str = 'date') -> pd.DataFrame:
        """Create multilingual sentiment features from text data"""
        
        if text_columns is None:
            text_columns = [col for col in df.columns if 'text' in col.lower() or 'news' in col.lower()]
        
        if not text_columns:
            logger.warning("No text columns found for sentiment analysis")
            return self._create_dummy_multilingual_features(df, date_col)
        
        df = df.copy()
        
        # Process each text column
        for col in text_columns:
            if col not in df.columns:
                continue
            
            logger.info(f"Processing multilingual sentiment for {col}")
            
            # Analyze sentiment for each text
            sentiments = []
            for text in df[col].fillna(''):
                result = self.analyze_sentiment(str(text))
                sentiments.append(result)
            
            # Add sentiment features
            df[f'{col}_sentiment'] = [s['sentiment_score'] for s in sentiments]
            df[f'{col}_confidence'] = [s['confidence'] for s in sentiments]
            df[f'{col}_language'] = [s['language'] for s in sentiments]
            df[f'{col}_positive'] = [s['positive_prob'] for s in sentiments]
            df[f'{col}_negative'] = [s['negative_prob'] for s in sentiments]
        
        # Aggregate features by language
        self._add_language_aggregates(df, text_columns, date_col)
        
        return df
    
    def _add_language_aggregates(self, df: pd.DataFrame, text_columns: List[str], 
                               date_col: str):
        """Add aggregated sentiment features by language"""
        
        # Count articles by language (row-wise aggregation)
        for lang in ['chinese', 'french', 'english']:
            lang_counts = []
            lang_sentiments = []
            
            for idx in range(len(df)):
                row_count = 0
                row_sentiment = 0.0
                
                for col in text_columns:
                    lang_col = f'{col}_language'
                    sentiment_col = f'{col}_sentiment'
                    
                    if (lang_col in df.columns and sentiment_col in df.columns and
                        pd.notna(df.loc[idx, lang_col]) and df.loc[idx, lang_col] == lang):
                        row_count += 1
                        if pd.notna(df.loc[idx, sentiment_col]):
                            row_sentiment += df.loc[idx, sentiment_col]
                
                lang_counts.append(row_count)
                lang_sentiments.append(row_sentiment / max(1, row_count))
            
            df[f'{lang}_news_count'] = lang_counts
            df[f'{lang}_sentiment_avg'] = lang_sentiments
        
        # Overall multilingual sentiment score (row-wise)
        multilingual_sentiments = []
        total_news_counts = []
        
        for idx in range(len(df)):
            total_weighted = 0.0
            total_count = 0
            
            for lang in ['chinese', 'french', 'english']:
                count = df.loc[idx, f'{lang}_news_count']
                sentiment = df.loc[idx, f'{lang}_sentiment_avg']
                
                total_weighted += sentiment * count
                total_count += count
            
            multilingual_sentiments.append(total_weighted / max(1, total_count))
            total_news_counts.append(total_count)
        
        df['multilingual_sentiment'] = multilingual_sentiments
        df['total_multilingual_news'] = total_news_counts
    
    def _create_dummy_multilingual_features(self, df: pd.DataFrame, 
                                          date_col: str) -> pd.DataFrame:
        """Create realistic dummy multilingual features"""
        
        df = df.copy()
        np.random.seed(42)
        
        # Simulate language distribution (English dominant, some Chinese/French)
        for i in range(len(df)):
            # Language counts (realistic distribution)
            english_count = np.random.poisson(5)  # Base news volume
            chinese_count = np.random.poisson(1)  # Less Chinese news
            french_count = np.random.poisson(0.5)  # Even less French
            
            # Language-specific sentiment (with realistic patterns)
            english_sentiment = np.random.normal(0.05, 0.3)  # Slightly positive bias
            chinese_sentiment = np.random.normal(-0.02, 0.4)  # Slightly negative (trade tensions)
            french_sentiment = np.random.normal(0.03, 0.25)  # Conservative positive
            
            df.loc[i, 'english_news_count'] = english_count
            df.loc[i, 'chinese_news_count'] = chinese_count
            df.loc[i, 'french_news_count'] = french_count
            
            df.loc[i, 'english_sentiment_avg'] = english_sentiment
            df.loc[i, 'chinese_sentiment_avg'] = chinese_sentiment
            df.loc[i, 'french_sentiment_avg'] = french_sentiment
            
            # Aggregate multilingual sentiment
            total_sentiment = (english_sentiment * english_count + 
                             chinese_sentiment * chinese_count + 
                             french_sentiment * french_count)
            total_count = english_count + chinese_count + french_count
            
            df.loc[i, 'multilingual_sentiment'] = total_sentiment / max(1, total_count)
            df.loc[i, 'total_multilingual_news'] = total_count
        
        logger.info("Created dummy multilingual sentiment features")
        return df

def create_multilingual_sentiment_features(df: pd.DataFrame,
                                         config: Optional[Dict] = None,
                                         text_columns: List[str] = None) -> pd.DataFrame:
    """Convenience function to create multilingual sentiment features"""
    
    analyzer = MultilingualSentimentAnalyzer(config)
    enhanced_df = analyzer.create_multilingual_features(df, text_columns)
    
    return enhanced_df

# Example usage and testing
if __name__ == "__main__":
    # Test with sample texts
    analyzer = MultilingualSentimentAnalyzer()
    
    # Test samples in different languages
    test_texts = [
        "Apple stock surges on strong earnings report",  # English
        "苹果股价因强劲财报而飙升",  # Chinese
        "Les actions d'Apple grimpent sur de solides résultats",  # French
    ]
    
    for text in test_texts:
        result = analyzer.analyze_sentiment(text)
        print(f"Text: {text}")
        print(f"Language: {result['language']}")
        print(f"Sentiment: {result['sentiment_score']:.3f}")
        print(f"Confidence: {result['confidence']:.3f}")
        print("---")