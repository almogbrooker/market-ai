#!/usr/bin/env python3
"""
LLM Sentiment Features Implementation
Based on chat-g.txt requirements: FinBERT + XLM-RoBERTa for multi-language sentiment
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, AutoModel
)
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class LLMSentimentEngine:
    """
    Multi-language sentiment analysis using finance-tuned models
    Implements chat-g.txt specifications for LLM feature engineering
    """
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"🧠 Initializing LLM Sentiment Engine on {self.device}")
        self._load_models()
    
    def _load_models(self):
        """Load finance-tuned models"""
        
        try:
            # FinBERT for English financial sentiment
            logger.info("📈 Loading FinBERT for English sentiment...")
            self.models['finbert'] = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # XLM-RoBERTa for multi-language sentiment
            logger.info("🌐 Loading XLM-RoBERTa for multi-language sentiment...")
            self.models['xlm_sentiment'] = pipeline(
                "sentiment-analysis", 
                model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Finance embeddings for narrative analysis
            logger.info("💼 Loading finance embeddings...")
            self.models['fin_embeddings'] = AutoModel.from_pretrained(
                "microsoft/DialoGPT-medium"  # Placeholder - use FinE5 when available
            ).to(self.device)
            
            logger.info("✅ All LLM models loaded successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not load all models: {e}")
            logger.info("📋 Using fallback sentiment analysis")
            self._load_fallback_models()
    
    def _load_fallback_models(self):
        """Load simpler models if advanced ones fail"""
        try:
            # Basic sentiment model
            self.models['basic_sentiment'] = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("✅ Fallback sentiment model loaded")
        except:
            logger.warning("⚠️ Using rule-based fallback")
            self.models = {}
    
    def analyze_text_batch(self, texts: List[str], language: str = 'en') -> List[Dict]:
        """
        Analyze batch of texts for sentiment
        
        Returns:
            List of dicts with: sentiment, confidence, positive_score, negative_score
        """
        
        if not texts:
            return []
        
        results = []
        
        try:
            # Choose model based on language and availability
            if language == 'en' and 'finbert' in self.models:
                # Use FinBERT for English financial text
                finbert_results = self.models['finbert'](texts)
                
                for result in finbert_results:
                    results.append({
                        'sentiment': result['label'].lower(),
                        'confidence': result['score'],
                        'positive_score': result['score'] if result['label'].lower() == 'positive' else 1 - result['score'],
                        'negative_score': result['score'] if result['label'].lower() == 'negative' else 1 - result['score'],
                        'neutral_score': result['score'] if result['label'].lower() == 'neutral' else 0.5
                    })
            
            elif 'xlm_sentiment' in self.models:
                # Use XLM-RoBERTa for multi-language
                xlm_results = self.models['xlm_sentiment'](texts)
                
                for result in xlm_results:
                    results.append({
                        'sentiment': result['label'].lower(),
                        'confidence': result['score'],
                        'positive_score': result['score'] if 'positive' in result['label'].lower() else 1 - result['score'],
                        'negative_score': result['score'] if 'negative' in result['label'].lower() else 1 - result['score'],
                        'neutral_score': 0.5
                    })
            
            else:
                # Fallback to rule-based
                for text in texts:
                    sentiment_score = self._rule_based_sentiment(text)
                    results.append({
                        'sentiment': 'positive' if sentiment_score > 0.1 else ('negative' if sentiment_score < -0.1 else 'neutral'),
                        'confidence': min(abs(sentiment_score) + 0.5, 0.9),
                        'positive_score': max(0, sentiment_score + 0.5),
                        'negative_score': max(0, -sentiment_score + 0.5),
                        'neutral_score': 0.5
                    })
        
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            # Return neutral sentiment for all texts
            results = [{
                'sentiment': 'neutral',
                'confidence': 0.5,
                'positive_score': 0.5,
                'negative_score': 0.5,
                'neutral_score': 0.5
            }] * len(texts)
        
        return results
    
    def _rule_based_sentiment(self, text: str) -> float:
        """Simple rule-based sentiment as fallback"""
        
        positive_words = [
            'buy', 'bull', 'up', 'rise', 'gain', 'profit', 'growth', 'strong', 
            'beat', 'exceed', 'positive', 'good', 'great', 'excellent', 'upgrade'
        ]
        
        negative_words = [
            'sell', 'bear', 'down', 'fall', 'loss', 'decline', 'weak', 'miss',
            'below', 'negative', 'bad', 'poor', 'terrible', 'downgrade', 'risk'
        ]
        
        text_lower = text.lower()
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count + neg_count == 0:
            return 0.0
        
        return (pos_count - neg_count) / (pos_count + neg_count)
    
    def daily_llm_features(self, df_text: pd.DataFrame) -> pd.DataFrame:
        """
        Main function to create daily LLM features from text data
        
        Expected df_text columns: Date, Ticker, lang, text, source
        Returns: Date, Ticker, fb_pos, fb_neg, ml_pos, ml_neg, n_docs, etc.
        """
        
        logger.info(f"🔄 Processing {len(df_text)} text documents for LLM features")
        
        if df_text.empty:
            return pd.DataFrame()
        
        all_features = []
        
        # Group by Date and Ticker
        for (date, ticker), group in df_text.groupby(['Date', 'Ticker']):
            
            # Get all texts for this date/ticker
            texts = group['text'].tolist()
            languages = group['lang'].tolist()
            sources = group['source'].tolist()
            
            # Analyze sentiment for all texts
            english_texts = [text for text, lang in zip(texts, languages) if lang == 'en']
            other_texts = [text for text, lang in zip(texts, languages) if lang != 'en']
            
            english_results = self.analyze_text_batch(english_texts, 'en') if english_texts else []
            other_results = self.analyze_text_batch(other_texts, 'other') if other_texts else []
            
            all_results = english_results + other_results
            
            if not all_results:
                continue
            
            # Aggregate features
            features = {
                'Date': date,
                'Ticker': ticker,
                
                # FinBERT features (English)
                'fb_pos': np.mean([r['positive_score'] for r in english_results]) if english_results else 0.5,
                'fb_neg': np.mean([r['negative_score'] for r in english_results]) if english_results else 0.5,
                
                # Multi-language features
                'ml_pos': np.mean([r['positive_score'] for r in all_results]),
                'ml_neg': np.mean([r['negative_score'] for r in all_results]),
                
                # Document counts
                'n_docs': len(texts),
                'n_news': sum(1 for s in sources if s == 'news'),
                'n_social': sum(1 for s in sources if s == 'social'),
                'n_edgar': sum(1 for s in sources if s == 'edgar'),
                
                # Sentiment statistics
                'senti_mean': np.mean([r['positive_score'] - r['negative_score'] for r in all_results]),
                'senti_std': np.std([r['positive_score'] - r['negative_score'] for r in all_results]),
                'senti_z': 0.0,  # Will calculate after aggregation
                
                # News volume z-score (abnormal news detection)
                'news_n_z': 0.0,  # Will calculate after aggregation
                
                # Confidence and uncertainty
                'avg_confidence': np.mean([r['confidence'] for r in all_results]),
                'sentiment_uncertainty': 1 - np.mean([r['confidence'] for r in all_results]),
                
                # Source diversity
                'source_diversity': len(set(sources)) / len(sources) if sources else 0,
                
                # Language diversity  
                'lang_diversity': len(set(languages)) / len(languages) if languages else 0
            }
            
            all_features.append(features)
        
        if not all_features:
            return pd.DataFrame()
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Calculate cross-sectional z-scores
        features_df = self._add_cross_sectional_features(features_df)
        
        # Shift features by +1 trading day (no same-day trading)
        features_df['Date'] = pd.to_datetime(features_df['Date'])
        features_df = features_df.sort_values(['Ticker', 'Date'])
        
        for ticker in features_df['Ticker'].unique():
            ticker_mask = features_df['Ticker'] == ticker
            features_df.loc[ticker_mask, 'Date'] = features_df.loc[ticker_mask, 'Date'].shift(-1)
        
        # Remove last day (no future data)
        features_df = features_df.dropna(subset=['Date'])
        
        logger.info(f"✅ Generated LLM features: {features_df.shape}")
        return features_df
    
    def _add_cross_sectional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-sectional z-scores and relative measures"""
        
        # Calculate z-scores by date
        for date in df['Date'].unique():
            date_mask = df['Date'] == date
            date_data = df[date_mask]
            
            if len(date_data) > 1:
                # Sentiment z-score
                senti_mean = date_data['senti_mean'].mean()
                senti_std = date_data['senti_mean'].std()
                if senti_std > 0:
                    df.loc[date_mask, 'senti_z'] = (date_data['senti_mean'] - senti_mean) / senti_std
                
                # News volume z-score
                news_mean = date_data['n_news'].mean()
                news_std = date_data['n_news'].std()
                if news_std > 0:
                    df.loc[date_mask, 'news_n_z'] = (date_data['n_news'] - news_mean) / news_std
        
        return df

def create_sample_text_data() -> pd.DataFrame:
    """Create sample text data for testing"""
    
    dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    sample_texts = [
        "Apple reports strong earnings with record iPhone sales",
        "Microsoft cloud revenue beats expectations significantly", 
        "Google announces major AI breakthrough in search",
        "Market volatility increases amid economic uncertainty",
        "Tech stocks rally on positive earnings reports",
        "Federal Reserve signals potential rate cuts ahead"
    ]
    
    data = []
    for date in dates:
        for ticker in tickers:
            for i in range(np.random.randint(1, 4)):  # 1-3 texts per ticker/date
                data.append({
                    'Date': date,
                    'Ticker': ticker,
                    'lang': np.random.choice(['en', 'zh', 'fr'], p=[0.7, 0.2, 0.1]),
                    'text': np.random.choice(sample_texts),
                    'source': np.random.choice(['news', 'social', 'edgar'], p=[0.6, 0.3, 0.1])
                })
    
    return pd.DataFrame(data)

def main():
    """Test the LLM sentiment engine"""
    
    print("🧠 Testing LLM Sentiment Engine")
    print("=" * 50)
    
    # Initialize engine
    engine = LLMSentimentEngine()
    
    # Create sample data
    df_text = create_sample_text_data()
    print(f"📄 Sample text data: {df_text.shape}")
    print(df_text.head())
    
    # Generate LLM features
    llm_features = engine.daily_llm_features(df_text)
    
    print(f"\n🔍 LLM Features Generated: {llm_features.shape}")
    print(llm_features.head())
    
    print(f"\n📊 Feature Summary:")
    print(f"   Tickers: {llm_features['Ticker'].nunique()}")
    print(f"   Date range: {llm_features['Date'].min()} to {llm_features['Date'].max()}")
    print(f"   Features: {[col for col in llm_features.columns if col not in ['Date', 'Ticker']]}")
    
    # Save sample
    llm_features.to_csv('sample_llm_features.csv', index=False)
    print(f"\n💾 Sample features saved to: sample_llm_features.csv")

if __name__ == "__main__":
    main()