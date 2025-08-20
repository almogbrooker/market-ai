#!/usr/bin/env python3
"""
LLM-based News Summarization and Analysis
Advanced news processing using Large Language Models for financial markets
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
import requests
import json
from datetime import datetime
import re
import time

logger = logging.getLogger(__name__)

class LLMNewsSummarizer:
    """LLM-powered news summarization and sentiment analysis"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize LLM news summarizer
        
        Args:
            config: Configuration dict with API keys and model settings
        """
        self.config = config or {}
        
        # LLM API configurations
        self.openai_api_key = self.config.get('openai_api_key')
        self.anthropic_api_key = self.config.get('anthropic_api_key')
        self.huggingface_api_key = self.config.get('huggingface_api_key')
        
        # Financial analysis prompts
        self.prompts = {
            'summary': """
Summarize this financial news article in 2-3 sentences, focusing on:
1. Key financial metrics or events
2. Market implications
3. Impact on stock price/company value

Article: {text}

Summary:""",
            
            'sentiment': """
Analyze the sentiment of this financial news for trading decisions.

Rate from -1 (very negative) to +1 (very positive) and provide confidence (0-1).

Article: {text}

Provide JSON response:
{{
    "sentiment_score": <-1 to 1>,
    "confidence": <0 to 1>,
    "reasoning": "<brief explanation>",
    "key_factors": ["<factor1>", "<factor2>"]
}}""",
            
            'key_extraction': """
Extract key financial information from this news article:

Article: {text}

Extract JSON:
{{
    "companies_mentioned": ["<ticker1>", "<ticker2>"],
    "financial_metrics": {{"revenue": "X", "earnings": "Y"}},
    "events": ["<event1>", "<event2>"],
    "market_impact": "<high/medium/low>",
    "time_horizon": "<immediate/short-term/long-term>"
}}""",
            
            'risk_assessment': """
Assess the market risk implications of this news:

Article: {text}

Provide risk assessment JSON:
{{
    "risk_level": "<low/medium/high>",
    "risk_factors": ["<factor1>", "<factor2>"],
    "affected_sectors": ["<sector1>", "<sector2>"],
    "volatility_impact": "<low/medium/high>",
    "recommendations": "<brief recommendation>"
}}"""
        }
    
    def _call_openai_api(self, prompt: str, max_tokens: int = 500) -> str:
        """Call OpenAI API for text generation"""
        
        if not self.openai_api_key:
            return self._fallback_analysis(prompt)
        
        try:
            headers = {
                'Authorization': f'Bearer {self.openai_api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': 'gpt-3.5-turbo',
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': max_tokens,
                'temperature': 0.3
            }
            
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                logger.warning(f"OpenAI API error: {response.status_code}")
                return self._fallback_analysis(prompt)
                
        except Exception as e:
            logger.warning(f"OpenAI API call failed: {e}")
            return self._fallback_analysis(prompt)
    
    def _call_huggingface_api(self, text: str, task: str = "summarization") -> str:
        """Call Hugging Face API for NLP tasks"""
        
        if not self.huggingface_api_key:
            return self._fallback_analysis(text)
        
        try:
            headers = {
                'Authorization': f'Bearer {self.huggingface_api_key}',
                'Content-Type': 'application/json'
            }
            
            # Use different models for different tasks
            models = {
                'summarization': 'facebook/bart-large-cnn',
                'sentiment': 'ProsusAI/finbert',
                'classification': 'microsoft/DialoGPT-medium'
            }
            
            model = models.get(task, 'facebook/bart-large-cnn')
            url = f'https://api-inference.huggingface.co/models/{model}'
            
            data = {'inputs': text}
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                if task == 'summarization' and isinstance(result, list):
                    return result[0].get('summary_text', '')
                elif task == 'sentiment' and isinstance(result, list):
                    return json.dumps(result[0])
                    
            logger.warning(f"Hugging Face API error: {response.status_code}")
            return self._fallback_analysis(text)
            
        except Exception as e:
            logger.warning(f"Hugging Face API call failed: {e}")
            return self._fallback_analysis(text)
    
    def _fallback_analysis(self, text: str) -> str:
        """Fallback rule-based analysis when APIs fail"""
        
        if len(text) < 100:
            return text
        
        # Simple extractive summarization
        sentences = text.split('. ')
        if len(sentences) <= 3:
            return text
        
        # Score sentences based on financial keywords
        financial_keywords = [
            'revenue', 'profit', 'earnings', 'sales', 'growth', 'market', 'stock',
            'investment', 'quarter', 'billion', 'million', 'percent', 'increase',
            'decrease', 'analyst', 'forecast', 'guidance', 'outlook'
        ]
        
        sentence_scores = []
        for sentence in sentences:
            score = sum(1 for keyword in financial_keywords if keyword.lower() in sentence.lower())
            sentence_scores.append((sentence, score))
        
        # Select top 2-3 sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in sentence_scores[:3]]
        
        return '. '.join(top_sentences) + '.'
    
    def summarize_article(self, text: str, method: str = 'auto') -> str:
        """Summarize a news article"""
        
        if not text or len(text.strip()) < 50:
            return text
        
        text = text.strip()[:4000]  # Limit length for API calls
        
        if method == 'openai' or (method == 'auto' and self.openai_api_key):
            prompt = self.prompts['summary'].format(text=text)
            return self._call_openai_api(prompt)
        
        elif method == 'huggingface' or (method == 'auto' and self.huggingface_api_key):
            return self._call_huggingface_api(text, 'summarization')
        
        else:
            return self._fallback_analysis(text)
    
    def analyze_sentiment_llm(self, text: str, method: str = 'auto') -> Dict:
        """Analyze sentiment using LLM"""
        
        if not text or len(text.strip()) < 20:
            return {
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'reasoning': 'Insufficient text',
                'key_factors': []
            }
        
        text = text.strip()[:3000]
        
        if method == 'openai' or (method == 'auto' and self.openai_api_key):
            prompt = self.prompts['sentiment'].format(text=text)
            response = self._call_openai_api(prompt)
            
            try:
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except:
                pass
        
        elif method == 'huggingface' or (method == 'auto' and self.huggingface_api_key):
            response = self._call_huggingface_api(text, 'sentiment')
            try:
                result = json.loads(response)
                if isinstance(result, dict) and 'score' in result:
                    # Convert FinBERT output to standard format
                    label = result.get('label', 'neutral').lower()
                    score = result.get('score', 0.5)
                    
                    sentiment_map = {'positive': 1, 'negative': -1, 'neutral': 0}
                    sentiment_score = sentiment_map.get(label, 0) * score
                    
                    return {
                        'sentiment_score': sentiment_score,
                        'confidence': score,
                        'reasoning': f'FinBERT classified as {label}',
                        'key_factors': [label]
                    }
            except:
                pass
        
        # Fallback to rule-based sentiment
        return self._fallback_sentiment(text)
    
    def _fallback_sentiment(self, text: str) -> Dict:
        """Rule-based sentiment analysis fallback"""
        
        positive_words = [
            'profit', 'growth', 'increase', 'rise', 'gain', 'strong', 'beat',
            'exceed', 'bullish', 'optimistic', 'upgrade', 'buy', 'outperform'
        ]
        
        negative_words = [
            'loss', 'decline', 'decrease', 'fall', 'weak', 'miss', 'disappoint',
            'bearish', 'pessimistic', 'downgrade', 'sell', 'underperform'
        ]
        
        text_lower = text.lower()
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total_sentiment = pos_count + neg_count
        
        if total_sentiment == 0:
            return {
                'sentiment_score': 0.0,
                'confidence': 0.1,
                'reasoning': 'No clear sentiment indicators',
                'key_factors': []
            }
        
        sentiment_score = (pos_count - neg_count) / total_sentiment
        confidence = min(total_sentiment / 10, 0.8)
        
        key_factors = []
        if pos_count > 0:
            key_factors.extend([w for w in positive_words if w in text_lower][:3])
        if neg_count > 0:
            key_factors.extend([w for w in negative_words if w in text_lower][:3])
        
        return {
            'sentiment_score': sentiment_score,
            'confidence': confidence,
            'reasoning': f'Rule-based: {pos_count} positive, {neg_count} negative words',
            'key_factors': key_factors
        }
    
    def extract_key_information(self, text: str) -> Dict:
        """Extract key financial information using LLM"""
        
        if not text:
            return {
                'companies_mentioned': [],
                'financial_metrics': {},
                'events': [],
                'market_impact': 'low',
                'time_horizon': 'short-term'
            }
        
        text = text.strip()[:3000]
        
        if self.openai_api_key:
            prompt = self.prompts['key_extraction'].format(text=text)
            response = self._call_openai_api(prompt)
            
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except:
                pass
        
        # Fallback extraction
        return self._fallback_extraction(text)
    
    def _fallback_extraction(self, text: str) -> Dict:
        """Rule-based information extraction"""
        
        # Extract ticker symbols
        ticker_pattern = r'\b[A-Z]{1,5}\b'
        potential_tickers = re.findall(ticker_pattern, text)
        
        # Filter common false positives
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'NEW', 'CEO', 'CFO'}
        tickers = [t for t in potential_tickers if t not in common_words and len(t) <= 4]
        
        # Extract financial metrics
        metrics = {}
        
        # Revenue patterns
        revenue_patterns = [
            r'revenue of \$?([\d,\.]+)\s?(million|billion)',
            r'\$?([\d,\.]+)\s?(million|billion) in revenue'
        ]
        
        for pattern in revenue_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount = match.group(1).replace(',', '')
                unit = match.group(2).lower()
                multiplier = 1000000 if unit == 'million' else 1000000000
                metrics['revenue'] = float(amount) * multiplier
                break
        
        # Simple event extraction
        events = []
        event_keywords = ['merger', 'acquisition', 'ipo', 'earnings', 'partnership', 'lawsuit']
        for keyword in event_keywords:
            if keyword in text.lower():
                events.append(keyword)
        
        # Market impact assessment
        impact_indicators = {
            'high': ['billion', 'merger', 'acquisition', 'bankruptcy'],
            'medium': ['million', 'earnings', 'partnership'],
            'low': ['minor', 'small', 'routine']
        }
        
        market_impact = 'low'
        for impact, indicators in impact_indicators.items():
            if any(indicator in text.lower() for indicator in indicators):
                market_impact = impact
                break
        
        return {
            'companies_mentioned': tickers[:5],  # Limit to 5
            'financial_metrics': metrics,
            'events': events,
            'market_impact': market_impact,
            'time_horizon': 'short-term'  # Default
        }
    
    def process_news_batch(self, news_df: pd.DataFrame,
                          text_column: str = 'text',
                          title_column: str = 'title') -> pd.DataFrame:
        """Process a batch of news articles with LLM analysis"""
        
        logger.info(f"Processing {len(news_df)} news articles with LLM")
        
        enhanced_df = news_df.copy()
        
        if text_column not in enhanced_df.columns:
            logger.warning(f"Text column '{text_column}' not found")
            return enhanced_df
        
        # Process each article
        summaries = []
        sentiments = []
        key_info = []
        
        for idx, row in enhanced_df.iterrows():
            text = str(row.get(text_column, ''))
            title = str(row.get(title_column, ''))
            
            # Combine title and text
            full_text = f"{title}. {text}" if title and title != 'nan' else text
            
            # Summarization
            summary = self.summarize_article(full_text)
            summaries.append(summary)
            
            # Sentiment analysis
            sentiment = self.analyze_sentiment_llm(full_text)
            sentiments.append(sentiment)
            
            # Key information extraction
            key_data = self.extract_key_information(full_text)
            key_info.append(key_data)
            
            # Progress logging
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(enhanced_df)} articles")
        
        # Add results to dataframe
        enhanced_df['llm_summary'] = summaries
        enhanced_df['llm_sentiment_score'] = [s['sentiment_score'] for s in sentiments]
        enhanced_df['llm_confidence'] = [s['confidence'] for s in sentiments]
        enhanced_df['llm_reasoning'] = [s['reasoning'] for s in sentiments]
        enhanced_df['llm_key_factors'] = [','.join(s['key_factors']) for s in sentiments]
        
        enhanced_df['mentioned_tickers'] = [','.join(k['companies_mentioned']) for k in key_info]
        enhanced_df['market_impact'] = [k['market_impact'] for k in key_info]
        enhanced_df['time_horizon'] = [k['time_horizon'] for k in key_info]
        enhanced_df['extracted_events'] = [','.join(k['events']) for k in key_info]
        
        logger.info("LLM processing complete")
        
        return enhanced_df
    
    def create_llm_aggregated_features(self, enhanced_df: pd.DataFrame,
                                     date_column: str = 'date') -> pd.DataFrame:
        """Create aggregated features from LLM analysis"""
        
        if enhanced_df.empty:
            return enhanced_df
        
        # Group by date for aggregation
        daily_features = enhanced_df.groupby(date_column).agg({
            'llm_sentiment_score': ['mean', 'std', 'count'],
            'llm_confidence': 'mean',
            'market_impact': lambda x: (x == 'high').sum(),  # Count high impact news
            'time_horizon': lambda x: (x == 'immediate').sum()  # Count immediate impact news
        }).round(4)
        
        # Flatten column names
        daily_features.columns = [f'llm_{col[0]}_{col[1]}' if col[1] else f'llm_{col[0]}'
                                for col in daily_features.columns]
        
        daily_features = daily_features.reset_index()
        
        # Add trend features
        if 'llm_llm_sentiment_score_mean' in daily_features.columns:
            daily_features['llm_sentiment_trend_3d'] = daily_features['llm_llm_sentiment_score_mean'].diff(3)
            daily_features['llm_sentiment_momentum'] = daily_features['llm_llm_sentiment_score_mean'].rolling(5).mean()
        
        return daily_features

def create_llm_news_features(news_df: pd.DataFrame,
                           config: Optional[Dict] = None,
                           text_column: str = 'text',
                           title_column: str = 'title') -> pd.DataFrame:
    """Convenience function to create LLM-based news features"""
    
    summarizer = LLMNewsSummarizer(config)
    
    # Process articles
    enhanced_df = summarizer.process_news_batch(news_df, text_column, title_column)
    
    # Create aggregated features if date column exists
    if 'date' in enhanced_df.columns:
        daily_features = summarizer.create_llm_aggregated_features(enhanced_df)
        return enhanced_df, daily_features
    
    return enhanced_df

# Example usage
if __name__ == "__main__":
    # Test with sample news
    sample_news = pd.DataFrame([
        {
            'date': '2024-01-15',
            'title': 'Apple Reports Record Q4 Earnings',
            'text': 'Apple Inc. reported record fourth-quarter earnings with revenue of $89.5 billion, beating analyst expectations of $87.2 billion. The company saw strong iPhone sales and growing services revenue.',
            'ticker': 'AAPL'
        },
        {
            'date': '2024-01-15',
            'title': 'Tesla Stock Falls on Production Concerns',
            'text': 'Tesla shares dropped 5% after the company reported lower than expected vehicle deliveries for the quarter. Production challenges at the Shanghai factory contributed to the shortfall.',
            'ticker': 'TSLA'
        }
    ])
    
    # Test without API keys (fallback mode)
    config = {}
    summarizer = LLMNewsSummarizer(config)
    
    enhanced_df = summarizer.process_news_batch(sample_news)
    print(f"Enhanced news shape: {enhanced_df.shape}")
    print(f"New columns: {[col for col in enhanced_df.columns if col.startswith('llm_')]}")
    
    # Test aggregated features
    daily_features = summarizer.create_llm_aggregated_features(enhanced_df)
    print(f"Daily features: {daily_features.columns.tolist()}")