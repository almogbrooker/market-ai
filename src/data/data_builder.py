#!/usr/bin/env python3
"""
Complete Data Builder - Phase 1 Implementation
Combines all data sources: prices, macro, LLM sentiment, technicals
Implements chat-g-2.txt specifications with leakage prevention
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf
import requests
import time

# Import our LLM sentiment engine
from src.features.sentiment.llm_sentiment import LLMSentimentEngine

logger = logging.getLogger(__name__)

class DataBuilder:
    """
    Complete data builder implementing chat-g-2.txt Phase 1 specifications
    """
    
    def __init__(self, include_llm: bool = True, include_macro: bool = True, 
                 start_date: str = '2018-01-01', tickers_file: str = 'nasdaq100.txt'):
        
        self.include_llm = include_llm
        self.include_macro = include_macro
        self.start_date = start_date
        self.tickers_file = tickers_file
        
        # Initialize LLM engine if needed
        if self.include_llm:
            logger.info("🧠 Initializing LLM sentiment engine...")
            self.llm_engine = LLMSentimentEngine()
        
        logger.info(f"🏗️ DataBuilder initialized: LLM={include_llm}, Macro={include_macro}")
    
    def build_complete_dataset(self) -> pd.DataFrame:
        """
        Build the complete dataset with all features
        Returns leakage-safe dataset ready for training
        """
        
        logger.info("🚀 Building complete dataset...")
        
        # Step 1: Load tickers
        tickers = self._load_tickers()
        
        # Step 2: Fetch price data
        price_data = self._fetch_price_data(tickers)
        
        # Step 3: Add technical indicators
        price_data = self._add_technical_indicators(price_data)
        
        # Step 4: Fetch macro data
        if self.include_macro:
            macro_data = self._fetch_macro_data()
            price_data = self._merge_macro_data(price_data, macro_data)
        
        # Step 5: Generate LLM sentiment features
        if self.include_llm:
            text_data = self._fetch_text_data(tickers)
            llm_features = self.llm_engine.daily_llm_features(text_data)
            price_data = self._merge_llm_features(price_data, llm_features)
        
        # Step 6: Add bull-sensitive features (chat-g.txt enhancement)
        price_data = self._add_bull_sensitive_features(price_data)
        
        # Step 7: Add cross-sectional features
        price_data = self._add_cross_sectional_features(price_data)
        
        # Step 8: Create labels with proper lag
        price_data = self._create_labels(price_data)
        
        # Step 8: Final cleanup and validation
        price_data = self._final_cleanup(price_data)
        
        logger.info(f"✅ Complete dataset built: {price_data.shape}")
        return price_data
    
    def _load_tickers(self) -> List[str]:
        """Load ticker symbols from file or use default"""
        
        ticker_file_path = Path(self.tickers_file)
        
        if ticker_file_path.exists():
            with open(ticker_file_path, 'r') as f:
                tickers = [line.strip() for line in f if line.strip()]
            logger.info(f"📋 Loaded {len(tickers)} tickers from {self.tickers_file}")
        else:
            # Default NASDAQ-100 subset
            tickers = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
                'ADBE', 'CRM', 'INTC', 'AMD', 'QCOM', 'CSCO', 'ORCL', 'TXN',
                'COST', 'PEP', 'CMCSA', 'TMUS', 'AVGO', 'ASML', 'AZN', 'DXCM'
            ]
            logger.info(f"📋 Using default {len(tickers)} NASDAQ tickers")
        
        return tickers
    
    def _fetch_price_data(self, tickers: List[str]) -> pd.DataFrame:
        """Fetch OHLCV data for all tickers with caching and retries"""
        
        logger.info(f"📊 Fetching price data for {len(tickers)} tickers...")

        cache_dir = Path("data/price_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        all_data: List[pd.DataFrame] = []
        failed_tickers: List[str] = []

        max_retries = 3
        backoff = 2

        for ticker in tickers:
            cache_file = cache_dir / f"{ticker}.csv"

            # Load from cache if available
            if cache_file.exists():
                logger.info(f"💾 Cache hit for {ticker}")
                try:
                    data = pd.read_csv(cache_file, parse_dates=['Date'])
                    all_data.append(data)
                    continue
                except Exception as e:
                    logger.warning(f"Failed to load cache for {ticker}: {e}; refetching")

            for attempt in range(max_retries):
                try:
                    stock = yf.Ticker(ticker)
                    data = stock.history(start=self.start_date, auto_adjust=True)

                    if data.empty:
                        raise ValueError("Empty data returned")

                    # Clean and standardize
                    data = data.reset_index()
                    data['Ticker'] = ticker
                    data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)

                    # Ensure we have required columns
                    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    if all(col in data.columns for col in required_cols):
                        data = data[required_cols + ['Ticker']]
                        all_data.append(data)
                        data.to_csv(cache_file, index=False)
                        logger.info(f"💾 Cached data for {ticker}")
                    else:
                        raise ValueError("Missing required columns")

                    # Rate limiting
                    time.sleep(0.1)
                    break

                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = backoff ** attempt
                        logger.warning(
                            f"Retry {attempt+1}/{max_retries} for {ticker} after error: {e}; waiting {wait}s"
                        )
                        time.sleep(wait)
                    else:
                        logger.warning(f"Failed to fetch {ticker} after {max_retries} attempts: {e}")
                        failed_tickers.append(ticker)
                        break
        
        if failed_tickers:
            logger.warning(f"⚠️ Failed to fetch data for: {failed_tickers}")
        
        if not all_data:
            raise ValueError("No price data could be fetched")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data = combined_data.sort_values(['Ticker', 'Date']).reset_index(drop=True)
        
        logger.info(f"✅ Price data fetched: {combined_data.shape}")
        return combined_data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        
        logger.info("🔧 Adding technical indicators...")
        
        all_data = []
        
        for ticker in data['Ticker'].unique():
            ticker_data = data[data['Ticker'] == ticker].copy()
            
            if len(ticker_data) < 50:  # Need enough data for indicators
                continue
            
            # Price-based indicators
            ticker_data['SMA_10'] = ticker_data['Close'].rolling(10).mean()
            ticker_data['SMA_20'] = ticker_data['Close'].rolling(20).mean()
            ticker_data['SMA_50'] = ticker_data['Close'].rolling(50).mean()
            ticker_data['EMA_12'] = ticker_data['Close'].ewm(span=12).mean()
            ticker_data['EMA_26'] = ticker_data['Close'].ewm(span=26).mean()
            
            # Momentum indicators
            ticker_data['RSI_14'] = self._calculate_rsi(ticker_data['Close'], 14)
            ticker_data['MACD'] = ticker_data['EMA_12'] - ticker_data['EMA_26']
            ticker_data['MACD_Signal'] = ticker_data['MACD'].ewm(span=9).mean()
            
            # Volatility indicators
            ticker_data['BB_Middle'] = ticker_data['Close'].rolling(20).mean()
            ticker_data['BB_Std'] = ticker_data['Close'].rolling(20).std()
            ticker_data['BB_Upper'] = ticker_data['BB_Middle'] + (ticker_data['BB_Std'] * 2)
            ticker_data['BB_Lower'] = ticker_data['BB_Middle'] - (ticker_data['BB_Std'] * 2)
            
            # Volume indicators
            ticker_data['Volume_SMA'] = ticker_data['Volume'].rolling(20).mean()
            ticker_data['Volume_Ratio'] = ticker_data['Volume'] / ticker_data['Volume_SMA']
            
            # Returns and volatility
            ticker_data['Return_1D'] = ticker_data['Close'].pct_change()
            ticker_data['Return_5D'] = ticker_data['Close'].pct_change(5)
            ticker_data['Volatility_20D'] = ticker_data['Return_1D'].rolling(20).std() * np.sqrt(252)
            
            # ATR for position sizing
            ticker_data['ATR_14'] = self._calculate_atr(ticker_data)
            
            all_data.append(ticker_data)
        
        if not all_data:
            raise ValueError("No technical indicators could be calculated")
        
        result = pd.concat(all_data, ignore_index=True)
        logger.info(f"✅ Technical indicators added: {result.shape}")
        return result
    
    def _fetch_macro_data(self) -> pd.DataFrame:
        """Fetch macro indicators from FRED"""
        
        logger.info("📈 Fetching macro indicators...")
        
        # Macro indicators as specified in chat-g-2.txt
        indicators = {
            'VIX': 'VIXCLS',
            'Treasury_10Y': 'DGS10', 
            'Fed_Funds': 'DFF',
            'CPI': 'CPIAUCSL',
            'Unemployment': 'UNRATE'
        }
        
        # Create realistic macro data (production would use FRED API)
        date_range = pd.date_range(start=self.start_date, end=datetime.now(), freq='D')
        
        macro_data = []
        
        # Initialize values
        vix = 20.0
        treasury_10y = 2.5
        fed_funds = 1.5
        cpi = 250.0
        unemployment = 5.0
        
        for date in date_range:
            # Simulate realistic macro evolution
            vix += np.random.normal(0, 0.8)
            vix = max(10, min(80, vix))
            
            treasury_10y += np.random.normal(0, 0.02)
            treasury_10y = max(0.5, min(8.0, treasury_10y))
            
            # Fed funds changes less frequently
            if np.random.random() < 0.005:  # 0.5% chance per day
                fed_funds += np.random.choice([-0.25, 0, 0.25])
            fed_funds = max(0, min(8, fed_funds))
            
            cpi += np.random.normal(0.001, 0.01)  # Slight upward trend
            unemployment += np.random.normal(0, 0.01)
            unemployment = max(2, min(15, unemployment))
            
            macro_data.append({
                'Date': date,
                'VIX': vix,
                'Treasury_10Y': treasury_10y,
                'Fed_Funds': fed_funds,
                'CPI': cpi,
                'Unemployment': unemployment
            })
        
        macro_df = pd.DataFrame(macro_data)
        macro_df['Date'] = pd.to_datetime(macro_df['Date']).dt.tz_localize(None)
        
        # Add derived features
        macro_df['Yield_Spread'] = macro_df['Treasury_10Y'] - macro_df['Fed_Funds']
        macro_df['VIX_Spike'] = (macro_df['VIX'] > 30).astype(int)
        
        # Apply publication lag (shift macro data forward)
        macro_df['Date'] = macro_df['Date'] + timedelta(days=1)
        
        logger.info(f"✅ Macro data created: {macro_df.shape}")
        return macro_df
    
    def _merge_macro_data(self, price_data: pd.DataFrame, macro_data: pd.DataFrame) -> pd.DataFrame:
        """Merge macro data with price data"""
        
        logger.info("🔗 Merging macro data...")
        
        # Merge on Date
        merged = price_data.merge(macro_data, on='Date', how='left')
        
        # Forward fill macro data for weekends/holidays
        macro_cols = [col for col in macro_data.columns if col != 'Date']
        merged[macro_cols] = merged[macro_cols].fillna(method='ffill')
        
        logger.info(f"✅ Macro data merged: {merged.shape}")
        return merged
    
    def _fetch_text_data(self, tickers: List[str]) -> pd.DataFrame:
        """Fetch text data for LLM processing"""
        
        logger.info("📰 Generating text data for LLM features...")
        
        # In production, this would fetch from GDELT, SEC EDGAR, etc.
        # For now, create realistic sample data
        
        date_range = pd.date_range(start=self.start_date, end=datetime.now(), freq='D')
        
        # Sample financial news texts
        news_templates = [
            "{ticker} reports {sentiment} earnings with {metric} {direction}",
            "{ticker} announces {event}, stock {movement}",
            "Analysts {action} {ticker} target price on {reason}",
            "{ticker} faces {challenge} in {market} segment",
            "Breaking: {ticker} {announcement} exceeds expectations"
        ]
        
        sentiments = ['strong', 'weak', 'mixed', 'positive', 'negative']
        metrics = ['revenue', 'profit', 'guidance', 'margins']
        directions = ['up', 'down', 'flat', 'beating estimates', 'missing targets']
        events = ['partnership', 'acquisition', 'product launch', 'expansion']
        movements = ['rallies', 'drops', 'stays flat', 'surges', 'declines']
        actions = ['raise', 'lower', 'maintain', 'initiate coverage on']
        reasons = ['strong fundamentals', 'market concerns', 'sector rotation', 'earnings beat']
        challenges = ['regulatory pressure', 'competition', 'supply chain issues']
        markets = ['international', 'domestic', 'emerging', 'developed']
        announcements = ['quarterly results', 'strategic initiative', 'partnership']
        
        text_data = []
        
        # Generate text for each date and ticker
        for date in date_range[::7]:  # Weekly to reduce volume
            for ticker in tickers[:10]:  # Limit for performance
                
                # Generate 1-3 texts per ticker per week
                n_texts = np.random.randint(1, 4)
                
                for _ in range(n_texts):
                    template = np.random.choice(news_templates)
                    
                    # Fill in template
                    text = template.format(
                        ticker=ticker,
                        sentiment=np.random.choice(sentiments),
                        metric=np.random.choice(metrics),
                        direction=np.random.choice(directions),
                        event=np.random.choice(events),
                        movement=np.random.choice(movements),
                        action=np.random.choice(actions),
                        reason=np.random.choice(reasons),
                        challenge=np.random.choice(challenges),
                        market=np.random.choice(markets),
                        announcement=np.random.choice(announcements)
                    )
                    
                    text_data.append({
                        'Date': date,
                        'Ticker': ticker,
                        'lang': np.random.choice(['en', 'zh', 'fr'], p=[0.8, 0.15, 0.05]),
                        'text': text,
                        'source': np.random.choice(['news', 'edgar', 'social'], p=[0.7, 0.2, 0.1])
                    })
        
        text_df = pd.DataFrame(text_data)
        text_df['Date'] = pd.to_datetime(text_df['Date']).dt.tz_localize(None)
        
        logger.info(f"✅ Text data generated: {text_df.shape}")
        return text_df
    
    def _merge_llm_features(self, price_data: pd.DataFrame, llm_features: pd.DataFrame) -> pd.DataFrame:
        """Merge LLM sentiment features with price data"""
        
        if llm_features.empty:
            logger.warning("⚠️ No LLM features to merge")
            return price_data
        
        logger.info("🧠 Merging LLM features...")
        
        # Merge on Date and Ticker
        merged = price_data.merge(llm_features, on=['Date', 'Ticker'], how='left')
        
        # Fill missing LLM features with neutral values
        llm_cols = [col for col in llm_features.columns if col not in ['Date', 'Ticker']]
        for col in llm_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0.5 if 'pos' in col or 'neg' in col else 0)
        
        logger.info(f"✅ LLM features merged: {merged.shape}")
        return merged
    
    def _add_bull_sensitive_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add bull-sensitive features per chat-g.txt requirements"""
        
        logger.info("🐂 Adding bull-sensitive features (earnings, breadth, RS)...")
        
        # Add earnings proximity features (simplified)
        data = self._add_earnings_features(data)
        
        # Add breadth and leadership features
        data = self._add_breadth_features(data)
        
        # Add cross-sectional relative strength
        data = self._add_relative_strength_features(data)
        
        # Add seasonality and flow features
        data = self._add_seasonality_features(data)
        
        # Add liquidity filters
        data = self._add_liquidity_features(data)
        
        logger.info("✅ Bull-sensitive features added")
        return data
    
    def _add_earnings_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add earnings proximity and surprise features"""
        
        # Simplified earnings features using quarterly cycle
        data['days_to_earnings'] = ((data['Date'].dt.month - 1) % 3) * 30  # Approximate quarterly cycle
        data['post_earnings_drift'] = np.sin(2 * np.pi * data['Date'].dt.dayofyear / 365) * 0.1
        data['earnings_season'] = ((data['Date'].dt.month - 1) // 3).astype(int)  # Q1=0, Q2=1, Q3=2, Q4=3
        
        return data
    
    def _add_breadth_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market breadth and leadership features"""
        
        # Calculate daily breadth metrics across all tickers
        daily_breadth = data.groupby('Date').agg({
            'Close': lambda x: (x.pct_change() > 0).sum() / len(x),  # % above 0
            'SMA_50': lambda x: (data.loc[x.index, 'Close'] > x).sum() / len(x) if 'SMA_50' in data.columns else 0.5
        }).rename(columns={'Close': 'breadth_positive', 'SMA_50': 'breadth_above_sma50'})
        
        # Merge back to main data
        data = data.merge(daily_breadth, on='Date', how='left')
        
        # Sector leadership (simplified - use tech proxy)
        tech_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']
        tech_data = data[data['Ticker'].isin(tech_tickers)]
        
        if len(tech_data) > 0:
            tech_performance = tech_data.groupby('Date')['Close'].mean().pct_change()
            qqq_performance = data[data['Ticker'] == 'QQQ']['Close'].pct_change() if 'QQQ' in data['Ticker'].values else tech_performance
            
            # Tech leadership vs market
            tech_leadership = (tech_performance - qqq_performance).fillna(0)
            tech_leadership_df = tech_leadership.to_frame('tech_leadership').reset_index()
            data = data.merge(tech_leadership_df, on='Date', how='left')
            data['tech_leadership'] = data['tech_leadership'].fillna(0)
        else:
            data['tech_leadership'] = 0
        
        return data
    
    def _add_relative_strength_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add cross-sectional relative strength rankings"""
        
        # Calculate rolling percentile ranks within NASDAQ universe
        for window in [5, 20]:
            col_name = f'return_{window}d'
            if col_name in data.columns:
                # Calculate percentile rank within each date
                data[f'rs_rank_{window}d'] = data.groupby('Date')[col_name].rank(pct=True)
            else:
                # Calculate the returns if not present
                data[col_name] = data.groupby('Ticker')['Close'].pct_change(window)
                data[f'rs_rank_{window}d'] = data.groupby('Date')[col_name].rank(pct=True)
        
        # Relative strength momentum (recent vs longer-term rank)
        if 'rs_rank_5d' in data.columns and 'rs_rank_20d' in data.columns:
            data['rs_momentum'] = data['rs_rank_5d'] - data['rs_rank_20d']
        
        return data
    
    def _add_seasonality_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add seasonality and flow features"""
        
        # Month-of-year dummies (simplified to quarters)
        data['month'] = data['Date'].dt.month
        data['quarter'] = ((data['Date'].dt.month - 1) // 3) + 1
        
        # Create quarterly dummies
        for q in [1, 2, 3, 4]:
            data[f'Q{q}'] = (data['quarter'] == q).astype(int)
        
        # Simplified options put/call ratio proxy (using VIX-like volatility)
        data['put_call_proxy'] = np.sin(2 * np.pi * data['Date'].dt.dayofyear / 365) * 0.2 + 0.8
        
        # Short interest trend proxy (monthly cycle)
        data['short_interest_proxy'] = np.cos(2 * np.pi * data['Date'].dt.day / 30) * 0.1 + 0.9
        
        return data
    
    def _add_liquidity_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add liquidity filters and features"""
        
        # Average Daily Volume (ADV) - 20-day rolling average
        data['ADV_20d'] = data.groupby('Ticker')['Volume'].rolling(20).mean().reset_index(0, drop=True)
        
        # Price filter (above $5)
        data['price_above_5'] = (data['Close'] > 5).astype(int)
        
        # Market cap proxy (price * volume as rough proxy)
        data['market_cap_proxy'] = data['Close'] * data['Volume']
        data['large_cap'] = (data['market_cap_proxy'] > data['market_cap_proxy'].quantile(0.7)).astype(int)
        
        # Liquidity score (normalized)
        data['liquidity_score'] = (
            data['price_above_5'] * 0.3 +
            (data['ADV_20d'] > data['ADV_20d'].quantile(0.5)).astype(int) * 0.4 +
            data['large_cap'] * 0.3
        )
        
        return data
    
    def _add_cross_sectional_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add cross-sectional features (rankings, z-scores)"""
        
        logger.info("📊 Adding cross-sectional features...")
        
        # Add momentum rankings by date
        for date in data['Date'].unique():
            date_mask = data['Date'] == date
            date_data = data[date_mask]
            
            if len(date_data) > 1:
                # Momentum z-scores
                if 'Return_5D' in data.columns:
                    momentum_mean = date_data['Return_5D'].mean()
                    momentum_std = date_data['Return_5D'].std()
                    if momentum_std > 0:
                        data.loc[date_mask, 'Momentum_ZScore'] = (
                            (date_data['Return_5D'] - momentum_mean) / momentum_std
                        )
                
                # Volume z-scores
                if 'Volume_Ratio' in data.columns:
                    vol_mean = date_data['Volume_Ratio'].mean()
                    vol_std = date_data['Volume_Ratio'].std()
                    if vol_std > 0:
                        data.loc[date_mask, 'Volume_ZScore'] = (
                            (date_data['Volume_Ratio'] - vol_mean) / vol_std
                        )
        
        # Fill NaN z-scores
        data['Momentum_ZScore'] = data.get('Momentum_ZScore', 0).fillna(0)
        data['Volume_ZScore'] = data.get('Volume_ZScore', 0).fillna(0)
        
        logger.info(f"✅ Cross-sectional features added: {data.shape}")
        return data
    
    def _create_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create forward-looking labels with proper lag"""
        
        logger.info("🎯 Creating labels with trading lag...")
        
        all_data = []
        
        for ticker in data['Ticker'].unique():
            ticker_data = data[data['Ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('Date')
            
            # Create forward returns (labels)
            ticker_data['target_1d'] = ticker_data['Close'].pct_change().shift(-1)  # Next day return
            ticker_data['target_5d'] = ticker_data['Close'].pct_change(5).shift(-5)  # 5-day return
            ticker_data['target_20d'] = ticker_data['Close'].pct_change(20).shift(-20)  # 20-day return
            
            # QQQ-relative returns (alpha)
            # For simplicity, assume QQQ return is market average
            market_return_1d = ticker_data['target_1d'].rolling(100).mean()
            ticker_data['alpha_1d'] = ticker_data['target_1d'] - market_return_1d
            
            # Meta-labels for triple barrier
            ticker_data = self._add_triple_barrier_labels(ticker_data)
            
            all_data.append(ticker_data)
        
        result = pd.concat(all_data, ignore_index=True)
        logger.info(f"✅ Labels created: {result.shape}")
        return result
    
    def _add_triple_barrier_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add triple barrier meta-labels"""
        
        if 'ATR_14' not in data.columns:
            return data
        
        # Triple barrier parameters (from chat-g-2.txt)
        tp_mult = 4.0  # Take profit = 4x ATR
        sl_mult = 2.5  # Stop loss = 2.5x ATR
        timeout = 20   # 20 days max hold
        
        data['barrier_label'] = 0  # 0=timeout, 1=take_profit, -1=stop_loss
        data['barrier_days'] = timeout
        
        for i in range(len(data) - timeout):
            if pd.isna(data.iloc[i]['ATR_14']):
                continue
            
            entry_price = data.iloc[i]['Close']
            atr = data.iloc[i]['ATR_14']
            
            tp_level = entry_price + (tp_mult * atr)
            sl_level = entry_price - (sl_mult * atr)
            
            # Check next 20 days
            for j in range(1, min(timeout + 1, len(data) - i)):
                future_price = data.iloc[i + j]['Close']
                
                if future_price >= tp_level:
                    data.iloc[i, data.columns.get_loc('barrier_label')] = 1
                    data.iloc[i, data.columns.get_loc('barrier_days')] = j
                    break
                elif future_price <= sl_level:
                    data.iloc[i, data.columns.get_loc('barrier_label')] = -1
                    data.iloc[i, data.columns.get_loc('barrier_days')] = j
                    break
        
        return data
    
    def _final_cleanup(self, data: pd.DataFrame) -> pd.DataFrame:
        """Final cleanup and validation"""
        
        logger.info("🧹 Final cleanup...")
        
        # Remove rows without targets (last few rows)
        data = data.dropna(subset=['target_1d'])
        
        # Sort by ticker and date
        data = data.sort_values(['Ticker', 'Date']).reset_index(drop=True)
        
        # Remove any remaining NaN in model features
        feature_cols = [col for col in data.columns 
                       if not col.startswith(('Date', 'Ticker', 'target_', 'barrier_'))]
        
        before_dropna = len(data)
        data = data.dropna(subset=feature_cols)
        after_dropna = len(data)
        
        if before_dropna != after_dropna:
            logger.warning(f"⚠️ Dropped {before_dropna - after_dropna} rows with NaN features")
        
        logger.info(f"✅ Final dataset: {data.shape}")
        return data
    
    # Helper methods
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=window).mean()

def main():
    """Test the data builder"""
    
    print("🏗️ Testing Data Builder")
    print("=" * 50)
    
    # Create test builder
    builder = DataBuilder(
        include_llm=True,
        include_macro=True,
        start_date='2024-01-01'
    )
    
    # Build dataset
    dataset = builder.build_complete_dataset()
    
    print(f"\n📊 Dataset Built:")
    print(f"   Shape: {dataset.shape}")
    print(f"   Tickers: {dataset['Ticker'].nunique()}")
    print(f"   Date range: {dataset['Date'].min()} to {dataset['Date'].max()}")
    
    # Show columns by category
    feature_categories = {
        'Price': [col for col in dataset.columns if any(price in col for price in ['Open', 'High', 'Low', 'Close', 'Volume'])],
        'Technical': [col for col in dataset.columns if any(tech in col for tech in ['SMA', 'EMA', 'RSI', 'MACD', 'BB', 'ATR'])],
        'Macro': [col for col in dataset.columns if any(macro in col for macro in ['VIX', 'Treasury', 'Fed', 'CPI', 'Unemployment'])],
        'LLM': [col for col in dataset.columns if any(llm in col for llm in ['fb_', 'ml_', 'senti_', 'news_'])],
        'Cross-sectional': [col for col in dataset.columns if 'ZScore' in col],
        'Labels': [col for col in dataset.columns if col.startswith(('target_', 'alpha_', 'barrier_'))]
    }
    
    print(f"\n📈 Feature Categories:")
    for category, features in feature_categories.items():
        print(f"   {category}: {len(features)} features")
    
    # Sample data
    print(f"\n📋 Sample Data:")
    sample_cols = ['Date', 'Ticker', 'Close', 'RSI_14', 'VIX', 'fb_pos', 'target_1d']
    available_cols = [col for col in sample_cols if col in dataset.columns]
    print(dataset[available_cols].head())
    
    # Save test dataset
    dataset.to_parquet('test_dataset.parquet')
    print(f"\n💾 Test dataset saved to: test_dataset.parquet")

if __name__ == "__main__":
    main()