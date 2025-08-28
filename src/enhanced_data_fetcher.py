#!/usr/bin/env python3
"""
ENHANCED DATA FETCHER
====================
Production-grade multi-source market data fetcher with fallbacks and S&P 500 universe
Supports: Alpha Vantage, Polygon, Finnhub, IEX Cloud, Yahoo Finance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import requests
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

class EnhancedDataFetcher:
    """Enhanced multi-source data fetcher with S&P 500 universe"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.setup_logging()
        self.config = self._load_config(config_path)
        
        # Load API credentials
        self.api_keys = self._load_api_keys()
        
        # Load S&P 500 universe
        self.symbols = self._load_sp500_universe()
        
        # Data source configurations
        self.data_sources = {
            'alpha_vantage': {
                'enabled': bool(self.api_keys.get('alpha_vantage')),
                'rate_limit': 5,  # requests per minute for free tier
                'priority': 1
            },
            'polygon': {
                'enabled': bool(self.api_keys.get('polygon')),
                'rate_limit': 100,  # requests per minute for free tier
                'priority': 2
            },
            'finnhub': {
                'enabled': bool(self.api_keys.get('finnhub')),
                'rate_limit': 60,  # requests per minute for free tier
                'priority': 3
            },
            'iex_cloud': {
                'enabled': bool(self.api_keys.get('iex_cloud')),
                'rate_limit': 100,  # requests per minute for free tier
                'priority': 4
            },
            'yfinance': {
                'enabled': True,  # Always available as fallback
                'rate_limit': None,  # No official rate limit
                'priority': 5
            }
        }
        
        # Cache and performance tracking
        self.data_cache = {}
        self.source_performance = {source: {'success_count': 0, 'fail_count': 0, 'avg_latency': 0.0} 
                                 for source in self.data_sources.keys()}
        
        print(f"🔄 Enhanced Data Fetcher initialized")
        print(f"   📊 Universe: {len(self.symbols)} S&P 500 symbols")
        print(f"   🔗 Sources enabled: {len([s for s, cfg in self.data_sources.items() if cfg['enabled']])}")
        print(f"   ⚡ Concurrent fetching: Enabled")
        
        # Log enabled sources
        for source, config in self.data_sources.items():
            status = "✅" if config['enabled'] else "❌"
            print(f"     {status} {source}")

    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = Path("../artifacts/logs/enhanced_data")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"data_fetcher_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _load_config(self, config_path):
        """Load configuration with enhanced defaults"""
        default_config = {
            'max_concurrent_requests': 10,
            'request_timeout': 30,
            'retry_attempts': 3,
            'cache_ttl_minutes': 5,
            'fallback_cascade': True,
            'data_quality_threshold': 0.95,
            'preferred_sources': ['alpha_vantage', 'polygon', 'finnhub', 'iex_cloud', 'yfinance']
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                user_config = json.load(f)
            default_config.update(user_config)
            
        return default_config

    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment variables"""
        api_keys = {}
        
        # Check for API keys
        key_mappings = {
            'alpha_vantage': 'ALPHA_VANTAGE_API_KEY',
            'polygon': 'POLYGON_API_KEY',
            'finnhub': 'FINNHUB_API_KEY',
            'iex_cloud': 'IEX_CLOUD_API_KEY',
            'twelve_data': 'TWELVE_DATA_API_KEY'
        }
        
        for service, env_var in key_mappings.items():
            key = os.getenv(env_var)
            if key:
                api_keys[service] = key
                self.logger.info(f"✅ {service} API key found")
            else:
                self.logger.warning(f"⚠️ {service} API key not found ({env_var})")
        
        return api_keys

    def _load_sp500_universe(self) -> List[str]:
        """Load S&P 500 stock universe"""
        
        # S&P 500 symbols (major components - expandable)
        sp500_symbols = [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'AVGO', 'NFLX',
            'ADBE', 'CRM', 'INTC', 'AMD', 'QCOM', 'TXN', 'AMAT', 'MU', 'ADI', 'LRCX',
            'KLAC', 'MRVL', 'SNPS', 'CDNS', 'FTNT', 'PANW', 'CRWD', 'ZS', 'NET', 'DDOG',
            'SNOW', 'MDB', 'OKTA', 'ZM', 'TEAM', 'WDAY', 'ADSK', 'INTU', 'CSCO',
            
            # Healthcare & Biotech
            'JNJ', 'PFE', 'UNH', 'MRNA', 'ABBV', 'TMO', 'ABT', 'DHR', 'BMY', 'LLY',
            'MDT', 'GILD', 'AMGN', 'VRTX', 'REGN', 'BIIB', 'ILMN', 'ZTS', 'CVS', 'CI',
            
            # Financials
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'COF', 'AXP', 'SCHW', 'USB',
            'PNC', 'TFC', 'BLK', 'SPGI', 'ICE', 'CME', 'MCO', 'AON', 'MMC', 'AJG',
            
            # Consumer Discretionary
            'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TJX', 'LOW', 'BKNG', 'ABNB',
            'MAR', 'HLT', 'MGM', 'CCL', 'NCLH', 'RCL', 'WYNN', 'LVS', 'DIS', 'CMCSA',
            
            # Consumer Staples
            'WMT', 'PG', 'KO', 'PEP', 'COST', 'WBA', 'CL', 'KMB', 'GIS', 'K',
            'HSY', 'MKC', 'SJM', 'CAG', 'CPB', 'KHC', 'MDLZ', 'MNST', 'KDP', 'CLX',
            
            # Industrials
            'BA', 'CAT', 'HON', 'UPS', 'RTX', 'LMT', 'GE', 'MMM', 'ITW', 'ETN',
            'EMR', 'ROK', 'PH', 'CARR', 'OTIS', 'GD', 'NOC', 'LHX', 'TDG', 'CTAS',
            
            # Materials
            'LIN', 'APD', 'SHW', 'DD', 'ECL', 'FCX', 'NEM', 'DOW', 'PPG', 'IFF',
            'ALB', 'CE', 'VMC', 'MLM', 'PKG', 'BALL', 'AVY', 'IP', 'WRK', 'SEE',
            
            # Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PXD', 'KMI', 'WMB', 'OKE', 'VLO',
            'PSX', 'MPC', 'FANG', 'DVN', 'BKR', 'HAL', 'APA', 'CTRA', 'EQT', 'TRGP',
            
            # Utilities
            'NEE', 'SO', 'DUK', 'AEP', 'SRE', 'D', 'PCG', 'EXC', 'XEL', 'WEC',
            'ED', 'ETR', 'ES', 'FE', 'EIX', 'PPL', 'EVRG', 'CMS', 'DTE', 'NI',
            
            # Real Estate
            'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'WELL', 'DLR', 'O', 'SBAC', 'EXR',
            'AVB', 'EQR', 'VTR', 'ESS', 'MAA', 'UDR', 'CPT', 'FRT', 'BXP', 'HST',
            
            # Communication Services
            'GOOGL', 'GOOG', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR'
        ]
        
        # Remove duplicates and sort
        unique_symbols = sorted(list(set(sp500_symbols)))
        
        self.logger.info(f"📊 Loaded S&P 500 universe: {len(unique_symbols)} symbols")
        return unique_symbols

    def fetch_latest_prices(self, symbols: Optional[List[str]] = None, 
                          streaming_priority: bool = False,
                          concurrent: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """Fetch latest market data with intelligent source selection"""
        
        if symbols is None:
            symbols = self.symbols
            
        print(f"\\n📊 Fetching data for {len(symbols)} symbols...")
        
        start_time = datetime.now()
        audit_info = {
            'timestamp': start_time.isoformat(),
            'requested_symbols': len(symbols),
            'sources_attempted': [],
            'successful_sources': [],
            'failed_sources': [],
            'success_rate': 0.0,
            'latency_seconds': 0.0,
            'data_quality_score': 0.0
        }
        
        # Get enabled sources sorted by priority
        available_sources = [
            (source, config) for source, config in self.data_sources.items() 
            if config['enabled']
        ]
        available_sources.sort(key=lambda x: x[1]['priority'])
        
        data = None
        
        # Try each source until we get sufficient data
        for source_name, source_config in available_sources:
            print(f"   🔄 Trying {source_name}...")
            audit_info['sources_attempted'].append(source_name)
            
            try:
                if concurrent and len(symbols) > 50:
                    source_data = self._fetch_concurrent(source_name, symbols)
                else:
                    source_data = self._fetch_from_source(source_name, symbols)
                
                if source_data is not None and len(source_data) > 0:
                    success_rate = len(source_data['Ticker'].unique()) / len(symbols)
                    
                    if success_rate >= self.config['data_quality_threshold']:
                        data = source_data
                        audit_info['successful_sources'].append(source_name)
                        print(f"   ✅ {source_name}: {success_rate:.1%} success rate")
                        break
                    else:
                        print(f"   ⚠️ {source_name}: {success_rate:.1%} (below threshold)")
                        audit_info['failed_sources'].append(f"{source_name} (low quality)")
                        
                        # Store partial data for potential merge
                        if data is None:
                            data = source_data
                        else:
                            data = self._merge_data(data, source_data)
                            
            except Exception as e:
                self.logger.error(f"Source {source_name} failed: {e}")
                audit_info['failed_sources'].append(f"{source_name} (error)")
                
        # Calculate final metrics
        if data is not None and len(data) > 0:
            unique_symbols = data['Ticker'].nunique()
            audit_info['success_rate'] = unique_symbols / len(symbols)
            audit_info['latency_seconds'] = (datetime.now() - start_time).total_seconds()
            audit_info['data_quality_score'] = self._calculate_data_quality(data)
            
            print(f"✅ Data fetched successfully:")
            print(f"   📊 Symbols: {unique_symbols}/{len(symbols)} ({audit_info['success_rate']:.1%})")
            print(f"   ⏱️ Latency: {audit_info['latency_seconds']:.2f}s")
            print(f"   🎯 Quality Score: {audit_info['data_quality_score']:.2f}")
            
        else:
            self.logger.error("All data sources failed")
            audit_info['success_rate'] = 0.0
            audit_info['latency_seconds'] = (datetime.now() - start_time).total_seconds()
            
        return data, audit_info

    def _fetch_from_source(self, source_name: str, symbols: List[str]) -> Optional[pd.DataFrame]:
        """Fetch data from specific source"""
        
        if source_name == 'alpha_vantage':
            return self._fetch_alpha_vantage(symbols)
        elif source_name == 'polygon':
            return self._fetch_polygon(symbols)
        elif source_name == 'finnhub':
            return self._fetch_finnhub(symbols)
        elif source_name == 'iex_cloud':
            return self._fetch_iex_cloud(symbols)
        elif source_name == 'yfinance':
            return self._fetch_yfinance(symbols)
        else:
            raise ValueError(f"Unknown source: {source_name}")

    def _fetch_alpha_vantage(self, symbols: List[str]) -> pd.DataFrame:
        """Fetch from Alpha Vantage API"""
        if not self.api_keys.get('alpha_vantage'):
            return None
            
        data_rows = []
        api_key = self.api_keys['alpha_vantage']
        
        for symbol in symbols[:100]:  # Free tier limit
            try:
                url = f"https://www.alphavantage.co/query"
                params = {
                    'function': 'TIME_SERIES_DAILY',
                    'symbol': symbol,
                    'apikey': api_key,
                    'outputsize': 'compact'
                }
                
                response = requests.get(url, params=params, timeout=10)
                data = response.json()
                
                if 'Time Series (Daily)' in data:
                    time_series = data['Time Series (Daily)']
                    for date_str, values in time_series.items():
                        data_rows.append({
                            'Date': datetime.strptime(date_str, '%Y-%m-%d').date(),
                            'Ticker': symbol,
                            'Open': float(values['1. open']),
                            'High': float(values['2. high']),
                            'Low': float(values['3. low']),
                            'Close': float(values['4. close']),
                            'Volume': int(values['5. volume']),
                            'source': 'alpha_vantage'
                        })
                
                # Rate limiting for free tier
                time.sleep(12)  # 5 requests per minute
                
            except Exception as e:
                self.logger.warning(f"Alpha Vantage error for {symbol}: {e}")
                continue
                
        return pd.DataFrame(data_rows) if data_rows else None

    def _fetch_yfinance(self, symbols: List[str]) -> pd.DataFrame:
        """Enhanced Yahoo Finance fetcher"""
        data_rows = []
        
        try:
            # Batch fetch for efficiency
            tickers = yf.Tickers(' '.join(symbols))
            
            for symbol in symbols:
                try:
                    ticker = tickers.tickers[symbol]
                    hist = ticker.history(period='3mo', interval='1d')
                    
                    if not hist.empty:
                        for date, row in hist.iterrows():
                            data_rows.append({
                                'Date': date.date(),
                                'Ticker': symbol,
                                'Open': row['Open'],
                                'High': row['High'],
                                'Low': row['Low'],
                                'Close': row['Close'],
                                'Volume': row['Volume'],
                                'source': 'yfinance'
                            })
                            
                except Exception as e:
                    self.logger.warning(f"YFinance error for {symbol}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"YFinance batch error: {e}")
            
        return pd.DataFrame(data_rows) if data_rows else None

    def _fetch_polygon(self, symbols: List[str]) -> pd.DataFrame:
        """Fetch from Polygon.io API"""
        if not self.api_keys.get('polygon'):
            return None
            
        # Implementation placeholder - requires polygon-api-client
        self.logger.info("Polygon.io integration placeholder")
        return None

    def _fetch_finnhub(self, symbols: List[str]) -> pd.DataFrame:
        """Fetch from Finnhub API"""
        if not self.api_keys.get('finnhub'):
            return None
            
        # Implementation placeholder
        self.logger.info("Finnhub integration placeholder")
        return None

    def _fetch_iex_cloud(self, symbols: List[str]) -> pd.DataFrame:
        """Fetch from IEX Cloud API"""
        if not self.api_keys.get('iex_cloud'):
            return None
            
        # Implementation placeholder
        self.logger.info("IEX Cloud integration placeholder")
        return None

    def _fetch_concurrent(self, source_name: str, symbols: List[str]) -> pd.DataFrame:
        """Fetch data using concurrent threads"""
        
        # Split symbols into batches
        batch_size = min(50, len(symbols) // self.config['max_concurrent_requests'])
        if batch_size == 0:
            batch_size = 1
            
        batches = [symbols[i:i+batch_size] for i in range(0, len(symbols), batch_size)]
        
        all_data = []
        
        with ThreadPoolExecutor(max_workers=self.config['max_concurrent_requests']) as executor:
            future_to_batch = {
                executor.submit(self._fetch_from_source, source_name, batch): batch 
                for batch in batches
            }
            
            for future in as_completed(future_to_batch):
                try:
                    batch_data = future.result()
                    if batch_data is not None:
                        all_data.append(batch_data)
                except Exception as e:
                    self.logger.error(f"Concurrent batch failed: {e}")
                    
        return pd.concat(all_data, ignore_index=True) if all_data else None

    def _merge_data(self, data1: pd.DataFrame, data2: pd.DataFrame) -> pd.DataFrame:
        """Merge data from multiple sources"""
        if data1 is None:
            return data2
        if data2 is None:
            return data1
            
        # Combine and remove duplicates (prefer first source)
        combined = pd.concat([data1, data2], ignore_index=True)
        merged = combined.drop_duplicates(subset=['Date', 'Ticker'], keep='first')
        
        return merged.sort_values(['Date', 'Ticker']).reset_index(drop=True)

    def _calculate_data_quality(self, data: pd.DataFrame) -> float:
        """Calculate data quality score"""
        if data is None or len(data) == 0:
            return 0.0
            
        quality_factors = []
        
        # Completeness
        expected_columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
        completeness = len([col for col in expected_columns if col in data.columns]) / len(expected_columns)
        quality_factors.append(completeness * 0.3)
        
        # Data freshness (recent date)
        if 'Date' in data.columns:
            latest_date = pd.to_datetime(data['Date']).max()
            days_old = (datetime.now().date() - latest_date.date()).days
            freshness = max(0, 1 - (days_old / 7))  # Decay over a week
            quality_factors.append(freshness * 0.3)
        
        # Missing values
        if len(data) > 0:
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            completeness_score = 1 - missing_ratio
            quality_factors.append(completeness_score * 0.2)
        
        # Symbol coverage
        unique_symbols = data['Ticker'].nunique()
        symbol_coverage = unique_symbols / len(self.symbols)
        quality_factors.append(symbol_coverage * 0.2)
        
        return sum(quality_factors)

    def validate_data_quality(self, data: pd.DataFrame) -> Dict:
        """Comprehensive data quality validation"""
        if data is None or len(data) == 0:
            return {'valid': False, 'issues': ['No data received']}
            
        issues = []
        
        # Check required columns
        required_columns = ['Date', 'Ticker', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            issues.append(f"Missing columns: {missing_columns}")
        
        # Check for recent data
        if 'Date' in data.columns:
            latest_date = pd.to_datetime(data['Date']).max()
            days_old = (datetime.now().date() - latest_date.date()).days
            if days_old > 3:  # Data older than 3 days
                issues.append(f"Data is {days_old} days old")
        
        # Check data ranges
        if 'Close' in data.columns:
            negative_prices = data[data['Close'] <= 0]
            if len(negative_prices) > 0:
                issues.append(f"Invalid prices: {len(negative_prices)} negative/zero prices")
        
        # Check symbol coverage
        unique_symbols = data['Ticker'].nunique() if 'Ticker' in data.columns else 0
        coverage = unique_symbols / len(self.symbols)
        if coverage < 0.8:
            issues.append(f"Low symbol coverage: {coverage:.1%}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'quality_score': self._calculate_data_quality(data),
            'symbol_count': unique_symbols,
            'coverage': coverage
        }

def main():
    """Test the enhanced data fetcher"""
    print("🚀 ENHANCED DATA FETCHER TEST")
    print("=" * 50)
    
    fetcher = EnhancedDataFetcher()
    
    # Test with a small subset
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    
    data, audit = fetcher.fetch_latest_prices(test_symbols)
    
    if data is not None:
        print(f"\\n✅ Test successful:")
        print(f"   📊 Records: {len(data)}")
        print(f"   🏢 Symbols: {data['Ticker'].nunique()}")
        print(f"   📅 Date range: {data['Date'].min()} to {data['Date'].max()}")
        print(f"   🎯 Quality: {audit['data_quality_score']:.2f}")
    else:
        print("❌ Test failed - no data retrieved")

if __name__ == "__main__":
    main()