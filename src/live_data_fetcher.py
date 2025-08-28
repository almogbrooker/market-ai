#!/usr/bin/env python3
"""
LIVE DATA FETCHER
=================
Production market data adapter with primary/fallback sources
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

class LiveDataFetcher:
    """Production market data fetcher with fallback sources"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.setup_logging()
        
        # Configuration
        self.config = self._load_config(config_path)
        self.symbols = self._load_universe()
        
        # Data sources priority order
        self.primary_source = self.config.get('primary_source', 'yfinance')
        self.fallback_sources = self.config.get('fallback_sources', ['yfinance'])
        
        # Initialize streaming attributes (simplified for current implementation)
        self.streaming_active = False
        self.stream_buffer = {}
        
        print(f"🔄 Live Data Fetcher initialized")
        print(f"   Primary source: {self.primary_source}")
        print(f"   Universe: {len(self.symbols)} symbols")
        print(f"   Streaming: ❌ Disabled (using batch data)")
        
    def setup_logging(self):
        """Setup logging for audit trail"""
        log_dir = Path("../artifacts/logs/live_data")
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
        """Load configuration"""
        default_config = {
            'primary_source': 'yfinance',
            'fallback_sources': ['yfinance'],
            'max_latency_minutes': 5,
            'timezone': 'America/New_York',
            'trading_hours': {'start': '09:30', 'end': '16:00'},
            'min_data_quality': 0.95  # Minimum data completeness
        }
        
        if config_path and Path(config_path).exists():
            import json
            with open(config_path) as f:
                user_config = json.load(f)
            default_config.update(user_config)
            
        return default_config
    
    def _load_universe(self):
        """Load trading universe"""
        # NASDAQ 100 symbols (subset for demo)
        symbols = [
            # Mega caps
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA',
            'AVGO', 'PEP', 'ASML', 'COST', 'NFLX', 'ADBE', 'TMUS', 'CSCO',
            # Large caps  
            'AMD', 'INTC', 'CMCSA', 'PDD', 'TXN', 'QCOM', 'INTU', 'AMAT',
            'MU', 'ADI', 'ISRG', 'BKNG', 'LRCX', 'GILD', 'KLAC', 'REGN',
            'PANW', 'SNPS', 'CDNS', 'MRVL', 'ORLY', 'CSX', 'ABNB', 'FTNT',
            # Growth stocks
            'CRWD', 'ADSK', 'NXPI', 'ROP', 'CPRT', 'MNST', 'FANG', 'TEAM',
            'WDAY', 'DDOG', 'ZS', 'OKTA', 'NET', 'SNOW', 'MDB', 'ZM',
            # Additional NASDAQ 100 stocks  
            'PCAR', 'PAYX', 'FAST', 'ODFL', 'DXCM', 'CTAS', 'VRSK', 'EXC',
            'AEP', 'XEL', 'DLTR', 'EBAY', 'BIIB', 'ILMN', 'KHC', 'WBD',
            'MRNA', 'ADP', 'ROST', 'KDP', 'CTSH', 'LULU',
            # More tech and biotech (removed delisted: SGEN, FISV, ATVI)
            'VRTX', 'CHTR', 'NTES', 'MELI', 'DASH', 'LCID', 'RIVN', 'ENPH',
            'HON', 'SBUX', 'MDLZ', 'MAR', 'PYPL', 'WMT', 'PFE'
        ]
        return symbols  # Full list (85 symbols - removed delisted stocks)
    
    def fetch_latest_prices(self, streaming_priority: bool = False) -> Tuple[pd.DataFrame, Dict]:
        """Fetch latest market data with audit trail"""
        print(f"\\n📊 Fetching latest market data...")
        
        start_time = datetime.now()
        audit_info = {
            'timestamp': start_time.isoformat(),
            'requested_symbols': len(self.symbols),
            'source_attempts': [],
            'success_rate': 0.0,
            'latency_seconds': 0.0,
            'streaming_used': False
        }
        
        data = None
        
        # Try streaming data first if enabled and requested
        if streaming_priority and self.streaming_active:
            stream_data = self._get_streaming_data()
            if stream_data is not None and len(stream_data) >= len(self.symbols) * 0.7:
                data = stream_data
                audit_info['streaming_used'] = True
                print(f"   ⚡ Using real-time stream data: {len(data)} symbols")
        
        # Fall back to traditional sources if streaming insufficient
        if data is None or len(data) < len(self.symbols) * 0.7:
            if data is not None:
                print(f"   📊 Stream data insufficient ({len(data)} symbols), falling back...")
            data = self._fetch_from_source(self.primary_source if not audit_info['streaming_used'] else 'alpaca_bars', audit_info)
        
        # Fall back if primary fails
        if data is None or len(data) < len(self.symbols) * 0.8:
            self.logger.warning(f"Primary source {self.primary_source} insufficient, trying fallbacks")
            
            for fallback_source in self.fallback_sources:
                if fallback_source != self.primary_source:
                    fallback_data = self._fetch_from_source(fallback_source, audit_info)
                    if fallback_data is not None:
                        data = fallback_data if data is None else self._merge_data(data, fallback_data)
                        break
        
        # Calculate metrics
        if data is not None:
            audit_info['success_rate'] = len(data) / len(self.symbols)
            audit_info['latency_seconds'] = (datetime.now() - start_time).total_seconds()
            
            print(f"✅ Data fetched successfully:")
            print(f"   📊 Symbols: {len(data)}/{len(self.symbols)} ({audit_info['success_rate']:.1%})")
            print(f"   ⏱️ Latency: {audit_info['latency_seconds']:.2f}s")
            
            self._log_audit_trail(audit_info)
            
            return data, audit_info
        else:
            self.logger.error("All data sources failed")
            return None, audit_info
    
    def _fetch_from_source(self, source: str, audit_info: Dict) -> Optional[pd.DataFrame]:
        """Fetch data from specific source"""
        attempt = {
            'source': source,
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'symbols_retrieved': 0,
            'error': None
        }
        
        try:
            if source == 'yfinance':
                data = self._fetch_yfinance()
            elif source == 'polygon':
                data = self._fetch_polygon()
            elif source == 'iex':
                data = self._fetch_iex()
            else:
                raise ValueError(f"Unknown source: {source}")
            
            if data is not None and len(data) > 0:
                attempt['success'] = True
                attempt['symbols_retrieved'] = len(data)
                audit_info['source_attempts'].append(attempt)
                return data
                
        except Exception as e:
            attempt['error'] = str(e)
            self.logger.error(f"Source {source} failed: {e}")
        
        audit_info['source_attempts'].append(attempt)
        return None
    
    def _fetch_yfinance(self) -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
        print(f"   📈 Fetching from Yahoo Finance...")
        
        # Get current prices
        tickers = yf.Tickers(' '.join(self.symbols))
        
        data_rows = []
        for symbol in self.symbols:
            try:
                ticker = tickers.tickers[symbol]
                info = ticker.info
                hist = ticker.history(period='3mo', interval='1d')  # Get 3 months of data for features
                
                if not hist.empty:
                    # Get all historical data for feature generation
                    for date, row in hist.iterrows():
                        data_rows.append({
                            'Date': date.date(),
                            'Ticker': symbol,
                            'Open': row['Open'],
                            'High': row['High'], 
                            'Low': row['Low'],
                            'Close': row['Close'],
                            'Volume': row['Volume'],
                            'source': 'yfinance',
                            'timestamp': datetime.now().isoformat()
                        })
            except:
                continue
        
        if data_rows:
            return pd.DataFrame(data_rows)
        return None
    
    def _fetch_polygon(self) -> Optional[pd.DataFrame]:
        """Fetch data from Polygon.io (requires API key)"""
        # Placeholder - would implement with actual Polygon API
        self.logger.info("Polygon source not configured")
        return None
    
    def _fetch_iex(self) -> Optional[pd.DataFrame]:
        """Fetch data from IEX Cloud (requires API key)"""
        # Placeholder - would implement with actual IEX API
        self.logger.info("IEX source not configured") 
        return None
    
    def _merge_data(self, primary: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
        """Merge data from multiple sources"""
        # Use primary data, fill gaps with fallback
        primary_symbols = set(primary['Ticker'])
        fallback_symbols = set(fallback['Ticker'])
        
        missing_symbols = fallback_symbols - primary_symbols
        if missing_symbols:
            missing_data = fallback[fallback['Ticker'].isin(missing_symbols)]
            return pd.concat([primary, missing_data], ignore_index=True)
        
        return primary
    
    def _log_audit_trail(self, audit_info: Dict):
        """Log comprehensive audit trail"""
        audit_file = Path("../artifacts/logs/live_data") / f"audit_trail_{datetime.now().strftime('%Y%m%d')}.json"
        
        try:
            import json
            audit_entry = {
                'run_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
                **audit_info
            }
            
            # Append to daily audit file
            audit_entries = []
            if audit_file.exists():
                with open(audit_file) as f:
                    audit_entries = json.load(f)
            
            audit_entries.append(audit_entry)
            
            with open(audit_file, 'w') as f:
                json.dump(audit_entries, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to log audit trail: {e}")
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict:
        """Validate data quality"""
        if data is None or len(data) == 0:
            return {'valid': False, 'issues': ['No data received']}
        
        issues = []
        
        # Check completeness
        completeness = len(data) / len(self.symbols)
        if completeness < self.config['min_data_quality']:
            issues.append(f"Low completeness: {completeness:.1%} < {self.config['min_data_quality']:.1%}")
        
        # Check for stale data
        if 'timestamp' in data.columns:
            latest_timestamp = pd.to_datetime(data['timestamp']).max()
            staleness = (datetime.now() - latest_timestamp).total_seconds() / 60
            
            if staleness > self.config['max_latency_minutes']:
                issues.append(f"Stale data: {staleness:.1f} min > {self.config['max_latency_minutes']} min")
        
        # Check for missing prices
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_prices = data[required_cols].isnull().sum().sum()
        if missing_prices > 0:
            issues.append(f"Missing price data: {missing_prices} fields")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'completeness': completeness,
            'symbol_count': len(data)
        }

def main():
    """Test live data fetcher"""
    print("🔄 Testing Live Data Fetcher")
    
    fetcher = LiveDataFetcher()
    data, audit = fetcher.fetch_latest_prices()
    
    if data is not None:
        quality = fetcher.validate_data_quality(data)
        print(f"\\n📊 Data Quality: {'✅ Valid' if quality['valid'] else '❌ Issues'}")
        if quality['issues']:
            for issue in quality['issues']:
                print(f"   ⚠️ {issue}")
        
        print(f"\\n📈 Sample data:")
        print(data.head())
    else:
        print("❌ Data fetch failed")

if __name__ == "__main__":
    main()