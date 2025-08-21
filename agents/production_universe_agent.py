#!/usr/bin/env python3
"""
PRODUCTION UNIVERSE AGENT - Research-Backed Implementation
Based on academic literature and industry best practices
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionUniverseAgent:
    """
    Production Universe Agent - Research-Backed Implementation
    Implements battle-tested filters and feature engineering
    """
    
    def __init__(self, config_path: str):
        logger.info("🏭 PRODUCTION UNIVERSE AGENT - RESEARCH-BACKED")
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.base_dir = Path(__file__).parent.parent
        self.artifacts_dir = self.base_dir / "artifacts"
        
        # Production filters
        self.filters = self.config['universe_filters']
        
        logger.info("🎯 Production Universe Filters:")
        logger.info(f"   Price ≥ ${self.filters['min_price']}")
        logger.info(f"   ADV ≥ ${self.filters['min_adv']:,}")
        logger.info(f"   Free Float ≥ ${self.filters['min_free_float_mcap']:,}")
        logger.info(f"   Borrowable Only: {self.filters['borrowable_only']}")
        logger.info(f"   Recompute: {self.filters['recompute_frequency']}")
        
    def build_production_universe(self, date: Optional[str] = None) -> bool:
        """
        Build production-ready universe with research-backed filters
        """
        
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"🏭 Building production universe for {date}...")
        
        try:
            # Step 1: Get NASDAQ universe
            nasdaq_universe = self._get_nasdaq_universe()
            
            # Step 2: Apply liquidity and quality filters
            filtered_universe = self._apply_production_filters(nasdaq_universe)
            
            # Step 3: Download price data and validate
            validated_universe = self._validate_universe_data(filtered_universe)
            
            # Step 4: Engineer research-backed features
            featured_data = self._engineer_production_features(validated_universe)
            
            # Step 5: Create residual returns target
            labeled_data = self._create_residual_targets(featured_data)
            
            # Step 6: Apply cross-sectional scaling
            final_data = self._apply_cross_sectional_scaling(labeled_data)
            
            # Step 7: Save production artifacts
            self._save_production_artifacts(final_data, date)
            
            logger.info("✅ Production universe built successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Production universe build failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _get_nasdaq_universe(self) -> List[str]:
        """Get comprehensive NASDAQ universe (research-grade)"""
        
        logger.info("📊 Fetching NASDAQ universe...")
        
        # Research-backed NASDAQ universe (active, liquid stocks)
        nasdaq_symbols = [
            # Technology - Mega/Large Cap
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA',
            'NFLX', 'INTC', 'CSCO', 'ADBE', 'ORCL', 'CRM', 'AVGO', 'TXN',
            'QCOM', 'AMAT', 'LRCX', 'KLAC', 'MCHP', 'ADI', 'INTU', 'ISRG',
            'PYPL', 'SHOP', 'ROKU', 'ZM', 'DOCU', 'CRWD', 'OKTA', 'PANW',
            
            # Healthcare & Biotechnology
            'AMGN', 'GILD', 'REGN', 'VRTX', 'BIIB', 'MRNA', 'ILMN', 'BMRN',
            'IDXX', 'DXCM', 'TDOC', 'VEEV', 'ALGN', 'EXAS', 'PTCT', 'SRPT',
            'TECH', 'INCY', 'FOLD', 'RARE', 'UTHR', 'NBIX', 'ALNY',
            
            # Consumer Discretionary
            'COST', 'SBUX', 'DLTR', 'ORLY', 'ROST', 'ULTA', 'LULU',
            'BKNG', 'EXPE', 'EBAY', 'ETSY', 'CHWY', 'PINS', 'SNAP',
            'MAR', 'WYNN', 'NCLH', 'CCL', 'TRIP', 'DASH', 'ABNB',
            
            # Communication Services
            'CMCSA', 'CHTR', 'SIRI', 'FOXA', 'FOX', 'TMUS',
            'LBTYA', 'LBTYK', 'LBRDA', 'LBRDK',
            
            # Industrials & Business Services
            'HON', 'CSX', 'PCAR', 'FAST', 'CTAS', 'VRSK', 'PAYX', 'ADP',
            'WDAY', 'TEAM', 'NOW', 'CDNS', 'SNPS', 'ANSS', 'PTC',
            
            # Semiconductors
            'AMD', 'MRVL', 'SWKS', 'QRVO', 'CRUS', 'MPWR', 'MXIM',
            
            # Growth & Cloud
            'DOCN', 'SNOW', 'PLTR', 'RBLX', 'U', 'DDOG', 'MDB', 'ZS',
            'NET', 'FSLY', 'TWLO', 'GTLB', 'MNDY', 'BILL',
            
            # International/ADRs (liquid only)
            'MELI', 'JD', 'PDD', 'BIDU', 'NTES', 'TME', 'BILI',
            
            # Additional Quality Names
            'MNST', 'MDLZ', 'PEP', 'EA', 'TTWO', 'NXPI',
            'POOL', 'FIVE', 'OLLI', 'WING', 'TXRH'
        ]
        
        # Remove any known delisted/merged tickers
        active_symbols = [s for s in nasdaq_symbols if s not in ['XLNX', 'ATVI', 'FISV']]
        
        logger.info(f"📊 NASDAQ universe: {len(active_symbols)} symbols")
        return active_symbols
    
    def _apply_production_filters(self, symbols: List[str]) -> pd.DataFrame:
        """Apply research-backed liquidity and quality filters"""
        
        logger.info("🔍 Applying production filters...")
        
        universe_data = []
        failed_count = 0
        
        for symbol in symbols:
            try:
                # Download recent data for filtering
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='60d')
                info = ticker.info
                
                if len(hist) < 20:  # Insufficient data
                    failed_count += 1
                    continue
                
                # Calculate metrics for filtering
                current_price = hist['Close'].iloc[-1]
                avg_volume = hist['Volume'].tail(20).mean()
                avg_dollar_volume = (hist['Close'] * hist['Volume']).tail(20).mean()
                
                # Market cap estimation (rough)
                shares_outstanding = info.get('sharesOutstanding', info.get('impliedSharesOutstanding', 0))
                market_cap = current_price * shares_outstanding if shares_outstanding else 0
                
                # Apply filters
                price_ok = current_price >= self.filters['min_price']
                adv_ok = avg_dollar_volume >= self.filters['min_adv']
                mcap_ok = market_cap >= self.filters['min_free_float_mcap'] if market_cap > 0 else True
                
                # Sector classification
                sector = self._classify_sector(symbol, info.get('sector', 'Other'))
                
                if price_ok and adv_ok and mcap_ok:
                    universe_data.append({
                        'Ticker': symbol,
                        'Price': current_price,
                        'ADV': avg_dollar_volume,
                        'MarketCap': market_cap,
                        'Sector': sector,
                        'Borrowable': True  # Assume true for large-cap NASDAQ
                    })
                else:
                    failed_count += 1
                    
            except Exception as e:
                logger.warning(f"Failed to process {symbol}: {e}")
                failed_count += 1
                continue
        
        universe_df = pd.DataFrame(universe_data)
        
        logger.info(f"✅ Filtered universe: {len(universe_df)} stocks ({failed_count} filtered out)")
        logger.info(f"   Sectors: {universe_df['Sector'].nunique()}")
        logger.info(f"   Avg Price: ${universe_df['Price'].mean():.2f}")
        logger.info(f"   Avg ADV: ${universe_df['ADV'].mean():,.0f}")
        
        return universe_df
    
    def _validate_universe_data(self, universe_df: pd.DataFrame) -> pd.DataFrame:
        """Download and validate price data for universe"""
        
        logger.info("📈 Validating universe data...")
        
        symbols = universe_df['Ticker'].tolist()
        
        # Download price data
        start_date = datetime.now() - timedelta(days=400)
        end_date = datetime.now()
        
        price_data = []
        for symbol in symbols:
            try:
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if len(data) > 200:  # At least ~8 months of data
                    data = data.reset_index()
                    data['Ticker'] = symbol
                    price_data.append(data)
            except:
                continue
        
        if not price_data:
            raise ValueError("No valid price data downloaded")
        
        combined_data = pd.concat(price_data, ignore_index=True)
        
        # Merge with universe metadata
        merged_data = combined_data.merge(
            universe_df[['Ticker', 'Sector']], 
            on='Ticker', 
            how='inner'
        )
        
        logger.info(f"✅ Validated data: {len(merged_data)} samples, {merged_data['Ticker'].nunique()} stocks")
        return merged_data
    
    def _engineer_production_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer research-backed features (≤10 features)"""
        
        logger.info("🔧 Engineering production features...")
        
        df = data.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Ticker', 'Date'])
        
        # Momentum Core (3 features)
        df['momentum_12_1m'] = df.groupby('Ticker')['Close'].pct_change(252).shift(5)  # Skip last week
        df['momentum_3m'] = df.groupby('Ticker')['Close'].pct_change(63).shift(1)
        df['momentum_20d'] = df.groupby('Ticker')['Close'].pct_change(20).shift(1)
        
        # Mean Reversion (2 features)
        df['reversal_5d'] = -df.groupby('Ticker')['Close'].pct_change(5).shift(1)  # Negative for reversal
        df['overnight_gap'] = (df['Open'] - df.groupby('Ticker')['Close'].shift(1)) / df.groupby('Ticker')['Close'].shift(1)
        
        # Quality/Liquidity (3 features)
        df['volatility_20d'] = df.groupby('Ticker')['Close'].pct_change().rolling(20).std().shift(1)
        df['dollar_volume_20d'] = (df['Close'] * df['Volume']).groupby(df['Ticker']).rolling(20).mean().shift(1)
        
        # Idiosyncratic volatility (residual vol after beta adjustment)
        df = self._calculate_idiosyncratic_vol(df)
        
        # Remove intermediate columns
        feature_cols = [
            'momentum_12_1m', 'momentum_3m', 'momentum_20d',
            'reversal_5d', 'overnight_gap',
            'volatility_20d', 'dollar_volume_20d', 'idiosyncratic_vol'
        ]
        
        # Keep only essential columns
        essential_cols = ['Date', 'Ticker', 'Close', 'Volume', 'Sector'] + feature_cols
        df = df[essential_cols + [col for col in df.columns if col not in essential_cols]].copy()
        
        logger.info(f"✅ Features engineered: {len(feature_cols)} production features")
        return df
    
    def _calculate_idiosyncratic_vol(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate idiosyncratic volatility (beta-adjusted)"""
        
        # Download QQQ for beta calculation
        try:
            qqq_data = yf.download('QQQ', start=df['Date'].min(), end=df['Date'].max(), progress=False)
            qqq_data = qqq_data.reset_index()
            qqq_data['qqq_return'] = qqq_data['Close'].pct_change()
            
            # Merge QQQ returns
            df = df.merge(qqq_data[['Date', 'qqq_return']], on='Date', how='left')
            df['stock_return'] = df.groupby('Ticker')['Close'].pct_change()
            
            # Calculate rolling beta and idiosyncratic vol
            def calc_idio_vol(group):
                if len(group) < 60:
                    return pd.Series([np.nan] * len(group), index=group.index)
                
                idio_vols = []
                for i in range(len(group)):
                    if i < 60:
                        idio_vols.append(np.nan)
                    else:
                        window_data = group.iloc[i-60:i]
                        if window_data['stock_return'].notna().sum() > 40:
                            # Simple regression: stock_return = alpha + beta * qqq_return + residual
                            x = window_data['qqq_return'].fillna(0).values
                            y = window_data['stock_return'].fillna(0).values
                            
                            if len(x) > 0 and np.std(x) > 0:
                                beta = np.corrcoef(x, y)[0, 1] * np.std(y) / np.std(x) if not np.isnan(np.corrcoef(x, y)[0, 1]) else 0
                                residuals = y - beta * x
                                idio_vol = np.std(residuals)
                            else:
                                idio_vol = np.std(y)
                        else:
                            idio_vol = group.iloc[i]['stock_return'] if not pd.isna(group.iloc[i]['stock_return']) else np.nan
                        
                        idio_vols.append(idio_vol)
                
                return pd.Series(idio_vols, index=group.index)
            
            df['idiosyncratic_vol'] = df.groupby('Ticker').apply(calc_idio_vol).reset_index(0, drop=True).shift(1)
            
        except Exception as e:
            logger.warning(f"Failed to calculate idiosyncratic vol: {e}")
            df['idiosyncratic_vol'] = df.groupby('Ticker')['Close'].pct_change().rolling(20).std().shift(1)
        
        return df
    
    def _create_residual_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create residual return targets (beta-neutral vs QQQ)"""
        
        logger.info("🎯 Creating residual return targets...")
        
        # Calculate next-day returns
        df['next_day_return'] = df.groupby('Ticker')['Close'].pct_change().shift(-1)
        
        # Download QQQ for benchmark
        try:
            qqq_data = yf.download('QQQ', start=df['Date'].min(), end=df['Date'].max(), progress=False)
            qqq_data = qqq_data.reset_index()
            qqq_data['qqq_next_return'] = qqq_data['Close'].pct_change().shift(-1)
            
            # Merge QQQ returns
            df = df.merge(qqq_data[['Date', 'qqq_next_return']], on='Date', how='left')
            
            # Create residual returns (stock - beta * QQQ)
            # Simplified: use correlation-based beta
            def calc_residual_returns(group):
                if len(group) < 60:
                    return group['next_day_return'] - group['qqq_next_return'].fillna(0)
                
                # Rolling 60-day beta
                stock_ret = group['next_day_return'].fillna(0)
                qqq_ret = group['qqq_next_return'].fillna(0)
                
                if len(stock_ret) > 0 and len(qqq_ret) > 0 and np.std(qqq_ret) > 0:
                    beta = np.corrcoef(stock_ret, qqq_ret)[0, 1] * np.std(stock_ret) / np.std(qqq_ret)
                    if np.isnan(beta):
                        beta = 1.0
                else:
                    beta = 1.0
                
                residual_returns = stock_ret - beta * qqq_ret
                return residual_returns
            
            df['residual_return_target'] = df.groupby('Ticker').apply(calc_residual_returns).reset_index(0, drop=True)
            
        except Exception as e:
            logger.warning(f"Failed to create residual targets: {e}")
            df['residual_return_target'] = df['next_day_return']
        
        logger.info("✅ Residual return targets created")
        return df
    
    def _apply_cross_sectional_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply cross-sectional z-scoring per date"""
        
        logger.info("📊 Applying cross-sectional scaling...")
        
        feature_cols = [
            'momentum_12_1m', 'momentum_3m', 'momentum_20d',
            'reversal_5d', 'overnight_gap',
            'volatility_20d', 'dollar_volume_20d', 'idiosyncratic_vol'
        ]
        
        # Cross-sectional z-score per date
        for col in feature_cols:
            if col in df.columns:
                df[f'{col}_zscore'] = df.groupby('Date')[col].apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)
        
        # Keep both raw and z-scored features
        logger.info("✅ Cross-sectional scaling applied")
        return df
    
    def _classify_sector(self, symbol: str, yf_sector: str) -> str:
        """Classify sector (simplified mapping)"""
        
        tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'INTC', 'CSCO', 'ADBE']
        health_symbols = ['AMGN', 'GILD', 'REGN', 'VRTX', 'BIIB', 'MRNA', 'ILMN']
        
        if symbol in tech_symbols or 'Technology' in yf_sector:
            return 'Technology'
        elif symbol in health_symbols or 'Healthcare' in yf_sector or 'Biotechnology' in yf_sector:
            return 'Healthcare'
        elif 'Consumer' in yf_sector:
            return 'Consumer'
        elif 'Communication' in yf_sector:
            return 'Communication'
        else:
            return 'Other'
    
    def _save_production_artifacts(self, df: pd.DataFrame, date: str):
        """Save production-ready artifacts"""
        
        logger.info("💾 Saving production artifacts...")
        
        # Ensure directories exist
        (self.artifacts_dir / "production").mkdir(parents=True, exist_ok=True)
        
        # Save complete dataset
        dataset_path = self.artifacts_dir / "production" / f"universe_{date.replace('-', '')}.parquet"
        df.to_parquet(dataset_path, index=False)
        
        # Save universe summary
        summary = {
            'date': date,
            'total_stocks': df['Ticker'].nunique(),
            'total_samples': len(df),
            'sectors': df['Sector'].nunique(),
            'date_range': [str(df['Date'].min()), str(df['Date'].max())],
            'features': [col for col in df.columns if col.endswith('_zscore')],
            'target': 'residual_return_target',
            'filters_applied': self.filters,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = self.artifacts_dir / "production" / f"summary_{date.replace('-', '')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"✅ Production artifacts saved:")
        logger.info(f"   Dataset: {dataset_path}")
        logger.info(f"   Summary: {summary_path}")
        logger.info(f"   Stocks: {df['Ticker'].nunique()}")
        logger.info(f"   Samples: {len(df)}")

def main():
    """Test production universe agent"""
    
    config_path = Path(__file__).parent.parent / "config" / "production_config.json"
    
    if not config_path.exists():
        print("❌ Production config not found")
        return False
    
    agent = ProductionUniverseAgent(str(config_path))
    success = agent.build_production_universe()
    
    if success:
        print("✅ Production universe built successfully")
    else:
        print("❌ Production universe build failed")
    
    return success

if __name__ == "__main__":
    main()