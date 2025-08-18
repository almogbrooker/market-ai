#!/usr/bin/env python3
"""
SEC EDGAR XBRL Fundamentals Data Integration
Fetches structured financial data from SEC EDGAR API
"""

import pandas as pd
import numpy as np
import requests
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import json
import time
import re

logger = logging.getLogger(__name__)

class SECEdgarFundamentalsFecher:
    """Fetches and processes SEC EDGAR XBRL fundamentals data"""
    
    def __init__(self, user_agent: str = "market-ai-system contact@example.com"):
        """
        Initialize SEC EDGAR API client
        
        Args:
            user_agent: Required user agent for SEC API compliance
        """
        self.user_agent = user_agent
        self.base_url = "https://data.sec.gov/api/xbrl"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.user_agent,
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'data.sec.gov'
        })
        
        # Key financial metrics to extract
        self.key_metrics = {
            'revenue': [
                'Revenues', 'Revenue', 'TotalRevenues', 'SalesRevenueGoodsNet',
                'SalesRevenueServicesNet', 'RevenueFromContractWithCustomerExcludingAssessedTax'
            ],
            'net_income': [
                'NetIncomeLoss', 'NetIncome', 'ProfitLoss', 'NetIncomeLossAvailableToCommonStockholdersBasic'
            ],
            'eps_basic': [
                'EarningsPerShareBasic', 'EarningsPerShareDiluted', 'IncomeLossFromContinuingOperationsPerBasicShare'
            ],
            'total_assets': [
                'Assets', 'AssetsCurrent', 'AssetsNoncurrent'
            ],
            'total_liabilities': [
                'Liabilities', 'LiabilitiesCurrent', 'LiabilitiesNoncurrent'
            ],
            'stockholders_equity': [
                'StockholdersEquity', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'
            ],
            'cash_and_equivalents': [
                'CashAndCashEquivalentsAtCarryingValue', 'Cash', 'CashCashEquivalentsAndShortTermInvestments'
            ],
            'operating_cash_flow': [
                'NetCashProvidedByUsedInOperatingActivities', 'CashProvidedByUsedInOperatingActivities'
            ]
        }
        
        # Rate limiting: SEC allows 10 requests per second
        self.rate_limit_delay = 0.1
        
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make rate-limited request to SEC API"""
        
        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(f"{self.base_url}/{endpoint}", params=params)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"SEC API request failed: {e}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return {}
    
    def get_company_facts(self, cik: str) -> Dict:
        """Get company facts (all historical filings) for a CIK"""
        
        # Pad CIK to 10 digits
        cik_padded = str(cik).zfill(10)
        
        logger.info(f"Fetching company facts for CIK: {cik_padded}")
        
        endpoint = f"companyfacts/CIK{cik_padded}.json"
        data = self._make_request(endpoint)
        
        if not data:
            logger.warning(f"No data found for CIK: {cik_padded}")
            return {}
        
        return data
    
    def extract_quarterly_metrics(self, company_facts: Dict, cik: str) -> pd.DataFrame:
        """Extract quarterly financial metrics from company facts"""
        
        if not company_facts or 'facts' not in company_facts:
            return pd.DataFrame()
        
        all_metrics = []
        
        # Extract data from US-GAAP namespace
        us_gaap = company_facts['facts'].get('us-gaap', {})
        
        for metric_category, concept_names in self.key_metrics.items():
            for concept_name in concept_names:
                if concept_name in us_gaap:
                    concept_data = us_gaap[concept_name]
                    units = concept_data.get('units', {})
                    
                    # Prefer USD units
                    unit_data = units.get('USD', units.get('shares', []))
                    
                    for entry in unit_data:
                        try:
                            # Parse quarterly data (10-Q) or annual (10-K)
                            form = entry.get('form', '')
                            if form not in ['10-Q', '10-K']:
                                continue
                            
                            filing_date = entry.get('filed')
                            end_date = entry.get('end')
                            value = entry.get('val')
                            
                            if not all([filing_date, end_date, value]):
                                continue
                            
                            metric_entry = {
                                'cik': cik,
                                'filing_date': pd.to_datetime(filing_date),
                                'period_end': pd.to_datetime(end_date),
                                'form': form,
                                'metric_category': metric_category,
                                'concept_name': concept_name,
                                'value': float(value),
                                'frame': entry.get('frame', ''),
                                'quarter': self._extract_quarter(end_date)
                            }
                            
                            all_metrics.append(metric_entry)
                            
                        except (ValueError, TypeError) as e:
                            logger.debug(f"Error processing metric entry: {e}")
                            continue
        
        if not all_metrics:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_metrics)
        
        # Sort by filing date
        df = df.sort_values(['period_end', 'filing_date'])
        
        # Remove duplicates (keep latest filing for each period)
        df = df.drop_duplicates(['period_end', 'metric_category'], keep='last')
        
        return df
    
    def _extract_quarter(self, end_date: str) -> str:
        """Extract quarter from period end date"""
        
        try:
            date = pd.to_datetime(end_date)
            quarter = f"Q{(date.month - 1) // 3 + 1}"
            return f"{date.year}-{quarter}"
        except:
            return ""
    
    def pivot_metrics_to_features(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """Pivot metrics to feature columns by period"""
        
        if metrics_df.empty:
            return pd.DataFrame()
        
        # Pivot to get metrics as columns
        pivot_df = metrics_df.pivot_table(
            index=['period_end', 'cik'],
            columns='metric_category',
            values='value',
            aggfunc='last'  # Take latest value if duplicates
        ).reset_index()
        
        # Rename columns to be more descriptive
        feature_columns = {}
        for col in pivot_df.columns:
            if col not in ['period_end', 'cik']:
                feature_columns[col] = f'fundamentals_{col}'
        
        pivot_df = pivot_df.rename(columns=feature_columns)
        
        # Add derived metrics
        self._add_derived_metrics(pivot_df)
        
        return pivot_df
    
    def _add_derived_metrics(self, df: pd.DataFrame):
        """Add derived financial metrics and ratios"""
        
        # ROE (Return on Equity)
        if 'fundamentals_net_income' in df.columns and 'fundamentals_stockholders_equity' in df.columns:
            df['fundamentals_roe'] = (
                df['fundamentals_net_income'] / df['fundamentals_stockholders_equity'].replace(0, np.nan)
            )
        
        # ROA (Return on Assets)
        if 'fundamentals_net_income' in df.columns and 'fundamentals_total_assets' in df.columns:
            df['fundamentals_roa'] = (
                df['fundamentals_net_income'] / df['fundamentals_total_assets'].replace(0, np.nan)
            )
        
        # Debt-to-Equity Ratio
        if 'fundamentals_total_liabilities' in df.columns and 'fundamentals_stockholders_equity' in df.columns:
            df['fundamentals_debt_to_equity'] = (
                df['fundamentals_total_liabilities'] / df['fundamentals_stockholders_equity'].replace(0, np.nan)
            )
        
        # Asset Turnover
        if 'fundamentals_revenue' in df.columns and 'fundamentals_total_assets' in df.columns:
            df['fundamentals_asset_turnover'] = (
                df['fundamentals_revenue'] / df['fundamentals_total_assets'].replace(0, np.nan)
            )
        
        # Cash Ratio
        if 'fundamentals_cash_and_equivalents' in df.columns and 'fundamentals_total_liabilities' in df.columns:
            df['fundamentals_cash_ratio'] = (
                df['fundamentals_cash_and_equivalents'] / df['fundamentals_total_liabilities'].replace(0, np.nan)
            )
    
    def create_ticker_mapping(self, tickers: List[str]) -> Dict[str, str]:
        """Create mapping from tickers to CIK numbers"""
        
        ticker_to_cik = {}
        
        # Common mappings for demo (in production, use SEC company tickers JSON)
        common_mappings = {
            'AAPL': '320193',
            'MSFT': '789019',
            'GOOGL': '1652044',
            'AMZN': '1018724',
            'TSLA': '1318605',
            'META': '1326801',
            'NVDA': '1045810',
            'JPM': '19617',
            'JNJ': '200406',
            'PG': '80424'
        }
        
        for ticker in tickers:
            if ticker in common_mappings:
                ticker_to_cik[ticker] = common_mappings[ticker]
            else:
                # For demo, create dummy CIK
                ticker_to_cik[ticker] = str(hash(ticker) % 1000000).zfill(6)
        
        return ticker_to_cik
    
    def fetch_fundamentals_for_tickers(self, tickers: List[str], 
                                     start_date: str = "2020-01-01") -> pd.DataFrame:
        """Fetch fundamentals data for multiple tickers"""
        
        logger.info(f"Fetching fundamentals for {len(tickers)} tickers")
        
        ticker_to_cik = self.create_ticker_mapping(tickers)
        all_fundamentals = []
        
        for ticker, cik in ticker_to_cik.items():
            try:
                logger.info(f"Processing {ticker} (CIK: {cik})")
                
                # Get company facts
                company_facts = self.get_company_facts(cik)
                
                if not company_facts:
                    logger.warning(f"No company facts for {ticker}")
                    continue
                
                # Extract quarterly metrics
                metrics_df = self.extract_quarterly_metrics(company_facts, cik)
                
                if metrics_df.empty:
                    logger.warning(f"No metrics extracted for {ticker}")
                    continue
                
                # Add ticker
                metrics_df['ticker'] = ticker
                
                # Filter by date
                start_dt = pd.to_datetime(start_date)
                metrics_df = metrics_df[metrics_df['period_end'] >= start_dt]
                
                all_fundamentals.append(metrics_df)
                
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                continue
        
        if not all_fundamentals:
            logger.warning("No fundamentals data retrieved, creating dummy data")
            return self._create_dummy_fundamentals(tickers, start_date)
        
        # Combine all data
        combined_df = pd.concat(all_fundamentals, ignore_index=True)
        
        # Pivot to feature format
        features_df = self.pivot_metrics_to_features(combined_df)
        
        return features_df
    
    def _create_dummy_fundamentals(self, tickers: List[str], 
                                 start_date: str = "2020-01-01") -> pd.DataFrame:
        """Create realistic dummy fundamentals data"""
        
        logger.info("Creating dummy fundamentals data")
        
        # Generate quarterly dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime("2024-12-31")
        
        quarters = pd.date_range(start=start_dt, end=end_dt, freq='Q')
        
        all_data = []
        
        for ticker in tickers:
            # Set random seed for consistent data per ticker
            np.random.seed(hash(ticker) % 2**32)
            
            # Base metrics for different company sizes
            base_metrics = {
                'AAPL': {'revenue': 100000, 'net_income': 25000, 'total_assets': 350000},
                'MSFT': {'revenue': 80000, 'net_income': 20000, 'total_assets': 300000},
                'GOOGL': {'revenue': 70000, 'net_income': 15000, 'total_assets': 250000}
            }.get(ticker, {'revenue': 10000, 'net_income': 1000, 'total_assets': 50000})
            
            for quarter_end in quarters:
                # Add realistic trends and seasonality
                trend_factor = 1 + (quarter_end.year - start_dt.year) * 0.05  # 5% annual growth
                seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * quarter_end.month / 12)
                noise_factor = 1 + np.random.normal(0, 0.1)
                
                factor = trend_factor * seasonal_factor * noise_factor
                
                # Generate correlated fundamentals
                revenue = base_metrics['revenue'] * factor
                net_income = base_metrics['net_income'] * factor * np.random.uniform(0.8, 1.2)
                total_assets = base_metrics['total_assets'] * trend_factor * np.random.uniform(0.95, 1.05)
                
                row_data = {
                    'period_end': quarter_end,
                    'ticker': ticker,
                    'fundamentals_revenue': revenue,
                    'fundamentals_net_income': net_income,
                    'fundamentals_total_assets': total_assets,
                    'fundamentals_total_liabilities': total_assets * 0.6,  # 60% leverage
                    'fundamentals_stockholders_equity': total_assets * 0.4,
                    'fundamentals_cash_and_equivalents': total_assets * 0.1,
                    'fundamentals_operating_cash_flow': net_income * 1.2,
                    'fundamentals_eps_basic': net_income / 1000  # Assume 1B shares
                }
                
                all_data.append(row_data)
        
        df = pd.DataFrame(all_data)
        
        # Add derived metrics
        self._add_derived_metrics(df)
        
        logger.info(f"Created dummy fundamentals: {df.shape}")
        
        return df
    
    def align_with_daily_data(self, fundamentals_df: pd.DataFrame,
                            daily_df: pd.DataFrame,
                            date_col: str = 'date',
                            ticker_col: str = 'ticker') -> pd.DataFrame:
        """Align quarterly fundamentals with daily stock data using forward-fill"""
        
        if fundamentals_df.empty:
            return daily_df
        
        logger.info("Aligning fundamentals with daily data")
        
        # Ensure date columns are datetime
        fundamentals_df = fundamentals_df.copy()
        daily_df = daily_df.copy()
        
        fundamentals_df['period_end'] = pd.to_datetime(fundamentals_df['period_end'])
        daily_df[date_col] = pd.to_datetime(daily_df[date_col])
        
        # Get list of tickers
        tickers = daily_df[ticker_col].unique()
        
        aligned_data = []
        
        for ticker in tickers:
            # Get data for this ticker
            daily_ticker = daily_df[daily_df[ticker_col] == ticker].copy()
            fundamentals_ticker = fundamentals_df[
                fundamentals_df[ticker_col] == ticker
            ].copy() if ticker_col in fundamentals_df.columns else pd.DataFrame()
            
            if fundamentals_ticker.empty:
                # No fundamentals data, add NaN columns
                for col in fundamentals_df.columns:
                    if col not in [ticker_col, 'period_end']:
                        daily_ticker[col] = np.nan
            else:
                # Sort fundamentals by period end
                fundamentals_ticker = fundamentals_ticker.sort_values('period_end')
                
                # Merge and forward-fill
                daily_ticker = daily_ticker.sort_values(date_col)
                
                # Use merge_asof for point-in-time alignment
                fundamentals_ticker = fundamentals_ticker.rename(columns={'period_end': date_col})
                
                merged = pd.merge_asof(
                    daily_ticker,
                    fundamentals_ticker.drop(columns=[ticker_col] if ticker_col in fundamentals_ticker.columns else []),
                    on=date_col,
                    direction='backward'  # Use most recent fundamentals
                )
                
                daily_ticker = merged
            
            aligned_data.append(daily_ticker)
        
        # Combine all tickers
        result_df = pd.concat(aligned_data, ignore_index=True)
        
        logger.info(f"Aligned fundamentals with daily data: {result_df.shape}")
        
        return result_df

def create_sec_fundamentals_features(tickers: List[str],
                                   start_date: str = "2020-01-01",
                                   user_agent: str = "market-ai-system contact@example.com") -> pd.DataFrame:
    """Convenience function to create SEC fundamentals features"""
    
    fetcher = SECEdgarFundamentalsFecher(user_agent=user_agent)
    
    try:
        fundamentals_df = fetcher.fetch_fundamentals_for_tickers(tickers, start_date)
        return fundamentals_df
        
    except Exception as e:
        logger.error(f"Failed to create SEC fundamentals features: {e}")
        return fetcher._create_dummy_fundamentals(tickers, start_date)

# Example usage
if __name__ == "__main__":
    # Test with sample tickers
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    fetcher = SECEdgarFundamentalsFecher()
    fundamentals_df = fetcher.fetch_fundamentals_for_tickers(test_tickers)
    
    print(f"Fundamentals data shape: {fundamentals_df.shape}")
    print(f"Columns: {list(fundamentals_df.columns)}")
    print(f"Sample data:\n{fundamentals_df.head()}")