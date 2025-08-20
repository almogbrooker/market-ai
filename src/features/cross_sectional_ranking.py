#!/usr/bin/env python3
"""
CROSS-SECTIONAL RANKING AND BETA-NEUTRAL TARGET GENERATION
Critical upgrade to remove market beta and generate pure alpha signals
Based on institutional factor modeling and cross-sectional analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import yfinance as yf
import logging
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class CrossSectionalRanker:
    """
    Cross-sectional ranking system for alpha generation
    Removes market beta and sector effects to isolate stock-specific alpha
    """
    
    def __init__(self, universe_size: int = 100):
        self.universe_size = universe_size
        self.market_data = {}
        self.sector_data = {}
        self.factor_loadings = {}
        
        # Standard factor definitions
        self.factors = {
            'market': 'SPY',  # Market factor
            'tech': 'XLK',    # Technology sector
            'finance': 'XLF', # Financial sector  
            'health': 'XLV',  # Healthcare sector
            'energy': 'XLE'   # Energy sector
        }
        
        logger.info("🎯 Cross-sectional ranker initialized")
    
    def load_factor_data(self, lookback_days: int = 252) -> Dict[str, pd.DataFrame]:
        """Load factor return data for beta calculation"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 30)
        
        factor_returns = {}
        
        for factor_name, ticker in self.factors.items():
            try:
                factor_data = yf.Ticker(ticker)
                df = factor_data.history(start=start_date, end=end_date)
                
                if len(df) > 50:
                    df['return_1d'] = df['Close'].pct_change()
                    factor_returns[factor_name] = df[['Close', 'return_1d']].dropna()
                    logger.debug(f"✅ Loaded {factor_name} factor: {len(df)} days")
                else:
                    logger.warning(f"⚠️ Insufficient data for {factor_name}")
                    
            except Exception as e:
                logger.error(f"❌ Error loading {factor_name}: {e}")
                continue
        
        return factor_returns
    
    def calculate_beta_exposures(self, stock_returns: pd.Series, 
                               factor_returns: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate beta exposures to various factors"""
        betas = {}
        
        # Align dates
        common_dates = stock_returns.index
        for factor_name in factor_returns:
            common_dates = common_dates.intersection(factor_returns[factor_name].index)
        
        if len(common_dates) < 30:
            logger.warning("Insufficient overlapping data for beta calculation")
            return {'market': 1.0, 'tech': 0.0, 'finance': 0.0, 'health': 0.0, 'energy': 0.0}
        
        # Calculate betas using linear regression
        aligned_stock_returns = stock_returns.loc[common_dates]
        
        for factor_name, factor_data in factor_returns.items():
            try:
                aligned_factor_returns = factor_data['return_1d'].loc[common_dates]
                
                # Remove NaN values
                valid_mask = ~(pd.isna(aligned_stock_returns) | pd.isna(aligned_factor_returns))
                if valid_mask.sum() < 20:
                    betas[factor_name] = 0.0
                    continue
                
                stock_clean = aligned_stock_returns[valid_mask]
                factor_clean = aligned_factor_returns[valid_mask]
                
                # Linear regression: stock_return = alpha + beta * factor_return + error
                if len(stock_clean) > 10 and factor_clean.std() > 1e-8:
                    beta = np.cov(stock_clean, factor_clean)[0, 1] / np.var(factor_clean)
                    betas[factor_name] = float(beta)
                else:
                    betas[factor_name] = 0.0
                    
            except Exception as e:
                logger.error(f"Error calculating {factor_name} beta: {e}")
                betas[factor_name] = 0.0
        
        return betas
    
    def neutralize_returns(self, stock_returns: Dict[str, pd.Series], 
                          symbols: List[str]) -> Dict[str, pd.Series]:
        """
        Create beta-neutral return targets by removing factor exposures
        This is the key function for generating pure alpha targets
        """
        logger.info("🔧 Creating beta-neutral targets...")
        
        # Load factor data
        factor_returns = self.load_factor_data()
        
        if not factor_returns:
            logger.warning("No factor data available - returning raw returns")
            return stock_returns
        
        neutralized_returns = {}
        
        for symbol in symbols:
            if symbol not in stock_returns:
                continue
                
            try:
                # Calculate beta exposures
                betas = self.calculate_beta_exposures(stock_returns[symbol], factor_returns)
                
                # Create neutralized returns by subtracting factor contributions
                original_returns = stock_returns[symbol].copy()
                neutralized = original_returns.copy()
                
                # Get common dates across all factors
                common_dates = original_returns.index
                for factor_data in factor_returns.values():
                    common_dates = common_dates.intersection(factor_data.index)
                
                if len(common_dates) < 30:
                    neutralized_returns[symbol] = original_returns
                    continue
                
                # Subtract each factor contribution: return_neutral = return_raw - beta * factor_return
                for factor_name, beta in betas.items():
                    if factor_name in factor_returns and abs(beta) > 1e-6:
                        factor_ret = factor_returns[factor_name]['return_1d'].loc[common_dates]
                        neutralized.loc[common_dates] -= beta * factor_ret
                
                neutralized_returns[symbol] = neutralized
                
                # Log neutralization impact
                if len(common_dates) > 30:
                    original_vol = original_returns.loc[common_dates].std()
                    neutral_vol = neutralized.loc[common_dates].std()
                    logger.debug(f"{symbol}: vol {original_vol:.4f} → {neutral_vol:.4f} (β_mkt: {betas.get('market', 0):.2f})")
                
            except Exception as e:
                logger.error(f"Error neutralizing {symbol}: {e}")
                neutralized_returns[symbol] = stock_returns[symbol]
        
        logger.info(f"✅ Neutralized returns for {len(neutralized_returns)} stocks")
        return neutralized_returns
    
    def rank_cross_sectional_signals(self, raw_signals: Dict[str, float], 
                                   date: datetime = None) -> Dict[str, float]:
        """
        Convert raw signals to cross-sectional ranks
        This removes market-wide effects and focuses on relative performance
        """
        if not raw_signals:
            return {}
        
        # Convert to pandas Series for easier manipulation
        signals_series = pd.Series(raw_signals)
        
        # Remove extreme outliers (beyond 3 standard deviations)
        mean_signal = signals_series.mean()
        std_signal = signals_series.std()
        
        if std_signal > 1e-8:
            # Cap extreme values
            signals_series = signals_series.clip(
                lower=mean_signal - 3 * std_signal,
                upper=mean_signal + 3 * std_signal
            )
        
        # Cross-sectional ranking (0 to 1)
        ranked_signals = signals_series.rank(pct=True) - 0.5  # Center around 0
        ranked_signals *= 2  # Scale to [-1, 1]
        
        # Convert back to dictionary
        ranked_dict = ranked_signals.to_dict()
        
        logger.debug(f"📊 Cross-sectional ranking: {len(raw_signals)} signals ranked")
        return ranked_dict
    
    def create_long_short_portfolio(self, ranked_signals: Dict[str, float],
                                  top_pct: float = 0.2, bottom_pct: float = 0.2) -> Dict[str, float]:
        """
        Create market-neutral long/short portfolio from ranked signals
        
        Args:
            ranked_signals: Cross-sectionally ranked signals [-1, 1]
            top_pct: Percentage of stocks to go long (top performers)
            bottom_pct: Percentage of stocks to go short (bottom performers)
        """
        if not ranked_signals:
            return {}
        
        signals_series = pd.Series(ranked_signals)
        n_stocks = len(signals_series)
        
        # Calculate thresholds
        long_threshold = signals_series.quantile(1 - top_pct)
        short_threshold = signals_series.quantile(bottom_pct)
        
        portfolio_weights = {}
        
        for symbol, signal in ranked_signals.items():
            if signal >= long_threshold:
                # Long position - scaled by signal strength
                portfolio_weights[symbol] = signal * 0.5  # Max 0.5 weight
            elif signal <= short_threshold:
                # Short position - scaled by signal strength
                portfolio_weights[symbol] = signal * 0.5  # Max -0.5 weight  
            else:
                # Neutral - no position
                portfolio_weights[symbol] = 0.0
        
        # Ensure dollar neutrality (long $ = short $)
        long_weights = {k: v for k, v in portfolio_weights.items() if v > 0}
        short_weights = {k: v for k, v in portfolio_weights.items() if v < 0}
        
        long_sum = sum(long_weights.values())
        short_sum = abs(sum(short_weights.values()))
        
        if long_sum > 0 and short_sum > 0:
            # Scale to ensure dollar neutrality
            scaling_factor = min(long_sum, short_sum)
            
            for symbol in portfolio_weights:
                if portfolio_weights[symbol] > 0:
                    portfolio_weights[symbol] *= scaling_factor / long_sum
                elif portfolio_weights[symbol] < 0:
                    portfolio_weights[symbol] *= scaling_factor / short_sum
        
        active_positions = {k: v for k, v in portfolio_weights.items() if abs(v) > 1e-6}
        
        logger.info(f"📊 Market-neutral portfolio: {len(active_positions)} positions")
        logger.info(f"   Long positions: {len([v for v in active_positions.values() if v > 0])}")
        logger.info(f"   Short positions: {len([v for v in active_positions.values() if v < 0])}")
        
        return portfolio_weights

class BetaNeutralTargetGenerator:
    """Generate beta-neutral training targets for model training"""
    
    def __init__(self):
        self.ranker = CrossSectionalRanker()
    
    def generate_training_targets(self, price_data: Dict[str, pd.DataFrame], 
                                forward_days: int = 5) -> pd.DataFrame:
        """
        Generate beta-neutral training targets
        
        Args:
            price_data: Dict of {symbol: OHLCV DataFrame}
            forward_days: Days ahead to predict
            
        Returns:
            DataFrame with columns: [date, symbol, raw_return, beta_neutral_return]
        """
        logger.info(f"🎯 Generating beta-neutral targets (forward_days={forward_days})")
        
        # Calculate raw returns
        stock_returns = {}
        
        for symbol, df in price_data.items():
            if len(df) < forward_days + 10:
                continue
                
            # Forward returns
            df = df.copy()
            df[f'forward_return_{forward_days}d'] = (
                df['Close'].shift(-forward_days) / df['Close'] - 1
            )
            
            stock_returns[symbol] = df[f'forward_return_{forward_days}d'].dropna()
        
        if not stock_returns:
            logger.error("No stock returns calculated")
            return pd.DataFrame()
        
        # Beta-neutralize returns
        symbols = list(stock_returns.keys())
        neutralized_returns = self.ranker.neutralize_returns(stock_returns, symbols)
        
        # Convert to training DataFrame
        training_data = []
        
        # Get common dates across all stocks
        all_dates = set()
        for returns in neutralized_returns.values():
            all_dates.update(returns.index)
        
        common_dates = sorted(all_dates)
        
        for date in common_dates:
            date_returns = {}
            
            for symbol in symbols:
                if symbol in stock_returns and date in stock_returns[symbol].index:
                    raw_ret = stock_returns[symbol].loc[date]
                    
                    if symbol in neutralized_returns and date in neutralized_returns[symbol].index:
                        neutral_ret = neutralized_returns[symbol].loc[date]
                    else:
                        neutral_ret = raw_ret
                    
                    if not (pd.isna(raw_ret) or pd.isna(neutral_ret)):
                        date_returns[symbol] = neutral_ret
            
            # Cross-sectional ranking for this date
            if len(date_returns) >= 10:  # Need minimum stocks for ranking
                ranked_returns = self.ranker.rank_cross_sectional_signals(date_returns, date)
                
                for symbol, ranked_return in ranked_returns.items():
                    if symbol in stock_returns and date in stock_returns[symbol].index:
                        training_data.append({
                            'date': date,
                            'symbol': symbol,
                            'raw_return': stock_returns[symbol].loc[date],
                            'beta_neutral_return': ranked_return,
                            'forward_days': forward_days
                        })
        
        training_df = pd.DataFrame(training_data)
        
        if len(training_df) > 0:
            logger.info(f"✅ Generated {len(training_df)} beta-neutral training samples")
            logger.info(f"   Date range: {training_df['date'].min()} to {training_df['date'].max()}")
            logger.info(f"   Symbols: {training_df['symbol'].nunique()}")
            
            # Save for inspection
            training_df.to_csv(f'beta_neutral_targets_{forward_days}d.csv', index=False)
            
        return training_df

# Factory functions for easy integration
def create_cross_sectional_ranker() -> CrossSectionalRanker:
    """Create a cross-sectional ranker with default settings"""
    return CrossSectionalRanker(universe_size=100)

def create_beta_neutral_targets(symbols: List[str], 
                              lookback_days: int = 365,
                              forward_days: int = 5) -> pd.DataFrame:
    """
    Quick function to generate beta-neutral targets
    
    Args:
        symbols: List of stock symbols
        lookback_days: Days of historical data to use
        forward_days: Days ahead to predict
    
    Returns:
        DataFrame with beta-neutral training targets
    """
    generator = BetaNeutralTargetGenerator()
    
    # Download price data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days + forward_days + 30)
    
    price_data = {}
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if len(df) > 50:
                price_data[symbol] = df
                
        except Exception as e:
            logger.error(f"Error downloading {symbol}: {e}")
            continue
    
    return generator.generate_training_targets(price_data, forward_days)