#!/usr/bin/env python3
"""
PORTFOLIO CONSTRUCTOR: Beta-Neutral Long/Short Implementation
Mission Brief Section 5 - Portfolio Simulation with Transaction Costs
"""

import pandas as pd
import numpy as np
import yfinance as yf
import sys
import os
from pathlib import Path
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PortfolioConstructor:
    """
    Beta-neutral long/short portfolio constructor with transaction costs
    """
    
    def __init__(self):
        logger.info("🎯 PORTFOLIO CONSTRUCTOR - BETA-NEUTRAL LONG/SHORT")
        
        # Mission Brief Parameters (Section 5)
        self.long_pct = 0.3             # Top 30% long
        self.short_pct = 0.3            # Bottom 30% short
        self.max_position_size = 0.08   # 8% max single name
        self.target_leverage = 1.0      # 100% gross exposure
        
        # Transaction costs (Mission Brief Section 1)
        self.fee_bps = 3.5              # 2-5 bps fee (mid-point)
        self.slippage_bps = 8.5         # 7-10 bps slippage (mid-point)
        self.short_borrow_rate = 0.01   # 100 bps annualized
        
        # Risk parameters (Mission Brief Section 5)
        self.vol_target = 0.12          # 12% volatility target
        self.max_sector_exposure = 0.3  # 30% sector cap
        self.beta_tolerance = 0.05      # ±5% beta tolerance
        
        # Setup directories
        self.artifacts_dir = Path(__file__).parent.parent / "artifacts" / "portfolio"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🎯 Portfolio Parameters:")
        logger.info(f"   Long/Short: {self.long_pct:.0%}/{self.short_pct:.0%}")
        logger.info(f"   Max position: {self.max_position_size:.0%}")
        logger.info(f"   Transaction costs: {self.fee_bps}bps fee + {self.slippage_bps}bps slippage")
        logger.info(f"   Short borrow: {self.short_borrow_rate:.1%} annualized")
    
    def load_sleeve_c_predictions(self) -> pd.DataFrame:
        """Load Sleeve C OOF predictions"""
        
        logger.info("📊 Loading Sleeve C predictions...")
        
        # Load metadata to get dataset info
        sleeve_c_dir = Path(__file__).parent.parent / "artifacts" / "sleeves" / "sleeve_c"
        metadata_path = sleeve_c_dir / "sleeve_c_metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError("Sleeve C metadata not found - run sleeve_c_trainer.py first")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        dataset_hash = metadata['dataset_hash']
        
        # Load original dataset
        nasdaq_dir = Path(__file__).parent.parent / "artifacts" / "nasdaq_picker"
        dataset_path = nasdaq_dir / f"nasdaq_dataset_{dataset_hash}.csv"
        
        df = pd.read_csv(dataset_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Load OOF predictions
        oof_path = sleeve_c_dir / "sleeve_c_oof_predictions.csv"
        oof_df = pd.read_csv(oof_path)
        
        # Merge predictions with original data
        df['oof_predictions'] = oof_df['oof_predictions']
        df['oof_mask'] = oof_df['oof_mask']
        
        # Keep only samples with predictions
        df = df[df['oof_mask'] == 1].copy()
        
        logger.info(f"✅ Loaded predictions: {len(df)} samples")
        logger.info(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
        logger.info(f"   Tickers: {df['Ticker'].nunique()}")
        
        return df
    
    def download_market_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Download market data for beta calculations"""
        
        logger.info("📈 Downloading market data for beta calculations...")
        
        # Download SPY as market proxy
        spy_data = yf.download('SPY', start=start_date, end=end_date, progress=False)
        spy_data = spy_data.reset_index()
        
        # Handle multi-index columns
        if isinstance(spy_data.columns, pd.MultiIndex):
            spy_data.columns = [col[0] if isinstance(col, tuple) else col for col in spy_data.columns]
        
        spy_data['SPY_Return'] = spy_data['Close'].pct_change()
        
        market_data = spy_data[['Date', 'SPY_Return']].dropna()
        
        logger.info(f"✅ Market data downloaded: {len(market_data)} days")
        
        return market_data
    
    def calculate_betas(self, df: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate 60-day rolling betas vs SPY"""
        
        logger.info("📊 Calculating rolling betas...")
        
        df = df.merge(market_data, on='Date', how='left')
        
        # Calculate 60-day rolling betas per ticker
        def rolling_beta(group):
            group = group.sort_values('Date')
            group['Return_1D'] = group['Close'].pct_change()
            
            # 60-day rolling beta
            group['Beta_60D'] = group['Return_1D'].rolling(60).corr(group['SPY_Return']) * \
                              (group['Return_1D'].rolling(60).std() / group['SPY_Return'].rolling(60).std())
            
            return group
        
        df = df.groupby('Ticker').apply(rolling_beta).reset_index(drop=True)
        
        # Fill missing betas with 1.0 (market beta)
        df['Beta_60D'] = df['Beta_60D'].fillna(1.0)
        
        logger.info("✅ Betas calculated")
        
        return df
    
    def construct_daily_portfolios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construct daily beta-neutral portfolios"""
        
        logger.info("🏗️ Constructing daily beta-neutral portfolios...")
        
        daily_portfolios = []
        
        for date, date_group in df.groupby('Date'):
            if len(date_group) < 10:  # Need minimum stocks
                continue
            
            date_group = date_group.copy()
            
            # Rank predictions (cross-sectional)
            date_group['prediction_rank'] = date_group['oof_predictions'].rank(pct=True)
            
            # Identify long/short candidates
            long_threshold = 1 - self.long_pct  # Top 30%
            short_threshold = self.short_pct    # Bottom 30%
            
            longs = date_group[date_group['prediction_rank'] >= long_threshold].copy()
            shorts = date_group[date_group['prediction_rank'] <= short_threshold].copy()
            
            if len(longs) == 0 or len(shorts) == 0:
                continue
            
            # Calculate equal weights
            n_longs = len(longs)
            n_shorts = len(shorts)
            
            long_weight = 0.5 / n_longs if n_longs > 0 else 0  # 50% total long exposure
            short_weight = -0.5 / n_shorts if n_shorts > 0 else 0  # 50% total short exposure
            
            # Apply position size limits
            long_weight = min(long_weight, self.max_position_size)
            short_weight = max(short_weight, -self.max_position_size)
            
            # Assign weights
            longs['weight'] = long_weight
            shorts['weight'] = short_weight
            
            # Combine portfolio
            portfolio = pd.concat([longs, shorts], ignore_index=True)
            
            # Calculate portfolio beta
            portfolio_beta = (portfolio['weight'] * portfolio['Beta_60D']).sum()
            
            # Beta neutralization (simple approach)
            if abs(portfolio_beta) > self.beta_tolerance:
                # Adjust weights to target zero beta
                beta_adjustment = -portfolio_beta / len(portfolio)
                portfolio['weight'] += beta_adjustment * portfolio['Beta_60D'] / portfolio['Beta_60D'].abs().sum()
            
            # Ensure no position exceeds max size after beta adjustment
            portfolio['weight'] = portfolio['weight'].clip(-self.max_position_size, self.max_position_size)
            
            # Portfolio statistics
            portfolio_info = {
                'Date': date,
                'n_positions': len(portfolio),
                'n_longs': len(longs),
                'n_shorts': len(shorts),
                'gross_exposure': portfolio['weight'].abs().sum(),
                'net_exposure': portfolio['weight'].sum(),
                'portfolio_beta': (portfolio['weight'] * portfolio['Beta_60D']).sum(),
                'max_position': portfolio['weight'].abs().max()
            }
            
            # Add portfolio info to each position
            for col, val in portfolio_info.items():
                portfolio[col] = val
            
            daily_portfolios.append(portfolio)
        
        if not daily_portfolios:
            raise ValueError("No valid portfolios constructed")
        
        portfolio_df = pd.concat(daily_portfolios, ignore_index=True)
        
        logger.info(f"✅ Daily portfolios constructed: {len(daily_portfolios)} dates")
        
        return portfolio_df
    
    def calculate_transaction_costs(self, portfolio_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate transaction costs with turnover"""
        
        logger.info("💰 Calculating transaction costs...")
        
        portfolio_df = portfolio_df.sort_values(['Date', 'Ticker']).copy()
        
        # Calculate position changes (turnover)
        portfolio_df['prev_weight'] = portfolio_df.groupby('Ticker')['weight'].shift(1).fillna(0)
        portfolio_df['weight_change'] = abs(portfolio_df['weight'] - portfolio_df['prev_weight'])
        portfolio_df['is_new_position'] = (portfolio_df['prev_weight'] == 0) & (portfolio_df['weight'] != 0)
        portfolio_df['is_closed_position'] = (portfolio_df['prev_weight'] != 0) & (portfolio_df['weight'] == 0)
        
        # Calculate costs
        # Entry/exit costs: fee + slippage
        portfolio_df['entry_exit_cost'] = portfolio_df['weight_change'] * (self.fee_bps + self.slippage_bps) / 10000
        
        # Short borrow costs (annualized)
        portfolio_df['short_borrow_cost'] = np.where(
            portfolio_df['weight'] < 0,
            abs(portfolio_df['weight']) * self.short_borrow_rate / 252,  # Daily rate
            0
        )
        
        # Total daily cost
        portfolio_df['total_cost'] = portfolio_df['entry_exit_cost'] + portfolio_df['short_borrow_cost']
        
        # Portfolio-level costs
        daily_costs = portfolio_df.groupby('Date').agg({
            'entry_exit_cost': 'sum',
            'short_borrow_cost': 'sum', 
            'total_cost': 'sum',
            'weight_change': 'sum'  # Total turnover
        }).rename(columns={'weight_change': 'daily_turnover'})
        
        portfolio_df = portfolio_df.merge(daily_costs, on='Date', suffixes=('', '_portfolio'))
        
        logger.info("✅ Transaction costs calculated")
        
        return portfolio_df
    
    def simulate_portfolio_performance(self, portfolio_df: pd.DataFrame) -> pd.DataFrame:
        """Simulate portfolio performance with costs"""
        
        logger.info("📈 Simulating portfolio performance...")
        
        # Calculate daily returns
        portfolio_df['Return_1D'] = portfolio_df.groupby('Ticker')['Close'].pct_change()
        
        # Position-level PnL
        portfolio_df['position_pnl'] = portfolio_df['weight'] * portfolio_df['Return_1D']
        
        # Daily portfolio performance
        daily_performance = portfolio_df.groupby('Date').agg({
            'position_pnl': 'sum',
            'total_cost_portfolio': 'first',
            'daily_turnover': 'first',
            'gross_exposure': 'first',
            'net_exposure': 'first',
            'portfolio_beta': 'first',
            'n_positions': 'first'
        }).rename(columns={'position_pnl': 'gross_return'})
        
        # Net returns (after costs)
        daily_performance['net_return'] = daily_performance['gross_return'] - daily_performance['total_cost_portfolio']
        
        # Cumulative performance
        daily_performance['cumulative_gross'] = (1 + daily_performance['gross_return']).cumprod()
        daily_performance['cumulative_net'] = (1 + daily_performance['net_return']).cumprod()
        
        # Performance statistics
        total_days = len(daily_performance)
        total_return_gross = daily_performance['cumulative_gross'].iloc[-1] - 1
        total_return_net = daily_performance['cumulative_net'].iloc[-1] - 1
        
        # Annualized metrics
        daily_vol_gross = daily_performance['gross_return'].std()
        daily_vol_net = daily_performance['net_return'].std()
        annualized_vol_gross = daily_vol_gross * np.sqrt(252)
        annualized_vol_net = daily_vol_net * np.sqrt(252)
        
        sharpe_gross = daily_performance['gross_return'].mean() / daily_vol_gross * np.sqrt(252) if daily_vol_gross > 0 else 0
        sharpe_net = daily_performance['net_return'].mean() / daily_vol_net * np.sqrt(252) if daily_vol_net > 0 else 0
        
        # Max drawdown
        peak_gross = daily_performance['cumulative_gross'].expanding().max()
        drawdown_gross = (daily_performance['cumulative_gross'] - peak_gross) / peak_gross
        max_drawdown_gross = drawdown_gross.min()
        
        peak_net = daily_performance['cumulative_net'].expanding().max()
        drawdown_net = (daily_performance['cumulative_net'] - peak_net) / peak_net
        max_drawdown_net = drawdown_net.min()
        
        # Cost attribution
        total_costs = daily_performance['total_cost_portfolio'].sum()
        avg_daily_turnover = daily_performance['daily_turnover'].mean()
        monthly_turnover = avg_daily_turnover * 21  # 21 trading days per month
        
        performance_stats = {
            'total_days': total_days,
            'total_return_gross': total_return_gross,
            'total_return_net': total_return_net,
            'annualized_vol_gross': annualized_vol_gross,
            'annualized_vol_net': annualized_vol_net,
            'sharpe_gross': sharpe_gross,
            'sharpe_net': sharpe_net,
            'max_drawdown_gross': max_drawdown_gross,
            'max_drawdown_net': max_drawdown_net,
            'total_costs': total_costs,
            'cost_impact': total_return_gross - total_return_net,
            'avg_daily_turnover': avg_daily_turnover,
            'monthly_turnover': monthly_turnover,
            'avg_gross_exposure': daily_performance['gross_exposure'].mean(),
            'avg_net_exposure': daily_performance['net_exposure'].mean(),
            'avg_portfolio_beta': daily_performance['portfolio_beta'].mean(),
            'avg_positions': daily_performance['n_positions'].mean()
        }
        
        logger.info(f"✅ Portfolio simulation completed:")
        logger.info(f"   Total Return (Gross): {total_return_gross:.2%}")
        logger.info(f"   Total Return (Net): {total_return_net:.2%}")
        logger.info(f"   Sharpe (Net): {sharpe_net:.2f}")
        logger.info(f"   Max Drawdown (Net): {max_drawdown_net:.2%}")
        logger.info(f"   Monthly Turnover: {monthly_turnover:.1%}")
        
        return daily_performance, performance_stats
    
    def compare_vs_benchmark(self, daily_performance: pd.DataFrame) -> Dict:
        """Compare vs cash baseline (Mission Brief requirement)"""
        
        logger.info("⚖️ Comparing vs cash baseline...")
        
        # Cash return = 0% (risk-free baseline)
        cash_return = 0.0
        
        portfolio_return = daily_performance['cumulative_net'].iloc[-1] - 1
        alpha = portfolio_return - cash_return
        
        # Also compare vs SPY for context
        start_date = daily_performance.index.min()
        end_date = daily_performance.index.max()
        
        try:
            spy_data = yf.download('SPY', start=start_date, end=end_date, progress=False)
            if isinstance(spy_data.columns, pd.MultiIndex):
                spy_data.columns = [col[0] if isinstance(col, tuple) else col for col in spy_data.columns]
            
            spy_return = (spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[0]) - 1
            spy_alpha = portfolio_return - spy_return
        except:
            spy_return = 0
            spy_alpha = 0
        
        comparison = {
            'portfolio_return': portfolio_return,
            'cash_return': cash_return,
            'alpha_vs_cash': alpha,
            'spy_return': spy_return,
            'alpha_vs_spy': spy_alpha,
            'outperformed_cash': alpha > 0,
            'outperformed_spy': spy_alpha > 0
        }
        
        logger.info(f"✅ Benchmark comparison:")
        logger.info(f"   Portfolio: {portfolio_return:.2%}")
        logger.info(f"   Cash: {cash_return:.2%}")
        logger.info(f"   Alpha vs Cash: {alpha:.2%}")
        logger.info(f"   Alpha vs SPY: {spy_alpha:.2%}")
        
        return comparison
    
    def save_portfolio_artifacts(self, portfolio_df: pd.DataFrame, daily_performance: pd.DataFrame, 
                               performance_stats: Dict, comparison: Dict):
        """Save portfolio artifacts (Mission Brief Section 6)"""
        
        logger.info("💾 Saving portfolio artifacts...")
        
        # Save full portfolio data
        portfolio_path = self.artifacts_dir / "daily_portfolios.csv"
        portfolio_df.to_csv(portfolio_path, index=False)
        
        # Save daily performance
        performance_path = self.artifacts_dir / "daily_performance.csv"
        daily_performance.to_csv(performance_path)
        
        # Save summary statistics
        summary = {
            'performance_stats': performance_stats,
            'benchmark_comparison': comparison,
            'mission_brief_compliance': {
                'beta_neutral': abs(performance_stats['avg_portfolio_beta']) <= 0.05,
                'turnover_reasonable': performance_stats['monthly_turnover'] <= 1.2,  # ≤120%/month
                'position_limits': True,  # Enforced in construction
                'transaction_costs_included': True
            },
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = self.artifacts_dir / "portfolio_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"✅ Portfolio artifacts saved to: {self.artifacts_dir}")

def main():
    """Run portfolio construction following Mission Brief"""
    
    logger.info("=" * 80)
    logger.info("🎯 PORTFOLIO CONSTRUCTOR - BETA-NEUTRAL LONG/SHORT")
    logger.info("=" * 80)
    
    try:
        # Initialize constructor
        constructor = PortfolioConstructor()
        
        # Load Sleeve C predictions
        df = constructor.load_sleeve_c_predictions()
        
        # Download market data for beta calculations
        start_date = df['Date'].min().strftime('%Y-%m-%d')
        end_date = df['Date'].max().strftime('%Y-%m-%d')
        tickers = df['Ticker'].unique().tolist()
        
        market_data = constructor.download_market_data(tickers, start_date, end_date)
        
        # Calculate betas
        df = constructor.calculate_betas(df, market_data)
        
        # Construct daily portfolios
        portfolio_df = constructor.construct_daily_portfolios(df)
        
        # Calculate transaction costs
        portfolio_df = constructor.calculate_transaction_costs(portfolio_df)
        
        # Simulate performance
        daily_performance, performance_stats = constructor.simulate_portfolio_performance(portfolio_df)
        
        # Compare vs benchmark
        comparison = constructor.compare_vs_benchmark(daily_performance)
        
        # Save artifacts
        constructor.save_portfolio_artifacts(portfolio_df, daily_performance, performance_stats, comparison)
        
        # Final assessment
        logger.info("=" * 80)
        logger.info("🏆 PORTFOLIO CONSTRUCTION RESULTS")
        logger.info("=" * 80)
        
        print(f"\n📊 PORTFOLIO PERFORMANCE")
        print(f"{'='*50}")
        print(f"🎯 Portfolio (Net):    {performance_stats['total_return_net']:.2%}")
        print(f"💰 Cash Baseline:      {comparison['cash_return']:.2%}")
        print(f"📈 Alpha vs Cash:      {comparison['alpha_vs_cash']:.2%}")
        print(f"⚡ Sharpe Ratio:       {performance_stats['sharpe_net']:.2f}")
        print(f"📉 Max Drawdown:       {performance_stats['max_drawdown_net']:.2%}")
        print(f"🔄 Monthly Turnover:   {performance_stats['monthly_turnover']:.1%}")
        print(f"⚖️ Avg Portfolio Beta: {performance_stats['avg_portfolio_beta']:.3f}")
        print(f"💸 Cost Impact:        {performance_stats['cost_impact']:.2%}")
        print()
        
        # Mission Brief compliance
        compliance = {
            'Beta Neutral': abs(performance_stats['avg_portfolio_beta']) <= 0.05,
            'Turnover ≤120%/mo': performance_stats['monthly_turnover'] <= 1.2,
            'Costs Included': True,
            'Beat Cash': comparison['alpha_vs_cash'] > 0
        }
        
        print(f"📋 MISSION BRIEF COMPLIANCE:")
        for criterion, passed in compliance.items():
            print(f"   {'✅' if passed else '❌'} {criterion}")
        
        all_passed = all(compliance.values())
        print(f"\n{'✅ PORTFOLIO READY FOR DEPLOYMENT' if all_passed else '⚠️ NEEDS IMPROVEMENT'}")
        
        return portfolio_df, daily_performance, performance_stats
        
    except Exception as e:
        logger.error(f"❌ Portfolio construction failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    portfolio_df, performance, stats = main()