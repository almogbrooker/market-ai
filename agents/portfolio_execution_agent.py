#!/usr/bin/env python3
"""
PORTFOLIO & EXECUTION AGENT - Chat-G.txt Section 5
Mission: Beta & sector neutral optimization with turnover caps
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PortfolioExecutionAgent:
    """
    Portfolio & Execution Agent - Chat-G.txt Section 5
    Beta & sector neutral optimization with turnover caps
    """
    
    def __init__(self, trading_config: Dict):
        logger.info("📊 PORTFOLIO & EXECUTION AGENT - MARKET NEUTRAL OPTIMIZATION")
        
        self.config = trading_config
        self.base_dir = Path(__file__).parent.parent
        self.artifacts_dir = self.base_dir / "artifacts"
        
        # Portfolio construction parameters
        self.portfolio_config = trading_config['portfolio_construction']
        
        logger.info("🎯 Portfolio Configuration:")
        logger.info(f"   Target Beta: {self.portfolio_config['target_beta']}")
        logger.info(f"   Max Position Size: {self.portfolio_config['max_position_size']:.1%}")
        logger.info(f"   Max Sector Exposure: {self.portfolio_config['max_sector_exposure']:.1%}")
        logger.info(f"   Max Daily Turnover: {self.portfolio_config['max_daily_turnover']:.1%}")
        logger.info(f"   Min Position Size: {self.portfolio_config['min_position_size']:.3%}")
        
    def construct_portfolio(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Construct market-neutral portfolio from model signals
        DoD: Beta within ±0.1, sector exposures <30%, turnover tracking, transaction costs
        """
        
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"🏗️ Constructing portfolio for {date}...")
        
        try:
            # Load model signals
            signals = self._load_model_signals(date)
            if signals is None or signals.empty:
                logger.error("No model signals available")
                return {'success': False, 'reason': 'No signals'}
            
            # Load market data
            market_data = self._load_market_data(signals['Ticker'].tolist())
            if market_data is None:
                logger.error("Failed to load market data")
                return {'success': False, 'reason': 'No market data'}
            
            # Merge signals with market data
            portfolio_data = self._prepare_portfolio_data(signals, market_data)
            
            # Load previous portfolio for turnover calculation
            prev_portfolio = self._load_previous_portfolio(date)
            
            # Optimize portfolio weights
            portfolio_weights = self._optimize_portfolio(portfolio_data, prev_portfolio)
            
            # Apply execution logic
            execution_plan = self._create_execution_plan(portfolio_weights, portfolio_data)
            
            # Calculate expected costs
            cost_analysis = self._calculate_transaction_costs(execution_plan, portfolio_data)
            
            # Generate portfolio analytics
            portfolio_analytics = self._generate_portfolio_analytics(portfolio_weights, portfolio_data)
            
            # Save portfolio artifacts
            self._save_portfolio_artifacts(portfolio_weights, execution_plan, cost_analysis, portfolio_analytics, date)
            
            result = {
                'success': True,
                'portfolio_weights': portfolio_weights,
                'execution_plan': execution_plan,
                'cost_analysis': cost_analysis,
                'analytics': portfolio_analytics,
                'date': date
            }
            
            logger.info("✅ Portfolio construction completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Portfolio construction failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'reason': f'Error: {e}'}
    
    def _load_model_signals(self, date: str) -> Optional[pd.DataFrame]:
        """Load model signals for portfolio construction"""
        
        logger.info("📡 Loading model signals...")
        
        # Load baseline ranker signals
        lgbm_oof_path = self.artifacts_dir / "oof" / "lgbm_oof.parquet"
        
        if not lgbm_oof_path.exists():
            logger.error("No baseline ranker signals found")
            return None
        
        # Load labels to get latest data
        labels_path = self.artifacts_dir / "labels" / "labels.parquet"
        if not labels_path.exists():
            logger.error("No labels data found")
            return None
        
        labels_df = pd.read_parquet(labels_path)
        labels_df['Date'] = pd.to_datetime(labels_df['Date'])
        
        # Get most recent date with complete data
        latest_date = labels_df['Date'].max()
        
        # Filter to latest universe
        latest_universe = labels_df[labels_df['Date'] == latest_date].copy()
        
        if len(latest_universe) == 0:
            logger.error("No universe data for latest date")
            return None
        
        # Load OOF predictions
        oof_data = pd.read_parquet(lgbm_oof_path)
        
        # Create signals dataframe
        signals = latest_universe[['Ticker', 'Date', 'sector']].copy()
        
        # Add model predictions as signals
        if len(oof_data) > 0:
            # Use the last predictions as current signals
            signals['signal'] = np.random.normal(0, 0.02, len(signals))  # Placeholder
            signals['signal_rank'] = signals['signal'].rank(pct=True) - 0.5  # Center around 0
        else:
            signals['signal'] = 0
            signals['signal_rank'] = 0
        
        # Add confidence scores
        signals['confidence'] = np.random.uniform(0.5, 1.0, len(signals))  # Placeholder
        
        logger.info(f"✅ Loaded signals for {len(signals)} stocks")
        return signals
    
    def _load_market_data(self, tickers: List[str]) -> Optional[pd.DataFrame]:
        """Load market data for portfolio construction"""
        
        logger.info("📈 Loading market data...")
        
        try:
            import yfinance as yf
            
            # Download recent data
            market_data = []
            
            for ticker in tickers[:50]:  # Limit for testing
                try:
                    data = yf.download(ticker, period='60d', progress=False)
                    if not data.empty:
                        latest = data.iloc[-1]
                        market_data.append({
                            'Ticker': ticker,
                            'Price': latest['Close'],
                            'Volume': latest['Volume'],
                            'Market_Cap': latest['Close'] * latest['Volume'] * 100,  # Rough estimate
                            'Beta': 1.0 + np.random.normal(0, 0.3),  # Placeholder
                            'ADV': data['Volume'].tail(20).mean() * latest['Close']
                        })
                except:
                    continue
            
            if not market_data:
                return None
            
            market_df = pd.DataFrame(market_data)
            
            logger.info(f"✅ Loaded market data for {len(market_df)} stocks")
            return market_df
            
        except Exception as e:
            logger.error(f"Failed to load market data: {e}")
            return None
    
    def _prepare_portfolio_data(self, signals: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare combined data for portfolio optimization"""
        
        # Merge signals with market data
        portfolio_data = signals.merge(market_data, on='Ticker', how='inner')
        
        # Calculate position sizes based on liquidity
        portfolio_data['max_position_pct'] = np.minimum(
            self.portfolio_config['max_position_size'],
            portfolio_data['ADV'] * 0.1 / portfolio_data['ADV'].sum()  # 10% of ADV
        )
        
        # Sector mapping
        portfolio_data['sector'] = portfolio_data['sector'].fillna('Other')
        
        logger.info(f"✅ Prepared portfolio data for {len(portfolio_data)} stocks")
        return portfolio_data
    
    def _load_previous_portfolio(self, current_date: str) -> Optional[pd.DataFrame]:
        """Load previous portfolio for turnover calculation"""
        
        # Look for previous portfolio file
        portfolio_dir = self.artifacts_dir / "portfolios"
        
        if not portfolio_dir.exists():
            return None
        
        # Find most recent portfolio before current date
        portfolio_files = list(portfolio_dir.glob("portfolio_*.parquet"))
        
        if not portfolio_files:
            return None
        
        # Sort by date and get most recent
        portfolio_files.sort()
        prev_portfolio_path = portfolio_files[-1]
        
        try:
            prev_portfolio = pd.read_parquet(prev_portfolio_path)
            logger.info(f"📂 Loaded previous portfolio: {prev_portfolio_path.name}")
            return prev_portfolio
        except:
            return None
    
    def _optimize_portfolio(self, portfolio_data: pd.DataFrame, prev_portfolio: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Optimize portfolio weights with constraints
        Chat-G.txt: Beta neutral, sector neutral, turnover caps
        """
        
        logger.info("⚖️ Optimizing portfolio weights...")
        
        n_stocks = len(portfolio_data)
        tickers = portfolio_data['Ticker'].tolist()
        
        # Objective: maximize signal * weight - turnover_penalty * |weight_change|
        signal_scores = portfolio_data['signal_rank'].values
        
        # Previous weights (0 if no previous portfolio)
        prev_weights = np.zeros(n_stocks)
        if prev_portfolio is not None:
            prev_dict = dict(zip(prev_portfolio['Ticker'], prev_portfolio['weight']))
            prev_weights = np.array([prev_dict.get(ticker, 0) for ticker in tickers])
        
        # Optimization bounds: [-max_position, +max_position]
        max_pos = portfolio_data['max_position_pct'].values
        bounds = [(-max_pos[i], max_pos[i]) for i in range(n_stocks)]
        
        # Constraints
        constraints = []
        
        # 1. Dollar neutral: sum of weights = 0
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w)
        })
        
        # 2. Beta neutral: beta^T * weights = target_beta
        betas = portfolio_data['Beta'].values
        target_beta = self.portfolio_config['target_beta']
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.dot(betas, w) - target_beta
        })
        
        # 3. Sector neutral: sector exposures < max_sector_exposure
        sectors = portfolio_data['sector'].unique()
        max_sector_exp = self.portfolio_config['max_sector_exposure']
        
        for sector in sectors:
            sector_mask = (portfolio_data['sector'] == sector).values
            constraints.append({
                'type': 'ineq',
                'fun': lambda w, mask=sector_mask: max_sector_exp - np.abs(np.sum(w[mask]))
            })
        
        # 4. Turnover constraint
        max_turnover = self.portfolio_config['max_daily_turnover']
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: max_turnover - np.sum(np.abs(w - prev_weights))
        })
        
        # Objective function
        turnover_penalty = 0.1  # Penalty for turnover
        
        def objective(weights):
            signal_return = -np.dot(signal_scores, weights)  # Negative for minimization
            turnover_cost = turnover_penalty * np.sum(np.abs(weights - prev_weights))
            return signal_return + turnover_cost
        
        # Initial guess: scale signals to respect constraints
        x0 = signal_scores * 0.01  # Small initial positions
        x0 = x0 - np.mean(x0)  # Make dollar neutral
        
        # Optimize
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-8}
            )
            
            if result.success:
                optimal_weights = result.x
                logger.info("✅ Portfolio optimization converged")
            else:
                logger.warning("⚠️ Optimization did not converge, using scaled signals")
                optimal_weights = x0
                
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            optimal_weights = x0
        
        # Create portfolio weights dataframe
        portfolio_weights = pd.DataFrame({
            'Ticker': tickers,
            'weight': optimal_weights,
            'signal': signal_scores,
            'prev_weight': prev_weights,
            'weight_change': optimal_weights - prev_weights
        })
        
        # Filter out tiny positions
        min_position = self.portfolio_config['min_position_size']
        portfolio_weights = portfolio_weights[np.abs(portfolio_weights['weight']) >= min_position].copy()
        
        logger.info(f"✅ Optimized portfolio: {len(portfolio_weights)} positions")
        logger.info(f"   Long positions: {(portfolio_weights['weight'] > 0).sum()}")
        logger.info(f"   Short positions: {(portfolio_weights['weight'] < 0).sum()}")
        logger.info(f"   Total turnover: {np.abs(portfolio_weights['weight_change']).sum():.2%}")
        
        return portfolio_weights
    
    def _create_execution_plan(self, portfolio_weights: pd.DataFrame, portfolio_data: pd.DataFrame) -> pd.DataFrame:
        """Create detailed execution plan with timing and slippage estimates"""
        
        logger.info("📋 Creating execution plan...")
        
        # Merge weights with market data
        execution_plan = portfolio_weights.merge(
            portfolio_data[['Ticker', 'Price', 'ADV', 'sector']], 
            on='Ticker', 
            how='left'
        )
        
        # Calculate notional amounts (assuming $10M portfolio)
        portfolio_value = 10_000_000  # $10M
        execution_plan['notional'] = execution_plan['weight'] * portfolio_value
        execution_plan['shares'] = execution_plan['notional'] / execution_plan['Price']
        
        # Execution timing based on position size
        execution_plan['execution_urgency'] = 'normal'
        execution_plan.loc[np.abs(execution_plan['weight']) > 0.02, 'execution_urgency'] = 'slow'  # Large positions
        execution_plan.loc[np.abs(execution_plan['weight_change']) > 0.01, 'execution_urgency'] = 'fast'  # Large changes
        
        # Expected slippage (bps)
        execution_plan['expected_slippage_bps'] = 5  # Base slippage
        execution_plan.loc[execution_plan['execution_urgency'] == 'fast', 'expected_slippage_bps'] = 8
        execution_plan.loc[execution_plan['execution_urgency'] == 'slow', 'expected_slippage_bps'] = 3
        
        # Order types
        execution_plan['order_type'] = 'limit'
        execution_plan.loc[np.abs(execution_plan['weight_change']) > 0.005, 'order_type'] = 'twap'
        
        logger.info(f"✅ Execution plan created for {len(execution_plan)} orders")
        return execution_plan
    
    def _calculate_transaction_costs(self, execution_plan: pd.DataFrame, portfolio_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive transaction cost analysis"""
        
        logger.info("💰 Calculating transaction costs...")
        
        # Spread costs (bid-ask spread)
        spread_bps = 3  # Average 3 bps spread
        spread_cost = np.abs(execution_plan['notional']).sum() * spread_bps / 10000
        
        # Slippage costs
        slippage_cost = np.sum(
            np.abs(execution_plan['notional']) * execution_plan['expected_slippage_bps'] / 10000
        )
        
        # Borrowing costs for shorts (annualized, daily cost)
        short_positions = execution_plan[execution_plan['weight'] < 0]
        borrow_rate = 0.02  # 2% annual
        borrow_cost = np.abs(short_positions['notional']).sum() * borrow_rate / 365
        
        # Commission costs
        commission_per_share = 0.005  # $0.005 per share
        commission_cost = np.abs(execution_plan['shares']).sum() * commission_per_share
        
        # Total costs
        total_cost = spread_cost + slippage_cost + borrow_cost + commission_cost
        
        cost_analysis = {
            'spread_cost': spread_cost,
            'slippage_cost': slippage_cost,
            'borrow_cost': borrow_cost,
            'commission_cost': commission_cost,
            'total_cost': total_cost,
            'cost_bps': total_cost / (np.abs(execution_plan['notional']).sum()) * 10000
        }
        
        logger.info(f"📊 Transaction Cost Analysis:")
        logger.info(f"   Spread Cost: ${spread_cost:,.0f}")
        logger.info(f"   Slippage Cost: ${slippage_cost:,.0f}")
        logger.info(f"   Borrow Cost: ${borrow_cost:,.0f}")
        logger.info(f"   Commission: ${commission_cost:,.0f}")
        logger.info(f"   Total Cost: ${total_cost:,.0f} ({cost_analysis['cost_bps']:.1f} bps)")
        
        return cost_analysis
    
    def _generate_portfolio_analytics(self, portfolio_weights: pd.DataFrame, portfolio_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive portfolio analytics"""
        
        logger.info("📊 Generating portfolio analytics...")
        
        # Merge for analysis
        analytics_data = portfolio_weights.merge(
            portfolio_data[['Ticker', 'Beta', 'sector', 'Market_Cap']], 
            on='Ticker', 
            how='left'
        )
        
        # Portfolio beta
        portfolio_beta = np.sum(analytics_data['weight'] * analytics_data['Beta'])
        
        # Sector exposures
        sector_exposures = analytics_data.groupby('sector')['weight'].sum().to_dict()
        
        # Long/short breakdown
        long_exposure = analytics_data[analytics_data['weight'] > 0]['weight'].sum()
        short_exposure = analytics_data[analytics_data['weight'] < 0]['weight'].sum()
        gross_exposure = long_exposure - short_exposure  # short_exposure is negative
        net_exposure = long_exposure + short_exposure
        
        # Position concentration
        position_sizes = np.abs(analytics_data['weight'])
        max_position = position_sizes.max()
        top5_concentration = position_sizes.nlargest(5).sum()
        
        # Turnover analysis
        total_turnover = np.abs(analytics_data['weight_change']).sum()
        
        analytics = {
            'portfolio_beta': portfolio_beta,
            'sector_exposures': sector_exposures,
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'gross_exposure': gross_exposure,
            'net_exposure': net_exposure,
            'max_position': max_position,
            'top5_concentration': top5_concentration,
            'total_turnover': total_turnover,
            'num_positions': len(analytics_data),
            'num_long': (analytics_data['weight'] > 0).sum(),
            'num_short': (analytics_data['weight'] < 0).sum()
        }
        
        logger.info(f"📊 Portfolio Analytics:")
        logger.info(f"   Portfolio Beta: {portfolio_beta:.3f}")
        logger.info(f"   Gross Exposure: {gross_exposure:.1%}")
        logger.info(f"   Net Exposure: {net_exposure:.1%}")
        logger.info(f"   Total Turnover: {total_turnover:.1%}")
        logger.info(f"   Positions: {analytics['num_long']} long, {analytics['num_short']} short")
        
        return analytics
    
    def _save_portfolio_artifacts(self, portfolio_weights: pd.DataFrame, execution_plan: pd.DataFrame, 
                                 cost_analysis: Dict, analytics: Dict, date: str):
        """Save all portfolio artifacts"""
        
        logger.info("💾 Saving portfolio artifacts...")
        
        # Ensure portfolios directory exists
        portfolio_dir = self.artifacts_dir / "portfolios"
        portfolio_dir.mkdir(parents=True, exist_ok=True)
        
        # Save portfolio weights
        portfolio_path = portfolio_dir / f"portfolio_{date.replace('-', '')}.parquet"
        portfolio_weights.to_parquet(portfolio_path, index=False)
        
        # Save execution plan
        execution_path = portfolio_dir / f"execution_plan_{date.replace('-', '')}.parquet"
        execution_plan.to_parquet(execution_path, index=False)
        
        # Save analytics and costs
        summary = {
            'date': date,
            'portfolio_analytics': analytics,
            'cost_analysis': cost_analysis,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = portfolio_dir / f"portfolio_summary_{date.replace('-', '')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"✅ Portfolio artifacts saved:")
        logger.info(f"   Weights: {portfolio_path}")
        logger.info(f"   Execution: {execution_path}")
        logger.info(f"   Summary: {summary_path}")

def main():
    """Test the portfolio execution agent"""
    
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "trading_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        logger.error("Trading config not found")
        return False
    
    # Initialize and run agent
    agent = PortfolioExecutionAgent(config)
    result = agent.construct_portfolio()
    
    if result['success']:
        print("✅ Portfolio construction completed successfully")
    else:
        print("❌ Portfolio construction failed")
        print(f"Reason: {result.get('reason', 'Unknown error')}")
    
    return result['success']

if __name__ == "__main__":
    main()