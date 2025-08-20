#!/usr/bin/env python3
"""
COMPREHENSIVE 2025 YTD BACKTEST vs QQQ
Tests maximum performance models from start of year until now
Full production simulation with realistic costs and execution
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import sys
from pathlib import Path
import logging
from typing import Dict, List, Tuple

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from update_bot_max_performance import MaxPerformanceModelLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveBacktest:
    """Comprehensive backtesting system with realistic execution"""
    
    def __init__(self):
        self.start_date = "2025-01-01"
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Trading parameters (realistic)
        self.initial_capital = 100000
        self.transaction_cost = 0.001  # 10 bps per trade
        self.max_position_size = 0.15   # Max 15% in any single stock
        self.rebalance_frequency = 5    # Days between rebalances
        
        # Risk management
        self.max_daily_loss = -0.03     # -3% daily stop loss
        self.portfolio_target_vol = 0.15  # 15% annual volatility target
        
        # Load maximum performance models
        try:
            self.model_loader = MaxPerformanceModelLoader()
            logger.info("✅ Maximum performance models loaded")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            self.model_loader = None
        
        # Portfolio tracking
        self.portfolio_history = []
        self.trade_history = []
        self.daily_returns = []
        
        logger.info(f"🚀 Comprehensive Backtest initialized")
        logger.info(f"   Period: {self.start_date} to {self.end_date}")
        logger.info(f"   Initial capital: ${self.initial_capital:,.0f}")
    
    def download_universe_data(self) -> Dict[str, pd.DataFrame]:
        """Download data for trading universe"""
        
        # High-quality trading universe
        symbols = [
            # Tech leaders
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD',
            'INTC', 'QCOM', 'ORCL', 'CRM', 'ADBE', 'NOW', 'PYPL', 'NFLX',
            
            # Established growth
            'UBER', 'CRWD', 'SNOW', 'DDOG', 'ZM', 'ROKU', 'SQ', 'SHOP',
            
            # Traditional leaders
            'JPM', 'BAC', 'UNH', 'JNJ', 'PG', 'HD', 'WMT', 'DIS', 'MCD', 'KO',
            
            # Add QQQ for benchmark
            'QQQ'
        ]
        
        logger.info(f"📊 Downloading data for {len(symbols)} symbols...")
        
        data = {}
        failed_symbols = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=self.start_date, end=self.end_date, interval='1d')
                
                if len(df) < 20:  # Minimum data requirement
                    failed_symbols.append(symbol)
                    continue
                
                # Calculate comprehensive features
                df = self._calculate_comprehensive_features(df)
                data[symbol] = df
                
                logger.debug(f"✅ {symbol}: {len(df)} days")
                
            except Exception as e:
                logger.error(f"❌ Failed to download {symbol}: {e}")
                failed_symbols.append(symbol)
                continue
        
        if failed_symbols:
            logger.warning(f"Failed to download: {failed_symbols}")
        
        logger.info(f"✅ Downloaded data for {len(data)} symbols")
        return data
    
    def _calculate_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the same features used in training"""
        df = df.copy()
        
        # Core returns
        df['return_1d'] = df['Close'].pct_change()
        df['return_5d'] = df['Close'].pct_change(5)
        df['return_20d'] = df['Close'].pct_change(20)
        
        # Moving averages and ratios
        for period in [10, 20, 50]:
            sma = df['Close'].rolling(period).mean()
            df[f'price_vs_sma_{period}'] = df['Close'] / sma - 1
        
        for period in [10, 20]:
            ema = df['Close'].ewm(span=period).mean()
            df[f'price_vs_ema_{period}'] = df['Close'] / ema - 1
        
        # RSI variants
        for rsi_period in [7, 14]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
            rs = gain / (loss + 1e-8)
            df[f'rsi_{rsi_period}'] = 100 - (100 / (1 + rs))
        
        # MACD system (normalized)
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        macd_histogram = macd - macd_signal
        
        # Z-score normalization
        df['macd_norm'] = (macd - macd.rolling(50).mean()) / (macd.rolling(50).std() + 1e-8)
        df['macd_signal_norm'] = (macd_signal - macd_signal.rolling(50).mean()) / (macd_signal.rolling(50).std() + 1e-8)
        df['macd_histogram_norm'] = (macd_histogram - macd_histogram.rolling(50).mean()) / (macd_histogram.rolling(50).std() + 1e-8)
        
        # Bollinger Bands
        bb_period = 20
        bb_middle = df['Close'].rolling(bb_period).mean()
        bb_std = df['Close'].rolling(bb_period).std()
        bb_upper = bb_middle + (2 * bb_std)
        bb_lower = bb_middle - (2 * bb_std)
        df['bb_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
        df['bb_width_norm'] = (bb_upper - bb_lower) / bb_middle
        
        # Volatility (normalized)
        for period in [10, 20]:
            vol = df['Close'].pct_change().rolling(period).std() * np.sqrt(252)
            vol_ma = vol.rolling(60).mean()
            df[f'volatility_{period}d_norm'] = vol / (vol_ma + 1e-8) - 1
        
        # Volume (normalized)
        volume_ma = df['Volume'].rolling(20).mean()
        df['volume_ratio_norm'] = np.log(df['Volume'] / (volume_ma + 1e-8) + 1e-8)
        
        # OBV (normalized)
        obv = (df['Volume'] * np.sign(df['Close'].diff())).cumsum()
        obv_ma = obv.rolling(20).mean()
        df['obv_ratio_norm'] = obv / (obv_ma + 1e-8) - 1
        
        # Momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}d'] = df['Close'] / df['Close'].shift(period) - 1
        
        # Price action
        df['high_low_pct'] = (df['High'] - df['Low']) / df['Close']
        df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)
        
        # ATR (normalized)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(14).mean()
        df['atr_norm'] = atr / df['Close']
        
        # ROC (normalized)
        roc = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10) * 100
        df['roc_10d_norm'] = (roc - roc.rolling(50).mean()) / (roc.rolling(50).std() + 1e-8)
        
        # Stochastic (normalized)
        low_14 = df['Low'].rolling(14).min()
        high_14 = df['High'].rolling(14).max()
        stoch_k = (df['Close'] - low_14) / (high_14 - low_14 + 1e-8) * 100
        df['stoch_k_norm'] = (stoch_k - 50) / 50  # [-1, 1]
        
        # Clean data
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Robust outlier treatment
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                q99 = df[col].quantile(0.99)
                q01 = df[col].quantile(0.01)
                df[col] = df[col].clip(lower=q01, upper=q99)
        
        return df
    
    def run_backtest(self) -> Dict:
        """Run comprehensive backtest"""
        
        # Download data
        data = self.download_universe_data()
        
        if 'QQQ' not in data:
            raise ValueError("QQQ benchmark data not available")
        
        # Get QQQ benchmark data
        qqq_data = data.pop('QQQ')
        
        # Get trading dates
        trading_dates = qqq_data.index.tolist()
        
        # Initialize portfolio
        portfolio_value = self.initial_capital
        positions = {}  # {symbol: shares}
        cash = self.initial_capital
        
        # Track performance
        portfolio_values = []
        qqq_values = []
        daily_returns = []
        qqq_returns = []
        
        # Initial QQQ value (for benchmark comparison)
        initial_qqq_price = qqq_data.iloc[0]['Close']
        qqq_shares = self.initial_capital / initial_qqq_price
        
        logger.info(f"🎯 Running backtest from {self.start_date} to {self.end_date}")
        logger.info(f"📊 Trading dates: {len(trading_dates)}")
        
        rebalance_counter = 0
        
        for i, date in enumerate(trading_dates):
            try:
                # Update portfolio value
                current_portfolio_value = cash
                
                for symbol, shares in positions.items():
                    if symbol in data and date in data[symbol].index:
                        current_price = data[symbol].loc[date]['Close']
                        current_portfolio_value += shares * current_price
                
                # Calculate daily return
                if i > 0:
                    daily_return = (current_portfolio_value / portfolio_value) - 1
                    daily_returns.append(daily_return)
                    
                    # QQQ return
                    qqq_return = qqq_data.loc[date]['Close'] / qqq_data.iloc[i-1]['Close'] - 1
                    qqq_returns.append(qqq_return)
                    
                    # Risk management: daily stop loss
                    if daily_return < self.max_daily_loss:
                        logger.warning(f"Daily stop loss triggered on {date}: {daily_return:.2%}")
                        # Liquidate positions (simplified)
                        positions = {}
                        cash = current_portfolio_value * 0.97  # 3% slippage penalty
                        current_portfolio_value = cash
                
                portfolio_value = current_portfolio_value
                
                # Track values
                portfolio_values.append(portfolio_value)
                qqq_benchmark_value = self.initial_capital * (qqq_data.loc[date]['Close'] / initial_qqq_price)
                qqq_values.append(qqq_benchmark_value)
                
                # Rebalancing logic
                rebalance_counter += 1
                if rebalance_counter >= self.rebalance_frequency and self.model_loader is not None:
                    rebalance_counter = 0
                    
                    # Generate signals using maximum performance models
                    signals = self._generate_signals(data, date)
                    
                    if signals:
                        # Rebalance portfolio
                        new_positions = self._rebalance_portfolio(
                            signals, data, date, cash, positions
                        )
                        
                        # Update positions and cash
                        positions, cash = new_positions
                
                # Log progress periodically
                if i % 30 == 0:
                    portfolio_return = (portfolio_value / self.initial_capital - 1) * 100
                    qqq_return_pct = (qqq_benchmark_value / self.initial_capital - 1) * 100
                    logger.info(f"📈 {date.strftime('%Y-%m-%d')}: Portfolio={portfolio_return:.1f}%, QQQ={qqq_return_pct:.1f}%")
            
            except Exception as e:
                logger.error(f"Error processing {date}: {e}")
                continue
        
        # Calculate final performance
        total_return = (portfolio_value / self.initial_capital) - 1
        qqq_total_return = (qqq_values[-1] / self.initial_capital) - 1
        
        # Calculate metrics
        portfolio_returns = np.array(daily_returns)
        qqq_returns_array = np.array(qqq_returns)
        
        # Annualized metrics
        trading_days = len(portfolio_returns)
        years = trading_days / 252
        
        portfolio_vol = portfolio_returns.std() * np.sqrt(252)
        qqq_vol = qqq_returns_array.std() * np.sqrt(252)
        
        portfolio_sharpe = (portfolio_returns.mean() * 252) / (portfolio_vol + 1e-8)
        qqq_sharpe = (qqq_returns_array.mean() * 252) / (qqq_vol + 1e-8)
        
        # Maximum drawdown
        portfolio_cumulative = np.cumprod(1 + portfolio_returns)
        portfolio_running_max = np.maximum.accumulate(portfolio_cumulative)
        portfolio_drawdown = (portfolio_cumulative - portfolio_running_max) / portfolio_running_max
        max_drawdown = portfolio_drawdown.min()
        
        # QQQ max drawdown
        qqq_cumulative = np.cumprod(1 + qqq_returns_array)
        qqq_running_max = np.maximum.accumulate(qqq_cumulative)
        qqq_drawdown = (qqq_cumulative - qqq_running_max) / qqq_running_max
        qqq_max_drawdown = qqq_drawdown.min()
        
        results = {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'trading_days': trading_days,
            'initial_capital': self.initial_capital,
            'final_portfolio_value': portfolio_value,
            'final_qqq_value': qqq_values[-1],
            'portfolio_total_return': total_return,
            'qqq_total_return': qqq_total_return,
            'excess_return': total_return - qqq_total_return,
            'portfolio_annualized_return': total_return / years,
            'qqq_annualized_return': qqq_total_return / years,
            'portfolio_volatility': portfolio_vol,
            'qqq_volatility': qqq_vol,
            'portfolio_sharpe': portfolio_sharpe,
            'qqq_sharpe': qqq_sharpe,
            'max_drawdown': max_drawdown,
            'qqq_max_drawdown': qqq_max_drawdown,
            'portfolio_values': portfolio_values,
            'qqq_values': qqq_values,
            'daily_returns': daily_returns,
            'qqq_returns': qqq_returns
        }
        
        return results
    
    def _generate_signals(self, data: Dict[str, pd.DataFrame], date: pd.Timestamp) -> List[Dict]:
        """Generate trading signals using maximum performance models"""
        if self.model_loader is None:
            return []
        
        # Prepare feature data for current date
        feature_data = {}
        
        for symbol, df in data.items():
            try:
                # Get data up to current date
                current_data = df[df.index <= date]
                
                if len(current_data) >= 50:  # Minimum data requirement
                    feature_data[symbol] = current_data
                    
            except Exception as e:
                continue
        
        if not feature_data:
            return []
        
        # Generate predictions
        try:
            predictions = self.model_loader.predict(feature_data)
            
            # Convert to signal format
            signals = []
            for symbol, prediction in predictions.items():
                # Only trade if signal is strong enough
                if abs(prediction) > 0.01:  # 1% threshold
                    signals.append({
                        'symbol': symbol,
                        'signal': prediction,
                        'confidence': min(abs(prediction) * 5, 1.0)
                    })
            
            # Sort by signal strength
            signals.sort(key=lambda x: abs(x['signal']), reverse=True)
            
            # Take top 10 signals
            return signals[:10]
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
    
    def _rebalance_portfolio(self, signals: List[Dict], data: Dict[str, pd.DataFrame], 
                           date: pd.Timestamp, cash: float, positions: Dict[str, int]) -> Tuple:
        """Rebalance portfolio based on signals"""
        
        # Calculate current portfolio value
        portfolio_value = cash
        for symbol, shares in positions.items():
            if symbol in data and date in data[symbol].index:
                current_price = data[symbol].loc[date]['Close']
                portfolio_value += shares * current_price
        
        # Liquidate current positions
        for symbol, shares in positions.items():
            if symbol in data and date in data[symbol].index:
                current_price = data[symbol].loc[date]['Close']
                proceeds = shares * current_price * (1 - self.transaction_cost)
                cash += proceeds
        
        # Clear positions
        new_positions = {}
        
        if not signals:
            return new_positions, cash
        
        # Allocate capital to new positions
        for signal in signals:
            symbol = signal['symbol']
            signal_strength = signal['signal']
            confidence = signal['confidence']
            
            if symbol not in data or date not in data[symbol].index:
                continue
            
            current_price = data[symbol].loc[date]['Close']
            
            # Position sizing based on signal strength and confidence
            base_allocation = min(self.max_position_size, abs(signal_strength) * confidence)
            position_value = portfolio_value * base_allocation
            
            # Account for transaction costs
            position_value *= (1 - self.transaction_cost)
            
            # Calculate shares
            shares = int(position_value / current_price)
            
            if shares > 0:
                actual_cost = shares * current_price * (1 + self.transaction_cost)
                
                if actual_cost <= cash:
                    new_positions[symbol] = shares
                    cash -= actual_cost
        
        return new_positions, cash
    
    def generate_report(self, results: Dict):
        """Generate comprehensive backtest report"""
        
        print("\n" + "="*100)
        print("🎯 COMPREHENSIVE 2025 YTD BACKTEST RESULTS")
        print("="*100)
        print(f"📅 Period: {results['start_date']} to {results['end_date']}")
        print(f"📊 Trading Days: {results['trading_days']}")
        print(f"💰 Initial Capital: ${results['initial_capital']:,.0f}")
        print()
        
        print("📈 PERFORMANCE COMPARISON")
        print("-" * 50)
        print(f"🤖 AI Trading Bot Final Value:    ${results['final_portfolio_value']:,.0f}")
        print(f"📊 QQQ Benchmark Final Value:     ${results['final_qqq_value']:,.0f}")
        print()
        
        print(f"🤖 AI Bot Total Return:           {results['portfolio_total_return']:.2%}")
        print(f"📊 QQQ Total Return:               {results['qqq_total_return']:.2%}")
        print(f"🎯 Excess Return (Alpha):          {results['excess_return']:.2%}")
        print()
        
        # Determine winner
        if results['excess_return'] > 0:
            print("🏆 WINNER: AI TRADING BOT BEATS QQQ! 🏆")
            outperformance = results['excess_return'] * 100
            print(f"   Outperformed QQQ by {outperformance:.1f} percentage points!")
        else:
            print("📊 QQQ benchmark outperformed the AI bot")
            underperformance = abs(results['excess_return']) * 100
            print(f"   Underperformed by {underperformance:.1f} percentage points")
        print()
        
        print("📊 RISK-ADJUSTED METRICS")
        print("-" * 50)
        print(f"🤖 AI Bot Volatility:             {results['portfolio_volatility']:.1%}")
        print(f"📊 QQQ Volatility:                 {results['qqq_volatility']:.1%}")
        print()
        print(f"🤖 AI Bot Sharpe Ratio:           {results['portfolio_sharpe']:.2f}")
        print(f"📊 QQQ Sharpe Ratio:               {results['qqq_sharpe']:.2f}")
        print()
        print(f"🤖 AI Bot Max Drawdown:           {results['max_drawdown']:.1%}")
        print(f"📊 QQQ Max Drawdown:               {results['qqq_max_drawdown']:.1%}")
        print()
        
        # Performance rating
        excess_return = results['excess_return']
        if excess_return > 0.05:
            print("⭐⭐⭐⭐⭐ EXCEPTIONAL PERFORMANCE!")
        elif excess_return > 0.02:
            print("⭐⭐⭐⭐ EXCELLENT PERFORMANCE!")
        elif excess_return > 0:
            print("⭐⭐⭐ GOOD PERFORMANCE!")
        elif excess_return > -0.02:
            print("⭐⭐ FAIR PERFORMANCE")
        else:
            print("⭐ NEEDS IMPROVEMENT")
        print()
        
        print("🎯 MODEL PERFORMANCE VALIDATION")
        print("-" * 50)
        if hasattr(self, 'model_loader') and self.model_loader:
            print(f"✅ Maximum Performance Models Used (IC=0.0324)")
            print(f"✅ LSTM + LightGBM Ensemble")
            print(f"✅ 40-day sequences with 27 features")
            print(f"✅ Beta-neutral cross-sectional ranking")
        else:
            print("⚠️  No models loaded - using simple strategy")
        print()
        
        print("="*100)
        
        # Create performance chart
        self._create_performance_chart(results)
    
    def _create_performance_chart(self, results: Dict):
        """Create performance comparison chart"""
        
        try:
            plt.style.use('dark_background')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # Cumulative returns
            dates = pd.date_range(start=results['start_date'], end=results['end_date'], freq='D')[:len(results['portfolio_values'])]
            
            portfolio_returns = np.array(results['portfolio_values']) / results['initial_capital'] * 100
            qqq_returns = np.array(results['qqq_values']) / results['initial_capital'] * 100
            
            ax1.plot(dates, portfolio_returns, label='AI Trading Bot', color='#00ff41', linewidth=2)
            ax1.plot(dates, qqq_returns, label='QQQ Benchmark', color='#ff6b6b', linewidth=2)
            ax1.set_title('📈 2025 YTD Performance: AI Bot vs QQQ Benchmark', fontsize=16, fontweight='bold')
            ax1.set_ylabel('Portfolio Value (%)', fontsize=12)
            ax1.legend(fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Daily returns distribution
            ax2.hist(results['daily_returns'], bins=50, alpha=0.7, label='AI Bot', color='#00ff41')
            ax2.hist(results['qqq_returns'], bins=50, alpha=0.7, label='QQQ', color='#ff6b6b')
            ax2.set_title('📊 Daily Returns Distribution', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Daily Return', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.legend(fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('ytd_backtest_results.png', dpi=300, bbox_inches='tight', facecolor='black')
            plt.show()
            
            print("📊 Performance chart saved as 'ytd_backtest_results.png'")
            
        except Exception as e:
            logger.error(f"Error creating chart: {e}")

def main():
    """Run comprehensive YTD backtest"""
    
    backtest = ComprehensiveBacktest()
    
    try:
        # Run backtest
        logger.info("🚀 Starting comprehensive YTD backtest...")
        results = backtest.run_backtest()
        
        # Generate report
        backtest.generate_report(results)
        
        # Save results
        import json
        with open('ytd_backtest_results.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = results.copy()
            json_results['portfolio_values'] = [float(x) for x in results['portfolio_values']]
            json_results['qqq_values'] = [float(x) for x in results['qqq_values']]
            json_results['daily_returns'] = [float(x) for x in results['daily_returns']]
            json_results['qqq_returns'] = [float(x) for x in results['qqq_returns']]
            
            json.dump(json_results, f, indent=2, default=str)
        
        logger.info("📁 Results saved to 'ytd_backtest_results.json'")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()