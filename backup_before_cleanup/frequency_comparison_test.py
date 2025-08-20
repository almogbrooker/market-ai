#!/usr/bin/env python3
"""
TRADING FREQUENCY COMPARISON TEST
Comprehensive comparison of 15min, 30min, and 60min trading frequencies vs QQQ benchmark
"""

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrequencyComparisonTest:
    """
    Test and compare different trading frequencies against QQQ benchmark
    Provides institutional-grade analysis of optimal trading intervals
    """
    
    def __init__(self):
        self.frequencies = [15, 30, 60]  # minutes
        self.test_period_days = 30  # Test period
        self.initial_capital = 100000  # $100K starting capital
        
        # Performance metrics storage
        self.results = {}
        
        logger.info("🚀 Trading Frequency Comparison Test Initialized")
        logger.info(f"📅 Test Period: {self.test_period_days} days")
        logger.info(f"💰 Initial Capital: ${self.initial_capital:,}")
    
    def fetch_market_data(self):
        """Fetch market data for testing"""
        logger.info("📊 Fetching market data...")
        
        # Get QQQ benchmark data
        qqq = yf.Ticker("QQQ")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.test_period_days + 10)  # Extra buffer
        
        self.qqq_data = qqq.history(start=start_date, end=end_date, interval="1h")
        
        # Get stock universe data
        stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD']
        self.stock_data = {}
        
        for symbol in stocks:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date, interval="1h")
                if len(data) > 0:
                    self.stock_data[symbol] = data
                    logger.info(f"✅ {symbol}: {len(data)} data points")
            except Exception as e:
                logger.warning(f"⚠️ Could not fetch {symbol}: {e}")
        
        logger.info(f"📈 Fetched data for {len(self.stock_data)} stocks and QQQ")
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for a dataframe"""
        # Returns
        df['return_1h'] = df['Close'].pct_change(1)
        df['return_4h'] = df['Close'].pct_change(4)
        df['return_24h'] = df['Close'].pct_change(24)
        
        # Moving averages
        df['sma_5'] = df['Close'].rolling(5).mean()
        df['sma_10'] = df['Close'].rolling(10).mean()
        df['sma_20'] = df['Close'].rolling(20).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume indicators
        df['volume_sma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        # Volatility
        df['volatility'] = df['return_1h'].rolling(24).std() * np.sqrt(24)
        
        return df
    
    def generate_frequency_signals(self, symbol: str, frequency: int, timestamp: datetime) -> float:
        """Generate trading signals based on frequency"""
        try:
            df = self.stock_data[symbol].copy()
            df = self.calculate_technical_indicators(df)
            
            # Get current data point
            current_idx = df.index.get_indexer([timestamp], method='nearest')[0]
            if current_idx < 20:  # Not enough data
                return 0.0
            
            current = df.iloc[current_idx]
            
            if frequency == 15:
                # High frequency - momentum focused
                signal = 0.0
                
                # Short-term momentum (60%)
                if current['return_1h'] > 0.005:
                    signal += 0.3
                elif current['return_1h'] < -0.005:
                    signal -= 0.3
                
                # Price vs SMA5 (30%)
                price_vs_sma5 = current['Close'] / current['sma_5'] - 1
                if price_vs_sma5 > 0.01:
                    signal += 0.2
                elif price_vs_sma5 < -0.01:
                    signal -= 0.2
                
                # RSI mean reversion (10%)
                if current['rsi'] < 30:
                    signal += 0.1
                elif current['rsi'] > 70:
                    signal -= 0.1
                
            elif frequency == 30:
                # Medium frequency - balanced
                signal = 0.0
                
                # Medium-term momentum (40%)
                if current['return_4h'] > 0.01:
                    signal += 0.25
                elif current['return_4h'] < -0.01:
                    signal -= 0.25
                
                # Trend (30%)
                price_vs_sma10 = current['Close'] / current['sma_10'] - 1
                if price_vs_sma10 > 0.015:
                    signal += 0.2
                elif price_vs_sma10 < -0.015:
                    signal -= 0.2
                
                # RSI (20%)
                if current['rsi'] < 35:
                    signal += 0.15
                elif current['rsi'] > 65:
                    signal -= 0.15
                
                # Volume (10%)
                if current['volume_ratio'] > 1.5:
                    signal += 0.05
                
            else:  # 60 minutes
                # Low frequency - institutional
                signal = 0.0
                
                # Long-term trend (35%)
                price_vs_sma20 = current['Close'] / current['sma_20'] - 1
                if price_vs_sma20 > 0.02:
                    signal += 0.25
                elif price_vs_sma20 < -0.02:
                    signal -= 0.25
                
                # Medium-term momentum (30%)
                if current['return_24h'] > 0.03:
                    signal += 0.2
                elif current['return_24h'] < -0.03:
                    signal -= 0.2
                
                # RSI positioning (25%)
                if current['rsi'] < 40:
                    signal += 0.18
                elif current['rsi'] > 60:
                    signal -= 0.18
                
                # Volatility adjustment (10%)
                if pd.notna(current['volatility']):
                    if current['volatility'] < 0.20:
                        signal += 0.05
                    elif current['volatility'] > 0.40:
                        signal -= 0.05
            
            return np.clip(signal, -1.0, 1.0)
            
        except Exception as e:
            logger.error(f"❌ Error generating signal for {symbol} at {timestamp}: {e}")
            return 0.0
    
    def backtest_frequency(self, frequency: int) -> dict:
        """Backtest a specific trading frequency"""
        logger.info(f"📈 Backtesting {frequency}min frequency...")
        
        # Initialize portfolio
        portfolio_value = self.initial_capital
        positions = {}
        portfolio_history = []
        trades = []
        
        # Set parameters based on frequency
        if frequency == 15:
            position_size = 0.05  # 5%
            min_confidence = 0.75
            max_positions = 8
        elif frequency == 30:
            position_size = 0.075  # 7.5%
            min_confidence = 0.65
            max_positions = 10
        else:  # 60
            position_size = 0.10  # 10%
            min_confidence = 0.60
            max_positions = 12
        
        # Get trading timestamps based on frequency
        qqq_timestamps = self.qqq_data.index
        trading_interval = frequency // 60 if frequency >= 60 else 1  # Convert to hours
        
        if frequency < 60:
            # For sub-hourly, take every nth hour
            step = 60 // frequency
            trading_timestamps = qqq_timestamps[::step]
        else:
            trading_timestamps = qqq_timestamps[::trading_interval]
        
        for timestamp in trading_timestamps:
            try:
                # Generate signals for all stocks
                signals = {}
                for symbol in self.stock_data.keys():
                    signal = self.generate_frequency_signals(symbol, frequency, timestamp)
                    if abs(signal) >= min_confidence:
                        signals[symbol] = signal
                
                # Execute trades
                cash = portfolio_value
                for symbol in positions:
                    if symbol in self.stock_data:
                        current_price = self.stock_data[symbol].loc[timestamp, 'Close'] if timestamp in self.stock_data[symbol].index else positions[symbol]['price']
                        cash -= positions[symbol]['shares'] * current_price
                
                # Close positions not in new signals
                for symbol in list(positions.keys()):
                    if symbol not in signals:
                        if symbol in self.stock_data and timestamp in self.stock_data[symbol].index:
                            sell_price = self.stock_data[symbol].loc[timestamp, 'Close']
                            cash += positions[symbol]['shares'] * sell_price
                            trades.append({
                                'timestamp': timestamp,
                                'symbol': symbol,
                                'action': 'sell',
                                'shares': positions[symbol]['shares'],
                                'price': sell_price
                            })
                        del positions[symbol]
                
                # Open new positions
                new_positions = len([s for s in signals.keys() if s not in positions])
                if new_positions > 0 and len(positions) < max_positions:
                    available_cash = cash * 0.95  # Keep 5% cash buffer
                    
                    for symbol, signal in signals.items():
                        if symbol not in positions and len(positions) < max_positions:
                            if symbol in self.stock_data and timestamp in self.stock_data[symbol].index:
                                current_price = self.stock_data[symbol].loc[timestamp, 'Close']
                                trade_value = available_cash * position_size * abs(signal)
                                shares = int(trade_value / current_price)
                                
                                if shares > 0:
                                    positions[symbol] = {
                                        'shares': shares * np.sign(signal),  # Negative for short
                                        'price': current_price,
                                        'entry_time': timestamp
                                    }
                                    cash -= shares * current_price * abs(np.sign(signal))
                                    trades.append({
                                        'timestamp': timestamp,
                                        'symbol': symbol,
                                        'action': 'buy' if signal > 0 else 'short',
                                        'shares': shares,
                                        'price': current_price
                                    })
                
                # Calculate portfolio value
                current_portfolio_value = cash
                for symbol, pos in positions.items():
                    if symbol in self.stock_data and timestamp in self.stock_data[symbol].index:
                        current_price = self.stock_data[symbol].loc[timestamp, 'Close']
                        current_portfolio_value += pos['shares'] * current_price
                
                portfolio_history.append({
                    'timestamp': timestamp,
                    'portfolio_value': current_portfolio_value,
                    'cash': cash,
                    'positions': len(positions)
                })
                
                portfolio_value = current_portfolio_value
                
            except Exception as e:
                logger.error(f"❌ Error at {timestamp}: {e}")
                continue
        
        # Calculate metrics
        if len(portfolio_history) > 1:
            start_value = portfolio_history[0]['portfolio_value']
            end_value = portfolio_history[-1]['portfolio_value']
            total_return = (end_value - start_value) / start_value
            
            # Calculate Sharpe ratio
            returns = []
            for i in range(1, len(portfolio_history)):
                prev_val = portfolio_history[i-1]['portfolio_value']
                curr_val = portfolio_history[i]['portfolio_value']
                returns.append((curr_val - prev_val) / prev_val)
            
            if len(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            else:
                sharpe = 0
            
            # Max drawdown
            peak = start_value
            max_dd = 0
            for point in portfolio_history:
                if point['portfolio_value'] > peak:
                    peak = point['portfolio_value']
                drawdown = (peak - point['portfolio_value']) / peak
                max_dd = max(max_dd, drawdown)
            
            results = {
                'frequency': frequency,
                'total_return': total_return,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'num_trades': len(trades),
                'portfolio_history': portfolio_history,
                'trades': trades,
                'final_value': end_value
            }
            
            logger.info(f"✅ {frequency}min: {total_return:.2%} return, {sharpe:.2f} Sharpe, {max_dd:.2%} max DD")
            return results
        
        return {}
    
    def calculate_qqq_benchmark(self) -> dict:
        """Calculate QQQ buy-and-hold benchmark"""
        logger.info("📊 Calculating QQQ benchmark...")
        
        if len(self.qqq_data) < 2:
            return {}
        
        start_price = self.qqq_data['Close'].iloc[0]
        end_price = self.qqq_data['Close'].iloc[-1]
        qqq_return = (end_price - start_price) / start_price
        
        # Calculate QQQ portfolio history
        qqq_shares = self.initial_capital / start_price
        qqq_history = []
        
        for timestamp, row in self.qqq_data.iterrows():
            portfolio_value = qqq_shares * row['Close']
            qqq_history.append({
                'timestamp': timestamp,
                'portfolio_value': portfolio_value
            })
        
        # QQQ Sharpe ratio
        qqq_returns = self.qqq_data['Close'].pct_change().dropna()
        qqq_sharpe = np.mean(qqq_returns) / np.std(qqq_returns) * np.sqrt(252) if np.std(qqq_returns) > 0 else 0
        
        # QQQ max drawdown
        qqq_peak = self.qqq_data['Close'].iloc[0]
        qqq_max_dd = 0
        for price in self.qqq_data['Close']:
            if price > qqq_peak:
                qqq_peak = price
            drawdown = (qqq_peak - price) / qqq_peak
            qqq_max_dd = max(qqq_max_dd, drawdown)
        
        logger.info(f"✅ QQQ: {qqq_return:.2%} return, {qqq_sharpe:.2f} Sharpe, {qqq_max_dd:.2%} max DD")
        
        return {
            'total_return': qqq_return,
            'sharpe_ratio': qqq_sharpe,
            'max_drawdown': qqq_max_dd,
            'portfolio_history': qqq_history,
            'final_value': qqq_shares * end_price
        }
    
    def run_comparison_test(self):
        """Run complete frequency comparison test"""
        logger.info("🚀 STARTING COMPREHENSIVE FREQUENCY COMPARISON TEST")
        logger.info("=" * 70)
        
        # Fetch data
        self.fetch_market_data()
        
        # Test each frequency
        for frequency in self.frequencies:
            self.results[frequency] = self.backtest_frequency(frequency)
        
        # Calculate QQQ benchmark
        self.results['QQQ'] = self.calculate_qqq_benchmark()
        
        # Generate report
        self.generate_comparison_report()
        
        # Create visualizations
        self.create_visualizations()
        
        # Save results
        self.save_results()
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        logger.info("\n" + "=" * 70)
        logger.info("📊 TRADING FREQUENCY COMPARISON REPORT")
        logger.info("=" * 70)
        
        # Table header
        print(f"{'Strategy':<12} {'Return':<10} {'Sharpe':<8} {'Max DD':<8} {'Trades':<8} {'Alpha':<8}")
        print("-" * 70)
        
        qqq_return = self.results.get('QQQ', {}).get('total_return', 0)
        
        # QQQ benchmark
        qqq_data = self.results.get('QQQ', {})
        if qqq_data:
            print(f"{'QQQ (BM)':<12} {qqq_data['total_return']:<10.2%} {qqq_data['sharpe_ratio']:<8.2f} {qqq_data['max_drawdown']:<8.2%} {'0':<8} {'0.00%':<8}")
        
        # Frequency results
        for freq in self.frequencies:
            if freq in self.results and self.results[freq]:
                data = self.results[freq]
                alpha = data['total_return'] - qqq_return
                print(f"{f'{freq}min':<12} {data['total_return']:<10.2%} {data['sharpe_ratio']:<8.2f} {data['max_drawdown']:<8.2%} {data['num_trades']:<8} {alpha:<8.2%}")
        
        print("-" * 70)
        
        # Best strategy
        best_freq = max(self.frequencies, 
                       key=lambda f: self.results.get(f, {}).get('total_return', -999))
        
        if best_freq in self.results and self.results[best_freq]:
            best_alpha = self.results[best_freq]['total_return'] - qqq_return
            print(f"\n🏆 BEST STRATEGY: {best_freq}min frequency")
            print(f"   📈 Total Return: {self.results[best_freq]['total_return']:.2%}")
            print(f"   🎯 Alpha vs QQQ: {best_alpha:.2%}")
            print(f"   📊 Sharpe Ratio: {self.results[best_freq]['sharpe_ratio']:.2f}")
            print(f"   📉 Max Drawdown: {self.results[best_freq]['max_drawdown']:.2%}")
            print(f"   🔄 Number of Trades: {self.results[best_freq]['num_trades']}")
        
        # Institutional recommendation
        print("\n" + "=" * 70)
        print("🏛️ INSTITUTIONAL RECOMMENDATION")
        print("=" * 70)
        
        if best_freq == 15:
            print("⚡ HIGH FREQUENCY OPTIMAL: 15-minute intervals recommended")
            print("   • Best for: Active management, momentum strategies")
            print("   • Risk: Higher transaction costs, more volatility")
            print("   • Suitable for: High-touch trading desks")
        elif best_freq == 30:
            print("⚖️ MEDIUM FREQUENCY OPTIMAL: 30-minute intervals recommended")
            print("   • Best for: Balanced risk/return, systematic strategies")
            print("   • Risk: Moderate transaction costs, balanced volatility")
            print("   • Suitable for: Institutional asset managers")
        else:
            print("🏛️ LOW FREQUENCY OPTIMAL: 60-minute intervals recommended")
            print("   • Best for: Long-term alpha, capacity, lower costs")
            print("   • Risk: Lower transaction costs, reduced noise")
            print("   • Suitable for: Large institutional portfolios")
    
    def create_visualizations(self):
        """Create performance visualization charts"""
        logger.info("📈 Creating visualization charts...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Portfolio Performance vs QQQ', 'Return Comparison', 
                          'Risk-Return Profile', 'Trade Frequency Analysis'),
            specs=[[{"secondary_y": False}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Portfolio performance chart
        colors = ['blue', 'green', 'red', 'black']
        
        # Add QQQ benchmark
        if 'QQQ' in self.results and self.results['QQQ']:
            qqq_history = self.results['QQQ']['portfolio_history']
            timestamps = [point['timestamp'] for point in qqq_history]
            values = [point['portfolio_value'] for point in qqq_history]
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=values, name='QQQ Benchmark', 
                          line=dict(color='black', width=2, dash='dash')),
                row=1, col=1
            )
        
        # Add frequency strategies
        for i, freq in enumerate(self.frequencies):
            if freq in self.results and self.results[freq]:
                history = self.results[freq]['portfolio_history']
                timestamps = [point['timestamp'] for point in history]
                values = [point['portfolio_value'] for point in history]
                
                fig.add_trace(
                    go.Scatter(x=timestamps, y=values, name=f'{freq}min Strategy',
                              line=dict(color=colors[i], width=2)),
                    row=1, col=1
                )
        
        # Return comparison bar chart
        strategies = ['QQQ'] + [f'{f}min' for f in self.frequencies]
        returns = []
        
        if 'QQQ' in self.results:
            returns.append(self.results['QQQ'].get('total_return', 0))
        else:
            returns.append(0)
        
        for freq in self.frequencies:
            if freq in self.results and self.results[freq]:
                returns.append(self.results[freq]['total_return'])
            else:
                returns.append(0)
        
        fig.add_trace(
            go.Bar(x=strategies, y=returns, name='Total Returns',
                   marker_color=['black'] + colors[:len(self.frequencies)]),
            row=1, col=2
        )
        
        # Risk-return scatter plot
        sharpe_ratios = []
        max_drawdowns = []
        strategy_names = []
        
        if 'QQQ' in self.results and self.results['QQQ']:
            sharpe_ratios.append(self.results['QQQ']['sharpe_ratio'])
            max_drawdowns.append(self.results['QQQ']['max_drawdown'])
            strategy_names.append('QQQ')
        
        for freq in self.frequencies:
            if freq in self.results and self.results[freq]:
                sharpe_ratios.append(self.results[freq]['sharpe_ratio'])
                max_drawdowns.append(self.results[freq]['max_drawdown'])
                strategy_names.append(f'{freq}min')
        
        fig.add_trace(
            go.Scatter(x=max_drawdowns, y=sharpe_ratios, mode='markers+text',
                      text=strategy_names, textposition="top center",
                      marker=dict(size=12, color=colors[:len(strategy_names)]),
                      name='Risk-Return'),
            row=2, col=1
        )
        
        # Trade frequency analysis
        trade_counts = []
        freq_labels = []
        
        for freq in self.frequencies:
            if freq in self.results and self.results[freq]:
                trade_counts.append(self.results[freq]['num_trades'])
                freq_labels.append(f'{freq}min')
        
        fig.add_trace(
            go.Bar(x=freq_labels, y=trade_counts, name='Number of Trades',
                   marker_color=colors[:len(self.frequencies)]),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Trading Frequency Comparison Analysis",
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_xaxes(title_text="Strategy", row=1, col=2)
        fig.update_yaxes(title_text="Total Return", row=1, col=2)
        fig.update_xaxes(title_text="Max Drawdown", row=2, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=1)
        fig.update_xaxes(title_text="Frequency", row=2, col=2)
        fig.update_yaxes(title_text="Number of Trades", row=2, col=2)
        
        # Save chart
        fig.write_html("frequency_comparison_analysis.html")
        logger.info("📊 Chart saved as frequency_comparison_analysis.html")
    
    def save_results(self):
        """Save detailed results to JSON"""
        # Convert datetime objects to strings for JSON serialization
        results_for_json = {}
        
        for key, value in self.results.items():
            if isinstance(value, dict):
                results_for_json[key] = {}
                for k, v in value.items():
                    if k in ['portfolio_history', 'trades']:
                        # Convert datetime objects in lists
                        if isinstance(v, list):
                            results_for_json[key][k] = []
                            for item in v:
                                if isinstance(item, dict):
                                    converted_item = {}
                                    for ik, iv in item.items():
                                        if isinstance(iv, datetime):
                                            converted_item[ik] = iv.isoformat()
                                        else:
                                            converted_item[ik] = iv
                                    results_for_json[key][k].append(converted_item)
                                else:
                                    results_for_json[key][k].append(item)
                        else:
                            results_for_json[key][k] = v
                    else:
                        results_for_json[key][k] = v
        
        with open('frequency_comparison_results.json', 'w') as f:
            json.dump(results_for_json, f, indent=2)
        
        logger.info("💾 Results saved to frequency_comparison_results.json")

def main():
    """Run the comprehensive frequency comparison test"""
    print("🤖 TRADING FREQUENCY OPTIMIZATION TEST")
    print("=" * 60)
    print("📊 Comparing 15min vs 30min vs 60min vs QQQ Benchmark")
    print("🎯 Institutional-grade analysis for optimal trading frequency")
    print()
    
    try:
        test = FrequencyComparisonTest()
        test.run_comparison_test()
        
        print("\n" + "=" * 60)
        print("✅ FREQUENCY COMPARISON TEST COMPLETED")
        print("📈 View results: frequency_comparison_analysis.html")
        print("💾 Data saved: frequency_comparison_results.json")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()