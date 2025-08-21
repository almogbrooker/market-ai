#!/usr/bin/env python3
"""
Daily Logging System for 6-Month Validation
Tracks predictions, realized returns, coverage, and regime changes
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class ValidationLogger:
    """
    Comprehensive daily logging for validation tracking
    """
    
    def __init__(self, logs_dir: Path):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.db_path = self.logs_dir / "validation_logs.db"
        self._init_database()
        
        # Daily logs storage
        self.daily_logs = []
        
    def _init_database(self):
        """Initialize SQLite database for efficient logging"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Daily predictions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_predictions (
                    date TEXT,
                    ticker TEXT,
                    prediction_score REAL,
                    position_size REAL,
                    regime TEXT,
                    regime_multiplier REAL,
                    tradeable INTEGER,
                    confidence_lower REAL,
                    confidence_upper REAL,
                    PRIMARY KEY (date, ticker)
                )
            """)
            
            # Daily returns table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_returns (
                    date TEXT,
                    ticker TEXT,
                    realized_return REAL,
                    predicted_return REAL,
                    position_pnl REAL,
                    in_coverage INTEGER,
                    PRIMARY KEY (date, ticker)
                )
            """)
            
            # Daily portfolio table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_portfolio (
                    date TEXT PRIMARY KEY,
                    portfolio_value REAL,
                    daily_pnl REAL,
                    gross_exposure REAL,
                    net_exposure REAL,
                    n_positions INTEGER,
                    n_tradeable INTEGER,
                    regime TEXT,
                    vix_level REAL,
                    kill_switches_active INTEGER
                )
            """)
            
            # Performance metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    date TEXT PRIMARY KEY,
                    cumulative_return REAL,
                    rolling_sharpe_30d REAL,
                    rolling_drawdown REAL,
                    hit_rate_longs REAL,
                    hit_rate_shorts REAL,
                    ic_daily REAL,
                    coverage_rate REAL
                )
            """)
    
    def log_daily_predictions(self, date: str, predictions: Dict, market_data: pd.DataFrame):
        """Log daily predictions to database"""
        
        with sqlite3.connect(self.db_path) as conn:
            for i, (_, stock) in enumerate(market_data.iterrows()):
                if i < len(predictions['final_scores']):
                    conn.execute("""
                        INSERT OR REPLACE INTO daily_predictions 
                        (date, ticker, prediction_score, position_size, regime, regime_multiplier, 
                         tradeable, confidence_lower, confidence_upper)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        date,
                        stock['Ticker'],
                        float(predictions['final_scores'][i]),
                        float(predictions['position_sizes'][i]),
                        predictions['regime'],
                        float(predictions['regime_multiplier']),
                        int(predictions['trade_filter'][i]),
                        float(predictions.get('lower_bound', [0])[i] if i < len(predictions.get('lower_bound', [])) else 0),
                        float(predictions.get('upper_bound', [0])[i] if i < len(predictions.get('upper_bound', [])) else 0)
                    ))
    
    def log_daily_returns(self, date: str, returns_data: List[Dict]):
        """Log realized returns vs predictions"""
        
        with sqlite3.connect(self.db_path) as conn:
            for return_data in returns_data:
                conn.execute("""
                    INSERT OR REPLACE INTO daily_returns 
                    (date, ticker, realized_return, predicted_return, position_pnl, in_coverage)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    date,
                    return_data['ticker'],
                    return_data['realized_return'],
                    return_data['predicted_return'],
                    return_data['position_pnl'],
                    return_data['in_coverage']
                ))
    
    def log_daily_portfolio(self, date: str, portfolio_metrics: Dict):
        """Log daily portfolio metrics"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO daily_portfolio 
                (date, portfolio_value, daily_pnl, gross_exposure, net_exposure, 
                 n_positions, n_tradeable, regime, vix_level, kill_switches_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                date,
                portfolio_metrics['portfolio_value'],
                portfolio_metrics['daily_pnl'],
                portfolio_metrics['gross_exposure'],
                portfolio_metrics['net_exposure'],
                portfolio_metrics['n_positions'],
                portfolio_metrics['n_tradeable'],
                portfolio_metrics['regime'],
                portfolio_metrics.get('vix_level', 20),
                int(portfolio_metrics['kill_switches_active'])
            ))
    
    def calculate_and_log_performance_metrics(self, date: str):
        """Calculate and log rolling performance metrics"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Get recent data for rolling calculations
            portfolio_df = pd.read_sql("""
                SELECT * FROM daily_portfolio 
                WHERE date <= ? 
                ORDER BY date DESC 
                LIMIT 60
            """, conn, params=[date])
            
            if len(portfolio_df) < 2:
                return
            
            # Calculate metrics
            portfolio_df = portfolio_df.sort_values('date')
            returns = portfolio_df['daily_pnl'] / portfolio_df['portfolio_value'].shift(1)
            returns = returns.dropna()
            
            if len(returns) == 0:
                return
            
            # Cumulative return
            cumulative_return = (portfolio_df['portfolio_value'].iloc[-1] / portfolio_df['portfolio_value'].iloc[0] - 1) * 100
            
            # Rolling Sharpe (30-day)
            if len(returns) >= 30:
                rolling_sharpe = (returns.tail(30).mean() * 252) / (returns.tail(30).std() * np.sqrt(252))
            else:
                rolling_sharpe = 0
            
            # Rolling drawdown
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = ((cumulative - rolling_max) / rolling_max * 100).iloc[-1]
            
            # Hit rates and IC
            returns_df = pd.read_sql("""
                SELECT * FROM daily_returns 
                WHERE date <= ? AND date > ?
                ORDER BY date DESC
            """, conn, params=[date, (pd.to_datetime(date) - timedelta(days=30)).strftime('%Y-%m-%d')])
            
            if len(returns_df) > 0:
                # Hit rates
                longs = returns_df[returns_df['predicted_return'] > 0]
                shorts = returns_df[returns_df['predicted_return'] < 0]
                
                hit_rate_longs = (longs['realized_return'] > 0).mean() if len(longs) > 0 else 0
                hit_rate_shorts = (shorts['realized_return'] < 0).mean() if len(shorts) > 0 else 0
                
                # Information Coefficient
                ic_daily = returns_df['realized_return'].corr(returns_df['predicted_return'])
                ic_daily = ic_daily if not pd.isna(ic_daily) else 0
                
                # Coverage rate
                coverage_rate = returns_df['in_coverage'].mean()
            else:
                hit_rate_longs = hit_rate_shorts = ic_daily = coverage_rate = 0
            
            # Save metrics
            conn.execute("""
                INSERT OR REPLACE INTO performance_metrics 
                (date, cumulative_return, rolling_sharpe_30d, rolling_drawdown, 
                 hit_rate_longs, hit_rate_shorts, ic_daily, coverage_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                date,
                cumulative_return,
                rolling_sharpe,
                drawdown,
                hit_rate_longs,
                hit_rate_shorts,
                ic_daily,
                coverage_rate
            ))
    
    def generate_daily_report(self, date: str) -> Dict:
        """Generate comprehensive daily report"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Portfolio summary
            portfolio_data = pd.read_sql("""
                SELECT * FROM daily_portfolio WHERE date = ?
            """, conn, params=[date])
            
            # Performance metrics
            performance_data = pd.read_sql("""
                SELECT * FROM performance_metrics WHERE date = ?
            """, conn, params=[date])
            
            # Recent predictions
            predictions_data = pd.read_sql("""
                SELECT * FROM daily_predictions WHERE date = ?
            """, conn, params=[date])
            
            report = {
                'date': date,
                'portfolio': portfolio_data.to_dict('records')[0] if len(portfolio_data) > 0 else {},
                'performance': performance_data.to_dict('records')[0] if len(performance_data) > 0 else {},
                'predictions_summary': {
                    'total_predictions': len(predictions_data),
                    'tradeable_positions': predictions_data['tradeable'].sum() if len(predictions_data) > 0 else 0,
                    'avg_prediction_score': predictions_data['prediction_score'].mean() if len(predictions_data) > 0 else 0,
                    'regime': predictions_data['regime'].iloc[0] if len(predictions_data) > 0 else 'unknown'
                }
            }
            
            return report
    
    def export_validation_summary(self, start_date: str, end_date: str) -> Dict:
        """Export comprehensive validation summary"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Load all data
            portfolio_df = pd.read_sql("""
                SELECT * FROM daily_portfolio 
                WHERE date BETWEEN ? AND ?
                ORDER BY date
            """, conn, params=[start_date, end_date])
            
            performance_df = pd.read_sql("""
                SELECT * FROM performance_metrics 
                WHERE date BETWEEN ? AND ?
                ORDER BY date
            """, conn, params=[start_date, end_date])
            
            returns_df = pd.read_sql("""
                SELECT * FROM daily_returns 
                WHERE date BETWEEN ? AND ?
            """, conn, params=[start_date, end_date])
            
            # Calculate summary statistics
            if len(portfolio_df) > 0:
                total_return = (portfolio_df['portfolio_value'].iloc[-1] / portfolio_df['portfolio_value'].iloc[0] - 1) * 100
                avg_daily_positions = portfolio_df['n_positions'].mean()
                avg_gross_exposure = portfolio_df['gross_exposure'].mean()
            else:
                total_return = avg_daily_positions = avg_gross_exposure = 0
            
            if len(performance_df) > 0:
                avg_sharpe = performance_df['rolling_sharpe_30d'].mean()
                max_drawdown = performance_df['rolling_drawdown'].min()
                avg_ic = performance_df['ic_daily'].mean()
                avg_coverage = performance_df['coverage_rate'].mean()
            else:
                avg_sharpe = max_drawdown = avg_ic = avg_coverage = 0
            
            summary = {
                'validation_period': {'start': start_date, 'end': end_date},
                'total_return_pct': total_return,
                'avg_sharpe_ratio': avg_sharpe,
                'max_drawdown_pct': max_drawdown,
                'avg_ic': avg_ic,
                'avg_coverage_rate': avg_coverage,
                'avg_daily_positions': avg_daily_positions,
                'avg_gross_exposure': avg_gross_exposure,
                'total_trading_days': len(portfolio_df),
                'data_quality': {
                    'portfolio_records': len(portfolio_df),
                    'performance_records': len(performance_df),
                    'return_records': len(returns_df)
                }
            }
            
            return summary
    
    def save_csv_exports(self):
        """Export all data to CSV for analysis"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Export all tables to CSV
            tables = ['daily_predictions', 'daily_returns', 'daily_portfolio', 'performance_metrics']
            
            for table in tables:
                df = pd.read_sql(f"SELECT * FROM {table}", conn)
                csv_path = self.logs_dir / f"{table}.csv"
                df.to_csv(csv_path, index=False)
                logger.info(f"Exported {table} to {csv_path}")

def main():
    """Test the logging system"""
    
    logs_dir = Path("validation/logs")
    logger_system = ValidationLogger(logs_dir)
    
    # Test logging
    test_predictions = {
        'final_scores': np.random.normal(0, 0.01, 10),
        'position_sizes': np.random.normal(0, 0.05, 10),
        'trade_filter': np.random.choice([True, False], 10),
        'regime': 'neutral',
        'regime_multiplier': 0.85
    }
    
    test_market_data = pd.DataFrame({
        'Ticker': [f'STOCK{i}' for i in range(10)],
        'Close': np.random.uniform(50, 200, 10)
    })
    
    date = '2023-01-01'
    logger_system.log_daily_predictions(date, test_predictions, test_market_data)
    
    # Test report generation
    report = logger_system.generate_daily_report(date)
    print(f"Daily report: {report}")

if __name__ == "__main__":
    main()