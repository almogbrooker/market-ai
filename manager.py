#!/usr/bin/env python3
"""
Unified AI Trading System Manager
Combines chat-g.txt achievements + chat-g-2.txt LLM enhancements

Single CLI that: builds data → trains with purged CV → backtests with costs → paper-trades (Alpaca)
Uses free data + LLM sentiment features. Prevents leakage. Adds uncertainty & risk controls.
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.data_builder import DataBuilder
from models.model_trainer import ModelTrainer  
from evaluation.backtester import Backtester
from trading.paper_trader import PaperTrader
from utils.logger import setup_logging
from utils.config import ensure_dirs

logger = logging.getLogger(__name__)

class TradingSystemManager:
    """Unified manager for the complete AI trading system"""
    
    def __init__(self):
        self.config = {
            'data_dir': 'artifacts',
            'model_dir': 'artifacts/models',
            'log_dir': 'logs'
        }
        
        # Ensure directories exist
        for dir_path in self.config.values():
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def build_data(self, args):
        """Phase 1: Build comprehensive dataset with LLM features"""
        logger.info("🚀 Phase 1: Building comprehensive dataset with LLM sentiment features")
        
        builder = DataBuilder(
            include_llm=args.include_llm,
            include_macro=args.include_macro,
            start_date=args.start,
            tickers_file=args.tickers_file
        )
        
        # Build the complete dataset
        dataset = builder.build_complete_dataset()
        
        # Save to parquet
        output_path = Path(self.config['data_dir']) / 'daily.parquet'
        dataset.to_parquet(output_path)
        
        # Validation checks
        self._validate_dataset(dataset, output_path)
        
        logger.info(f"✅ Dataset built: {output_path}")
        return output_path
    
    def train(self, args):
        """Phase 2: Train hybrid ensemble with uncertainty"""
        logger.info("🧠 Phase 2: Training hybrid ensemble with uncertainty")
        
        trainer = ModelTrainer(
            dataset_path=args.dataset,
            cv_type=args.cv,
            folds=args.folds,
            purge_days=args.purge,
            embargo_days=args.embargo,
            models=args.models.split(','),
            meta_model=args.meta_model
        )
        
        # Train all models
        results = trainer.train_all()
        
        # Save models
        model_dir = Path(self.config['model_dir']) / 'best'
        trainer.save_models(model_dir)
        
        logger.info(f"✅ Models trained and saved to: {model_dir}")
        return results
    
    def backtest(self, args):
        """Phase 4: Backtest with realistic costs"""
        logger.info("📊 Phase 4: Backtesting with realistic costs")
        
        backtester = Backtester(
            dataset_path=args.dataset,
            model_dir=args.model_dir,
            trade_at=args.trade_at,
            fee_bps=args.fee_bps,
            slip_bps=args.slip_bps,
            short_borrow_bps=args.short_borrow_bps,
            tp_mult=args.tp_mult,
            sl_mult=args.sl_mult,
            timeout=args.timeout
        )
        
        # Run backtest
        results = backtester.run_backtest()
        
        # Generate report
        report_path = Path(self.config['data_dir']) / 'backtest_report.html'
        backtester.generate_report(results, report_path)
        
        logger.info(f"✅ Backtest completed: {report_path}")
        return results
    
    def paper_trade(self, args):
        """Phase 5: Paper trading with Alpaca"""
        logger.info("📈 Phase 5: Starting paper trading")
        
        trader = PaperTrader(
            model_dir=args.model_dir,
            max_gross=args.max_gross,
            max_per_name=args.max_per_name
        )
        
        # Start paper trading
        trader.start_trading()
        
        logger.info("✅ Paper trading started")
    
    def _validate_dataset(self, dataset, output_path):
        """Validate dataset meets requirements"""
        logger.info("🔍 Validating dataset...")
        
        # Check row count
        min_rows = 100000  # Minimum for NASDAQ-100 2018+
        if len(dataset) < min_rows:
            logger.warning(f"Dataset has {len(dataset)} rows, expected >{min_rows}")
        
        # Check for NaNs in model features
        feature_cols = [col for col in dataset.columns if not col.startswith(('Date', 'Ticker', 'target_'))]
        nan_count = dataset[feature_cols].isna().sum().sum()
        if nan_count > 0:
            logger.error(f"Found {nan_count} NaNs in model features")
            raise ValueError("Dataset contains NaN values in model features")
        
        # Check date range
        date_range = dataset['Date'].max() - dataset['Date'].min()
        logger.info(f"Dataset: {len(dataset)} rows, {dataset['Ticker'].nunique()} tickers")
        logger.info(f"Date range: {dataset['Date'].min()} to {dataset['Date'].max()} ({date_range.days} days)")
        
        # Sample rows
        logger.info("Sample data:")
        logger.info(dataset.head(3).to_string())
        
        logger.info("✅ Dataset validation passed")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='AI Trading System Manager')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Build data command
    build_parser = subparsers.add_parser('build-data', help='Build dataset with LLM features')
    build_parser.add_argument('--tickers-file', default='nasdaq100.txt', help='File with ticker symbols')
    build_parser.add_argument('--start', default='2018-01-01', help='Start date')
    build_parser.add_argument('--include-llm', action='store_true', help='Include LLM sentiment features')
    build_parser.add_argument('--include-macro', action='store_true', help='Include macro indicators')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train models with purged CV')
    train_parser.add_argument('--dataset', default='artifacts/daily.parquet', help='Dataset path')
    train_parser.add_argument('--cv', default='purged', choices=['purged', 'time_series'], help='CV type')
    train_parser.add_argument('--folds', type=int, default=5, help='Number of folds')
    train_parser.add_argument('--purge', type=int, default=30, help='Purge days')
    train_parser.add_argument('--embargo', type=int, default=5, help='Embargo days')
    train_parser.add_argument('--models', default='patchtst,itransformer,lstm_small', help='Models to train')
    train_parser.add_argument('--meta-model', default='mlp', help='Meta model type')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Backtest with costs')
    backtest_parser.add_argument('--dataset', default='artifacts/daily.parquet', help='Dataset path')
    backtest_parser.add_argument('--model-dir', default='artifacts/models/best', help='Model directory')
    backtest_parser.add_argument('--trade-at', default='next_open', help='Trade timing')
    backtest_parser.add_argument('--fee-bps', type=float, default=2.0, help='Fee in basis points')
    backtest_parser.add_argument('--slip-bps', type=float, default=7.0, help='Slippage in basis points')
    backtest_parser.add_argument('--short-borrow-bps', type=float, default=100.0, help='Short borrow cost')
    backtest_parser.add_argument('--tp-mult', type=float, default=4.0, help='Take profit multiplier')
    backtest_parser.add_argument('--sl-mult', type=float, default=2.5, help='Stop loss multiplier')
    backtest_parser.add_argument('--timeout', type=int, default=20, help='Position timeout days')
    
    # Paper trade command
    paper_parser = subparsers.add_parser('paper', help='Start paper trading')
    paper_parser.add_argument('--model-dir', default='artifacts/models/best', help='Model directory')
    paper_parser.add_argument('--max-gross', type=float, default=0.6, help='Max gross exposure')
    paper_parser.add_argument('--max-per-name', type=float, default=0.08, help='Max per name exposure')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging and ensure required directories exist
    setup_logging()
    ensure_dirs()

    # Initialize manager
    manager = TradingSystemManager()
    
    # Route to appropriate method
    if args.command == 'build-data':
        manager.build_data(args)
    elif args.command == 'train':
        manager.train(args)
    elif args.command == 'backtest':
        manager.backtest(args)
    elif args.command == 'paper':
        manager.paper_trade(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        parser.print_help()

if __name__ == "__main__":
    main()
