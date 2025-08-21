#!/usr/bin/env python3
"""
PAPER TRADING AGENT - wraps src.trading.paper_trader.PaperTrader
Exposes methods to start/stop trading and report P&L
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from src.trading.paper_trader import PaperTrader
from agents.monitoring_reporting_agent import MonitoringReportingAgent

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PaperTradingAgent:
    """Agent to manage paper trading using PaperTrader"""

    def __init__(self, trading_config: Dict[str, Any]):
        logger.info("📝 PAPER TRADING AGENT INIT")

        self.config = trading_config
        paper_cfg = trading_config.get('paper_trading', {})

        # Validation window (default 6 months)
        self.validation_window_months = paper_cfg.get('validation_window_months', 6)
        logger.info(f"   Validation window: {self.validation_window_months} months")

        # Alpaca credentials
        alpaca_cfg = paper_cfg.get('alpaca', {})
        api_key = alpaca_cfg.get('api_key')
        secret_key = alpaca_cfg.get('secret_key')
        base_url = alpaca_cfg.get('base_url')

        # Initialize PaperTrader
        model_dir = paper_cfg.get('model_dir', 'artifacts/models/best')
        self.trader = PaperTrader(
            model_dir=model_dir,
            max_gross=paper_cfg.get('max_gross', 0.6),
            max_per_name=paper_cfg.get('max_per_name', 0.08),
            api_key=api_key,
            secret_key=secret_key,
            base_url=base_url,
        )

        # Monitoring/reporting
        self.monitor = MonitoringReportingAgent(trading_config)
        self.peak_portfolio_value = self.trader.portfolio_value

    def start_trading(self) -> None:
        """Begin paper trading"""
        logger.info("🚀 Starting paper trading")
        self.trader.start_trading()

    def stop_trading(self) -> None:
        """Stop paper trading"""
        logger.info("🛑 Stopping paper trading")
        self.trader.is_trading = False
        self.trader._cleanup()

    def report_pnl(self) -> Dict[str, Any]:
        """Report current P&L and capture monitoring metrics"""
        self.trader._update_portfolio_info()

        # Calculate exposures
        exposure = self._calculate_exposure()

        # Update drawdown
        drawdown = self._calculate_drawdown()

        # Slippage placeholder from config
        slippage_bps = self.config.get('transaction_costs', {}).get('spread_slippage_bps', 0)

        # Capture metrics via monitoring agent
        self.monitor.capture_trading_metrics(slippage_bps, exposure, drawdown)

        pnl_report = {
            'timestamp': datetime.utcnow().isoformat(),
            'portfolio_value': self.trader.portfolio_value,
            'gross_exposure': exposure['gross'],
            'net_exposure': exposure['net'],
            'drawdown': drawdown,
            'slippage_bps': slippage_bps,
        }
        logger.info(f"📊 P&L Report: {pnl_report}")
        return pnl_report

    def _calculate_exposure(self) -> Dict[str, float]:
        total = 0.0
        net = 0.0
        if self.trader.positions:
            for pos in self.trader.positions.values():
                mv = float(getattr(pos, 'market_value', 0))
                total += abs(mv)
                net += mv
        if self.trader.portfolio_value:
            gross = total / self.trader.portfolio_value
            net_exp = net / self.trader.portfolio_value
        else:
            gross = 0.0
            net_exp = 0.0
        return {'gross': gross, 'net': net_exp}

    def _calculate_drawdown(self) -> float:
        pv = self.trader.portfolio_value
        if pv > self.peak_portfolio_value:
            self.peak_portfolio_value = pv
        if self.peak_portfolio_value == 0:
            return 0.0
        return (pv - self.peak_portfolio_value) / self.peak_portfolio_value


def main():
    """Manual test for the paper trading agent"""
    import json
    config_path = Path(__file__).parent.parent / 'config' / 'trading_config.json'
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    agent = PaperTradingAgent(cfg)
    agent.report_pnl()


if __name__ == "__main__":
    main()
