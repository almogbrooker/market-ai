#!/usr/bin/env python3
"""
Streamlit dashboard for monitoring live trading metrics.
Displays positions, exposure, PnL, gate coverage, and recent decisions.
Includes alerts for risk limit breaches. Uses read-only Alpaca API access.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from alpaca.trading.client import TradingClient

# Risk limits
MAX_GROSS = 0.60  # 60% gross exposure
MAX_PER_NAME = 0.08  # 8% per-name exposure
DAILY_LOSS_LIMIT = 0.03  # 3% daily loss
GATE_TARGET = 0.15  # 15% gate accept rate


def _get_account(client: TradingClient):
    """Fetch account information."""
    return client.get_account()


def _get_positions_df(client: TradingClient, portfolio_value: float) -> pd.DataFrame:
    """Return current positions as DataFrame with exposure metrics."""
    positions = client.get_all_positions()
    rows = []
    for pos in positions:
        mv = float(pos.market_value)
        exposure = abs(mv) / portfolio_value if portfolio_value else 0.0
        rows.append({
            "symbol": pos.symbol,
            "qty": float(pos.qty),
            "market_value": mv,
            "unrealized_pl": float(pos.unrealized_pl),
            "exposure_pct": exposure,
        })
    return pd.DataFrame(rows)


def _load_gate_metrics(log_dir: str | Path):
    """Load gate coverage and recent decisions from decision logs."""
    log_path = Path(log_dir)
    csv_files = sorted(log_path.glob("decisions_*.csv"))
    if not csv_files:
        return None, None

    latest = csv_files[-1]
    df = pd.read_csv(latest)
    coverage = df["gate_passed"].mean() if "gate_passed" in df.columns else None
    recent = df.sort_values("timestamp", ascending=False).head(20)
    return coverage, recent


def main():
    st.title("📊 Market AI Trading Dashboard")

    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        st.error("Alpaca API credentials not found in environment")
        st.stop()

    client = TradingClient(api_key, api_secret, paper=True)

    # Account metrics
    account = _get_account(client)
    portfolio_value = float(account.portfolio_value)
    equity = float(account.equity)
    last_equity = float(getattr(account, "last_equity", equity))
    daily_pnl = equity - last_equity

    st.subheader("Account Summary")
    cols = st.columns(3)
    cols[0].metric("Equity", f"${equity:,.2f}")
    cols[1].metric("Daily PnL", f"${daily_pnl:,.2f}")
    cols[2].metric("Buying Power", f"${float(account.buying_power):,.2f}")

    if daily_pnl < -DAILY_LOSS_LIMIT * last_equity:
        st.error("Daily loss limit breached")

    # Position metrics
    st.subheader("Positions")
    positions_df = _get_positions_df(client, portfolio_value)
    if positions_df.empty:
        st.info("No open positions")
    else:
        st.dataframe(positions_df)
        gross_exposure = positions_df["market_value"].abs().sum() / portfolio_value
        st.metric("Gross Exposure", f"{gross_exposure:.1%}")
        if gross_exposure > MAX_GROSS:
            st.error("Gross exposure exceeds limit")

        breaches = positions_df[positions_df["exposure_pct"] > MAX_PER_NAME]
        for _, row in breaches.iterrows():
            st.warning(f"{row['symbol']} exceeds per-name limit ({row['exposure_pct']:.1%})")

    # Gate metrics
    st.subheader("Gate Coverage")
    coverage, decisions = _load_gate_metrics("PRODUCTION/logs/decisions")
    if coverage is not None:
        st.metric("Accept Rate", f"{coverage:.1%}")
        if coverage < GATE_TARGET:
            st.warning("Gate coverage below target")
    else:
        st.info("No gate metrics available")

    st.subheader("Recent Decisions")
    if decisions is not None:
        st.dataframe(decisions)
    else:
        st.info("No decision logs found")


if __name__ == "__main__":
    main()
