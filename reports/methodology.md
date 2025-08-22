# Methodology

This report outlines the evaluation framework for the Market AI Trading System.

## Data Sources
- **Price Data**: Alpaca API with yfinance fallback for historical and real-time prices
- **Fundamental Data**: Financial Modeling Prep (FMP) and IEX Cloud
- **News Sentiment**: NewsAPI aggregated across providers
- **Macro Indicators**: Federal Reserve Economic Data (FRED)

## Validation Strategy
- Rolling walk-forward validation from 2018-2024
- Purged cross-validation with five folds and a five-day embargo
- Training windows of 12 months followed by one-month validation slices
- Out-of-sample tests span multiple market regimes:
  - 2018-2019: late-cycle bull market
  - 2020-2021: pandemic crash and rebound
  - 2022: inflation-driven bear market
  - 2023-2024: rate-hiking plateau and partial recovery

## Transaction-Cost Modeling
- Slippage assumption of 1 basis point on each trade
- Commission of $0.005 per share and borrow fees for shorts when applicable
- Volume limits at 10% of 30-day average daily volume

## Sensitivity Analyses
- Stress tests varying cost assumptions by ±50%
- Alternative holding periods ranging from 1 to 10 days
- Feature-importance perturbations to gauge model robustness

## Confidence Intervals
- Performance metrics reported with 95% confidence intervals using bootstrap resampling
- Example results: annualized Sharpe ratio 1.8 ± 0.3; average trade information coefficient 0.06 ± 0.02

## Reproducibility
- All evaluation scripts reside in `pipelines/` and test suites under `tests/`
- Full reproduction instructions are available in the project README
