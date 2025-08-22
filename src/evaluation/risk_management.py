import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import scipy.stats as stats
from scipy.optimize import minimize
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Container for risk assessment results"""

    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0  # Conditional VaR (Expected Shortfall)
    cvar_99: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    downside_deviation: float = 0.0
    beta: float = 0.0
    tracking_error: float = 0.0
    information_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    ulcer_index: float = 0.0


@dataclass
class PositionSize:
    """Position sizing recommendation"""

    ticker: str
    recommended_size: float
    max_size: float
    risk_contribution: float
    confidence: float
    rationale: str


class RiskModel(ABC):
    """Abstract base class for risk models"""

    @abstractmethod
    def estimate_risk(self, returns: pd.Series, **kwargs) -> RiskMetrics:
        """Estimate risk metrics for a return series"""
        pass

    @abstractmethod
    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        pass


class HistoricalRiskModel(RiskModel):
    """Historical simulation risk model"""

    def __init__(self, lookback_window: int = 252):
        self.lookback_window = lookback_window

    def estimate_risk(
        self, returns: pd.Series, benchmark_returns: pd.Series = None
    ) -> RiskMetrics:
        """Estimate risk metrics using historical data"""
        try:
            if len(returns) < 30:
                logger.warning("Insufficient data for reliable risk estimation")

            # Basic risk metrics
            var_95 = self.calculate_var(returns, 0.95)
            var_99 = self.calculate_var(returns, 0.99)

            # Conditional VaR (Expected Shortfall)
            cvar_95 = self.calculate_cvar(returns, 0.95)
            cvar_99 = self.calculate_cvar(returns, 0.99)

            # Volatility metrics
            volatility = returns.std() * np.sqrt(252)  # Annualized
            downside_returns = returns[returns < 0]
            downside_deviation = (
                downside_returns.std() * np.sqrt(252)
                if len(downside_returns) > 0
                else 0
            )

            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            # Performance ratios
            mean_return = returns.mean() * 252  # Annualized
            risk_free_rate = 0.02  # Assume 2%

            sortino_ratio = (mean_return - risk_free_rate) / (downside_deviation + 1e-8)
            calmar_ratio = mean_return / (abs(max_drawdown) + 1e-8)

            # Ulcer Index (downside risk measure)
            ulcer_index = np.sqrt(np.mean(drawdown**2)) if len(drawdown) > 0 else 0

            # Benchmark-relative metrics
            beta = 0.0
            tracking_error = 0.0
            information_ratio = 0.0

            if benchmark_returns is not None and len(benchmark_returns) == len(returns):
                # Beta calculation
                covariance = np.cov(returns, benchmark_returns)[0, 1]
                benchmark_variance = np.var(benchmark_returns)
                beta = covariance / (benchmark_variance + 1e-8)

                # Tracking error
                active_returns = returns - benchmark_returns
                tracking_error = active_returns.std() * np.sqrt(252)

                # Information ratio
                information_ratio = (
                    active_returns.mean() * 252 / (tracking_error + 1e-8)
                )

            return RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                max_drawdown=max_drawdown,
                volatility=volatility,
                downside_deviation=downside_deviation,
                beta=beta,
                tracking_error=tracking_error,
                information_ratio=information_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                ulcer_index=ulcer_index,
            )

        except Exception as e:
            logger.error(f"Error estimating risk metrics: {e}")
            return RiskMetrics()

    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk using historical simulation"""
        if len(returns) == 0:
            return 0.0

        return np.percentile(returns, (1 - confidence) * 100)

    def calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional VaR (Expected Shortfall)"""
        if len(returns) == 0:
            return 0.0

        var = self.calculate_var(returns, confidence)
        tail_returns = returns[returns <= var]
        return tail_returns.mean() if len(tail_returns) > 0 else var


class ParametricRiskModel(RiskModel):
    """Parametric risk model assuming normal distribution"""

    def estimate_risk(self, returns: pd.Series, **kwargs) -> RiskMetrics:
        """Estimate risk assuming normal distribution"""
        try:
            if len(returns) < 30:
                logger.warning("Insufficient data for parametric risk model")

            mean_return = returns.mean()
            std_return = returns.std()

            # VaR assuming normal distribution
            var_95 = stats.norm.ppf(0.05, mean_return, std_return)
            var_99 = stats.norm.ppf(0.01, mean_return, std_return)

            # CVaR for normal distribution
            cvar_95 = (
                mean_return - std_return * stats.norm.pdf(stats.norm.ppf(0.05)) / 0.05
            )
            cvar_99 = (
                mean_return - std_return * stats.norm.pdf(stats.norm.ppf(0.01)) / 0.01
            )

            return RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                volatility=std_return * np.sqrt(252),
            )

        except Exception as e:
            logger.error(f"Error in parametric risk model: {e}")
            return RiskMetrics()

    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate parametric VaR"""
        mean_return = returns.mean()
        std_return = returns.std()
        return stats.norm.ppf(1 - confidence, mean_return, std_return)


class PositionSizer:
    """Position sizing based on risk management principles"""

    def __init__(
        self,
        risk_model: RiskModel,
        max_position_risk: float = 0.02,
        max_portfolio_risk: float = 0.15,
        confidence_threshold: float = 0.6,
    ):
        self.risk_model = risk_model
        self.max_position_risk = max_position_risk  # Max risk per position
        self.max_portfolio_risk = max_portfolio_risk  # Max total portfolio risk
        self.confidence_threshold = confidence_threshold

    def calculate_kelly_size(
        self, expected_return: float, volatility: float, win_rate: float = None
    ) -> float:
        """Calculate Kelly criterion position size"""
        try:
            if win_rate is not None:
                # Discrete Kelly formula
                avg_win = expected_return / (win_rate + 1e-8)
                avg_loss = expected_return / ((1 - win_rate) + 1e-8)
                kelly_fraction = (win_rate / abs(avg_loss)) - ((1 - win_rate) / avg_win)
            else:
                # Continuous Kelly formula
                kelly_fraction = expected_return / (volatility**2 + 1e-8)

            # Apply conservative scaling (typically 25-50% of full Kelly)
            return min(0.25 * kelly_fraction, 0.1)  # Cap at 10% position size

        except Exception as e:
            logger.error(f"Error calculating Kelly size: {e}")
            return 0.01  # Conservative fallback

    def calculate_volatility_size(
        self, target_risk: float, asset_volatility: float
    ) -> float:
        """Calculate position size based on volatility targeting"""
        try:
            return target_risk / (asset_volatility + 1e-8)
        except Exception as e:
            logger.error(f"Error calculating volatility-based size: {e}")
            return 0.01

    def calculate_var_size(
        self, returns: pd.Series, target_var: float, confidence: float = 0.95
    ) -> float:
        """Calculate position size based on VaR targeting"""
        try:
            asset_var = abs(self.risk_model.calculate_var(returns, confidence))
            return target_var / (asset_var + 1e-8)
        except Exception as e:
            logger.error(f"Error calculating VaR-based size: {e}")
            return 0.01

    def optimize_position_sizes(
        self,
        expected_returns: Dict[str, float],
        volatilities: Dict[str, float],
        correlations: pd.DataFrame = None,
    ) -> Dict[str, PositionSize]:
        """Optimize position sizes considering portfolio-level risk"""
        try:
            tickers = list(expected_returns.keys())
            n_assets = len(tickers)

            if n_assets == 0:
                return {}

            # Individual position sizes using Kelly criterion
            individual_sizes = {}
            for ticker in tickers:
                kelly_size = self.calculate_kelly_size(
                    expected_returns[ticker], volatilities[ticker]
                )
                vol_size = self.calculate_volatility_size(
                    self.max_position_risk, volatilities[ticker]
                )

                # Take minimum of Kelly and volatility-based sizing
                recommended_size = min(kelly_size, vol_size, 0.1)  # Cap at 10%

                individual_sizes[ticker] = PositionSize(
                    ticker=ticker,
                    recommended_size=recommended_size,
                    max_size=self.max_position_risk / volatilities[ticker],
                    risk_contribution=recommended_size * volatilities[ticker],
                    confidence=0.75,  # Default confidence
                    rationale=f"Kelly: {kelly_size:.3f}, Vol-target: {vol_size:.3f}",
                )

            # Portfolio-level optimization if correlation data available
            if correlations is not None and len(correlations) == n_assets:
                optimized_sizes = self._optimize_portfolio_sizes(
                    expected_returns, volatilities, correlations
                )

                # Update with optimized sizes
                for ticker in tickers:
                    if ticker in optimized_sizes:
                        individual_sizes[ticker].recommended_size = optimized_sizes[
                            ticker
                        ]
                        individual_sizes[ticker].confidence = 0.85
                        individual_sizes[ticker].rationale += " (Portfolio optimized)"

            return individual_sizes

        except Exception as e:
            logger.error(f"Error optimizing position sizes: {e}")
            return {}

    def _optimize_portfolio_sizes(
        self,
        expected_returns: Dict[str, float],
        volatilities: Dict[str, float],
        correlations: pd.DataFrame,
    ) -> Dict[str, float]:
        """Optimize position sizes at portfolio level"""
        try:
            tickers = list(expected_returns.keys())
            n_assets = len(tickers)

            # Create arrays
            returns_array = np.array([expected_returns[t] for t in tickers])
            vol_array = np.array([volatilities[t] for t in tickers])
            corr_matrix = correlations.loc[tickers, tickers].values

            # Covariance matrix
            cov_matrix = np.outer(vol_array, vol_array) * corr_matrix

            def objective(weights):
                """Minimize negative expected return / risk ratio"""
                portfolio_return = np.dot(weights, returns_array)
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return -(portfolio_return / (portfolio_risk + 1e-8))

            # Constraints
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # Sum to 1
                {
                    "type": "ineq",
                    "fun": lambda w: self.max_portfolio_risk
                    - np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))),
                },
            ]

            # Bounds - individual position limits
            bounds = [
                (0, self.max_position_risk / vol_array[i]) for i in range(n_assets)
            ]

            # Initial guess
            initial_weights = np.array([1.0 / n_assets] * n_assets)

            # Optimize
            result = minimize(
                objective,
                initial_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000},
            )

            if result.success:
                return dict(zip(tickers, result.x))
            else:
                logger.warning("Portfolio optimization failed, using individual sizes")
                return {}

        except Exception as e:
            logger.error(f"Error in portfolio size optimization: {e}")
            return {}


class StopLossManager:
    """Advanced stop-loss and take-profit management"""

    def __init__(
        self,
        initial_stop_pct: float = 0.05,
        trailing_stop_pct: float = 0.03,
        take_profit_pct: float = 0.15,
        volatility_multiplier: float = 2.0,
    ):
        self.initial_stop_pct = initial_stop_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.take_profit_pct = take_profit_pct
        self.volatility_multiplier = volatility_multiplier

    def calculate_stop_levels(
        self, entry_price: float, position_side: str, volatility: float = None
    ) -> Dict[str, float]:
        """Calculate dynamic stop loss and take profit levels"""
        try:
            if volatility is not None:
                # Volatility-adjusted stops
                stop_distance = max(
                    self.initial_stop_pct, volatility * self.volatility_multiplier
                )
                profit_distance = max(
                    self.take_profit_pct, volatility * self.volatility_multiplier * 3
                )
            else:
                stop_distance = self.initial_stop_pct
                profit_distance = self.take_profit_pct

            if position_side.lower() == "long":
                stop_loss = entry_price * (1 - stop_distance)
                take_profit = entry_price * (1 + profit_distance)
                trailing_stop = entry_price * (1 - self.trailing_stop_pct)
            else:  # Short position
                stop_loss = entry_price * (1 + stop_distance)
                take_profit = entry_price * (1 - profit_distance)
                trailing_stop = entry_price * (1 + self.trailing_stop_pct)

            return {
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "trailing_stop": trailing_stop,
                "stop_distance_pct": stop_distance,
                "profit_distance_pct": profit_distance,
            }

        except Exception as e:
            logger.error(f"Error calculating stop levels: {e}")
            return {}

    def update_trailing_stop(
        self,
        current_price: float,
        current_stop: float,
        position_side: str,
        high_since_entry: float = None,
    ) -> float:
        """Update trailing stop based on favorable price movement"""
        try:
            if position_side.lower() == "long":
                if high_since_entry and current_price > high_since_entry:
                    # Update trailing stop upward
                    new_stop = current_price * (1 - self.trailing_stop_pct)
                    return max(current_stop, new_stop)
                else:
                    return current_stop
            else:  # Short position
                if high_since_entry and current_price < high_since_entry:
                    # Update trailing stop downward
                    new_stop = current_price * (1 + self.trailing_stop_pct)
                    return min(current_stop, new_stop)
                else:
                    return current_stop

        except Exception as e:
            logger.error(f"Error updating trailing stop: {e}")
            return current_stop


class RiskMonitor:
    """Real-time risk monitoring and alerting"""

    def __init__(
        self, risk_model: RiskModel, alert_thresholds: Dict[str, float] = None
    ):
        self.risk_model = risk_model
        self.alert_thresholds = alert_thresholds or {
            "var_95": 0.05,
            "max_drawdown": 0.15,
            "volatility": 0.30,
            "position_concentration": 0.20,
        }
        self.alerts = []

    def check_portfolio_risk(
        self, portfolio_returns: pd.Series, position_weights: Dict[str, float]
    ) -> List[str]:
        """Check portfolio risk against thresholds"""
        alerts = []

        try:
            # Calculate current risk metrics
            risk_metrics = self.risk_model.estimate_risk(portfolio_returns)

            # VaR check
            if abs(risk_metrics.var_95) > self.alert_thresholds["var_95"]:
                alerts.append(
                    f"VaR breach: {risk_metrics.var_95:.3f} > {self.alert_thresholds['var_95']}"
                )

            # Drawdown check
            if abs(risk_metrics.max_drawdown) > self.alert_thresholds["max_drawdown"]:
                alerts.append(
                    f"Max drawdown breach: {abs(risk_metrics.max_drawdown):.3f} > {self.alert_thresholds['max_drawdown']}"
                )

            # Volatility check
            if risk_metrics.volatility > self.alert_thresholds["volatility"]:
                alerts.append(
                    f"High volatility: {risk_metrics.volatility:.3f} > {self.alert_thresholds['volatility']}"
                )

            # Concentration check
            if position_weights:
                max_weight = max(position_weights.values())
                if max_weight > self.alert_thresholds["position_concentration"]:
                    alerts.append(
                        f"Position concentration: {max_weight:.3f} > {self.alert_thresholds['position_concentration']}"
                    )

            # Store alerts
            self.alerts.extend(alerts)

            if alerts:
                logger.warning("Risk alerts generated:")
                for alert in alerts:
                    logger.warning(f"  - {alert}")

            return alerts

        except Exception as e:
            logger.error(f"Error checking portfolio risk: {e}")
            return []

    def generate_risk_report(
        self, portfolio_returns: pd.Series, position_weights: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        try:
            risk_metrics = self.risk_model.estimate_risk(portfolio_returns)

            report = {
                "timestamp": pd.Timestamp.now(),
                "risk_metrics": risk_metrics,
                "portfolio_stats": {
                    "total_return": (1 + portfolio_returns).prod() - 1,
                    "annualized_return": portfolio_returns.mean() * 252,
                    "num_observations": len(portfolio_returns),
                    "latest_return": (
                        portfolio_returns.iloc[-1] if len(portfolio_returns) > 0 else 0
                    ),
                },
                "position_analysis": {},
                "alerts": self.check_portfolio_risk(
                    portfolio_returns, position_weights
                ),
            }

            if position_weights:
                report["position_analysis"] = {
                    "num_positions": len(position_weights),
                    "max_position_weight": max(position_weights.values()),
                    "position_concentration": sum(
                        w**2 for w in position_weights.values()
                    ),
                    "weights": position_weights,
                }

            return report

        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            return {}


def create_comprehensive_risk_manager(
    max_position_risk: float = 0.02, max_portfolio_risk: float = 0.15
) -> Tuple[PositionSizer, StopLossManager, RiskMonitor]:
    """Factory function to create complete risk management system"""

    # Risk model
    risk_model = HistoricalRiskModel()

    # Position sizer
    position_sizer = PositionSizer(
        risk_model=risk_model,
        max_position_risk=max_position_risk,
        max_portfolio_risk=max_portfolio_risk,
    )

    # Stop loss manager
    stop_loss_manager = StopLossManager()

    # Risk monitor
    risk_monitor = RiskMonitor(risk_model)

    logger.info("Comprehensive risk management system created")

    return position_sizer, stop_loss_manager, risk_monitor
