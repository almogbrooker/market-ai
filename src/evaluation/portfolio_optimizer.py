import numpy as np
import pandas as pd
import scipy.optimize as sco
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Container for portfolio optimization results"""
    weights: np.ndarray
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    optimization_success: bool
    message: str
    metrics: Dict[str, float]

class PortfolioOptimizer(ABC):
    """Abstract base class for portfolio optimizers"""
    
    @abstractmethod
    def optimize(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray,
                 **kwargs) -> OptimizationResult:
        pass

class MeanVarianceOptimizer(PortfolioOptimizer):
    """Markowitz mean-variance optimization"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def optimize(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray,
                 target_return: Optional[float] = None, max_weight: float = 0.4,
                 min_weight: float = 0.0) -> OptimizationResult:
        
        try:
            n_assets = len(expected_returns)
            
            # Objective function for maximum Sharpe ratio
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
                return -(portfolio_return - self.risk_free_rate) / (portfolio_risk + 1e-8)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
            ]
            
            # Add target return constraint if specified
            if target_return is not None:
                constraints.append({
                    'type': 'eq', 
                    'fun': lambda w: np.dot(w, expected_returns) - target_return
                })
            
            # Bounds for each weight
            bounds = tuple([(min_weight, max_weight) for _ in range(n_assets)])
            
            # Initial guess (equal weights)
            initial_weights = np.array([1.0 / n_assets] * n_assets)
            
            # Optimize
            result = sco.minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'disp': False, 'maxiter': 1000}
            )
            
            if result.success:
                weights = result.x
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / (portfolio_risk + 1e-8)
                
                metrics = {
                    'concentration': np.sum(weights**2),  # Herfindahl index
                    'max_weight': np.max(weights),
                    'min_weight': np.min(weights),
                    'num_nonzero_weights': np.sum(weights > 1e-4),
                    'turnover': 0.0  # Would need previous weights to calculate
                }
                
                return OptimizationResult(
                    weights=weights,
                    expected_return=portfolio_return,
                    expected_risk=portfolio_risk,
                    sharpe_ratio=sharpe_ratio,
                    optimization_success=True,
                    message="Optimization completed successfully",
                    metrics=metrics
                )
            else:
                logger.error(f"Optimization failed: {result.message}")
                equal_weights = np.array([1.0 / n_assets] * n_assets)
                return self._fallback_result(equal_weights, expected_returns, covariance_matrix, result.message)
                
        except Exception as e:
            logger.error(f"Error in mean-variance optimization: {e}")
            equal_weights = np.array([1.0 / len(expected_returns)] * len(expected_returns))
            return self._fallback_result(equal_weights, expected_returns, covariance_matrix, str(e))
    
    def _fallback_result(self, weights: np.ndarray, expected_returns: np.ndarray,
                        covariance_matrix: np.ndarray, message: str) -> OptimizationResult:
        """Create fallback result with equal weights"""
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / (portfolio_risk + 1e-8)
        
        return OptimizationResult(
            weights=weights,
            expected_return=portfolio_return,
            expected_risk=portfolio_risk,
            sharpe_ratio=sharpe_ratio,
            optimization_success=False,
            message=f"Optimization failed, using equal weights: {message}",
            metrics={}
        )

class RiskParityOptimizer(PortfolioOptimizer):
    """Risk parity optimization - equal risk contribution"""
    
    def optimize(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray,
                 **kwargs) -> OptimizationResult:
        
        try:
            n_assets = len(expected_returns)
            
            def risk_parity_objective(weights):
                """Minimize the sum of squared differences in risk contributions"""
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
                marginal_contrib = np.dot(covariance_matrix, weights)
                contrib = weights * marginal_contrib / (portfolio_vol + 1e-8)
                target_contrib = portfolio_vol / n_assets
                return np.sum((contrib - target_contrib)**2)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            ]
            
            # Bounds (all positive weights)
            bounds = tuple([(0.001, 0.5) for _ in range(n_assets)])
            
            # Initial guess
            initial_weights = np.array([1.0 / n_assets] * n_assets)
            
            result = sco.minimize(
                risk_parity_objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'disp': False}
            )
            
            if result.success:
                weights = result.x
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
                sharpe_ratio = portfolio_return / (portfolio_risk + 1e-8)
                
                return OptimizationResult(
                    weights=weights,
                    expected_return=portfolio_return,
                    expected_risk=portfolio_risk,
                    sharpe_ratio=sharpe_ratio,
                    optimization_success=True,
                    message="Risk parity optimization completed successfully",
                    metrics={}
                )
            else:
                equal_weights = np.array([1.0 / n_assets] * n_assets)
                return self._fallback_result(equal_weights, expected_returns, covariance_matrix, result.message)
                
        except Exception as e:
            logger.error(f"Error in risk parity optimization: {e}")
            equal_weights = np.array([1.0 / len(expected_returns)] * len(expected_returns))
            return self._fallback_result(equal_weights, expected_returns, covariance_matrix, str(e))
    
    def _fallback_result(self, weights: np.ndarray, expected_returns: np.ndarray,
                        covariance_matrix: np.ndarray, message: str) -> OptimizationResult:
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        return OptimizationResult(
            weights=weights,
            expected_return=portfolio_return,
            expected_risk=portfolio_risk,
            sharpe_ratio=portfolio_return / (portfolio_risk + 1e-8),
            optimization_success=False,
            message=f"Risk parity optimization failed: {message}",
            metrics={}
        )

class BlackLittermanOptimizer(PortfolioOptimizer):
    """Black-Litterman optimization with views"""
    
    def __init__(self, risk_free_rate: float = 0.02, tau: float = 0.025):
        self.risk_free_rate = risk_free_rate
        self.tau = tau  # Uncertainty parameter
    
    def optimize(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray,
                 market_caps: Optional[np.ndarray] = None,
                 views_matrix: Optional[np.ndarray] = None,
                 views_returns: Optional[np.ndarray] = None,
                 views_uncertainty: Optional[np.ndarray] = None) -> OptimizationResult:
        
        try:
            n_assets = len(expected_returns)
            
            # Market equilibrium weights (if market caps not provided, use equal weights)
            if market_caps is None:
                w_market = np.array([1.0 / n_assets] * n_assets)
            else:
                w_market = market_caps / np.sum(market_caps)
            
            # Implied equilibrium returns
            risk_aversion = 1.0  # Simplification
            pi = risk_aversion * np.dot(covariance_matrix, w_market)
            
            # If no views, return market portfolio
            if views_matrix is None:
                portfolio_return = np.dot(w_market, expected_returns)
                portfolio_risk = np.sqrt(np.dot(w_market.T, np.dot(covariance_matrix, w_market)))
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / (portfolio_risk + 1e-8)
                
                return OptimizationResult(
                    weights=w_market,
                    expected_return=portfolio_return,
                    expected_risk=portfolio_risk,
                    sharpe_ratio=sharpe_ratio,
                    optimization_success=True,
                    message="Black-Litterman with market weights (no views)",
                    metrics={}
                )
            
            # Black-Litterman with views
            tau_sigma = self.tau * covariance_matrix
            
            if views_uncertainty is None:
                # Default uncertainty
                omega = np.diag(np.diag(np.dot(views_matrix, np.dot(tau_sigma, views_matrix.T))))
            else:
                omega = np.diag(views_uncertainty)
            
            # Black-Litterman formula
            M1 = np.linalg.inv(tau_sigma)
            M2 = np.dot(views_matrix.T, np.dot(np.linalg.inv(omega), views_matrix))
            M3 = np.dot(np.linalg.inv(tau_sigma), pi)
            M4 = np.dot(views_matrix.T, np.dot(np.linalg.inv(omega), views_returns))
            
            bl_returns = np.dot(np.linalg.inv(M1 + M2), M3 + M4)
            bl_cov = np.linalg.inv(M1 + M2)
            
            # Optimize with Black-Litterman inputs
            mv_optimizer = MeanVarianceOptimizer(self.risk_free_rate)
            result = mv_optimizer.optimize(bl_returns, bl_cov)
            result.message = "Black-Litterman optimization completed"
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Black-Litterman optimization: {e}")
            equal_weights = np.array([1.0 / len(expected_returns)] * len(expected_returns))
            portfolio_return = np.dot(equal_weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(equal_weights.T, np.dot(covariance_matrix, equal_weights)))
            
            return OptimizationResult(
                weights=equal_weights,
                expected_return=portfolio_return,
                expected_risk=portfolio_risk,
                sharpe_ratio=(portfolio_return - self.risk_free_rate) / (portfolio_risk + 1e-8),
                optimization_success=False,
                message=f"Black-Litterman optimization failed: {e}",
                metrics={}
            )

class PortfolioManager:
    """Main class for portfolio management and optimization"""
    
    def __init__(self, tickers: List[str], risk_free_rate: float = 0.02):
        self.tickers = tickers
        self.risk_free_rate = risk_free_rate
        self.optimizers = {
            'mean_variance': MeanVarianceOptimizer(risk_free_rate),
            'risk_parity': RiskParityOptimizer(),
            'black_litterman': BlackLittermanOptimizer(risk_free_rate)
        }
        
    def estimate_parameters(self, returns_df: pd.DataFrame, 
                          method: str = 'historical') -> Tuple[np.ndarray, np.ndarray]:
        """Estimate expected returns and covariance matrix"""
        try:
            if method == 'historical':
                expected_returns = returns_df.mean().values * 252  # Annualize
                covariance_matrix = returns_df.cov().values * 252  # Annualize
                
            elif method == 'shrinkage':
                # Ledoit-Wolf shrinkage estimator
                from sklearn.covariance import LedoitWolf
                
                lw = LedoitWolf()
                covariance_matrix = lw.fit(returns_df.fillna(0)).covariance_ * 252
                expected_returns = returns_df.mean().values * 252
                
            elif method == 'ewma':
                # Exponentially weighted moving average
                alpha = 0.94
                expected_returns = returns_df.ewm(alpha=alpha).mean().iloc[-1].values * 252
                covariance_matrix = returns_df.ewm(alpha=alpha).cov().iloc[-len(self.tickers):].values * 252
                
            else:
                raise ValueError(f"Unknown estimation method: {method}")
            
            # Ensure positive definite covariance matrix
            eigenvals, eigenvecs = np.linalg.eigh(covariance_matrix)
            eigenvals = np.maximum(eigenvals, 1e-6)
            covariance_matrix = np.dot(eigenvecs, np.dot(np.diag(eigenvals), eigenvecs.T))
            
            return expected_returns, covariance_matrix
            
        except Exception as e:
            logger.error(f"Error estimating parameters: {e}")
            raise
    
    def optimize_portfolio(self, returns_df: pd.DataFrame, 
                          method: str = 'mean_variance',
                          estimation_method: str = 'historical',
                          **kwargs) -> OptimizationResult:
        """Optimize portfolio using specified method"""
        try:
            if method not in self.optimizers:
                raise ValueError(f"Unknown optimization method: {method}")
            
            # Estimate parameters
            expected_returns, covariance_matrix = self.estimate_parameters(
                returns_df, estimation_method
            )
            
            # Optimize
            optimizer = self.optimizers[method]
            result = optimizer.optimize(expected_returns, covariance_matrix, **kwargs)
            
            # Add ticker information
            result.tickers = self.tickers
            
            logger.info(f"Portfolio optimization completed using {method} method")
            logger.info(f"Expected return: {result.expected_return:.4f}, "
                       f"Risk: {result.expected_risk:.4f}, "
                       f"Sharpe: {result.sharpe_ratio:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            raise
    
    def efficient_frontier(self, returns_df: pd.DataFrame, 
                          num_portfolios: int = 100) -> pd.DataFrame:
        """Generate efficient frontier"""
        try:
            expected_returns, covariance_matrix = self.estimate_parameters(returns_df)
            
            min_ret = np.min(expected_returns)
            max_ret = np.max(expected_returns)
            target_returns = np.linspace(min_ret, max_ret, num_portfolios)
            
            results = []
            optimizer = self.optimizers['mean_variance']
            
            for target_return in target_returns:
                try:
                    result = optimizer.optimize(
                        expected_returns, covariance_matrix, 
                        target_return=target_return
                    )
                    if result.optimization_success:
                        results.append({
                            'target_return': target_return,
                            'expected_return': result.expected_return,
                            'expected_risk': result.expected_risk,
                            'sharpe_ratio': result.sharpe_ratio,
                            'weights': result.weights
                        })
                except:
                    continue
            
            return pd.DataFrame(results)
            
        except Exception as e:
            logger.error(f"Error generating efficient frontier: {e}")
            return pd.DataFrame()
    
    def backtest_portfolio(self, returns_df: pd.DataFrame, weights: np.ndarray,
                          rebalance_freq: str = 'M') -> Dict[str, float]:
        """Backtest portfolio performance"""
        try:
            # Create portfolio returns
            portfolio_returns = (returns_df * weights).sum(axis=1)
            
            # Rebalancing logic (simplified)
            if rebalance_freq == 'M':
                # Monthly rebalancing
                portfolio_returns = portfolio_returns.resample('M').apply(
                    lambda x: (1 + x).prod() - 1
                )
            
            # Performance metrics
            total_return = (1 + portfolio_returns).prod() - 1
            annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility
            max_drawdown = (portfolio_returns.cumsum() - portfolio_returns.cumsum().cummax()).min()
            
            # Downside metrics
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252)
            sortino_ratio = (annualized_return - self.risk_free_rate) / (downside_volatility + 1e-8)
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': annualized_return / (abs(max_drawdown) + 1e-8)
            }
            
        except Exception as e:
            logger.error(f"Error in portfolio backtesting: {e}")
            return {}

def create_sample_portfolio(tickers: List[str], returns_df: pd.DataFrame) -> OptimizationResult:
    """Create a sample optimized portfolio"""
    try:
        manager = PortfolioManager(tickers)
        result = manager.optimize_portfolio(returns_df, method='mean_variance')
        
        # Display results
        weights_df = pd.DataFrame({
            'Ticker': tickers,
            'Weight': result.weights,
            'Weight_Pct': result.weights * 100
        })
        weights_df = weights_df.sort_values('Weight', ascending=False)
        
        logger.info("\nOptimal Portfolio Weights:")
        for _, row in weights_df.iterrows():
            logger.info(f"{row['Ticker']}: {row['Weight_Pct']:.2f}%")
        
        return result
        
    except Exception as e:
        logger.error(f"Error creating sample portfolio: {e}")
        raise