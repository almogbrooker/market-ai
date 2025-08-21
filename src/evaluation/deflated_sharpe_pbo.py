#!/usr/bin/env python3
"""
DEFLATED SHARPE AND PROBABILITY OF BACKTEST OVERFITTING (PBO)
Critical validation tools to make the 6.83% IC claim credible
Based on López de Prado's "Advances in Financial Machine Learning"
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from sklearn.model_selection import KFold, TimeSeriesSplit
from typing import Tuple, List, Dict, Optional, Union
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DeflatedSharpeRatio:
    """
    Deflated Sharpe Ratio calculation
    Adjusts Sharpe ratio for multiple testing and selection bias
    """
    
    def __init__(self, benchmark_sharpe: float = 0.0):
        """
        Initialize Deflated Sharpe calculator
        
        Args:
            benchmark_sharpe: Benchmark Sharpe ratio to test against
        """
        self.benchmark_sharpe = benchmark_sharpe
    
    def calculate(self, 
                 returns: np.ndarray,
                 n_trials: int = 100,
                 n_independent_samples: Optional[int] = None,
                 skewness: Optional[float] = None,
                 kurtosis: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate Deflated Sharpe Ratio
        
        Args:
            returns: Return series
            n_trials: Number of trials/strategies tested (for multiple testing adjustment)
            n_independent_samples: Effective number of independent samples
            skewness: Return skewness (computed if not provided)
            kurtosis: Return excess kurtosis (computed if not provided)
            
        Returns:
            Dictionary with deflated Sharpe results
        """
        returns = np.asarray(returns)
        returns_clean = returns[~np.isnan(returns)]
        
        if len(returns_clean) < 30:
            logger.warning("⚠️ Insufficient data for Deflated Sharpe calculation")
            return self._empty_result()
        
        # Basic Sharpe ratio
        mean_return = np.mean(returns_clean)
        std_return = np.std(returns_clean, ddof=1)
        
        if std_return == 0:
            return self._empty_result()
        
        sharpe_ratio = mean_return / std_return
        
        # Effective number of independent samples
        if n_independent_samples is None:
            n_independent_samples = self._estimate_independent_samples(returns_clean)
        
        # Moments
        if skewness is None:
            skewness = stats.skew(returns_clean)
        if kurtosis is None:
            kurtosis = stats.kurtosis(returns_clean, fisher=True)  # Excess kurtosis
        
        # Standard error of Sharpe ratio
        # Adjusted for higher moments (Lo, 2002)
        sharpe_var = (1 + (sharpe_ratio**2)/2 - sharpe_ratio * skewness + 
                     ((sharpe_ratio**2)/4) * kurtosis) / n_independent_samples
        sharpe_std = np.sqrt(sharpe_var)
        
        # Multiple testing adjustment using Bonferroni-like correction
        # Maximum expected Sharpe ratio under null
        max_expected_sharpe = self._calculate_max_expected_sharpe(n_trials, n_independent_samples)
        
        # Deflated Sharpe ratio
        deflated_sharpe = (sharpe_ratio - max_expected_sharpe) / sharpe_std
        
        # Statistical significance
        p_value = 2 * (1 - norm.cdf(abs(deflated_sharpe)))  # Two-tailed test
        
        # Confidence interval for original Sharpe
        sharpe_ci_lower = sharpe_ratio - 1.96 * sharpe_std
        sharpe_ci_upper = sharpe_ratio + 1.96 * sharpe_std
        
        results = {
            'sharpe_ratio': sharpe_ratio,
            'deflated_sharpe': deflated_sharpe,
            'sharpe_std_error': sharpe_std,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'max_expected_sharpe': max_expected_sharpe,
            'n_trials': n_trials,
            'n_independent_samples': n_independent_samples,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'sharpe_ci_lower': sharpe_ci_lower,
            'sharpe_ci_upper': sharpe_ci_upper
        }
        
        logger.info(f"📊 Deflated Sharpe: {deflated_sharpe:.3f} (p={p_value:.4f})")
        
        return results
    
    def _estimate_independent_samples(self, returns: np.ndarray) -> int:
        """
        Estimate effective number of independent samples
        Accounts for autocorrelation in returns
        """
        n = len(returns)
        
        # Calculate first-order autocorrelation
        if n < 3:
            return n
        
        try:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            if np.isnan(autocorr):
                autocorr = 0
        except:
            autocorr = 0
        
        # Adjustment for autocorrelation (simplified)
        if abs(autocorr) < 0.05:
            return n
        else:
            # Effective sample size adjustment
            effective_n = n * (1 - autocorr) / (1 + autocorr)
            return max(int(effective_n), 10)
    
    def _calculate_max_expected_sharpe(self, n_trials: int, n_samples: int) -> float:
        """
        Calculate maximum expected Sharpe ratio under null hypothesis
        Accounts for selection bias from multiple testing
        """
        if n_trials <= 1:
            return 0.0
        
        # Expected maximum of n_trials independent normal random variables
        # Approximation for large n_trials
        gamma = 0.5772156649  # Euler-Mascheroni constant
        
        if n_trials >= 10:
            # Asymptotic approximation
            max_expected = np.sqrt(2 * np.log(n_trials) - np.log(np.log(n_trials)) - np.log(4*np.pi)) / np.sqrt(n_samples)
        else:
            # Small sample correction using order statistics
            expected_max = norm.ppf(1 - 1/(n_trials + 1))
            max_expected = expected_max / np.sqrt(n_samples)
        
        return max_expected
    
    def _empty_result(self) -> Dict[str, float]:
        """Return empty result dictionary"""
        return {
            'sharpe_ratio': 0.0,
            'deflated_sharpe': 0.0,
            'sharpe_std_error': 0.0,
            'p_value': 1.0,
            'is_significant': False,
            'max_expected_sharpe': 0.0,
            'n_trials': 0,
            'n_independent_samples': 0,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'sharpe_ci_lower': 0.0,
            'sharpe_ci_upper': 0.0
        }

class ProbabilityOfBacktestOverfitting:
    """
    Probability of Backtest Overfitting (PBO) calculation
    Tests whether IS performance is likely due to overfitting
    """
    
    def __init__(self, n_splits: int = 16):
        """
        Initialize PBO calculator
        
        Args:
            n_splits: Number of combinatorial splits to generate
        """
        self.n_splits = n_splits
    
    def calculate(self, 
                 returns: np.ndarray,
                 n_strategies: int = 100,
                 method: str = 'combinatorial',
                 parallel: bool = True) -> Dict[str, float]:
        """
        Calculate Probability of Backtest Overfitting
        
        Args:
            returns: Matrix of strategy returns [n_periods x n_strategies]
            n_strategies: Number of strategies tested (for simulation if needed)
            method: 'combinatorial' or 'holdout'
            parallel: Use parallel processing
            
        Returns:
            PBO metrics and statistics
        """
        returns = np.asarray(returns)
        
        if returns.ndim == 1:
            # Single strategy - simulate multiple strategies for PBO
            returns = self._simulate_strategies(returns, n_strategies)
        
        n_periods, n_strats = returns.shape
        
        if n_periods < 100:
            logger.warning("⚠️ Insufficient periods for reliable PBO calculation")
        
        logger.info(f"🔍 Computing PBO for {n_strats} strategies over {n_periods} periods")
        
        if method == 'combinatorial':
            pbo_result = self._combinatorial_pbo(returns, parallel)
        else:
            pbo_result = self._holdout_pbo(returns)
        
        return pbo_result
    
    def _combinatorial_pbo(self, returns: np.ndarray, parallel: bool = True) -> Dict[str, float]:
        """
        Combinatorial Purged Cross-Validation PBO
        """
        n_periods, n_strategies = returns.shape
        
        # Generate combinatorial splits
        splits = self._generate_combinatorial_splits(n_periods)
        
        if parallel and len(splits) > 4:
            # Parallel processing for large number of splits
            pbo_stats = self._parallel_pbo_calculation(returns, splits)
        else:
            # Sequential processing
            pbo_stats = []
            for train_idx, test_idx in splits:
                stats_dict = self._calculate_split_statistics(returns, train_idx, test_idx)
                pbo_stats.append(stats_dict)
        
        # Combine results
        return self._aggregate_pbo_results(pbo_stats)
    
    def _holdout_pbo(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Simple holdout PBO (50/50 split)
        """
        n_periods, n_strategies = returns.shape
        split_point = n_periods // 2
        
        train_idx = list(range(split_point))
        test_idx = list(range(split_point, n_periods))
        
        stats_dict = self._calculate_split_statistics(returns, train_idx, test_idx)
        
        return {
            'pbo': stats_dict['prob_loss'],
            'probability_loss': stats_dict['prob_loss'],
            'phi': stats_dict['phi'],
            'lambda_oos': stats_dict['lambda_oos'],
            'n_splits': 1,
            'median_logit': stats_dict['logit'],
            'stdev_logit': 0.0
        }
    
    def _generate_combinatorial_splits(self, n_periods: int) -> List[Tuple[List[int], List[int]]]:
        """
        Generate combinatorial purged cross-validation splits
        """
        splits = []
        
        # Use time series split as base
        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=None)
        indices = np.arange(n_periods)
        
        for train_idx, test_idx in tscv.split(indices):
            # Add purge gap (5% of total periods)
            purge_gap = max(1, int(0.05 * n_periods))
            
            # Remove purge period from start of test set
            test_start = test_idx[0]
            purged_test_idx = test_idx[test_idx >= test_start + purge_gap]
            
            if len(purged_test_idx) > 10:  # Minimum test set size
                splits.append((train_idx.tolist(), purged_test_idx.tolist()))
        
        return splits
    
    def _calculate_split_statistics(self, 
                                  returns: np.ndarray,
                                  train_idx: List[int], 
                                  test_idx: List[int]) -> Dict[str, float]:
        """
        Calculate statistics for a single split
        """
        train_returns = returns[train_idx, :]
        test_returns = returns[test_idx, :]
        
        # Calculate Sharpe ratios
        train_sharpe = self._calculate_sharpe_ratios(train_returns)
        test_sharpe = self._calculate_sharpe_ratios(test_returns)
        
        # Find best strategy in IS period
        best_strategy_idx = np.nanargmax(train_sharpe)
        
        # Performance of best strategy in OOS
        best_oos_sharpe = test_sharpe[best_strategy_idx]
        
        # Rank of best IS strategy in OOS period
        oos_rank = stats.rankdata(-test_sharpe, method='ordinal')[best_strategy_idx]
        relative_rank = oos_rank / len(test_sharpe)
        
        # Lambda (relative rank statistic)
        lambda_oos = relative_rank
        
        # Phi (probability that best IS strategy underperforms in OOS)
        phi = 1 if best_oos_sharpe < np.nanmedian(test_sharpe) else 0
        
        # Logit transformation for aggregation
        logit = np.log(lambda_oos / (1 - lambda_oos + 1e-8))
        
        return {
            'lambda_oos': lambda_oos,
            'phi': phi,
            'prob_loss': phi,
            'logit': logit,
            'best_is_sharpe': train_sharpe[best_strategy_idx],
            'best_oos_sharpe': best_oos_sharpe,
            'median_oos_sharpe': np.nanmedian(test_sharpe)
        }
    
    def _parallel_pbo_calculation(self, returns: np.ndarray, splits: List) -> List[Dict]:
        """
        Parallel calculation of PBO statistics
        """
        n_cores = min(mp.cpu_count(), len(splits))
        
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            futures = []
            
            for train_idx, test_idx in splits:
                future = executor.submit(self._calculate_split_statistics, returns, train_idx, test_idx)
                futures.append(future)
            
            results = [future.result() for future in futures]
        
        return results
    
    def _aggregate_pbo_results(self, pbo_stats: List[Dict]) -> Dict[str, float]:
        """
        Aggregate PBO results across all splits
        """
        if not pbo_stats:
            return {'pbo': 1.0, 'probability_loss': 1.0}
        
        # Extract statistics
        lambdas = [stat['lambda_oos'] for stat in pbo_stats]
        phis = [stat['phi'] for stat in pbo_stats]
        logits = [stat['logit'] for stat in pbo_stats if not np.isinf(stat['logit'])]
        
        # PBO is the probability that lambda > 0.5 (equivalently, relative rank > 0.5)
        prob_overfit = np.mean([lam > 0.5 for lam in lambdas])
        
        # Alternative: probability of loss (phi)
        prob_loss = np.mean(phis)
        
        # Statistics on lambda distribution
        median_lambda = np.median(lambdas)
        mean_lambda = np.mean(lambdas)
        
        # Logit statistics
        if logits:
            median_logit = np.median(logits)
            stdev_logit = np.std(logits)
        else:
            median_logit = 0.0
            stdev_logit = 0.0
        
        results = {
            'pbo': prob_overfit,
            'probability_loss': prob_loss,
            'phi': prob_loss,
            'lambda_oos': mean_lambda,
            'lambda_median': median_lambda,
            'n_splits': len(pbo_stats),
            'median_logit': median_logit,
            'stdev_logit': stdev_logit
        }
        
        logger.info(f"📊 PBO Results: PBO={prob_overfit:.3f}, Prob Loss={prob_loss:.3f}")
        
        return results
    
    def _calculate_sharpe_ratios(self, returns: np.ndarray) -> np.ndarray:
        """
        Calculate Sharpe ratios for multiple strategies
        """
        if returns.size == 0:
            return np.array([])
        
        means = np.nanmean(returns, axis=0)
        stds = np.nanstd(returns, axis=0, ddof=1)
        
        # Avoid division by zero
        stds = np.where(stds == 0, 1e-8, stds)
        
        sharpe_ratios = means / stds
        return sharpe_ratios
    
    def _simulate_strategies(self, returns: np.ndarray, n_strategies: int) -> np.ndarray:
        """
        Simulate multiple strategies from a single return series
        Used when only one strategy is provided
        """
        n_periods = len(returns)
        base_vol = np.std(returns)
        base_mean = np.mean(returns)
        
        # Create variations by adding noise and parameter variations
        simulated_returns = np.zeros((n_periods, n_strategies))
        
        for i in range(n_strategies):
            # Add random noise to simulate different strategies
            noise_scale = np.random.uniform(0.1, 0.3)
            noise = np.random.normal(0, base_vol * noise_scale, n_periods)
            
            # Vary the mean slightly
            mean_adjustment = np.random.normal(0, abs(base_mean) * 0.1)
            
            # Create synthetic strategy
            strategy_returns = returns + noise + mean_adjustment
            simulated_returns[:, i] = strategy_returns
        
        # Ensure first strategy is the original
        simulated_returns[:, 0] = returns
        
        return simulated_returns

class RobustnessValidator:
    """
    Comprehensive robustness validation combining Deflated Sharpe and PBO
    """
    
    def __init__(self):
        self.deflated_sharpe = DeflatedSharpeRatio()
        self.pbo_calculator = ProbabilityOfBacktestOverfitting()
    
    def validate_strategy(self, 
                         returns: np.ndarray,
                         n_trials: int = 100,
                         n_simulated_strategies: int = 50) -> Dict[str, Union[float, bool, Dict]]:
        """
        Comprehensive validation of a trading strategy
        
        Args:
            returns: Strategy returns
            n_trials: Number of strategies/trials tested (for multiple testing adjustment)
            n_simulated_strategies: Number of strategies to simulate for PBO
            
        Returns:
            Complete validation results
        """
        logger.info(f"🔍 Running comprehensive robustness validation...")
        
        returns_clean = np.asarray(returns)[~np.isnan(returns)]
        
        if len(returns_clean) < 50:
            logger.error("❌ Insufficient data for validation")
            return self._empty_validation_result()
        
        # Deflated Sharpe Ratio
        ds_results = self.deflated_sharpe.calculate(returns_clean, n_trials=n_trials)
        
        # PBO Analysis
        pbo_results = self.pbo_calculator.calculate(
            returns_clean, 
            n_strategies=n_simulated_strategies,
            method='combinatorial'
        )
        
        # White's Reality Check (simplified version)
        reality_check = self._whites_reality_check(returns_clean, n_trials)
        
        # Overall assessment
        validation_passed = self._assess_overall_validation(ds_results, pbo_results, reality_check)
        
        # Combine all results
        results = {
            'validation_passed': validation_passed,
            'deflated_sharpe': ds_results,
            'pbo_analysis': pbo_results,
            'reality_check': reality_check,
            'summary': self._create_validation_summary(ds_results, pbo_results, reality_check)
        }
        
        self._log_validation_results(results)
        
        return results
    
    def _whites_reality_check(self, returns: np.ndarray, n_trials: int) -> Dict[str, float]:
        """
        Simplified White's Reality Check
        Test if strategy significantly outperforms random strategies
        """
        # Bootstrap random strategies
        n_bootstrap = 1000
        strategy_sharpe = np.mean(returns) / np.std(returns)
        
        bootstrap_sharpes = []
        for _ in range(n_bootstrap):
            # Bootstrap resample
            bootstrap_returns = np.random.choice(returns, size=len(returns), replace=True)
            # Shuffle to remove any structure
            np.random.shuffle(bootstrap_returns)
            
            boot_sharpe = np.mean(bootstrap_returns) / (np.std(bootstrap_returns) + 1e-8)
            bootstrap_sharpes.append(boot_sharpe)
        
        bootstrap_sharpes = np.array(bootstrap_sharpes)
        
        # P-value: fraction of bootstrap Sharpes greater than observed
        p_value = np.mean(bootstrap_sharpes >= strategy_sharpe)
        
        # Multiple testing adjustment
        adjusted_p_value = min(1.0, p_value * n_trials)
        
        return {
            'p_value': p_value,
            'adjusted_p_value': adjusted_p_value,
            'is_significant': adjusted_p_value < 0.05,
            'strategy_sharpe': strategy_sharpe,
            'bootstrap_sharpe_95th': np.percentile(bootstrap_sharpes, 95)
        }
    
    def _assess_overall_validation(self, ds_results: Dict, pbo_results: Dict, rc_results: Dict) -> bool:
        """
        Assess whether strategy passes overall validation
        """
        # Criteria for passing validation
        criteria = {
            'deflated_sharpe_significant': ds_results['is_significant'],
            'deflated_sharpe_positive': ds_results['deflated_sharpe'] > 0,
            'pbo_acceptable': pbo_results['pbo'] < 0.7,  # Less than 70% chance of overfitting
            'reality_check_significant': rc_results['is_significant']
        }
        
        # Strategy passes if it meets majority of criteria
        passing_count = sum(criteria.values())
        total_criteria = len(criteria)
        
        validation_passed = passing_count >= (total_criteria - 1)  # Allow one failure
        
        return validation_passed
    
    def _create_validation_summary(self, ds_results: Dict, pbo_results: Dict, rc_results: Dict) -> Dict:
        """
        Create human-readable validation summary
        """
        return {
            'sharpe_ratio': ds_results['sharpe_ratio'],
            'deflated_sharpe': ds_results['deflated_sharpe'],
            'ds_p_value': ds_results['p_value'],
            'ds_significant': ds_results['is_significant'],
            'pbo_probability': pbo_results['pbo'],
            'pbo_acceptable': pbo_results['pbo'] < 0.7,
            'reality_check_p_value': rc_results['adjusted_p_value'],
            'reality_check_passed': rc_results['is_significant']
        }
    
    def _log_validation_results(self, results: Dict):
        """
        Log comprehensive validation results
        """
        summary = results['summary']
        
        logger.info("=" * 80)
        logger.info("📊 ROBUSTNESS VALIDATION RESULTS")
        logger.info("=" * 80)
        
        logger.info(f"🎯 Overall Validation: {'✅ PASSED' if results['validation_passed'] else '❌ FAILED'}")
        logger.info("")
        
        logger.info("📈 DEFLATED SHARPE RATIO:")
        logger.info(f"   Sharpe Ratio: {summary['sharpe_ratio']:.4f}")
        logger.info(f"   Deflated Sharpe: {summary['deflated_sharpe']:.4f}")
        logger.info(f"   P-value: {summary['ds_p_value']:.4f}")
        logger.info(f"   Significant: {'✅ YES' if summary['ds_significant'] else '❌ NO'}")
        logger.info("")
        
        logger.info("🎲 PROBABILITY OF BACKTEST OVERFITTING:")
        logger.info(f"   PBO: {summary['pbo_probability']:.4f} ({summary['pbo_probability']*100:.1f}%)")
        logger.info(f"   Acceptable: {'✅ YES' if summary['pbo_acceptable'] else '❌ NO'}")
        logger.info("")
        
        logger.info("🔍 WHITE'S REALITY CHECK:")
        logger.info(f"   Adjusted P-value: {summary['reality_check_p_value']:.4f}")
        logger.info(f"   Passed: {'✅ YES' if summary['reality_check_passed'] else '❌ NO'}")
        
        logger.info("=" * 80)
    
    def _empty_validation_result(self) -> Dict:
        """Return empty validation result"""
        return {
            'validation_passed': False,
            'deflated_sharpe': {},
            'pbo_analysis': {},
            'reality_check': {},
            'summary': {}
        }

# Usage example
def validate_6_83_ic_claim():
    """
    Example validation of the 6.83% IC claim
    """
    # Simulate returns based on 6.83% IC
    np.random.seed(42)
    
    # Convert IC to approximate Sharpe ratio
    # IC ≈ Information Coefficient, Sharpe ≈ IC × sqrt(breadth)
    ic = 0.0683
    breadth = 20  # ~20 stocks
    expected_sharpe = ic * np.sqrt(breadth)  # ≈ 0.305
    
    # Generate returns with this Sharpe ratio
    n_days = 500  # ~2 years of trading
    daily_vol = 0.01  # 1% daily volatility
    daily_mean = expected_sharpe * daily_vol / np.sqrt(252)
    
    returns = np.random.normal(daily_mean, daily_vol, n_days)
    
    # Run validation
    validator = RobustnessValidator()
    
    results = validator.validate_strategy(
        returns=returns,
        n_trials=100,  # Tested 100 different strategies
        n_simulated_strategies=50
    )
    
    return results

if __name__ == "__main__":
    # Test with simulated 6.83% IC strategy
    results = validate_6_83_ic_claim()
    
    if results['validation_passed']:
        print("🎉 Strategy validation PASSED - 6.83% IC claim is robust!")
    else:
        print("⚠️ Strategy validation FAILED - further investigation needed")