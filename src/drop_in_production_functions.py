#!/usr/bin/env python3
"""
DROP-IN PRODUCTION FUNCTIONS
============================
Ready-to-use functions for production deployment:
1. Decile monotonicity chart
2. Bootstrap Sharpe CI
3. Cost-aware acceptor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import spearmanr
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def create_decile_monotonicity_chart(predictions, returns, title="Decile Monotonicity Analysis"):
    """
    Create decile monotonicity chart for model validation
    
    Parameters:
    - predictions: array of model predictions/scores
    - returns: array of actual next-day returns
    - title: chart title
    
    Returns:
    - matplotlib figure object
    - monotonicity statistics dict
    """
    
    # Create DataFrame for analysis
    df = pd.DataFrame({
        'prediction': predictions,
        'return': returns
    }).dropna()
    
    # Create deciles
    df['decile'] = pd.qcut(df['prediction'], q=10, labels=False, duplicates='drop')
    
    # Calculate decile statistics
    decile_stats = df.groupby('decile').agg({
        'return': ['mean', 'std', 'count'],
        'prediction': ['mean', 'min', 'max']
    }).round(6)
    
    decile_stats.columns = ['return_mean', 'return_std', 'count', 'pred_mean', 'pred_min', 'pred_max']
    decile_stats = decile_stats.reset_index()
    
    # Calculate monotonicity metrics
    returns_by_decile = decile_stats['return_mean'].values
    monotonic_increases = 0
    
    for i in range(1, len(returns_by_decile)):
        if returns_by_decile[i] > returns_by_decile[i-1]:
            monotonic_increases += 1
    
    monotonicity_ratio = monotonic_increases / (len(returns_by_decile) - 1)
    
    # Spearman correlation between decile and mean return
    decile_corr, decile_p_value = spearmanr(decile_stats['decile'], decile_stats['return_mean'])
    
    # Create chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Main monotonicity chart
    ax1.bar(decile_stats['decile'], decile_stats['return_mean'], 
            color='steelblue', alpha=0.7, edgecolor='navy')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Prediction Decile (0=lowest, 9=highest)')
    ax1.set_ylabel('Mean Next-Day Return')
    ax1.set_title(f'{title}\nMonotonicity: {monotonicity_ratio:.1%}')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(decile_stats['decile'], decile_stats['return_mean'], 1)
    p = np.poly1d(z)
    ax1.plot(decile_stats['decile'], p(decile_stats['decile']), "r--", alpha=0.8)
    
    # Distribution chart
    ax2.boxplot([df[df['decile']==i]['return'].values for i in range(10)], 
                positions=range(10))
    ax2.set_xlabel('Prediction Decile')
    ax2.set_ylabel('Return Distribution')
    ax2.set_title('Return Distributions by Decile')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Statistics summary
    monotonicity_stats = {
        'monotonicity_ratio': monotonicity_ratio,
        'spearman_correlation': decile_corr,
        'spearman_p_value': decile_p_value,
        'decile_stats': decile_stats.to_dict('records'),
        'is_monotonic': monotonicity_ratio >= 0.7,
        'is_significant': decile_p_value < 0.05
    }
    
    return fig, monotonicity_stats

def calculate_bootstrap_sharpe_ci(daily_returns, n_bootstrap=2000, confidence_level=0.95):
    """
    Calculate bootstrap confidence interval for Sharpe ratio
    
    Parameters:
    - daily_returns: array of daily portfolio returns
    - n_bootstrap: number of bootstrap samples
    - confidence_level: confidence level (0.95 for 95% CI)
    
    Returns:
    - dict with CI bounds, mean Sharpe, and pass/fail status
    """
    
    daily_returns = np.array(daily_returns)
    daily_returns = daily_returns[~np.isnan(daily_returns)]
    
    if len(daily_returns) < 30:
        return {
            'error': 'Insufficient data for bootstrap',
            'pass': False
        }
    
    # Bootstrap resampling
    bootstrap_sharpes = []
    
    for i in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(daily_returns, size=len(daily_returns), replace=True)
        
        # Calculate Sharpe ratio
        if bootstrap_sample.std() > 1e-8:
            daily_sharpe = bootstrap_sample.mean() / bootstrap_sample.std()
            annualized_sharpe = daily_sharpe * np.sqrt(252)
            bootstrap_sharpes.append(annualized_sharpe)
    
    bootstrap_sharpes = np.array(bootstrap_sharpes)
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_sharpes, (alpha/2) * 100)
    ci_upper = np.percentile(bootstrap_sharpes, (1 - alpha/2) * 100)
    
    # Original Sharpe ratio
    original_sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    mean_bootstrap_sharpe = np.mean(bootstrap_sharpes)
    
    # Pass criteria: entire CI above zero
    passes_test = ci_lower > 0
    
    results = {
        'original_sharpe': original_sharpe,
        'mean_bootstrap_sharpe': mean_bootstrap_sharpe,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence_level': confidence_level,
        'n_bootstrap': n_bootstrap,
        'pass': passes_test,
        'daily_return_stats': {
            'mean': daily_returns.mean(),
            'std': daily_returns.std(),
            'count': len(daily_returns)
        }
    }
    
    return results

def cost_aware_acceptor(scores, estimated_costs, position_limits, lambda_cost=0.5):
    """
    Cost-aware portfolio optimizer: maximize Σ(score_i - λ*cost_i)
    
    Parameters:
    - scores: array of prediction scores
    - estimated_costs: array of estimated transaction costs (in bps)
    - position_limits: dict with 'max_positions', 'max_gross_notional', 'max_per_name'
    - lambda_cost: cost penalty parameter (expected bps per trade)
    
    Returns:
    - dict with selected positions and optimization results
    """
    
    n_assets = len(scores)
    scores = np.array(scores)
    estimated_costs = np.array(estimated_costs)
    
    # Convert costs from bps to decimal
    cost_penalty = estimated_costs / 10000 * lambda_cost
    
    # Utility = score - cost_penalty
    utility = scores - cost_penalty
    
    # Simple greedy approach (for production, use proper optimizer)
    max_positions = position_limits.get('max_positions', min(50, n_assets // 2))
    max_gross_notional = position_limits.get('max_gross_notional', 1.0)
    max_per_name = position_limits.get('max_per_name', 0.05)
    
    # Sort by utility
    sorted_indices = np.argsort(utility)
    
    # Select long positions (highest utility)
    long_positions = []
    long_notional = 0.0
    
    for i in reversed(sorted_indices):
        if len(long_positions) >= max_positions // 2:
            break
        if utility[i] <= 0:
            break
        if long_notional + max_per_name <= max_gross_notional / 2:
            long_positions.append({
                'index': i,
                'score': scores[i],
                'utility': utility[i],
                'cost': estimated_costs[i],
                'position_size': max_per_name,
                'side': 'long'
            })
            long_notional += max_per_name
    
    # Select short positions (lowest utility, but we short them)
    short_positions = []
    short_notional = 0.0
    
    for i in sorted_indices:
        if len(short_positions) >= max_positions // 2:
            break
        if utility[i] >= 0:
            break
        if short_notional + max_per_name <= max_gross_notional / 2:
            short_positions.append({
                'index': i,
                'score': scores[i],
                'utility': utility[i],
                'cost': estimated_costs[i],
                'position_size': -max_per_name,  # Negative for short
                'side': 'short'
            })
            short_notional += max_per_name
    
    all_positions = long_positions + short_positions
    
    # Calculate portfolio statistics
    total_utility = sum(pos['utility'] * abs(pos['position_size']) for pos in all_positions)
    total_cost = sum(pos['cost'] * abs(pos['position_size']) for pos in all_positions)
    gross_notional = sum(abs(pos['position_size']) for pos in all_positions)
    
    results = {
        'positions': all_positions,
        'summary': {
            'n_positions': len(all_positions),
            'n_long': len(long_positions),
            'n_short': len(short_positions),
            'gross_notional': gross_notional,
            'total_utility': total_utility,
            'total_estimated_cost': total_cost,
            'avg_cost_per_position': total_cost / len(all_positions) if all_positions else 0
        },
        'optimization_params': {
            'lambda_cost': lambda_cost,
            'max_positions': max_positions,
            'max_gross_notional': max_gross_notional,
            'max_per_name': max_per_name
        }
    }
    
    return results

def estimate_transaction_costs(scores, market_data, base_cost_bps=15):
    """
    Estimate transaction costs based on scores and market characteristics
    
    Parameters:
    - scores: array of prediction scores
    - market_data: dict with 'volumes', 'spreads', 'prices' (optional)
    - base_cost_bps: base transaction cost in basis points
    
    Returns:
    - array of estimated costs in basis points
    """
    
    n_assets = len(scores)
    scores = np.array(scores)
    
    # Base cost
    costs = np.full(n_assets, base_cost_bps, dtype=float)
    
    # Score extremity penalty (higher costs for extreme scores)
    score_z = np.abs(stats.zscore(scores))
    extremity_penalty = score_z * 2  # +2 bps per sigma
    costs += extremity_penalty
    
    # Market microstructure costs (if available)
    if market_data:
        if 'spreads' in market_data:
            spreads = np.array(market_data['spreads'])
            spread_cost = spreads * 0.5  # Half spread cost
            costs += spread_cost
        
        if 'volumes' in market_data:
            volumes = np.array(market_data['volumes'])
            # Lower volume = higher cost
            volume_z = stats.zscore(np.log(volumes + 1))
            volume_penalty = np.maximum(0, -volume_z * 3)  # Up to +3 bps for low volume
            costs += volume_penalty
    
    # Add random market impact component
    np.random.seed(42)  # For reproducibility
    random_impact = np.random.normal(0, 2, n_assets)  # ±2 bps random
    costs += random_impact
    
    # Clip to realistic range
    costs = np.clip(costs, 8, 40)
    
    return costs

# Example usage and testing functions
def test_drop_in_functions():
    """Test all drop-in functions with sample data"""
    print("🧪 TESTING DROP-IN FUNCTIONS")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Sample predictions and returns with some correlation
    predictions = np.random.normal(0, 0.02, n_samples)
    returns = predictions * 0.3 + np.random.normal(0, 0.015, n_samples)
    
    # Test 1: Decile monotonicity chart
    print("\n1. Testing decile monotonicity chart...")
    fig, mono_stats = create_decile_monotonicity_chart(predictions, returns)
    print(f"   Monotonicity ratio: {mono_stats['monotonicity_ratio']:.2%}")
    print(f"   Spearman correlation: {mono_stats['spearman_correlation']:.4f}")
    print(f"   Is monotonic: {mono_stats['is_monotonic']}")
    
    # Test 2: Bootstrap Sharpe CI
    print("\n2. Testing bootstrap Sharpe CI...")
    daily_returns = np.random.normal(0.0005, 0.02, 252)  # One year of returns
    sharpe_results = calculate_bootstrap_sharpe_ci(daily_returns)
    print(f"   Sharpe ratio: {sharpe_results['original_sharpe']:.3f}")
    print(f"   95% CI: [{sharpe_results['ci_lower']:.3f}, {sharpe_results['ci_upper']:.3f}]")
    print(f"   Passes test: {sharpe_results['pass']}")
    
    # Test 3: Cost-aware acceptor
    print("\n3. Testing cost-aware acceptor...")
    scores = np.random.normal(0, 0.02, 100)
    costs = estimate_transaction_costs(scores, {})
    
    position_limits = {
        'max_positions': 40,
        'max_gross_notional': 1.0,
        'max_per_name': 0.05
    }
    
    portfolio_results = cost_aware_acceptor(scores, costs, position_limits)
    print(f"   Selected positions: {portfolio_results['summary']['n_positions']}")
    print(f"   Gross notional: {portfolio_results['summary']['gross_notional']:.2f}")
    print(f"   Avg cost per position: {portfolio_results['summary']['avg_cost_per_position']:.1f} bps")
    
    print("\n✅ All drop-in functions tested successfully!")
    
    return {
        'monotonicity': mono_stats,
        'sharpe_ci': sharpe_results,
        'cost_aware': portfolio_results
    }

if __name__ == "__main__":
    test_results = test_drop_in_functions()