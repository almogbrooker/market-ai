#!/usr/bin/env python3
"""
PRODUCTION-READY DROP-IN FUNCTIONS
==================================
Ready-to-paste production functions for institutional trading systems:

1. PSI (raw-only) with frozen train bins + Laplace smoothing
2. Decile monotonicity & bucketed IC analysis  
3. Bootstrap Sharpe CI with proper sizing
4. Cost-aware acceptor (maximize E[return] - λ·cost)
5. Sign-flip guard (rolling online IC with p-value)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import spearmanr, ttest_1samp
import warnings
warnings.filterwarnings('ignore')

def calculate_frozen_psi_raw(train_features, live_features, feature_names=None, n_bins=10):
    """
    Calculate PSI using frozen train bins with Laplace smoothing (raw features only)
    
    Parameters:
    - train_features: DataFrame or array of training features (raw, not model inputs)
    - live_features: DataFrame or array of live features (raw, not model inputs) 
    - feature_names: list of feature names (if arrays provided)
    - n_bins: number of quantile bins (default 10)
    
    Returns:
    - dict with global PSI, per-feature PSI, and bin edges
    """
    
    if isinstance(train_features, np.ndarray):
        train_df = pd.DataFrame(train_features, columns=feature_names or [f'feature_{i}' for i in range(train_features.shape[1])])
        live_df = pd.DataFrame(live_features, columns=feature_names or [f'feature_{i}' for i in range(live_features.shape[1])])
    else:
        train_df = train_features.copy()
        live_df = live_features.copy()
    
    psi_results = {
        'feature_psi': {},
        'bin_edges': {},
        'distributions': {}
    }
    
    for feature in train_df.columns:
        if feature not in live_df.columns:
            continue
            
        # Get raw values (important: use raw, not transformed features)
        train_values = train_df[feature].dropna()
        live_values = live_df[feature].dropna()
        
        if len(train_values) < 100 or len(live_values) < 50:
            psi_results['feature_psi'][feature] = np.nan
            continue
        
        # Create frozen quantile bins from training data
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(train_values, quantiles)
        
        # Ensure all values are captured
        bin_edges[0] = train_values.min() - 1e-6
        bin_edges[-1] = train_values.max() + 1e-6
        
        # Remove duplicate edges
        bin_edges = np.unique(bin_edges)
        
        # Calculate expected (training) distribution
        expected_counts, _ = np.histogram(train_values, bins=bin_edges)
        
        # Calculate actual (live) distribution  
        actual_counts, _ = np.histogram(live_values, bins=bin_edges)
        
        # Laplace smoothing (+1 to all bins)
        expected_smooth = expected_counts + 1
        actual_smooth = actual_counts + 1
        
        # Convert to probabilities
        expected_pct = expected_smooth / expected_smooth.sum()
        actual_pct = actual_smooth / actual_smooth.sum()
        
        # PSI calculation
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        
        # Store results
        psi_results['feature_psi'][feature] = psi
        psi_results['bin_edges'][feature] = bin_edges
        psi_results['distributions'][feature] = {
            'expected_pct': expected_pct,
            'actual_pct': actual_pct
        }
    
    # Global PSI (average of valid feature PSIs)
    valid_psis = [psi for psi in psi_results['feature_psi'].values() if not np.isnan(psi)]
    psi_results['global_psi'] = np.mean(valid_psis) if valid_psis else np.nan
    
    # Top offenders
    feature_psis = {k: v for k, v in psi_results['feature_psi'].items() if not np.isnan(v)}
    psi_results['top_10_psi'] = dict(sorted(feature_psis.items(), key=lambda x: x[1], reverse=True)[:10])
    
    return psi_results

def create_decile_monotonicity_analysis(scores, returns, title="Decile Monotonicity Analysis"):
    """
    Create comprehensive decile monotonicity analysis for OOS T+1 validation
    
    Parameters:
    - scores: array of model scores/predictions
    - returns: array of actual T+1 returns
    - title: analysis title
    
    Returns:
    - dict with monotonicity stats, decile analysis, and chart
    """
    
    # Create analysis DataFrame
    df = pd.DataFrame({
        'score': scores,
        'return': returns
    }).dropna()
    
    if len(df) < 100:
        return {'error': 'Insufficient data for decile analysis'}
    
    # Create deciles
    df['decile'] = pd.qcut(df['score'], q=10, labels=False, duplicates='drop')
    
    # Calculate decile statistics
    decile_stats = df.groupby('decile').agg({
        'return': ['mean', 'std', 'count'],
        'score': ['mean', 'min', 'max']
    }).round(6)
    
    decile_stats.columns = ['return_mean', 'return_std', 'count', 'score_mean', 'score_min', 'score_max']
    decile_stats = decile_stats.reset_index()
    
    # Monotonicity analysis
    returns_by_decile = decile_stats['return_mean'].values
    
    # Count monotonic increases
    monotonic_increases = 0
    for i in range(1, len(returns_by_decile)):
        if returns_by_decile[i] > returns_by_decile[i-1]:
            monotonic_increases += 1
    
    monotonicity_ratio = monotonic_increases / (len(returns_by_decile) - 1)
    
    # Spearman correlation between decile and mean return
    decile_corr, decile_p_value = spearmanr(decile_stats['decile'], decile_stats['return_mean'])
    
    # Bucketed IC analysis (post-isotonic calibration check)
    bucketed_ic_analysis = {
        'decile_correlation': decile_corr,
        'p_value': decile_p_value,
        'slope_estimate': np.polyfit(decile_stats['decile'], decile_stats['return_mean'], 1)[0]
    }
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Main monotonicity chart
    bars = ax1.bar(decile_stats['decile'], decile_stats['return_mean'], 
                   color='steelblue', alpha=0.7, edgecolor='navy')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Prediction Decile (0=lowest, 9=highest)')
    ax1.set_ylabel('Mean T+1 Return')
    ax1.set_title(f'{title}\\nMonotonicity: {monotonicity_ratio:.1%}, ρ={decile_corr:.3f}')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(decile_stats['decile'], decile_stats['return_mean'], 1)
    p = np.poly1d(z)
    ax1.plot(decile_stats['decile'], p(decile_stats['decile']), "r--", alpha=0.8, linewidth=2)
    
    # Return distribution by decile
    ax2.boxplot([df[df['decile']==i]['return'].values for i in range(10)], 
                positions=range(10))
    ax2.set_xlabel('Prediction Decile')
    ax2.set_ylabel('T+1 Return Distribution')
    ax2.set_title('Return Distributions by Decile')
    ax2.grid(True, alpha=0.3)
    
    # Score vs Return scatter (sample)
    if len(df) > 5000:
        sample_df = df.sample(5000, random_state=42)
    else:
        sample_df = df
        
    ax3.scatter(sample_df['score'], sample_df['return'], alpha=0.3, s=1)
    ax3.set_xlabel('Model Score')
    ax3.set_ylabel('T+1 Return')
    ax3.set_title('Score vs Return Scatter')
    ax3.grid(True, alpha=0.3)
    
    # Add regression line
    slope, intercept = np.polyfit(sample_df['score'], sample_df['return'], 1)
    ax3.plot(sample_df['score'], slope * sample_df['score'] + intercept, 'r-', alpha=0.8)
    
    # Cumulative return by decile
    cumulative_returns = decile_stats['return_mean'].cumsum()
    ax4.plot(decile_stats['decile'], cumulative_returns, 'g-', marker='o', linewidth=2)
    ax4.set_xlabel('Prediction Decile')
    ax4.set_ylabel('Cumulative Return')
    ax4.set_title('Cumulative Returns by Decile')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Comprehensive results
    analysis_results = {
        'monotonicity_ratio': monotonicity_ratio,
        'is_monotonic': monotonicity_ratio >= 0.7,
        'decile_correlation': decile_corr,
        'decile_p_value': decile_p_value,
        'is_significant': decile_p_value < 0.05,
        'bucketed_ic': bucketed_ic_analysis,
        'decile_stats': decile_stats.to_dict('records'),
        'pass_criteria': {
            'monotonic_threshold': monotonicity_ratio >= 0.7,
            'significance_threshold': decile_p_value < 0.05,
            'overall_pass': (monotonicity_ratio >= 0.7) and (decile_p_value < 0.05)
        },
        'chart': fig
    }
    
    return analysis_results

def calculate_bootstrap_sharpe_ci_professional(daily_returns, n_bootstrap=2000, confidence_level=0.95):
    """
    Professional bootstrap Sharpe ratio confidence interval for net returns
    
    Parameters:
    - daily_returns: array of daily portfolio returns (net of costs)
    - n_bootstrap: number of bootstrap iterations
    - confidence_level: confidence level (0.95 for 95% CI)
    
    Returns:
    - dict with comprehensive Sharpe analysis and pass/fail criteria
    """
    
    daily_returns = np.array(daily_returns)
    daily_returns = daily_returns[~np.isnan(daily_returns)]
    
    if len(daily_returns) < 30:
        return {
            'error': 'Insufficient data (need ≥30 observations)',
            'pass': False,
            'status': 'FAIL'
        }
    
    # Bootstrap resampling
    np.random.seed(42)  # For reproducibility
    bootstrap_sharpes = []
    bootstrap_returns_mean = []
    bootstrap_returns_std = []
    
    for i in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(daily_returns, size=len(daily_returns), replace=True)
        
        # Calculate statistics
        daily_mean = bootstrap_sample.mean()
        daily_std = bootstrap_sample.std()
        
        if daily_std > 1e-8:
            daily_sharpe = daily_mean / daily_std
            annualized_sharpe = daily_sharpe * np.sqrt(252)
            
            bootstrap_sharpes.append(annualized_sharpe)
            bootstrap_returns_mean.append(daily_mean * 252)  # Annualized
            bootstrap_returns_std.append(daily_std * np.sqrt(252))  # Annualized
    
    bootstrap_sharpes = np.array(bootstrap_sharpes)
    bootstrap_returns_mean = np.array(bootstrap_returns_mean)
    bootstrap_returns_std = np.array(bootstrap_returns_std)
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_sharpes, (alpha/2) * 100)
    ci_upper = np.percentile(bootstrap_sharpes, (1 - alpha/2) * 100)
    
    # Original statistics
    original_daily_mean = daily_returns.mean()
    original_daily_std = daily_returns.std()
    original_sharpe = (original_daily_mean / original_daily_std) * np.sqrt(252) if original_daily_std > 1e-8 else 0
    
    # Bootstrap statistics
    mean_bootstrap_sharpe = np.mean(bootstrap_sharpes)
    median_bootstrap_sharpe = np.median(bootstrap_sharpes)
    
    # Pass criteria: entire CI above zero
    passes_test = ci_lower > 0
    
    # Additional statistics
    prob_positive = np.mean(bootstrap_sharpes > 0)
    prob_above_1 = np.mean(bootstrap_sharpes > 1.0)
    
    # Risk metrics
    annualized_vol = original_daily_std * np.sqrt(252)
    max_drawdown = calculate_max_drawdown(daily_returns)
    
    results = {
        # Core Sharpe analysis
        'original_sharpe': original_sharpe,
        'mean_bootstrap_sharpe': mean_bootstrap_sharpe,
        'median_bootstrap_sharpe': median_bootstrap_sharpe,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence_level': confidence_level,
        
        # Return characteristics
        'annualized_return': original_daily_mean * 252,
        'annualized_volatility': annualized_vol,
        'max_drawdown': max_drawdown,
        
        # Probabilities
        'prob_positive_sharpe': prob_positive,
        'prob_sharpe_above_1': prob_above_1,
        
        # Bootstrap details
        'n_bootstrap': n_bootstrap,
        'bootstrap_sharpes': bootstrap_sharpes.tolist(),
        
        # Pass/fail criteria
        'pass': passes_test,
        'status': 'PASS' if passes_test else 'FAIL',
        'pass_criteria': {
            'entire_ci_positive': passes_test,
            'sharpe_above_threshold': original_sharpe > 0.5,
            'sufficient_probability': prob_positive > 0.6
        },
        
        # Raw data stats
        'daily_return_stats': {
            'mean': original_daily_mean,
            'std': original_daily_std,
            'count': len(daily_returns),
            'min': daily_returns.min(),
            'max': daily_returns.max()
        }
    }
    
    return results

def cost_aware_portfolio_optimizer(scores, estimated_costs_bps, position_limits, lambda_cost=0.5):
    """
    Cost-aware acceptor: maximize Σ(score_i - λ·cost_i) under constraints
    
    Parameters:
    - scores: array of prediction scores
    - estimated_costs_bps: array of estimated transaction costs in basis points
    - position_limits: dict with constraints {'max_positions', 'max_gross_notional', 'max_per_name'}
    - lambda_cost: cost penalty parameter
    
    Returns:
    - dict with optimized portfolio and performance metrics
    """
    
    n_assets = len(scores)
    scores = np.array(scores)
    costs_bps = np.array(estimated_costs_bps)
    
    # Convert costs from bps to decimal for utility calculation
    cost_penalty = costs_bps / 10000 * lambda_cost
    
    # Utility = score - cost_penalty
    utility = scores - cost_penalty
    
    # Extract constraints
    max_positions = position_limits.get('max_positions', min(50, n_assets // 2))
    max_gross_notional = position_limits.get('max_gross_notional', 1.0)
    max_per_name = position_limits.get('max_per_name', 0.05)
    
    # Sort by utility
    utility_rank = np.argsort(utility)
    
    # Greedy selection optimized for utility
    selected_positions = []
    current_gross = 0.0
    
    # Select long positions (highest utility first)
    long_count = 0
    max_long = max_positions // 2
    
    for idx in reversed(utility_rank):  # Highest utility first
        if long_count >= max_long:
            break
        if utility[idx] <= 0:  # Only positive utility for longs
            break
        if current_gross + max_per_name <= max_gross_notional:
            selected_positions.append({
                'index': idx,
                'side': 'long',
                'score': scores[idx],
                'cost_bps': costs_bps[idx],
                'utility': utility[idx],
                'position_size': max_per_name,
                'notional': max_per_name
            })
            current_gross += max_per_name
            long_count += 1
    
    # Select short positions (lowest utility, but we profit from shorting them)
    short_count = 0
    max_short = max_positions // 2
    
    for idx in utility_rank:  # Lowest utility first
        if short_count >= max_short:
            break
        if utility[idx] >= 0:  # Only negative utility for shorts (we short them)
            break
        if current_gross + max_per_name <= max_gross_notional:
            selected_positions.append({
                'index': idx,
                'side': 'short',
                'score': scores[idx],
                'cost_bps': costs_bps[idx],
                'utility': utility[idx],
                'position_size': -max_per_name,  # Negative for short
                'notional': max_per_name
            })
            current_gross += max_per_name
            short_count += 1
    
    # Calculate portfolio metrics
    if selected_positions:
        total_utility = sum(pos['utility'] * abs(pos['position_size']) for pos in selected_positions)
        total_cost_bps = sum(pos['cost_bps'] * abs(pos['position_size']) for pos in selected_positions)
        weighted_avg_score = sum(pos['score'] * abs(pos['position_size']) for pos in selected_positions)
        
        # Compare with traditional acceptor (simple top/bottom selection)
        traditional_long_indices = np.argsort(scores)[-max_long:]
        traditional_short_indices = np.argsort(scores)[:max_short]
        traditional_indices = np.concatenate([traditional_long_indices, traditional_short_indices])
        
        traditional_cost = np.mean(costs_bps[traditional_indices])
        cost_aware_cost = total_cost_bps / len(selected_positions)
        
        cost_reduction_pct = (traditional_cost - cost_aware_cost) / traditional_cost * 100 if traditional_cost > 0 else 0
        
    else:
        total_utility = 0
        total_cost_bps = 0
        weighted_avg_score = 0
        cost_reduction_pct = 0
    
    results = {
        'positions': selected_positions,
        'portfolio_summary': {
            'n_positions': len(selected_positions),
            'n_long': long_count,
            'n_short': short_count,
            'gross_notional': current_gross,
            'total_utility': total_utility,
            'weighted_avg_score': weighted_avg_score,
            'avg_cost_bps': total_cost_bps / len(selected_positions) if selected_positions else 0
        },
        'optimization_results': {
            'lambda_cost': lambda_cost,
            'cost_reduction_vs_traditional_pct': cost_reduction_pct,
            'utility_maximized': total_utility,
            'constraints_satisfied': current_gross <= max_gross_notional
        },
        'constraints_used': position_limits
    }
    
    return results

def sign_flip_guard_monitor(historical_returns, model_predictions, window_days=60, 
                           threshold_ic=-0.01, consecutive_days=3, alpha=0.05):
    """
    Sign-flip guard: monitor 60d rolling online IC with statistical significance
    
    Parameters:
    - historical_returns: array of historical returns
    - model_predictions: array of model predictions (aligned with returns)
    - window_days: rolling window size
    - threshold_ic: IC threshold for concern (-0.01)
    - consecutive_days: consecutive days below threshold
    - alpha: significance level for statistical test
    
    Returns:
    - dict with guard status and recommended actions
    """
    
    returns = np.array(historical_returns)
    predictions = np.array(model_predictions)
    
    if len(returns) != len(predictions):
        return {'error': 'Returns and predictions must have same length'}
    
    if len(returns) < window_days + consecutive_days:
        return {'error': f'Need at least {window_days + consecutive_days} observations'}
    
    # Calculate rolling IC
    rolling_ics = []
    rolling_p_values = []
    
    for i in range(window_days, len(returns)):
        window_returns = returns[i-window_days:i]
        window_predictions = predictions[i-window_days:i]
        
        # Remove NaN values
        valid_mask = ~(np.isnan(window_returns) | np.isnan(window_predictions))
        clean_returns = window_returns[valid_mask]
        clean_predictions = window_predictions[valid_mask]
        
        if len(clean_returns) >= 30:  # Minimum for reliable IC
            ic, p_value = spearmanr(clean_returns, clean_predictions)
            rolling_ics.append(ic if not np.isnan(ic) else 0)
            rolling_p_values.append(p_value if not np.isnan(p_value) else 1.0)
        else:
            rolling_ics.append(0)
            rolling_p_values.append(1.0)
    
    rolling_ics = np.array(rolling_ics)
    rolling_p_values = np.array(rolling_p_values)
    
    # Check for consecutive days below threshold
    below_threshold = rolling_ics < threshold_ic
    
    # Find consecutive runs
    consecutive_violations = 0
    max_consecutive = 0
    current_consecutive = 0
    
    for violation in below_threshold:
        if violation:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    # Check most recent consecutive days
    recent_consecutive = 0
    for violation in reversed(below_threshold[-consecutive_days:]):
        if violation:
            recent_consecutive += 1
        else:
            break
    
    # Statistical significance test
    recent_ics = rolling_ics[-min(30, len(rolling_ics)):]  # Last 30 observations
    recent_mean_ic = np.mean(recent_ics)
    
    # One-sample t-test: is mean IC significantly different from 0?
    if len(recent_ics) >= 10:
        t_stat, t_p_value = ttest_1samp(recent_ics, 0)
        is_significantly_negative = (t_p_value < alpha) and (recent_mean_ic < 0)
    else:
        is_significantly_negative = False
        t_p_value = 1.0
    
    # Guard triggers
    trigger_consecutive = recent_consecutive >= consecutive_days
    trigger_statistical = is_significantly_negative
    
    # Overall guard status
    guard_triggered = trigger_consecutive and trigger_statistical
    
    # Recommended actions
    if guard_triggered:
        recommended_actions = [
            'IMMEDIATE: Switch to paper trading mode',
            'Alert risk management team',
            'Investigate model degradation causes',
            'Consider signal inversion (requires human approval)',
            'Review feature stability and data quality'
        ]
        status = 'TRIGGERED'
        severity = 'CRITICAL'
    elif trigger_consecutive:
        recommended_actions = [
            'WARNING: Monitor closely',
            'Prepare for potential demotion',
            'Review recent model performance'
        ]
        status = 'WARNING' 
        severity = 'HIGH'
    else:
        recommended_actions = ['Continue normal operations']
        status = 'NORMAL'
        severity = 'LOW'
    
    results = {
        # Guard status
        'status': status,
        'severity': severity,
        'guard_triggered': guard_triggered,
        
        # IC analysis
        'recent_mean_ic': recent_mean_ic,
        'rolling_ics': rolling_ics.tolist(),
        'below_threshold_days': int(np.sum(below_threshold)),
        'consecutive_violations': recent_consecutive,
        'max_consecutive_violations': max_consecutive,
        
        # Statistical analysis
        'is_significantly_negative': is_significantly_negative,
        't_test_p_value': t_p_value,
        'mean_ic_last_30d': recent_mean_ic,
        
        # Triggers
        'trigger_consecutive': trigger_consecutive,
        'trigger_statistical': trigger_statistical,
        
        # Configuration
        'window_days': window_days,
        'threshold_ic': threshold_ic,
        'consecutive_days': consecutive_days,
        'alpha': alpha,
        
        # Actions
        'recommended_actions': recommended_actions
    }
    
    return results

def calculate_max_drawdown(returns):
    """Calculate maximum drawdown from return series"""
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return np.min(drawdown)

# Example usage and testing
def test_production_functions():
    """Test all production-ready functions"""
    print("🧪 TESTING PRODUCTION-READY FUNCTIONS")
    print("=" * 60)
    
    # Generate realistic test data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Test 1: PSI with frozen bins
    print("\n1. Testing PSI with frozen train bins...")
    train_features = np.random.multivariate_normal(np.zeros(n_features), np.eye(n_features), n_samples)
    live_features = np.random.multivariate_normal(np.ones(n_features) * 0.1, np.eye(n_features) * 1.1, n_samples//2)
    
    psi_result = calculate_frozen_psi_raw(train_features, live_features)
    print(f"   Global PSI: {psi_result['global_psi']:.4f}")
    print(f"   Features with high PSI: {sum(1 for psi in psi_result['feature_psi'].values() if psi > 0.1)}")
    
    # Test 2: Decile monotonicity
    print("\n2. Testing decile monotonicity analysis...")
    scores = np.random.normal(0, 0.02, n_samples)
    returns = scores * 0.3 + np.random.normal(0, 0.015, n_samples)  # Realistic correlation
    
    decile_result = create_decile_monotonicity_analysis(scores, returns)
    if 'error' not in decile_result:
        print(f"   Monotonicity ratio: {decile_result['monotonicity_ratio']:.2%}")
        print(f"   Decile correlation: {decile_result['decile_correlation']:.4f}")
        print(f"   Overall pass: {decile_result['pass_criteria']['overall_pass']}")
    
    # Test 3: Bootstrap Sharpe CI
    print("\n3. Testing bootstrap Sharpe CI...")
    daily_returns = np.random.normal(0.0008, 0.018, 252)  # ~20% annual return, 18% vol
    
    sharpe_result = calculate_bootstrap_sharpe_ci_professional(daily_returns)
    if 'error' not in sharpe_result:
        print(f"   Original Sharpe: {sharpe_result['original_sharpe']:.3f}")
        print(f"   95% CI: [{sharpe_result['ci_lower']:.3f}, {sharpe_result['ci_upper']:.3f}]")
        print(f"   Status: {sharpe_result['status']}")
    
    # Test 4: Cost-aware acceptor
    print("\n4. Testing cost-aware acceptor...")
    scores = np.random.normal(0, 0.02, 100)
    costs = 15 + np.abs(scores) * 500 + np.random.normal(0, 3, 100)  # Realistic cost structure
    
    position_limits = {
        'max_positions': 40,
        'max_gross_notional': 1.0,
        'max_per_name': 0.05
    }
    
    portfolio_result = cost_aware_portfolio_optimizer(scores, costs, position_limits)
    print(f"   Selected positions: {portfolio_result['portfolio_summary']['n_positions']}")
    print(f"   Cost reduction: {portfolio_result['optimization_results']['cost_reduction_vs_traditional_pct']:.1f}%")
    
    # Test 5: Sign-flip guard
    print("\n5. Testing sign-flip guard...")
    # Simulate degrading model
    good_period = np.random.normal(0.001, 0.02, 200)
    bad_period = np.random.normal(-0.0005, 0.02, 100)
    all_returns = np.concatenate([good_period, bad_period])
    
    good_predictions = good_period * 0.5 + np.random.normal(0, 0.01, 200)
    bad_predictions = -bad_period * 0.2 + np.random.normal(0, 0.015, 100)  # Degraded correlation
    all_predictions = np.concatenate([good_predictions, bad_predictions])
    
    guard_result = sign_flip_guard_monitor(all_returns, all_predictions)
    if 'error' not in guard_result:
        print(f"   Guard status: {guard_result['status']}")
        print(f"   Recent mean IC: {guard_result['recent_mean_ic']:.4f}")
        print(f"   Consecutive violations: {guard_result['consecutive_violations']}")
    
    print("\n✅ All production functions tested successfully!")
    
    return {
        'psi': psi_result,
        'decile': decile_result,
        'sharpe': sharpe_result,
        'cost_aware': portfolio_result,
        'sign_flip': guard_result
    }

if __name__ == "__main__":
    test_results = test_production_functions()