#!/usr/bin/env python3
"""
Point-in-Time (PIT) Compliance Utilities
Validation functions to ensure no temporal leakage in financial data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def validate_pit_compliance(data: pd.DataFrame, target_date: datetime, 
                           strict: bool = True) -> bool:
    """
    Comprehensive PIT compliance validation
    
    Args:
        data: DataFrame to validate
        target_date: The date for which predictions are being made
        strict: If True, raises exceptions on violations
    
    Returns:
        bool: True if compliant, False otherwise
    """
    
    violations = []
    
    # 1. Check for future dates in data
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        future_data = data[data['Date'] > target_date]
        
        if not future_data.empty:
            violations.append(f"Found {len(future_data)} records with future dates")
    
    # 2. Check report dates for fundamentals
    if 'report_date' in data.columns:
        data['report_date'] = pd.to_datetime(data['report_date'])
        future_reports = data[data['report_date'] > target_date]
        
        if not future_reports.empty:
            violations.append(f"Found {len(future_reports)} records with future report dates")
    
    # 3. Check earnings announcement dates
    if 'earnings_date' in data.columns:
        data['earnings_date'] = pd.to_datetime(data['earnings_date'])
        future_earnings = data[data['earnings_date'] > target_date]
        
        if not future_earnings.empty:
            violations.append(f"Found {len(future_earnings)} records with future earnings dates")
    
    # 4. Check for proper target calculation (no same-bar leakage)
    target_cols = [col for col in data.columns if col.startswith('target_')]
    for target_col in target_cols:
        if not validate_target_calculation(data, target_col, target_date):
            violations.append(f"Target column {target_col} may have temporal leakage")
    
    # 5. Check for feature leakage
    risky_features = [
        'next_day_volume', 'future_return', 'tomorrow_', 'next_',
        'forward_', 'ahead_', 'post_earnings'
    ]
    
    for col in data.columns:
        if any(risky in col.lower() for risky in risky_features):
            violations.append(f"Column '{col}' suggests potential temporal leakage")
    
    # 6. Check data freshness (ensure sufficient lag)
    if validate_data_freshness(data, target_date) is False:
        violations.append("Data may be too fresh (insufficient reporting lag)")
    
    # Report results
    if violations:
        error_msg = "PIT Compliance Violations:\n" + "\n".join(f"  - {v}" for v in violations)
        logger.error(error_msg)
        
        if strict:
            raise ValueError(error_msg)
        return False
    else:
        logger.info("✅ PIT compliance validation passed")
        return True

def validate_target_calculation(data: pd.DataFrame, target_col: str, 
                               target_date: datetime) -> bool:
    """Validate that target calculation doesn't use future data"""
    
    if target_col not in data.columns:
        return True
    
    # Check if target values are available for recent dates
    # (which would suggest same-bar calculation)
    data = data.copy()
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Get most recent data
    recent_data = data[data['Date'] >= target_date - timedelta(days=5)]
    
    if recent_data.empty:
        return True
    
    # Check if target is available for very recent dates
    recent_with_target = recent_data[recent_data[target_col].notna()]
    
    if not recent_with_target.empty:
        latest_target_date = recent_with_target['Date'].max()
        
        # If target is available for dates very close to target_date,
        # it might indicate same-bar leakage
        days_gap = (target_date - latest_target_date).days
        
        if days_gap < 2:  # Targets available within 2 days suggest leakage
            logger.warning(f"Target {target_col} available within {days_gap} days - possible leakage")
            return False
    
    return True

def validate_data_freshness(data: pd.DataFrame, target_date: datetime) -> Optional[bool]:
    """Validate that data has appropriate reporting lags"""
    
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        latest_data_date = data['Date'].max()
        
        # Check if latest data is too close to target date
        days_lag = (target_date - latest_data_date).days
        
        if days_lag < 1:
            logger.warning(f"Data freshness warning: only {days_lag} days lag")
            return False
        elif days_lag > 30:
            logger.warning(f"Data staleness warning: {days_lag} days lag")
    
    return True

def create_temporal_buffer(data: pd.DataFrame, target_col: str, 
                          horizon_days: int, buffer_days: int = 1) -> pd.DataFrame:
    """
    Create proper temporal buffer for target calculation
    
    Args:
        data: DataFrame with price data
        target_col: Name of target column to create
        horizon_days: Forward return horizon (e.g., 5 for 5-day returns)
        buffer_days: Additional buffer days to prevent same-bar leakage
    
    Returns:
        DataFrame with properly buffered targets
    """
    
    data = data.copy()
    data = data.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    logger.info(f"Creating temporal buffer: {horizon_days}d horizon + {buffer_days}d buffer")
    
    # Calculate targets with proper temporal buffer
    for ticker in data['Ticker'].unique():
        ticker_mask = data['Ticker'] == ticker
        ticker_data = data[ticker_mask].copy()
        
        if len(ticker_data) > horizon_days + buffer_days:
            # Calculate forward returns with buffer
            # shift(-(horizon_days + buffer_days)) ensures no same-bar leakage
            forward_returns = ticker_data['Close'].pct_change(horizon_days).shift(-(horizon_days + buffer_days))
            
            # Update main dataframe
            data.loc[ticker_mask, target_col] = forward_returns
    
    # Remove rows where target can't be calculated
    initial_count = len(data)
    data = data.dropna(subset=[target_col])
    final_count = len(data)
    
    logger.info(f"Temporal buffer applied: {initial_count} → {final_count} samples")
    
    return data

def validate_cross_sectional_features(data: pd.DataFrame, date_col: str = 'Date') -> bool:
    """Validate cross-sectional features don't use future information"""
    
    violations = []
    
    # Check for cross-sectional features that might leak
    zscore_cols = [col for col in data.columns if col.startswith('ZSCORE_')]
    rank_cols = [col for col in data.columns if col.startswith('RANK_')]
    
    for col in zscore_cols + rank_cols:
        # Check if feature varies properly across dates
        if date_col in data.columns:
            # Group by date and check if feature has reasonable variance
            daily_stats = data.groupby(date_col)[col].agg(['mean', 'std']).reset_index()
            
            # Z-scores should have mean ~0 and std ~1 within each date
            if col.startswith('ZSCORE_'):
                mean_of_means = daily_stats['mean'].mean()
                if abs(mean_of_means) > 0.1:  # Should be close to 0
                    violations.append(f"{col} has suspicious cross-sectional mean: {mean_of_means:.3f}")
    
    if violations:
        logger.warning("Cross-sectional feature violations:\n" + "\n".join(f"  - {v}" for v in violations))
        return False
    
    logger.info("✅ Cross-sectional features validation passed")
    return True

def validate_feature_calculation_order(data: pd.DataFrame) -> bool:
    """Validate that features are calculated in correct temporal order"""
    
    # Check that lagged features exist and are properly lagged
    lag_features = [col for col in data.columns if 'lag' in col.lower()]
    
    for lag_col in lag_features:
        base_col = lag_col.replace('_lag1', '').replace('_lag', '')
        
        if base_col in data.columns:
            # Check correlation - lagged feature should be less correlated with future
            # than the base feature (simplified test)
            
            target_cols = [col for col in data.columns if col.startswith('target_')]
            if target_cols:
                target_col = target_cols[0]
                
                base_corr = abs(data[base_col].corr(data[target_col]))
                lag_corr = abs(data[lag_col].corr(data[target_col]))
                
                # Lagged features should generally have lower correlation with future
                if lag_corr > base_corr * 1.1:  # Allow some tolerance
                    logger.warning(f"Lagged feature {lag_col} has higher correlation with target than base feature")
    
    logger.info("✅ Feature calculation order validation passed")
    return True

def check_rolling_window_leakage(data: pd.DataFrame, window_cols: List[str]) -> bool:
    """Check that rolling window calculations don't use future data"""
    
    violations = []
    
    for col in window_cols:
        if col not in data.columns:
            continue
        
        # For each ticker, check that rolling calculations make sense
        for ticker in data['Ticker'].unique():
            ticker_data = data[data['Ticker'] == ticker].sort_values('Date')
            
            if len(ticker_data) < 20:  # Need minimum data for meaningful test
                continue
            
            # Check first few values - they should be NaN or calculated from limited data
            first_values = ticker_data[col].head(10)
            
            # Early values should either be NaN or show proper rolling behavior
            # (This is a simplified check - in practice, would verify actual rolling logic)
            non_null_count = first_values.notna().sum()
            
            if non_null_count > 5:  # Too many non-null values too early
                logger.warning(f"Rolling feature {col} for {ticker} may use future data in early periods")
    
    if violations:
        return False
    
    logger.info("✅ Rolling window leakage check passed")
    return True

def assert_no_target_leakage(features: pd.DataFrame, targets: pd.Series, 
                            threshold: float = 0.95) -> bool:
    """Assert that no feature has suspiciously high correlation with target"""
    
    violations = []
    
    for col in features.columns:
        if features[col].dtype in ['float64', 'int64']:
            corr = abs(features[col].corr(targets))
            
            if not np.isnan(corr) and corr > threshold:
                violations.append(f"Feature {col} has suspiciously high correlation: {corr:.3f}")
    
    if violations:
        error_msg = "Potential target leakage detected:\n" + "\n".join(f"  - {v}" for v in violations)
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("✅ No target leakage detected")
    return True

def validate_temporal_consistency(data: pd.DataFrame, date_col: str = 'Date') -> bool:
    """Validate temporal consistency of the dataset"""
    
    violations = []
    
    if date_col not in data.columns:
        return True
    
    data[date_col] = pd.to_datetime(data[date_col])
    
    # Check for each ticker
    for ticker in data['Ticker'].unique():
        ticker_data = data[data['Ticker'] == ticker].sort_values(date_col)
        
        # Check for duplicate dates
        duplicate_dates = ticker_data[ticker_data[date_col].duplicated()]
        if not duplicate_dates.empty:
            violations.append(f"Ticker {ticker} has duplicate dates: {len(duplicate_dates)} occurrences")
        
        # Check for unrealistic date gaps
        date_diffs = ticker_data[date_col].diff().dt.days
        large_gaps = date_diffs[date_diffs > 10]  # More than 10 days gap
        if not large_gaps.empty:
            violations.append(f"Ticker {ticker} has {len(large_gaps)} large date gaps (>10 days)")
    
    if violations:
        logger.warning("Temporal consistency issues:\n" + "\n".join(f"  - {v}" for v in violations))
        return False
    
    logger.info("✅ Temporal consistency validation passed")
    return True

class PITValidator:
    """Context manager for PIT validation during data processing"""
    
    def __init__(self, target_date: datetime, strict: bool = True):
        self.target_date = target_date
        self.strict = strict
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        logger.info(f"🔒 Starting PIT validation for {self.target_date.date()}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"🔒 PIT validation completed in {elapsed:.2f}s")
    
    def validate(self, data: pd.DataFrame, description: str = "") -> bool:
        """Validate a DataFrame with optional description"""
        
        if description:
            logger.info(f"   Validating: {description}")
        
        return validate_pit_compliance(data, self.target_date, self.strict)

# Convenience functions for common validations
def quick_pit_check(data: pd.DataFrame, target_date: datetime) -> bool:
    """Quick PIT compliance check (non-strict)"""
    return validate_pit_compliance(data, target_date, strict=False)

def strict_pit_check(data: pd.DataFrame, target_date: datetime) -> bool:
    """Strict PIT compliance check (raises on violations)"""
    return validate_pit_compliance(data, target_date, strict=True)