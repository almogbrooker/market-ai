#!/usr/bin/env python3
"""
N-AWARE MONITORING FUNCTIONS
============================
Principled monitoring with binomial statistics and universe size validation
"""

import numpy as np
from scipy.stats import binom

class NAwareMonitoring:
    def __init__(self):
        self.target_accept_rate = 0.18
        self.min_universe_size = 20
        self.small_universe_threshold = 100
        self.min_accepted_for_ic = 10
        self.min_accepted_for_skew = 5
        self.min_fills_for_slippage = 10
        
    def validate_universe_size(self, N):
        """Validate universe size and determine monitoring mode"""
        if N < self.min_universe_size:
            return False, "hard_floor", f"Universe size {N} < {self.min_universe_size} (data too thin)"
        elif N < self.small_universe_threshold:
            return True, "small_universe", f"Small universe mode (N={N})"
        else:
            return True, "normal_universe", f"Normal universe mode (N={N})"
    
    def get_binomial_acceptance_band(self, N, p=None):
        """Get 95% binomial confidence band for acceptance count"""
        if p is None:
            p = self.target_accept_rate
        
        # Two-sided 95% confidence interval
        lower_k = binom.ppf(0.025, N, p)
        upper_k = binom.ppf(0.975, N, p)
        
        return int(lower_k), int(upper_k)
    
    def validate_gate_acceptance(self, accepted_count, universe_size):
        """N-aware gate acceptance validation"""
        valid, mode, message = self.validate_universe_size(universe_size)
        
        if not valid:
            return False, "CRITICAL", message
        
        if mode == "small_universe":
            # Use binomial statistics for small universes
            lower_k, upper_k = self.get_binomial_acceptance_band(universe_size)
            
            if accepted_count < lower_k:
                p_under = binom.cdf(accepted_count, universe_size, self.target_accept_rate)
                return False, "WARNING", f"Under-acceptance: K={accepted_count} < {lower_k} (p={p_under:.3f})"
            elif accepted_count > upper_k:
                p_over = 1 - binom.cdf(accepted_count - 1, universe_size, self.target_accept_rate)
                return False, "WARNING", f"Over-acceptance: K={accepted_count} > {upper_k} (p={p_over:.3f})"
            else:
                return True, "INFO", f"Acceptance normal: K={accepted_count} ∈ [{lower_k}, {upper_k}] (N={universe_size})"
        
        else:  # normal_universe
            # Use percentage bands for large universes
            accept_rate = accepted_count / universe_size
            
            if accept_rate < 0.15 or accept_rate > 0.25:
                return False, "WARNING", f"Accept rate {accept_rate:.1%} outside [15%, 25%]"
            else:
                return True, "INFO", f"Accept rate {accept_rate:.1%} within normal range"
    
    def validate_coverage_skew(self, long_count, short_count, accepted_count, universe_size):
        """N-aware coverage skew validation"""
        if accepted_count < self.min_accepted_for_skew:
            return True, "INFO", f"Coverage skew check skipped (K={accepted_count} < {self.min_accepted_for_skew})"
        
        long_pct = long_count / accepted_count if accepted_count > 0 else 0
        short_pct = short_count / accepted_count if accepted_count > 0 else 0
        
        # Use relaxed thresholds for small universes
        valid, mode, _ = self.validate_universe_size(universe_size)
        threshold = 0.20 if mode == "small_universe" else 0.35
        
        alerts = []
        
        if long_pct < threshold:
            alerts.append(f"Low long coverage: {long_pct:.1%} < {threshold:.1%}")
        
        if short_pct < threshold:
            alerts.append(f"Low short coverage: {short_pct:.1%} < {threshold:.1%}")
        
        if alerts:
            return False, "WARNING", "; ".join(alerts)
        
        return True, "INFO", f"Coverage skew normal: {long_pct:.1%} long, {short_pct:.1%} short"
    
    def validate_gated_ic(self, gated_ic, accepted_count, baseline_ic):
        """N-aware gated IC validation"""
        if accepted_count < self.min_accepted_for_ic:
            return True, "INFO", f"Gated IC check skipped (K={accepted_count} < {self.min_accepted_for_ic})"
        
        alert_floor = baseline_ic * 0.70  # -30% threshold
        
        if gated_ic < alert_floor:
            return False, "WARNING", f"Gated IC {gated_ic:.6f} < alert floor {alert_floor:.6f}"
        
        return True, "INFO", f"Gated IC {gated_ic:.6f} above alert floor {alert_floor:.6f}"
    
    def validate_execution_quality(self, avg_slippage_bps, fill_count, avg_fill_time_ms):
        """N-aware execution quality validation"""
        alerts = []
        
        if fill_count >= self.min_fills_for_slippage:
            if avg_slippage_bps > 6.0:
                alerts.append(f"High avg slippage: {avg_slippage_bps:.2f} bps")
        else:
            # Don't alert on slippage with few fills
            pass
        
        if avg_fill_time_ms > 100:
            alerts.append(f"Slow fills: {avg_fill_time_ms:.1f}ms avg")
        
        if alerts:
            return False, "WARNING", "; ".join(alerts)
        
        return True, "INFO", f"Execution quality normal (slippage: {avg_slippage_bps:.2f} bps, fills: {fill_count})"
    
    def validate_data_separation(self, model_input_columns):
        """Validate that raw macro features are not in model input"""
        deny_list = ["Yield_Spread", "Treasury_10Y"]
        raw_features_found = [col for col in model_input_columns if col in deny_list]
        
        if raw_features_found:
            return False, "CRITICAL", f"Raw features in model input: {raw_features_found}"
        
        # Check that ranked versions are present
        required_ranked = ["Yield_Spread_rank", "Treasury_10Y_rank"]
        missing_ranked = [col for col in required_ranked if col not in model_input_columns]
        
        if missing_ranked:
            return False, "CRITICAL", f"Missing ranked features: {missing_ranked}"
        
        return True, "INFO", "Data separation validated: raw features excluded, ranked features present"

def create_monitoring_report(universe_size, accepted_count, long_count, short_count, 
                           gated_ic, baseline_ic, avg_slippage_bps, fill_count, 
                           avg_fill_time_ms, model_input_columns):
    """Create comprehensive N-aware monitoring report"""
    monitor = NAwareMonitoring()
    
    # Universe validation
    universe_valid, mode, universe_msg = monitor.validate_universe_size(universe_size)
    
    # All validations
    validations = {
        "universe_size": (universe_valid, "CRITICAL" if not universe_valid else "INFO", universe_msg),
        "gate_acceptance": monitor.validate_gate_acceptance(accepted_count, universe_size),
        "coverage_skew": monitor.validate_coverage_skew(long_count, short_count, accepted_count, universe_size),
        "gated_ic": monitor.validate_gated_ic(gated_ic, accepted_count, baseline_ic),
        "execution_quality": monitor.validate_execution_quality(avg_slippage_bps, fill_count, avg_fill_time_ms),
        "data_separation": monitor.validate_data_separation(model_input_columns)
    }
    
    # Compile results
    results = {
        "universe_mode": mode,
        "universe_size": universe_size,
        "monitoring_results": {},
        "alerts": {
            "critical": [],
            "warning": [],
            "info": []
        }
    }
    
    for check_name, (is_valid, severity, message) in validations.items():
        results["monitoring_results"][check_name] = {
            "valid": is_valid,
            "severity": severity,
            "message": message
        }
        
        results["alerts"][severity.lower()].append({
            "check": check_name,
            "message": message
        })
    
    return results
