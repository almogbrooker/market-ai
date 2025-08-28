#!/usr/bin/env python3
"""
PRODUCTION GUARDRAILS
=====================
Auto-monitoring and response system with P@10/P@20, PSI clipping, and rollback rules
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')

class ProductionGuardrails:
    """Production monitoring and auto-response system"""
    
    def __init__(self):
        self.base_dir = Path("../artifacts")
        
        # Thresholds for automatic actions (updated production gates)
        self.ic_thresholds = {
            'de_risk': 0.000,      # De-risk to 50% if 20d IC < 0
            'rollback': -0.005,    # Auto-rollback if 20d IC < -0.5%
            'min_acceptable': 0.002 # Min daily IC to continue
        }
        
        self.precision_thresholds = {
            'p10_min': 0.51,       # P@10 < 51% → turnover shrink
            'p20_min': 0.51,       # P@20 < 51% → turnover shrink
            'turnover_shrink': 0.05 # Reduce turnover by 5pp
        }
        
        self.drift_thresholds = {
            'psi_warning': 0.15,   # Flag for review
            'psi_action': 0.20,    # Auto-clip and re-standardize
            'ks_warning': 0.05,    # KS test p-value warning
            'ks_action': 0.01      # KS test p-value action
        }
        
        self.cost_thresholds = {
            'borrow_spike': 0.005,     # 50 bps daily borrow cost spike
            'slippage_spike': 0.003,   # 30 bps slippage spike
            'realized_cost_multiplier': 1.5,  # 1.5x assumed costs on 5d window
            'turnover_reduction': 0.05  # Reduce turnover by 5pp
        }
        
        self.risk_thresholds = {
            'drawdown_multiplier': 2.0,  # 2x expected vol triggers action
            'beta_exposure': 0.10,       # Max 10% beta exposure
            'sector_exposure': 0.15      # Max 15% sector exposure
        }
        
        print("🛡️ Production Guardrails initialized")
        print(f"   📊 IC rollback threshold: {self.ic_thresholds['rollback']:.3f}")
        print(f"   🌊 PSI action threshold: {self.drift_thresholds['psi_action']:.2f}")
        print(f"   💰 Cost spike threshold: {self.cost_thresholds['borrow_spike']:.3f}")
    
    def wilson_confidence_interval(self, successes: int, trials: int, confidence: float = 0.95) -> tuple:
        """Calculate Wilson score confidence interval for binomial proportion"""
        if trials == 0:
            return (np.nan, np.nan)
        
        z = 1.96 if confidence == 0.95 else 1.645  # Z-score for confidence level
        p = successes / trials
        denominator = 1 + z**2 / trials
        centre = (p + z**2 / (2 * trials)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denominator
        
        return (max(0, centre - margin), min(1, centre + margin))
    
    def calculate_precision_at_k(self, predictions: pd.Series, returns: pd.Series, k: int = 10) -> dict:
        """Calculate Precision@K (P@K) metric with confidence intervals"""
        if len(predictions) < k or len(returns) < k:
            return {'precision': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}
        
        # Sort by predictions (descending) and get top K
        sorted_idx = predictions.sort_values(ascending=False).index[:k]
        top_k_returns = returns.loc[sorted_idx]
        
        # Calculate precision = fraction of top K with positive returns
        successes = (top_k_returns > 0).sum()
        precision = successes / k
        
        # Wilson confidence interval
        ci_lower, ci_upper = self.wilson_confidence_interval(successes, k)
        
        return {
            'precision': precision,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'successes': successes,
            'trials': k
        }
    
    def calculate_top_bottom_spread(self, predictions: pd.Series, returns: pd.Series, top_pct: float = 0.1, bottom_pct: float = 0.1) -> dict:
        """Calculate top-bottom spread (mean return difference and Sharpe)"""
        if len(predictions) < 20:  # Need sufficient data
            return {'spread_mean': np.nan, 'spread_sharpe': np.nan}
        
        n_top = max(1, int(len(predictions) * top_pct))
        n_bottom = max(1, int(len(predictions) * bottom_pct))
        
        # Sort by predictions
        sorted_data = pd.DataFrame({'pred': predictions, 'ret': returns}).sort_values('pred', ascending=False)
        
        top_returns = sorted_data.head(n_top)['ret']
        bottom_returns = sorted_data.tail(n_bottom)['ret']
        
        # Calculate spread
        top_mean = top_returns.mean()
        bottom_mean = bottom_returns.mean()
        spread_mean = top_mean - bottom_mean
        
        # Calculate Sharpe of spread (assuming daily data)
        spread_std = np.sqrt(top_returns.var()/len(top_returns) + bottom_returns.var()/len(bottom_returns))
        spread_sharpe = spread_mean / spread_std if spread_std > 0 else np.nan
        
        return {
            'spread_mean': spread_mean,
            'spread_sharpe': spread_sharpe,
            'top_mean': top_mean,
            'bottom_mean': bottom_mean,
            'spread_std': spread_std
        }
    
    def calculate_rolling_performance_metrics(self, days: int = 20) -> dict:
        """Calculate rolling performance metrics"""
        print(f"📊 Calculating {days}d rolling metrics...")
        
        # Placeholder - would load actual performance history
        # For demo, generate synthetic data
        dates = pd.date_range(end=datetime.now().date(), periods=days, freq='D')
        
        # Synthetic IC data (declining trend to test thresholds)
        base_ic = 0.008
        trend = -0.0003  # Declining trend
        noise = np.random.normal(0, 0.003, days)
        
        daily_ics = [base_ic + trend * i + noise[i] for i in range(days)]
        daily_returns = np.random.normal(0.0005, 0.02, days)  # Synthetic daily returns
        
        # Create synthetic precision and spread data
        daily_p10_results = []
        daily_p20_results = []
        daily_spreads = []
        
        for i in range(days):
            # Simulate daily predictions and returns
            n_stocks = 96
            pred_sim = np.random.normal(0, 1, n_stocks)
            ret_sim = pred_sim * 0.02 + np.random.normal(0, 0.03, n_stocks)  # Some signal + noise
            
            pred_series = pd.Series(pred_sim)
            ret_series = pd.Series(ret_sim)
            
            # Calculate P@K with confidence intervals
            p10_result = self.calculate_precision_at_k(pred_series, ret_series, 10)
            p20_result = self.calculate_precision_at_k(pred_series, ret_series, 20)
            
            # Calculate top-bottom spread
            spread_result = self.calculate_top_bottom_spread(pred_series, ret_series)
            
            daily_p10_results.append(p10_result)
            daily_p20_results.append(p20_result)
            daily_spreads.append(spread_result)
        
        performance_df = pd.DataFrame({
            'date': dates,
            'daily_ic': daily_ics,
            'daily_return': daily_returns,
            'p10_precision': [r['precision'] for r in daily_p10_results],
            'p10_ci_lower': [r['ci_lower'] for r in daily_p10_results],
            'p10_ci_upper': [r['ci_upper'] for r in daily_p10_results],
            'p20_precision': [r['precision'] for r in daily_p20_results],
            'p20_ci_lower': [r['ci_lower'] for r in daily_p20_results],
            'p20_ci_upper': [r['ci_upper'] for r in daily_p20_results],
            'spread_mean': [s['spread_mean'] for s in daily_spreads],
            'spread_sharpe': [s['spread_sharpe'] for s in daily_spreads]
        })
        
        # Calculate rolling metrics
        rolling_ic = performance_df['daily_ic'].mean()
        ic_trend = np.polyfit(range(days), daily_ics, 1)[0]  # Linear trend
        
        # Enhanced metrics with confidence intervals and spread
        avg_p10 = performance_df['p10_precision'].mean()
        avg_p20 = performance_df['p20_precision'].mean()
        avg_p10_ci_lower = performance_df['p10_ci_lower'].mean()
        avg_p10_ci_upper = performance_df['p10_ci_upper'].mean()
        avg_p20_ci_lower = performance_df['p20_ci_lower'].mean()
        avg_p20_ci_upper = performance_df['p20_ci_upper'].mean()
        
        avg_spread_mean = performance_df['spread_mean'].mean()
        avg_spread_sharpe = performance_df['spread_sharpe'].mean()
        
        metrics = {
            'rolling_ic': rolling_ic,
            'ic_trend': ic_trend,
            'current_ic': daily_ics[-1],
            'avg_p_at_10': avg_p10,
            'avg_p_at_20': avg_p20,
            'p10_ci_lower': avg_p10_ci_lower,
            'p10_ci_upper': avg_p10_ci_upper,
            'p20_ci_lower': avg_p20_ci_lower,
            'p20_ci_upper': avg_p20_ci_upper,
            'avg_spread_mean': avg_spread_mean,
            'avg_spread_sharpe': avg_spread_sharpe,
            'daily_sharpe': np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252),
            'performance_df': performance_df
        }
        
        print(f"   📈 {days}d IC: {rolling_ic:.4f} (trend: {ic_trend:.4f}/day)")
        print(f"   🎯 P@10: {avg_p10:.3f} [95% CI: {avg_p10_ci_lower:.3f}-{avg_p10_ci_upper:.3f}]")
        print(f"   🎯 P@20: {avg_p20:.3f} [95% CI: {avg_p20_ci_lower:.3f}-{avg_p20_ci_upper:.3f}]")
        print(f"   💰 Top-Bottom Spread: {avg_spread_mean:.4f} (Sharpe: {avg_spread_sharpe:.2f})")
        
        return metrics
    
    def check_ic_performance(self, metrics: dict) -> tuple:
        """Check IC performance and recommend actions"""
        rolling_ic = metrics['rolling_ic']
        current_ic = metrics['current_ic']
        
        actions = []
        alerts = []
        
        # Check rollback condition
        if rolling_ic < self.ic_thresholds['rollback']:
            actions.append({
                'type': 'rollback_model',
                'reason': f"20d IC {rolling_ic:.4f} < rollback threshold {self.ic_thresholds['rollback']:.4f}",
                'severity': 'CRITICAL'
            })
        
        # Check de-risk condition
        elif rolling_ic < self.ic_thresholds['de_risk']:
            actions.append({
                'type': 'de_risk_portfolio',
                'reason': f"20d IC {rolling_ic:.4f} < de-risk threshold {self.ic_thresholds['de_risk']:.4f}",
                'severity': 'HIGH',
                'target_exposure': 0.5  # Reduce to 50%
            })
        
        # Check daily minimum
        if current_ic < self.ic_thresholds['min_acceptable']:
            alerts.append({
                'type': 'daily_ic_low',
                'reason': f"Daily IC {current_ic:.4f} < minimum {self.ic_thresholds['min_acceptable']:.4f}",
                'severity': 'MEDIUM'
            })
        
        return actions, alerts
    
    def check_precision_performance(self, metrics: dict) -> tuple:
        """Check P@K performance and recommend actions"""
        actions = []
        alerts = []
        
        p10 = metrics.get('avg_p_at_10', 0.5)
        p20 = metrics.get('avg_p_at_20', 0.5)
        
        # Check P@10 threshold
        if p10 < self.precision_thresholds['p10_min']:
            actions.append({
                'type': 'reduce_turnover_precision',
                'reason': f"P@10 {p10:.3f} < threshold {self.precision_thresholds['p10_min']:.3f}",
                'severity': 'MEDIUM',
                'turnover_reduction': self.precision_thresholds['turnover_shrink']
            })
        
        # Check P@20 threshold  
        if p20 < self.precision_thresholds['p20_min']:
            actions.append({
                'type': 'reduce_turnover_precision',
                'reason': f"P@20 {p20:.3f} < threshold {self.precision_thresholds['p20_min']:.3f}",
                'severity': 'MEDIUM', 
                'turnover_reduction': self.precision_thresholds['turnover_shrink']
            })
        
        return actions, alerts
    
    def check_feature_drift(self) -> tuple:
        """Check for feature drift and auto-clip if needed"""
        print("🌊 Checking feature drift...")
        
        actions = []
        alerts = []
        
        # Simulate feature drift analysis
        feature_names = [
            'momentum_5d_lag3_rank', 'momentum_10d_lag3_rank', 'volatility_10d_lag3_rank',
            'mean_rev_10d_lag3_rank', 'rsi_14d_lag3_rank'
        ]
        
        for feature in feature_names:
            # Simulate PSI calculation
            psi_value = np.random.uniform(0.05, 0.25)  # Random PSI for demo
            
            if psi_value > self.drift_thresholds['psi_action']:
                actions.append({
                    'type': 'auto_clip_feature',
                    'feature': feature,
                    'psi_value': psi_value,
                    'reason': f"PSI {psi_value:.3f} > action threshold {self.drift_thresholds['psi_action']:.2f}",
                    'severity': 'HIGH'
                })
            elif psi_value > self.drift_thresholds['psi_warning']:
                alerts.append({
                    'type': 'feature_drift_warning',
                    'feature': feature,
                    'psi_value': psi_value,
                    'reason': f"PSI {psi_value:.3f} > warning threshold {self.drift_thresholds['psi_warning']:.2f}",
                    'severity': 'MEDIUM'
                })
        
        print(f"   📊 Checked {len(feature_names)} features")
        print(f"   ⚠️ {len([a for a in actions if a['type'] == 'auto_clip_feature'])} require auto-clipping")
        
        return actions, alerts
    
    def auto_clip_feature(self, feature_name: str, psi_value: float):
        """Auto-clip feature distribution and re-standardize"""
        print(f"✂️ Auto-clipping feature: {feature_name} (PSI: {psi_value:.3f})")
        
        # Placeholder for actual feature clipping logic
        clip_record = {
            'timestamp': datetime.now().isoformat(),
            'feature': feature_name,
            'psi_value': psi_value,
            'action': 'auto_clip_and_restandardize',
            'clip_percentiles': [1, 99],  # Clip at 1st and 99th percentiles
            'success': True
        }
        
        # Log the clipping action
        clip_log_file = self.base_dir / "logs" / "feature_clipping.json"
        clip_log_file.parent.mkdir(exist_ok=True)
        
        try:
            if clip_log_file.exists():
                with open(clip_log_file) as f:
                    log_entries = json.load(f)
            else:
                log_entries = []
            
            log_entries.append(clip_record)
            
            with open(clip_log_file, 'w') as f:
                json.dump(log_entries, f, indent=2)
                
            print(f"   ✅ Feature {feature_name} clipped and logged")
            
        except Exception as e:
            print(f"   ❌ Failed to log clipping action: {e}")
    
    def check_cost_spikes(self) -> tuple:
        """Check for cost spikes and adjust turnover"""
        print("💰 Checking cost conditions...")
        
        actions = []
        alerts = []
        
        # Simulate cost data
        current_borrow_cost = np.random.uniform(0.001, 0.008)  # 10-80 bps
        current_slippage = np.random.uniform(0.0005, 0.004)    # 5-40 bps
        
        # Simulate 5-day realized costs vs assumptions
        assumed_daily_cost = 0.0019  # ~0.19 bps per day (48 bps annually)
        realized_5d_costs = [np.random.uniform(0.001, 0.005) for _ in range(5)]  # Last 5 days
        avg_realized_cost = np.mean(realized_5d_costs)
        
        # Check borrow cost spike
        if current_borrow_cost > self.cost_thresholds['borrow_spike']:
            actions.append({
                'type': 'reduce_turnover',
                'reason': f"Borrow cost spike: {current_borrow_cost:.3f} > {self.cost_thresholds['borrow_spike']:.3f}",
                'severity': 'MEDIUM',
                'turnover_reduction': self.cost_thresholds['turnover_reduction']
            })
        
        # Check slippage spike
        if current_slippage > self.cost_thresholds['slippage_spike']:
            actions.append({
                'type': 'reduce_turnover',
                'reason': f"Slippage spike: {current_slippage:.3f} > {self.cost_thresholds['slippage_spike']:.3f}",
                'severity': 'MEDIUM',
                'turnover_reduction': self.cost_thresholds['turnover_reduction']
            })
        
        # Check 5-day realized cost vs assumptions
        if avg_realized_cost > assumed_daily_cost * self.cost_thresholds['realized_cost_multiplier']:
            actions.append({
                'type': 'tighten_turnover_costs',
                'reason': f"5d avg cost {avg_realized_cost:.3f} > {self.cost_thresholds['realized_cost_multiplier']:.1f}x assumed {assumed_daily_cost:.3f}",
                'severity': 'MEDIUM',
                'turnover_reduction': self.cost_thresholds['turnover_reduction']
            })
        
        print(f"   📊 Current: borrow {current_borrow_cost:.3f}, slippage {current_slippage:.3f}")
        print(f"   📊 5d avg realized: {avg_realized_cost:.3f} vs assumed {assumed_daily_cost:.3f}")
        
        return actions, alerts
    
    def check_risk_exposures(self) -> tuple:
        """Check risk exposures and sector neutrality"""
        print("⚖️ Checking risk exposures...")
        
        actions = []
        alerts = []
        
        # Simulate risk exposure data
        current_beta = np.random.uniform(-0.05, 0.12)
        max_sector_exp = np.random.uniform(0.08, 0.18)
        
        # Check beta exposure
        if abs(current_beta) > self.risk_thresholds['beta_exposure']:
            alerts.append({
                'type': 'beta_exposure_high',
                'reason': f"Beta exposure {current_beta:.3f} > threshold {self.risk_thresholds['beta_exposure']:.3f}",
                'severity': 'MEDIUM'
            })
        
        # Check sector exposure
        if max_sector_exp > self.risk_thresholds['sector_exposure']:
            alerts.append({
                'type': 'sector_exposure_high',
                'reason': f"Max sector exposure {max_sector_exp:.3f} > threshold {self.risk_thresholds['sector_exposure']:.3f}",
                'severity': 'MEDIUM'
            })
        
        print(f"   📊 Beta: {current_beta:.3f}, Max sector: {max_sector_exp:.3f}")
        
        return actions, alerts
    
    def execute_action(self, action: dict):
        """Execute automatic action"""
        action_type = action['type']
        
        print(f"🤖 Executing action: {action_type}")
        print(f"   📋 Reason: {action['reason']}")
        
        if action_type == 'rollback_model':
            self._rollback_to_previous_model()
        elif action_type == 'de_risk_portfolio':
            self._reduce_portfolio_exposure(action['target_exposure'])
        elif action_type == 'auto_clip_feature':
            self.auto_clip_feature(action['feature'], action['psi_value'])
        elif action_type == 'reduce_turnover':
            self._reduce_turnover_limit(action['turnover_reduction'])
        elif action_type == 'reduce_turnover_precision':
            self._reduce_turnover_limit(action['turnover_reduction'], reason="precision")
        elif action_type == 'tighten_turnover_costs':
            self._reduce_turnover_limit(action['turnover_reduction'], reason="cost_spike")
        else:
            print(f"   ⚠️ Unknown action type: {action_type}")
    
    def _rollback_to_previous_model(self):
        """Rollback to previous model version"""
        print("   🔄 Initiating model rollback...")
        
        # Placeholder for actual rollback logic
        rollback_record = {
            'timestamp': datetime.now().isoformat(),
            'action': 'model_rollback',
            'previous_model': 'enhanced_model_v2',
            'current_model': 'enhanced_model_v1',
            'reason': 'IC performance below rollback threshold'
        }
        
        # Would implement actual model switching logic here
        print("   ✅ Model rollback completed (placeholder)")
    
    def _reduce_portfolio_exposure(self, target_exposure: float):
        """Reduce portfolio exposure"""
        print(f"   📉 Reducing exposure to {target_exposure:.1%}...")
        
        # Placeholder for actual exposure reduction
        print("   ✅ Portfolio exposure reduced (placeholder)")
    
    def _reduce_turnover_limit(self, reduction: float, reason: str = "general"):
        """Reduce turnover limit"""
        print(f"   🔄 Reducing turnover limit by {reduction:.1%} (reason: {reason})...")
        
        # Placeholder for turnover adjustment
        print("   ✅ Turnover limit reduced (placeholder)")
    
    def run_full_monitoring_cycle(self):
        """Run complete monitoring and response cycle"""
        print("🛡️ RUNNING FULL MONITORING CYCLE")
        print("=" * 60)
        
        all_actions = []
        all_alerts = []
        
        # 1. Check IC performance
        metrics = self.calculate_rolling_performance_metrics()
        ic_actions, ic_alerts = self.check_ic_performance(metrics)
        all_actions.extend(ic_actions)
        all_alerts.extend(ic_alerts)
        
        # 2. Check P@K precision performance 
        precision_actions, precision_alerts = self.check_precision_performance(metrics)
        all_actions.extend(precision_actions)
        all_alerts.extend(precision_alerts)
        
        # 3. Check feature drift
        drift_actions, drift_alerts = self.check_feature_drift()
        all_actions.extend(drift_actions)
        all_alerts.extend(drift_alerts)
        
        # 4. Check cost spikes
        cost_actions, cost_alerts = self.check_cost_spikes()
        all_actions.extend(cost_actions)
        all_alerts.extend(cost_alerts)
        
        # 5. Check risk exposures
        risk_actions, risk_alerts = self.check_risk_exposures()
        all_actions.extend(risk_actions)
        all_alerts.extend(risk_alerts)
        
        # Summary
        print(f"\\n📊 MONITORING SUMMARY:")
        print(f"   🚨 Critical actions required: {len([a for a in all_actions if a['severity'] == 'CRITICAL'])}")
        print(f"   ⚠️ High priority actions: {len([a for a in all_actions if a['severity'] == 'HIGH'])}")
        print(f"   📋 Medium priority actions: {len([a for a in all_actions if a['severity'] == 'MEDIUM'])}")
        print(f"   📢 Alerts: {len(all_alerts)}")
        
        # Execute critical and high priority actions automatically
        critical_actions = [a for a in all_actions if a['severity'] in ['CRITICAL', 'HIGH']]
        
        if critical_actions:
            print(f"\\n🤖 EXECUTING {len(critical_actions)} AUTOMATIC ACTIONS:")
            for action in critical_actions:
                self.execute_action(action)
        
        # Save monitoring report
        self._save_monitoring_report(all_actions, all_alerts, metrics)
        
        print(f"\\n✅ MONITORING CYCLE COMPLETED")
        return len(critical_actions) == 0  # Return True if no critical issues
    
    def _save_monitoring_report(self, actions: list, alerts: list, metrics: dict):
        """Save monitoring report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'rolling_ic': metrics['rolling_ic'],
                'current_ic': metrics['current_ic'],
                'ic_trend': metrics['ic_trend'],
                'avg_p_at_10': metrics['avg_p_at_10'],
                'avg_p_at_20': metrics['avg_p_at_20'],
                'p10_ci_lower': metrics.get('p10_ci_lower', np.nan),
                'p10_ci_upper': metrics.get('p10_ci_upper', np.nan),
                'p20_ci_lower': metrics.get('p20_ci_lower', np.nan), 
                'p20_ci_upper': metrics.get('p20_ci_upper', np.nan),
                'avg_spread_mean': metrics.get('avg_spread_mean', np.nan),
                'avg_spread_sharpe': metrics.get('avg_spread_sharpe', np.nan),
                'daily_sharpe': metrics['daily_sharpe']
            },
            'production_gates': {
                'ic_de_risk_threshold': self.ic_thresholds['de_risk'],
                'ic_rollback_threshold': self.ic_thresholds['rollback'],
                'p10_min_threshold': self.precision_thresholds['p10_min'],
                'p20_min_threshold': self.precision_thresholds['p20_min'],
                'psi_action_threshold': self.drift_thresholds['psi_action'],
                'cost_multiplier_threshold': self.cost_thresholds['realized_cost_multiplier']
            },
            'actions': actions,
            'alerts': alerts,
            'system_health': 'HEALTHY' if len([a for a in actions if a['severity'] == 'CRITICAL']) == 0 else 'ATTENTION_REQUIRED'
        }
        
        report_file = self.base_dir / "monitoring" / f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"   💾 Report saved: {report_file}")

def main():
    """Run production guardrails monitoring"""
    print("🚀 Starting Production Guardrails")
    
    guardrails = ProductionGuardrails()
    success = guardrails.run_full_monitoring_cycle()
    
    if success:
        print("\\n✅ All systems healthy - no critical issues detected")
    else:
        print("\\n⚠️ Critical issues detected and addressed")

if __name__ == "__main__":
    main()