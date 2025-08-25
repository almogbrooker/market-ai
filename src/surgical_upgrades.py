#!/usr/bin/env python3
"""
SURGICAL UPGRADES
=================
Two fast-win upgrades for production readiness:
A) Post-hoc sizing calibration (isotonic regression)
B) Cost-aware acceptor for optimal portfolio construction
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import spearmanr
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class SurgicalUpgrades:
    """Implement surgical upgrades for production system"""
    
    def __init__(self):
        print("🛠️ SURGICAL UPGRADES")
        print("=" * 70)
        
        self.base_dir = Path("../artifacts")
        self.models_dir = self.base_dir / "models"
        self.processed_dir = self.base_dir / "processed"
        self.validation_dir = self.base_dir / "validation"
        
        # Load trained model
        self.model_dir = sorted([d for d in self.models_dir.iterdir() if d.is_dir()])[-1]
        self.model = joblib.load(self.model_dir / "model.pkl")
        self.scaler = joblib.load(self.model_dir / "scaler.pkl")
        
        with open(self.model_dir / "model_card.json", 'r') as f:
            self.model_card = json.load(f)
    
    def load_calibration_data(self):
        """Load last 90-120 days for calibration"""
        df = pd.read_parquet(self.processed_dir / "train_institutional.parquet")
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Use validation data (last 30%) as calibration set
        val_split_idx = int(len(df) * 0.7)
        calib_df = df.iloc[val_split_idx:].copy()
        
        feature_cols = [col for col in df.columns if col.endswith('_t1')]
        X_calib = calib_df[feature_cols].fillna(0)
        y_calib = calib_df['target_forward'].fillna(0)
        
        # Generate predictions
        X_calib_scaled = self.scaler.transform(X_calib)
        predictions = self.model.predict(X_calib_scaled)
        
        calib_df['raw_score'] = predictions
        calib_df['realized_return'] = y_calib
        
        return calib_df
    
    def upgrade_a_isotonic_calibration(self):
        """A) Post-hoc sizing calibration using isotonic regression"""
        print("\n🎯 UPGRADE A: ISOTONIC CALIBRATION")
        print("-" * 50)
        
        calib_df = self.load_calibration_data()
        
        # Filter to accepted subset only (top/bottom quintiles)
        accepted_mask = (
            (calib_df['raw_score'] >= calib_df['raw_score'].quantile(0.8)) |
            (calib_df['raw_score'] <= calib_df['raw_score'].quantile(0.2))
        )
        
        accepted_df = calib_df[accepted_mask].copy()
        
        if len(accepted_df) < 100:
            print("   ❌ Insufficient data for calibration")
            return None
        
        # Fit isotonic regression: raw_score -> realized_return
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(accepted_df['raw_score'], accepted_df['realized_return'])
        
        # Test calibration on the accepted subset
        calibrated_scores = iso_reg.predict(accepted_df['raw_score'])
        
        # Calculate improvement in IC
        raw_ic, _ = spearmanr(accepted_df['realized_return'], accepted_df['raw_score'])
        calibrated_ic, _ = spearmanr(accepted_df['realized_return'], calibrated_scores)
        
        ic_improvement = (abs(calibrated_ic) - abs(raw_ic)) / abs(raw_ic) * 100 if raw_ic != 0 else 0
        
        print(f"   📊 Calibration data: {len(accepted_df)} samples")
        print(f"   📈 Raw IC: {raw_ic:.4f}")
        print(f"   📈 Calibrated IC: {calibrated_ic:.4f}")
        print(f"   🚀 IC improvement: {ic_improvement:+.1f}%")
        
        # Save calibrator
        calibrator_file = self.model_dir / "isotonic_calibrator.pkl"
        joblib.dump(iso_reg, calibrator_file)
        
        print(f"   💾 Calibrator saved: {calibrator_file}")
        
        # Demonstrate calibrated predictions
        test_scores = np.linspace(accepted_df['raw_score'].min(), accepted_df['raw_score'].max(), 10)
        calibrated_expectations = iso_reg.predict(test_scores)
        
        print(f"   📋 Sample calibration mapping:")
        for raw, calib in zip(test_scores[:5], calibrated_expectations[:5]):
            print(f"      Raw: {raw:.4f} → Calibrated: {calib:.6f}")
        
        calibration_results = {
            'raw_ic': raw_ic,
            'calibrated_ic': calibrated_ic,
            'ic_improvement_pct': ic_improvement,
            'calibrator_path': str(calibrator_file),
            'samples_used': len(accepted_df)
        }
        
        return calibration_results
    
    def upgrade_b_cost_aware_acceptor(self):
        """B) Cost-aware acceptor for optimal portfolio construction"""
        print("\n💰 UPGRADE B: COST-AWARE ACCEPTOR")
        print("-" * 50)
        
        calib_df = self.load_calibration_data()
        
        # Simulate daily portfolio optimization
        optimization_results = []
        
        unique_dates = calib_df['Date'].unique()[:10]  # Test on first 10 days
        
        for date in unique_dates:
            date_data = calib_df[calib_df['Date'] == date].copy()
            
            if len(date_data) < 20:
                continue
            
            # Simulate transaction costs (based on score density and position size)
            date_data['estimated_cost_bps'] = self._estimate_transaction_costs(date_data)
            
            # Traditional acceptor (simple threshold)
            traditional_long = date_data.nlargest(max(1, len(date_data) // 5), 'raw_score')
            traditional_short = date_data.nsmallest(max(1, len(date_data) // 5), 'raw_score')
            traditional_positions = pd.concat([traditional_long, traditional_short])
            
            # Cost-aware acceptor
            cost_aware_positions = self._optimize_cost_aware_portfolio(date_data)
            
            # Compare performance
            traditional_return = (
                traditional_long['realized_return'].mean() - 
                traditional_short['realized_return'].mean()
            )
            traditional_cost = traditional_positions['estimated_cost_bps'].mean()
            traditional_net = traditional_return - traditional_cost / 10000
            
            cost_aware_return = (
                cost_aware_positions[cost_aware_positions['position'] > 0]['realized_return'].mean() -
                cost_aware_positions[cost_aware_positions['position'] < 0]['realized_return'].mean()
            )
            cost_aware_cost = cost_aware_positions['estimated_cost_bps'].mean()
            cost_aware_net = cost_aware_return - cost_aware_cost / 10000
            
            optimization_results.append({
                'date': date,
                'traditional_gross': traditional_return,
                'traditional_cost': traditional_cost,
                'traditional_net': traditional_net,
                'cost_aware_gross': cost_aware_return,
                'cost_aware_cost': cost_aware_cost,
                'cost_aware_net': cost_aware_net,
                'traditional_positions': len(traditional_positions),
                'cost_aware_positions': len(cost_aware_positions)
            })
        
        if not optimization_results:
            print("   ❌ No optimization results available")
            return None
        
        # Aggregate results
        results_df = pd.DataFrame(optimization_results)
        
        avg_traditional_net = results_df['traditional_net'].mean()
        avg_cost_aware_net = results_df['cost_aware_net'].mean()
        
        net_improvement = (avg_cost_aware_net - avg_traditional_net) / abs(avg_traditional_net) * 100 if avg_traditional_net != 0 else 0
        
        avg_traditional_cost = results_df['traditional_cost'].mean()
        avg_cost_aware_cost = results_df['cost_aware_cost'].mean()
        cost_reduction = (avg_traditional_cost - avg_cost_aware_cost) / avg_traditional_cost * 100 if avg_traditional_cost != 0 else 0
        
        print(f"   📊 Optimization tests: {len(optimization_results)} days")
        print(f"   💰 Traditional net return: {avg_traditional_net:.6f}")
        print(f"   💰 Cost-aware net return: {avg_cost_aware_net:.6f}")
        print(f"   🚀 Net return improvement: {net_improvement:+.1f}%")
        print(f"   💸 Cost reduction: {cost_reduction:+.1f}%")
        
        cost_aware_results = {
            'net_improvement_pct': net_improvement,
            'cost_reduction_pct': cost_reduction,
            'avg_traditional_net': avg_traditional_net,
            'avg_cost_aware_net': avg_cost_aware_net,
            'tests_run': len(optimization_results)
        }
        
        return cost_aware_results
    
    def _estimate_transaction_costs(self, date_data):
        """Estimate transaction costs based on score density and position characteristics"""
        # Simulate realistic transaction costs
        base_cost = 15  # 15 bps base
        
        # Higher costs for extreme scores (liquidity premium)
        score_extremity = np.abs(stats.zscore(date_data['raw_score']))
        extremity_cost = score_extremity * 2  # +2 bps per sigma
        
        # Random component for market impact
        np.random.seed(42)
        random_cost = np.random.normal(0, 3, len(date_data))  # ±3 bps random
        
        total_cost = base_cost + extremity_cost + random_cost
        return np.clip(total_cost, 10, 30)  # Clip to realistic range
    
    def _optimize_cost_aware_portfolio(self, date_data, lambda_cost=0.5):
        """Optimize portfolio to maximize score - λ*cost"""
        # Simple greedy optimization for demonstration
        date_data = date_data.copy()
        date_data['utility'] = date_data['raw_score'] - lambda_cost * date_data['estimated_cost_bps'] / 10000
        
        # Select top utility positions (both long and short)
        n_positions = max(2, len(date_data) // 8)  # Slightly fewer positions
        
        # Long positions (positive utility)
        long_candidates = date_data[date_data['utility'] > 0].nlargest(n_positions, 'utility')
        # Short positions (negative utility, but we short them)
        short_candidates = date_data[date_data['utility'] < 0].nsmallest(n_positions, 'utility')
        
        # Combine and add position indicators
        long_candidates = long_candidates.copy()
        short_candidates = short_candidates.copy()
        
        long_candidates['position'] = 1
        short_candidates['position'] = -1
        
        optimized_portfolio = pd.concat([long_candidates, short_candidates])
        
        return optimized_portfolio if len(optimized_portfolio) > 0 else date_data.head(0)
    
    def run_surgical_upgrades(self):
        """Run both surgical upgrades"""
        print("\n🎯 RUNNING SURGICAL UPGRADES")
        print("=" * 70)
        
        # Run Upgrade A
        calibration_results = self.upgrade_a_isotonic_calibration()
        
        # Run Upgrade B  
        cost_aware_results = self.upgrade_b_cost_aware_acceptor()
        
        # Combined results
        print("\n" + "=" * 70)
        print("📊 SURGICAL UPGRADES SUMMARY")
        print("=" * 70)
        
        if calibration_results:
            print(f"✅ Isotonic Calibration:")
            print(f"   IC improvement: {calibration_results['ic_improvement_pct']:+.1f}%")
            print(f"   Calibrated IC: {calibration_results['calibrated_ic']:.4f}")
        else:
            print(f"❌ Isotonic Calibration: Failed")
        
        if cost_aware_results:
            print(f"✅ Cost-Aware Acceptor:")
            print(f"   Net return improvement: {cost_aware_results['net_improvement_pct']:+.1f}%")
            print(f"   Cost reduction: {cost_aware_results['cost_reduction_pct']:+.1f}%")
        else:
            print(f"❌ Cost-Aware Acceptor: Failed")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.validation_dir / f"surgical_upgrades_{timestamp}.json"
        
        upgrade_report = {
            'timestamp': datetime.now().isoformat(),
            'calibration_upgrade': calibration_results,
            'cost_aware_upgrade': cost_aware_results,
            'summary': {
                'calibration_success': calibration_results is not None,
                'cost_aware_success': cost_aware_results is not None,
                'total_ic_improvement': calibration_results['ic_improvement_pct'] if calibration_results else 0,
                'total_cost_reduction': cost_aware_results['cost_reduction_pct'] if cost_aware_results else 0
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(upgrade_report, f, indent=2, default=str)
        
        print(f"\n📄 Upgrade results saved: {results_file}")
        
        return upgrade_report

def main():
    """Run surgical upgrades"""
    upgrades = SurgicalUpgrades()
    results = upgrades.run_surgical_upgrades()
    
    print("\n🚀 SURGICAL UPGRADES COMPLETE")
    
    return results

if __name__ == "__main__":
    upgrade_results = main()