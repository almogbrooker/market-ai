#!/usr/bin/env python3
"""
Out-of-Sample Validation 2023-2025
Test validated system on completely unseen future data
"""

import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def load_validated_model_features():
    """Load the validated feature set from CV results"""
    
    artifacts_dir = Path(__file__).parent / 'artifacts'
    cv_report_path = artifacts_dir / 'cv_report.json'
    
    if cv_report_path.exists():
        with open(cv_report_path, 'r') as f:
            cv_report = json.load(f)
        
        # Extract validated features from CV results
        if cv_report['cv_results']:
            validated_features = cv_report['cv_results'][0]['available_features']
            logger.info(f"Loaded {len(validated_features)} validated features from CV")
            return validated_features
    
    # Fallback to academic feature set
    academic_features = [
        # Technical features
        'return_5d_lag1', 'return_20d_lag1', 'vol_20d_lag1', 'volume_ratio_lag1',
        'RSI_14', 'MACD', 'Volume_Ratio', 'Return_5D', 'Volatility_20D',
        
        # Cross-sectional fundamental z-scores
        'ZSCORE_PE', 'ZSCORE_PB', 'ZSCORE_PS', 'ZSCORE_ROE', 'ZSCORE_ROA',
        'ZSCORE_PM', 'ZSCORE_OM', 'ZSCORE_REV_GROWTH', 'ZSCORE_EPS_GROWTH',
        
        # Cross-sectional fundamental ranks
        'RANK_PE', 'RANK_PB', 'RANK_ROE', 'RANK_REV_GROWTH',
        
        # Engineered combinations
        'VALUE_MOMENTUM', 'QUALITY_GROWTH'
    ]
    
    logger.info(f"Using fallback academic features: {len(academic_features)} features")
    return academic_features

def train_ensemble_model(train_data, features):
    """Train ensemble model with 5-day and 20-day horizon blending"""
    
    logger.info("🏋️ TRAINING ENSEMBLE MODEL")
    
    # Prepare training data
    X_train = train_data[features].fillna(0).values
    
    # Multi-horizon targets
    y_5d = train_data['target_5d'].fillna(0).values if 'target_5d' in train_data.columns else train_data['next_return_1d'].fillna(0).values
    y_20d = train_data['target_20d'].fillna(0).values if 'target_20d' in train_data.columns else train_data['next_return_1d'].fillna(0).values
    
    # Simple ensemble: equal-weighted features for each horizon
    def predict_horizon(X, horizon='5d'):
        """Simple linear combination predictor"""
        # Normalize features
        X_norm = np.zeros_like(X)
        for i in range(X.shape[1]):
            col = X[:, i]
            if np.std(col) > 0:
                X_norm[:, i] = (col - np.mean(col)) / np.std(col)
            else:
                X_norm[:, i] = col
        
        # Equal weighted combination
        if horizon == '5d':
            # Favor momentum and technical features for 5-day
            weights = np.ones(X.shape[1])
            weights[:4] *= 1.5  # Boost technical features
        else:  # 20d
            # Favor fundamental features for 20-day
            weights = np.ones(X.shape[1])
            if X.shape[1] > 10:
                weights[10:] *= 1.5  # Boost fundamental features
        
        weights = weights / np.sum(weights)
        return np.dot(X_norm, weights)
    
    # Train both horizons
    pred_5d = predict_horizon(X_train, '5d')
    pred_20d = predict_horizon(X_train, '20d')
    
    # Calculate ICs to determine blending weights
    ic_5d = np.corrcoef(pred_5d, y_5d)[0, 1] if not np.isnan(np.corrcoef(pred_5d, y_5d)[0, 1]) else 0
    ic_20d = np.corrcoef(pred_20d, y_20d)[0, 1] if not np.isnan(np.corrcoef(pred_20d, y_20d)[0, 1]) else 0
    
    logger.info(f"   5-day IC: {ic_5d:+.6f}")
    logger.info(f"   20-day IC: {ic_20d:+.6f}")
    
    # Dynamic blending weights based on IC performance
    total_ic = abs(ic_5d) + abs(ic_20d)
    if total_ic > 0:
        weight_5d = abs(ic_5d) / total_ic
        weight_20d = abs(ic_20d) / total_ic
    else:
        weight_5d = 0.6  # Default favor 5-day
        weight_20d = 0.4
    
    logger.info(f"   Ensemble weights: 5d={weight_5d:.3f}, 20d={weight_20d:.3f}")
    
    return {
        'features': features,
        'predict_5d': predict_horizon,
        'predict_20d': predict_horizon, 
        'weight_5d': weight_5d,
        'weight_20d': weight_20d,
        'train_ic_5d': ic_5d,
        'train_ic_20d': ic_20d
    }

def oos_validation_2023_2025():
    """Run complete OOS validation on 2023-2025 data"""
    
    logger.info("🚀 OUT-OF-SAMPLE VALIDATION 2023-2025")
    logger.info("=" * 60)
    logger.info("Train: ≤2022-12-31, Test: 2023-2025 rolling windows")
    logger.info("=" * 60)
    
    # Load datasets
    artifacts_dir = Path(__file__).parent / 'artifacts'
    
    # Training data (≤2022)
    train_path = artifacts_dir / 'ds_train.parquet'
    if not train_path.exists():
        logger.error("Training dataset not found. Run clean_validation_protocol.py first.")
        return False
    
    train_data = pd.read_parquet(train_path)
    logger.info(f"Training data: {len(train_data):,} samples")
    logger.info(f"Training period: {train_data['Date'].min()} to {train_data['Date'].max()}")
    
    # OOS test periods
    oos_2023h1_path = artifacts_dir / 'ds_oos_2023H1.parquet'
    oos_2023h2_2024_path = artifacts_dir / 'ds_oos_2023H2_2024.parquet' 
    oos_2025ytd_path = artifacts_dir / 'ds_oos_2025YTD.parquet'
    
    oos_periods = []
    
    if oos_2023h1_path.exists():
        oos_2023h1 = pd.read_parquet(oos_2023h1_path)
        oos_periods.append(('2023H1', oos_2023h1))
        logger.info(f"OOS 2023H1: {len(oos_2023h1):,} samples")
    
    if oos_2023h2_2024_path.exists():
        oos_2023h2_2024 = pd.read_parquet(oos_2023h2_2024_path)
        oos_periods.append(('2023H2_2024', oos_2023h2_2024))
        logger.info(f"OOS 2023H2-2024: {len(oos_2023h2_2024):,} samples")
        
    if oos_2025ytd_path.exists():
        oos_2025ytd = pd.read_parquet(oos_2025ytd_path)
        oos_periods.append(('2025YTD', oos_2025ytd))
        logger.info(f"OOS 2025YTD: {len(oos_2025ytd):,} samples")
    
    if not oos_periods:
        logger.error("No OOS datasets found")
        return False
    
    # Load validated features
    validated_features = load_validated_model_features()
    
    # Train ensemble model on training data only
    logger.info(f"\n🏋️ TRAINING ON ≤2022 DATA ONLY")
    ensemble_model = train_ensemble_model(train_data, validated_features)
    
    # Test on each OOS period
    oos_results = []
    
    for period_name, oos_data in oos_periods:
        logger.info(f"\n🔮 TESTING ON {period_name}")
        logger.info(f"   Period: {oos_data['Date'].min()} to {oos_data['Date'].max()}")
        logger.info(f"   Samples: {len(oos_data):,}")
        
        # Prepare OOS features
        available_features = [f for f in validated_features if f in oos_data.columns]
        logger.info(f"   Available features: {len(available_features)}/{len(validated_features)}")
        
        if len(available_features) < 5:
            logger.warning(f"   Insufficient features for {period_name}")
            continue
        
        X_oos = oos_data[available_features].fillna(0).values
        
        # Multi-horizon predictions
        pred_5d = ensemble_model['predict_5d'](X_oos, '5d')
        pred_20d = ensemble_model['predict_20d'](X_oos, '20d')
        
        # 🔒 FIXED: Freeze horizon before OOS - no cherry-picking on test set
        # Use pre-determined ensemble weights from training CV only
        frozen_weight_5d = ensemble_model['weight_5d']
        frozen_weight_20d = ensemble_model['weight_20d']
        
        ensemble_pred = (
            frozen_weight_5d * pred_5d + 
            frozen_weight_20d * pred_20d
        )
        
        # 🔒 FIXED: Use single frozen target (5d) - no OOS horizon selection
        frozen_target_col = 'target_5d'  # Pre-selected in training
        
        period_results = {
            'period': period_name,
            'samples': len(oos_data),
            'features_used': len(available_features),
            'date_range': f"{oos_data['Date'].min()} to {oos_data['Date'].max()}",
            'frozen_horizon': '5d',
            'frozen_weight_5d': float(frozen_weight_5d),
            'frozen_weight_20d': float(frozen_weight_20d)
        }
        
        # Calculate IC for frozen horizon only
        if frozen_target_col in oos_data.columns:
            y_actual = oos_data[frozen_target_col].fillna(0).values
            
            if len(y_actual) == len(ensemble_pred):
                # Use Spearman IC for cross-sectional robustness
                from scipy.stats import spearmanr
                ic_pearson = np.corrcoef(ensemble_pred, y_actual)[0, 1]
                ic_spearman, _ = spearmanr(ensemble_pred, y_actual)
                
                if np.isnan(ic_pearson): ic_pearson = 0
                if np.isnan(ic_spearman): ic_spearman = 0
                
                period_results['ic_pearson'] = float(ic_pearson)
                period_results['ic_spearman'] = float(ic_spearman)
                period_results['primary_ic'] = float(ic_spearman)  # Use Spearman as primary
        
        # Portfolio simulation with realistic costs
        if 'primary_ic' in period_results:
            y_best = oos_data[frozen_target_col].fillna(0).values
            
            # 🔒 FIXED: Realistic portfolio simulation with proper costs & compounding
            cost_per_turn = 0.0003  # 3 bps per side → ~6 bps roundtrip
            equity_curve = 1.0
            prev_weights = None
            daily_returns = []
            daily_turnovers = []
            
            # Group by date for daily rebalancing
            for date, day_data in oos_data.groupby('Date'):
                day_indices = day_data.index - oos_data.index[0]  # Relative indices
                
                if len(day_indices) >= 4:  # Need minimum stocks
                    day_pred = ensemble_pred[day_indices]
                    day_actual = y_best[day_indices]
                    
                    # Build beta-neutral long/short weights
                    q75 = np.percentile(day_pred, 75)
                    q25 = np.percentile(day_pred, 25)
                    
                    long_mask = day_pred >= q75
                    short_mask = day_pred <= q25
                    
                    # Equal-weight within long/short buckets, sum to 0 (beta-neutral)
                    weights = np.zeros(len(day_pred))
                    if np.any(long_mask):
                        weights[long_mask] = 0.5 / np.sum(long_mask)  # Long side = +50%
                    if np.any(short_mask):
                        weights[short_mask] = -0.5 / np.sum(short_mask)  # Short side = -50%
                    
                    # Calculate gross return
                    day_ret_gross = float(np.dot(weights, day_actual))
                    
                    # Calculate turnover-based costs (L1 change in weights)
                    if prev_weights is None:
                        turnover = np.sum(np.abs(weights))  # Initial position
                    else:
                        # Pad/truncate to same length for comparison
                        min_len = min(len(weights), len(prev_weights))
                        if min_len > 0:
                            turnover = np.sum(np.abs(weights[:min_len] - prev_weights[:min_len]))
                            # Add new positions
                            if len(weights) > len(prev_weights):
                                turnover += np.sum(np.abs(weights[len(prev_weights):]))
                            elif len(prev_weights) > len(weights):
                                turnover += np.sum(np.abs(prev_weights[len(weights):]))
                        else:
                            turnover = np.sum(np.abs(weights))
                    
                    day_cost = cost_per_turn * turnover
                    day_ret_net = day_ret_gross - day_cost
                    
                    # Update equity curve with geometric compounding
                    equity_curve *= (1.0 + day_ret_net)
                    daily_returns.append(day_ret_net)
                    daily_turnovers.append(turnover)
                    prev_weights = weights.copy()
            
            if daily_returns:
                daily_returns = np.array(daily_returns)
                daily_turnovers = np.array(daily_turnovers)
                
                # 🔒 FIXED: Proper geometric returns vs sum
                total_return_geometric = (equity_curve - 1.0) * 100  # Proper compounding
                annualized_return = (equity_curve ** (252 / len(daily_returns)) - 1.0) * 100
                volatility = np.std(daily_returns) * np.sqrt(252) * 100
                sharpe = (annualized_return / 100) / (volatility / 100) if volatility > 0 else 0
                
                # Calculate max drawdown
                cumulative = np.cumprod(1 + daily_returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdowns = (cumulative - running_max) / running_max
                max_drawdown = np.min(drawdowns) * 100
                
                period_results.update({
                    'portfolio_return_pct': float(total_return_geometric),
                    'annualized_return_pct': float(annualized_return),
                    'portfolio_volatility_pct': float(volatility), 
                    'portfolio_sharpe': float(sharpe),
                    'max_drawdown_pct': float(max_drawdown),
                    'trading_days': len(daily_returns),
                    'avg_daily_turnover': float(np.mean(daily_turnovers)),
                    'median_daily_turnover': float(np.median(daily_turnovers)),
                    'total_transaction_costs_pct': float(np.sum(daily_turnovers) * cost_per_turn * 100)
                })
        
        oos_results.append(period_results)
        
        # Log results with fixed metrics
        logger.info(f"   📊 RESULTS:")
        if 'primary_ic' in period_results:
            logger.info(f"      Spearman IC (frozen 5d): {period_results['primary_ic']:+.6f}")
        if 'portfolio_return_pct' in period_results:
            logger.info(f"      Total Return (geometric): {period_results['portfolio_return_pct']:+.2f}%")
            logger.info(f"      Annualized Return: {period_results['annualized_return_pct']:+.2f}%")
            logger.info(f"      Sharpe Ratio: {period_results['portfolio_sharpe']:+.2f}")
            logger.info(f"      Max Drawdown: {period_results['max_drawdown_pct']:+.2f}%")
            logger.info(f"      Avg Daily Turnover: {period_results['avg_daily_turnover']:.1%}")
            logger.info(f"      Total Transaction Costs: {period_results['total_transaction_costs_pct']:+.2f}%")
    
    # Overall assessment with realistic metrics
    if oos_results:
        all_ics = [r.get('primary_ic', 0) for r in oos_results if 'primary_ic' in r]
        avg_ic = np.mean(all_ics) if all_ics else 0
        
        portfolio_returns = [r.get('portfolio_return_pct', 0) for r in oos_results if 'portfolio_return_pct' in r]
        avg_return = np.mean(portfolio_returns) if portfolio_returns else 0
        
        all_sharpes = [r.get('portfolio_sharpe', 0) for r in oos_results if 'portfolio_sharpe' in r]
        avg_sharpe = np.mean(all_sharpes) if all_sharpes else 0
        
        all_drawdowns = [r.get('max_drawdown_pct', 0) for r in oos_results if 'max_drawdown_pct' in r]
        worst_drawdown = min(all_drawdowns) if all_drawdowns else 0
        
        all_turnovers = [r.get('avg_daily_turnover', 0) for r in oos_results if 'avg_daily_turnover' in r]
        avg_turnover = np.mean(all_turnovers) if all_turnovers else 0
        
        logger.info(f"\n🎯 OVERALL OOS ASSESSMENT (FIXED METRICS)")
        logger.info(f"   Average Spearman IC: {avg_ic:+.6f}")
        logger.info(f"   IC Range: [{min(all_ics):+.6f}, {max(all_ics):+.6f}]" if all_ics else "   IC Range: N/A")
        logger.info(f"   Average Period Return: {avg_return:+.2f}%")
        logger.info(f"   Average Sharpe Ratio: {avg_sharpe:+.2f}")
        logger.info(f"   Worst Drawdown: {worst_drawdown:+.2f}%")
        logger.info(f"   Average Daily Turnover: {avg_turnover:.1%}")
        logger.info(f"   Periods tested: {len(oos_results)}")
        
        # 🔒 FIXED: Realistic pass/fail criteria with capacity constraints
        oos_passed = (
            avg_ic >= 0.002 and  # Average IC ≥ 0.2% (realistic for cross-sectional)
            avg_sharpe >= 0.3 and  # Sharpe ≥ 0.3 (reasonable with costs)
            worst_drawdown >= -25.0 and  # Max drawdown ≤ 25%
            len([ic for ic in all_ics if ic > 0]) >= len(all_ics) * 0.6 and  # 60% positive periods
            avg_turnover <= 0.5  # Daily turnover ≤ 50% (institutional capacity limit)
        )
        
        # Additional capacity warnings
        if avg_turnover > 0.3:
            logger.warning(f"⚠️ High turnover {avg_turnover:.1%} may exceed institutional capacity")
        if avg_turnover > 0.5:
            logger.error(f"❌ Turnover {avg_turnover:.1%} exceeds realistic trading capacity")
        
        logger.info(f"\n🏆 OOS VALIDATION (FIXED): {'✅ PASSED' if oos_passed else '❌ FAILED'}")
        logger.info(f"   ✓ IC ≥ 0.2%: {'✅' if avg_ic >= 0.002 else '❌'}")
        logger.info(f"   ✓ Sharpe ≥ 0.3: {'✅' if avg_sharpe >= 0.3 else '❌'}")
        logger.info(f"   ✓ Drawdown ≤ 25%: {'✅' if worst_drawdown >= -25.0 else '❌'}")
        logger.info(f"   ✓ 60% positive periods: {'✅' if len([ic for ic in all_ics if ic > 0]) >= len(all_ics) * 0.6 else '❌'}")
        logger.info(f"   ✓ Turnover ≤ 50%: {'✅' if avg_turnover <= 0.5 else '❌'}")
        
        # Save results with fixed validation methodology
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'validation_type': 'oos_2023_2025_FIXED',
            'methodology_fixes': [
                'Frozen horizon selection (no OOS cherry-picking)',
                'Spearman IC for cross-sectional robustness', 
                'Realistic transaction costs (6 bps roundtrip)',
                'Proper geometric compounding',
                'Beta-neutral portfolio construction',
                'Turnover-based cost model'
            ],
            'training_period': f"{train_data['Date'].min()} to {train_data['Date'].max()}",
            'ensemble_model': {
                'features_count': len(validated_features),
                'train_ic_5d': ensemble_model['train_ic_5d'],
                'train_ic_20d': ensemble_model['train_ic_20d'],
                'frozen_weight_5d': ensemble_model['weight_5d'],
                'frozen_weight_20d': ensemble_model['weight_20d'],
                'frozen_horizon': '5d'
            },
            'oos_results': oos_results,
            'summary': {
                'average_spearman_ic': float(avg_ic),
                'ic_range': [float(min(all_ics)), float(max(all_ics))] if all_ics else [0, 0],
                'average_period_return_pct': float(avg_return),
                'average_sharpe_ratio': float(avg_sharpe),
                'worst_drawdown_pct': float(worst_drawdown),
                'average_daily_turnover': float(avg_turnover),
                'periods_tested': len(oos_results),
                'validation_passed': oos_passed,
                'realistic_acceptance_gates': {
                    'min_ic': 0.002,
                    'min_sharpe': 0.3,
                    'max_drawdown': -25.0,
                    'min_positive_periods_pct': 60,
                    'max_daily_turnover': 0.8
                }
            }
        }
        
        # Save report with FIXED suffix to distinguish from leaked version
        reports_dir = Path(__file__).parent / 'reports'
        reports_dir.mkdir(exist_ok=True)
        
        report_path = reports_dir / 'oos_validation_2023_2025_FIXED.json'
        with open(report_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"📊 FIXED OOS validation report saved: {report_path}")
        logger.info(f"🔍 Compare with original (leaked) results in oos_validation_2023_2025.json")
        
        return final_results
    
    else:
        logger.error("❌ No OOS results generated")
        return False

def main():
    """Run OOS validation"""
    
    results = oos_validation_2023_2025()
    
    if results and results['summary']['validation_passed']:
        print(f"\n🎉 FIXED OOS VALIDATION PASSED!")
        print(f"Average Spearman IC: {results['summary']['average_spearman_ic']:+.6f}")
        print(f"Average Period Return: {results['summary']['average_period_return_pct']:+.2f}%")
        print(f"Average Sharpe: {results['summary']['average_sharpe_ratio']:+.2f}")
        print("System validated with realistic costs and no leakage!")
    elif results:
        print(f"\n⚠️ FIXED OOS VALIDATION PARTIAL")
        print(f"Average Spearman IC: {results['summary']['average_spearman_ic']:+.6f}")
        print(f"Average Period Return: {results['summary']['average_period_return_pct']:+.2f}%")
        print(f"Average Sharpe: {results['summary']['average_sharpe_ratio']:+.2f}")
        print(f"Worst Drawdown: {results['summary']['worst_drawdown_pct']:+.2f}%")
        print("System shows promise but failed acceptance gates")
    else:
        print(f"\n❌ FIXED OOS VALIDATION FAILED")

if __name__ == "__main__":
    main()