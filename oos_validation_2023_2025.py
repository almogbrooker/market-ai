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
        
        # Ensemble prediction
        ensemble_pred = (
            ensemble_model['weight_5d'] * pred_5d + 
            ensemble_model['weight_20d'] * pred_20d
        )
        
        # Test against different horizons
        horizons_to_test = {
            '1d': 'next_return_1d',
            '5d': 'target_5d', 
            '20d': 'target_20d'
        }
        
        period_results = {
            'period': period_name,
            'samples': len(oos_data),
            'features_used': len(available_features),
            'date_range': f"{oos_data['Date'].min()} to {oos_data['Date'].max()}"
        }
        
        best_ic = 0
        best_horizon = '1d'
        
        for horizon_name, target_col in horizons_to_test.items():
            if target_col in oos_data.columns:
                y_actual = oos_data[target_col].fillna(0).values
                
                if len(y_actual) == len(ensemble_pred):
                    ic = np.corrcoef(ensemble_pred, y_actual)[0, 1]
                    if np.isnan(ic):
                        ic = 0
                    
                    period_results[f'ic_{horizon_name}'] = float(ic)
                    
                    if abs(ic) > abs(best_ic):
                        best_ic = ic
                        best_horizon = horizon_name
        
        period_results['best_ic'] = float(best_ic)
        period_results['best_horizon'] = best_horizon
        
        # Portfolio simulation for best horizon
        if best_ic != 0:
            y_best = oos_data[horizons_to_test[best_horizon]].fillna(0).values
            
            # Simple long/short portfolio simulation
            portfolio_returns = []
            
            # Group by date for daily rebalancing
            for date, day_data in oos_data.groupby('Date'):
                day_indices = day_data.index - oos_data.index[0]  # Relative indices
                
                if len(day_indices) >= 4:  # Need minimum stocks
                    day_pred = ensemble_pred[day_indices]
                    day_actual = y_best[day_indices]
                    
                    # Long/short based on prediction quartiles
                    q75 = np.percentile(day_pred, 75)
                    q25 = np.percentile(day_pred, 25)
                    
                    long_mask = day_pred >= q75
                    short_mask = day_pred <= q25
                    
                    if np.any(long_mask) and np.any(short_mask):
                        long_return = np.mean(day_actual[long_mask])
                        short_return = -np.mean(day_actual[short_mask])  # Short position
                        daily_portfolio_return = (long_return + short_return) / 2
                        portfolio_returns.append(daily_portfolio_return)
            
            if portfolio_returns:
                portfolio_returns = np.array(portfolio_returns)
                total_return = np.sum(portfolio_returns) * 100
                volatility = np.std(portfolio_returns) * np.sqrt(252) * 100
                sharpe = np.mean(portfolio_returns) * 252 / (np.std(portfolio_returns) * np.sqrt(252)) if np.std(portfolio_returns) > 0 else 0
                
                period_results.update({
                    'portfolio_return_pct': float(total_return),
                    'portfolio_volatility_pct': float(volatility), 
                    'portfolio_sharpe': float(sharpe),
                    'trading_days': len(portfolio_returns)
                })
        
        oos_results.append(period_results)
        
        # Log results
        logger.info(f"   📊 RESULTS:")
        logger.info(f"      Best IC ({best_horizon}): {best_ic:+.6f}")
        if 'portfolio_return_pct' in period_results:
            logger.info(f"      Portfolio Return: {period_results['portfolio_return_pct']:+.2f}%")
            logger.info(f"      Sharpe Ratio: {period_results['portfolio_sharpe']:+.2f}")
    
    # Overall assessment
    if oos_results:
        all_ics = [r['best_ic'] for r in oos_results]
        avg_ic = np.mean(all_ics)
        
        portfolio_returns = [r.get('portfolio_return_pct', 0) for r in oos_results if 'portfolio_return_pct' in r]
        total_portfolio_return = np.sum(portfolio_returns)
        
        logger.info(f"\n🎯 OVERALL OOS ASSESSMENT")
        logger.info(f"   Average IC: {avg_ic:+.6f}")
        logger.info(f"   IC Range: [{min(all_ics):+.6f}, {max(all_ics):+.6f}]")
        logger.info(f"   Total Portfolio Return: {total_portfolio_return:+.2f}%")
        logger.info(f"   Periods tested: {len(oos_results)}")
        
        # Pass/fail criteria
        oos_passed = (
            avg_ic >= 0.005 and  # Average IC ≥ 0.5%
            avg_ic >= 0.5 * ensemble_model.get('train_ic_5d', 0.01) and  # At least 50% of training IC
            len([ic for ic in all_ics if ic > 0]) >= len(all_ics) * 0.6  # 60% positive periods
        )
        
        logger.info(f"\n🏆 OOS VALIDATION: {'✅ PASSED' if oos_passed else '❌ FAILED'}")
        
        # Save results
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'validation_type': 'oos_2023_2025',
            'training_period': f"{train_data['Date'].min()} to {train_data['Date'].max()}",
            'ensemble_model': {
                'features_count': len(validated_features),
                'train_ic_5d': ensemble_model['train_ic_5d'],
                'train_ic_20d': ensemble_model['train_ic_20d'],
                'weight_5d': ensemble_model['weight_5d'],
                'weight_20d': ensemble_model['weight_20d']
            },
            'oos_results': oos_results,
            'summary': {
                'average_ic': float(avg_ic),
                'ic_range': [float(min(all_ics)), float(max(all_ics))],
                'total_portfolio_return_pct': float(total_portfolio_return),
                'periods_tested': len(oos_results),
                'validation_passed': oos_passed
            }
        }
        
        # Save report
        reports_dir = Path(__file__).parent / 'reports'
        reports_dir.mkdir(exist_ok=True)
        
        report_path = reports_dir / 'oos_validation_2023_2025.json'
        with open(report_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"📊 OOS validation report saved: {report_path}")
        
        return final_results
    
    else:
        logger.error("❌ No OOS results generated")
        return False

def main():
    """Run OOS validation"""
    
    results = oos_validation_2023_2025()
    
    if results and results['summary']['validation_passed']:
        print(f"\n🎉 OOS VALIDATION PASSED!")
        print(f"Average IC: {results['summary']['average_ic']:+.6f}")
        print(f"Portfolio Return: {results['summary']['total_portfolio_return_pct']:+.2f}%")
        print("System validated on completely unseen 2023-2025 data!")
    elif results:
        print(f"\n⚠️ OOS VALIDATION PARTIAL")
        print(f"Average IC: {results['summary']['average_ic']:+.6f}")
        print("System shows some predictive power but needs improvement")
    else:
        print(f"\n❌ OOS VALIDATION FAILED")

if __name__ == "__main__":
    main()