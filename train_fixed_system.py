#!/usr/bin/env python3
"""
Train Fixed System - Train the complete system using the IC-fixed dataset
"""

import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Import system components
from src.models.tiered_system import TieredAlphaSystem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def train_system_with_fixed_data():
    """Train the complete system using the IC-fixed dataset"""
    
    logger.info("🚀 TRAINING SYSTEM WITH FIXED IC DATASET")
    logger.info("=" * 60)
    
    # Load fixed dataset
    fixed_data_path = Path(__file__).parent / 'data' / 'training_data_enhanced_fixed.csv'
    if not fixed_data_path.exists():
        logger.error(f"Fixed dataset not found: {fixed_data_path}")
        logger.info("Run 'python implement_ic_fixes.py' first to create the fixed dataset")
        return False
    
    data = pd.read_csv(fixed_data_path)
    data['Date'] = pd.to_datetime(data['Date'])
    
    logger.info(f"Loaded fixed dataset: {len(data):,} samples")
    logger.info(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    logger.info(f"Columns: {len(data.columns)}")
    
    # Use data before 2023 for training (keeps 2023-2024 for validation)
    train_data = data[data['Date'] < '2023-01-01']
    
    logger.info(f"Training samples: {len(train_data):,}")
    logger.info(f"Training period: {train_data['Date'].min()} to {train_data['Date'].max()}")
    
    # Initialize system configuration
    system_config = {
        'lstm': {'enabled': True, 'max_epochs': 50, 'sequence_length': 30},
        'regime': {'enabled': False},  # Disabled to avoid external dependencies
        'meta': {'combiner_type': 'ridge'}
    }
    
    logger.info(f"System configuration: {system_config}")
    
    # Initialize and train system
    alpha_system = TieredAlphaSystem(system_config)
    
    try:
        logger.info("\n🏋️ TRAINING COMPLETE TIERED SYSTEM...")
        training_results = alpha_system.train_system(train_data)
        
        logger.info("\n✅ TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)
        
        # Display training results
        for model_name, results in training_results.items():
            if isinstance(results, dict):
                logger.info(f"📊 {model_name.upper()} RESULTS:")
                if 'val_ic' in results:
                    logger.info(f"   Validation IC: {results['val_ic']:+.6f}")
                if 'val_mse' in results:
                    logger.info(f"   Validation MSE: {results['val_mse']:.6f}")
                if 'best_epoch' in results:
                    logger.info(f"   Best Epoch: {results['best_epoch']}")
            else:
                logger.info(f"📊 {model_name.upper()}: {results}")
        
        # Test a quick prediction to ensure everything works
        logger.info("\n🔮 TESTING QUICK PREDICTION...")
        test_sample = train_data.tail(100)  # Use last 100 samples from training
        
        predictions = alpha_system.predict_alpha(test_sample)
        
        final_scores = predictions.get('final_scores', [])
        n_tradeable = predictions.get('n_tradeable', 0)
        
        logger.info(f"   Test prediction samples: {len(final_scores)}")
        logger.info(f"   Tradeable positions: {n_tradeable}")
        logger.info(f"   Score range: [{np.min(final_scores):.6f}, {np.max(final_scores):.6f}]")
        
        # Save system state indication
        models_dir = Path(__file__).parent / 'models'
        models_dir.mkdir(exist_ok=True)
        
        status_file = models_dir / 'training_status.txt'
        with open(status_file, 'w') as f:
            f.write("SYSTEM_TRAINED_WITH_FIXED_DATASET\n")
            f.write(f"Training completed: {pd.Timestamp.now()}\n")
            f.write(f"Training samples: {len(train_data):,}\n")
            f.write(f"IC fix applied: YES\n")
        
        logger.info(f"✅ Training status saved: {status_file}")
        
        logger.info("\n🎉 SYSTEM READY FOR 6-MONTH VALIDATION!")
        logger.info("Next steps:")
        logger.info("1. Run: python run_offline_validation.py")
        logger.info("2. Or run: python validation/paper_trading_engine.py")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main training function"""
    
    success = train_system_with_fixed_data()
    
    if success:
        print("\n🚀 TRAINING SUCCESSFUL!")
        print("System is now trained with IC-fixed dataset and ready for validation.")
    else:
        print("\n❌ TRAINING FAILED!")
        print("Check the logs above for error details.")
    
    return success

if __name__ == "__main__":
    main()