#!/usr/bin/env python3
"""
INSTITUTIONAL TRADING SYSTEM
============================
Organized system with proper directory structure and path checking
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import joblib
import warnings
warnings.filterwarnings('ignore')

class InstitutionalSystem:
    """Complete institutional trading system with organized structure"""
    
    def __init__(self):
        print("🏛️ INSTITUTIONAL TRADING SYSTEM")
        print("=" * 60)
        
        # Organized directory structure
        self.base_dir = Path("../artifacts")
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models" 
        self.processed_dir = self.base_dir / "processed"
        self.validation_dir = self.base_dir / "validation"
        
        # Create directories if needed
        for dir_path in [self.data_dir, self.models_dir, self.processed_dir, self.validation_dir]:
            dir_path.mkdir(exist_ok=True)
        
        print(f"📁 Base directory: {self.base_dir}")
        print(f"📊 Data directory: {self.data_dir}")
        print(f"🎯 Models directory: {self.models_dir}")
        print(f"🔧 Processed directory: {self.processed_dir}")
        print(f"✅ Validation directory: {self.validation_dir}")
        
        # System configuration
        self.current_model_name = None
        self.model_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def check_data_status(self) -> dict:
        """Check what data is available"""
        print("\n📊 CHECKING DATA STATUS")
        print("-" * 40)
        
        data_status = {}
        
        # Check raw training data
        train_file = self.base_dir / "ds_train.parquet"
        if train_file.exists():
            df = pd.read_parquet(train_file)
            data_status['raw_training'] = {
                'exists': True,
                'path': str(train_file),
                'shape': df.shape,
                'date_range': f"{df['Date'].min()} to {df['Date'].max()}" if 'Date' in df.columns else 'N/A',
                'tickers': df['Ticker'].nunique() if 'Ticker' in df.columns else 'N/A'
            }
            print(f"   ✅ Raw training data: {df.shape[0]:,} rows, {df.shape[1]} columns")
            print(f"      📅 Period: {data_status['raw_training']['date_range']}")
            print(f"      🏢 Tickers: {data_status['raw_training']['tickers']}")
        else:
            data_status['raw_training'] = {'exists': False, 'path': str(train_file)}
            print(f"   ❌ Raw training data not found: {train_file}")
        
        # Check processed data
        processed_train = self.processed_dir / "train_institutional.parquet"
        if processed_train.exists():
            df = pd.read_parquet(processed_train)
            data_status['processed_training'] = {
                'exists': True,
                'path': str(processed_train),
                'shape': df.shape
            }
            print(f"   ✅ Processed training data: {df.shape[0]:,} rows, {df.shape[1]} columns")
        else:
            data_status['processed_training'] = {'exists': False, 'path': str(processed_train)}
            print(f"   ❌ Processed training data not found: {processed_train}")
        
        # Check feature config
        feature_config = self.processed_dir / "feature_config.json"
        if feature_config.exists():
            with open(feature_config, 'r') as f:
                config = json.load(f)
            data_status['feature_config'] = {
                'exists': True,
                'features_count': len(config.get('selected_features', []))
            }
            print(f"   ✅ Feature config: {data_status['feature_config']['features_count']} features")
        else:
            data_status['feature_config'] = {'exists': False}
            print(f"   ❌ Feature config not found: {feature_config}")
        
        return data_status
    
    def check_models_status(self) -> dict:
        """Check what models are available"""
        print("\n🎯 CHECKING MODELS STATUS")
        print("-" * 40)
        
        models_status = {}
        
        # List all model directories
        if self.models_dir.exists():
            model_dirs = [d for d in self.models_dir.iterdir() if d.is_dir()]
            
            if model_dirs:
                print(f"   📊 Found {len(model_dirs)} model directories:")
                
                for model_dir in sorted(model_dirs, reverse=True):  # Most recent first
                    model_card_path = model_dir / "model_card.json"
                    
                    if model_card_path.exists():
                        with open(model_card_path, 'r') as f:
                            model_card = json.load(f)
                        
                        model_info = {
                            'name': model_dir.name,
                            'path': str(model_dir),
                            'type': model_card.get('model_info', {}).get('model_type', 'unknown'),
                            'ic': model_card.get('performance', {}).get('validation_ic', 0.0),
                            'approved': model_card.get('deployment_ready', False),
                            'date': model_card.get('model_info', {}).get('training_date', 'unknown')
                        }
                        
                        models_status[model_dir.name] = model_info
                        
                        status_icon = "✅" if model_info['approved'] else "⚠️"
                        print(f"      {status_icon} {model_dir.name}")
                        print(f"         Type: {model_info['type']}, IC: {model_info['ic']:.4f}")
                        print(f"         Approved: {'✅' if model_info['approved'] else '❌'}")
                    else:
                        print(f"      ⚠️ {model_dir.name} (no model card)")
                        models_status[model_dir.name] = {
                            'name': model_dir.name,
                            'path': str(model_dir),
                            'incomplete': True
                        }
            else:
                print(f"   📊 No model directories found")
                models_status = {}
        else:
            print(f"   ❌ Models directory does not exist: {self.models_dir}")
            models_status = {}
        
        return models_status
    
    def get_latest_approved_model(self) -> dict:
        """Get the latest institutionally approved model"""
        models_status = self.check_models_status()
        
        # Filter approved models
        approved_models = {
            name: info for name, info in models_status.items() 
            if info.get('approved', False) and not info.get('incomplete', False)
        }
        
        if approved_models:
            # Get most recent approved model
            latest_model_name = max(approved_models.keys())
            latest_model = approved_models[latest_model_name]
            
            print(f"\n🏆 LATEST APPROVED MODEL")
            print("-" * 40)
            print(f"   📊 Model: {latest_model['name']}")
            print(f"   🎯 Type: {latest_model['type']}")
            print(f"   📈 IC: {latest_model['ic']:.4f}")
            print(f"   📅 Date: {latest_model['date']}")
            print(f"   📁 Path: {latest_model['path']}")
            
            self.current_model_name = latest_model_name
            return latest_model
        else:
            print(f"\n⚠️ NO APPROVED MODELS FOUND")
            return {}
    
    def load_model_for_prediction(self, model_name: str = None) -> dict:
        """Load a specific model for making predictions"""
        if model_name is None:
            model_info = self.get_latest_approved_model()
            if not model_info:
                raise ValueError("No approved model available")
            model_name = model_info['name']
        
        model_dir = self.models_dir / model_name
        
        print(f"\n🔧 LOADING MODEL: {model_name}")
        print("-" * 40)
        
        # Load model card
        with open(model_dir / "model_card.json", 'r') as f:
            model_card = json.load(f)
        
        model_type = model_card['model_info']['model_type']
        
        # Load model components
        if model_type in ['ridge', 'lasso']:
            model = joblib.load(model_dir / "model.pkl")
            scaler = joblib.load(model_dir / "scaler.pkl")
            print(f"   ✅ Loaded {model_type} model and scaler")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Load features
        with open(model_dir / "features.json", 'r') as f:
            features = json.load(f)
        
        print(f"   📊 Features loaded: {len(features)}")
        print(f"   🎯 Model IC: {model_card['performance']['validation_ic']:.4f}")
        
        return {
            'model': model,
            'scaler': scaler,
            'features': features,
            'model_card': model_card,
            'model_name': model_name
        }
    
    def validate_system_integrity(self) -> dict:
        """Comprehensive system integrity check"""
        print("\n🔍 SYSTEM INTEGRITY VALIDATION")
        print("=" * 50)
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {}
        }
        
        # 1. Check data integrity
        print("\n📊 Data Integrity Check:")
        data_status = self.check_data_status()
        
        data_check = {
            'raw_data_available': data_status['raw_training']['exists'],
            'processed_data_available': data_status['processed_training']['exists'],
            'feature_config_available': data_status['feature_config']['exists']
        }
        
        data_integrity_pass = all(data_check.values())
        print(f"   Data integrity: {'✅ PASS' if data_integrity_pass else '❌ FAIL'}")
        
        validation_results['checks']['data_integrity'] = {
            'status': 'PASS' if data_integrity_pass else 'FAIL',
            'details': data_check
        }
        
        # 2. Check models integrity
        print("\n🎯 Models Integrity Check:")
        models_status = self.check_models_status()
        
        approved_models_count = sum(1 for info in models_status.values() if info.get('approved', False))
        latest_approved = self.get_latest_approved_model()
        
        models_check = {
            'models_directory_exists': self.models_dir.exists(),
            'approved_models_available': approved_models_count > 0,
            'latest_approved_loadable': bool(latest_approved)
        }
        
        models_integrity_pass = all(models_check.values())
        print(f"   Models integrity: {'✅ PASS' if models_integrity_pass else '❌ FAIL'}")
        
        validation_results['checks']['models_integrity'] = {
            'status': 'PASS' if models_integrity_pass else 'FAIL',
            'details': models_check,
            'approved_models_count': approved_models_count
        }
        
        # 3. Model loading test
        print("\n🧪 Model Loading Test:")
        try:
            if latest_approved:
                model_components = self.load_model_for_prediction()
                
                # Test prediction capability
                test_features = np.random.randn(1, len(model_components['features']))
                test_features_scaled = model_components['scaler'].transform(test_features)
                test_prediction = model_components['model'].predict(test_features_scaled)
                
                loading_test_pass = True
                print(f"   Model loading test: ✅ PASS")
                print(f"   Test prediction: {test_prediction[0]:.6f}")
            else:
                loading_test_pass = False
                print(f"   Model loading test: ❌ FAIL (no approved model)")
                
        except Exception as e:
            loading_test_pass = False
            print(f"   Model loading test: ❌ FAIL ({str(e)})")
        
        validation_results['checks']['model_loading'] = {
            'status': 'PASS' if loading_test_pass else 'FAIL'
        }
        
        # 4. Overall system status
        overall_pass = data_integrity_pass and models_integrity_pass and loading_test_pass
        
        validation_results['overall_status'] = 'PASS' if overall_pass else 'FAIL'
        validation_results['deployment_ready'] = overall_pass
        
        print(f"\n🏛️ OVERALL SYSTEM STATUS")
        print("=" * 50)
        print(f"📊 Data integrity: {'✅' if data_integrity_pass else '❌'}")
        print(f"🎯 Models integrity: {'✅' if models_integrity_pass else '❌'}")
        print(f"🧪 Model loading: {'✅' if loading_test_pass else '❌'}")
        print(f"🎉 Overall status: {'✅ SYSTEM READY' if overall_pass else '❌ SYSTEM NOT READY'}")
        
        # Save validation results
        validation_file = self.validation_dir / f"system_validation_{self.model_timestamp}.json"
        with open(validation_file, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        print(f"📄 Validation report saved: {validation_file}")
        
        return validation_results
    
    def create_prediction_interface(self) -> dict:
        """Create interface for making predictions"""
        print("\n🚀 CREATING PREDICTION INTERFACE")
        print("=" * 50)
        
        # Ensure we have an approved model
        model_components = self.load_model_for_prediction()
        
        def make_prediction(feature_data: dict) -> dict:
            """Make a single prediction"""
            try:
                # Convert features to array in correct order
                feature_values = []
                for feature in model_components['features']:
                    if feature in feature_data:
                        feature_values.append(feature_data[feature])
                    else:
                        raise ValueError(f"Missing feature: {feature}")
                
                # Scale and predict
                feature_array = np.array(feature_values).reshape(1, -1)
                feature_scaled = model_components['scaler'].transform(feature_array)
                prediction = model_components['model'].predict(feature_scaled)[0]
                
                return {
                    'prediction': prediction,
                    'model_name': model_components['model_name'],
                    'model_ic': model_components['model_card']['performance']['validation_ic'],
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                return {
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        print(f"   ✅ Prediction interface created")
        print(f"   🎯 Model: {model_components['model_name']}")
        print(f"   📊 Features required: {len(model_components['features'])}")
        
        interface_info = {
            'model_name': model_components['model_name'],
            'required_features': model_components['features'],
            'model_ic': model_components['model_card']['performance']['validation_ic'],
            'prediction_function': make_prediction
        }
        
        return interface_info
    
    def run_complete_system_check(self) -> dict:
        """Run complete system check and setup"""
        print("🏛️ RUNNING COMPLETE SYSTEM CHECK")
        print("=" * 80)
        
        try:
            # 1. System integrity validation
            validation_results = self.validate_system_integrity()
            
            if not validation_results['deployment_ready']:
                print(f"\n❌ SYSTEM NOT READY FOR DEPLOYMENT")
                return validation_results
            
            # 2. Create prediction interface
            prediction_interface = self.create_prediction_interface()
            
            # 3. Final summary
            system_summary = {
                'system_status': 'READY',
                'validation_results': validation_results,
                'prediction_interface': {
                    'model_name': prediction_interface['model_name'],
                    'required_features': prediction_interface['required_features'],
                    'model_performance': prediction_interface['model_ic']
                },
                'directories': {
                    'data': str(self.data_dir),
                    'models': str(self.models_dir),
                    'processed': str(self.processed_dir),
                    'validation': str(self.validation_dir)
                }
            }
            
            print(f"\n🎉 SYSTEM CHECK COMPLETE")
            print("=" * 50)
            print(f"✅ System ready for institutional deployment")
            print(f"🎯 Active model: {prediction_interface['model_name']}")
            print(f"📈 Model IC: {prediction_interface['model_ic']:.4f}")
            print(f"🔧 Features required: {len(prediction_interface['required_features'])}")
            print(f"📁 All directories organized and validated")
            
            return system_summary
            
        except Exception as e:
            print(f"❌ System check failed: {str(e)}")
            return {'system_status': 'FAILED', 'error': str(e)}

def main():
    """Main system check execution"""
    system = InstitutionalSystem()
    results = system.run_complete_system_check()
    
    if results.get('system_status') == 'READY':
        print(f"\n🎯 INSTITUTIONAL SYSTEM: ✅ READY FOR DEPLOYMENT")
        return True
    else:
        print(f"\n🚨 INSTITUTIONAL SYSTEM: ❌ NOT READY")
        return False

if __name__ == "__main__":
    success = main()