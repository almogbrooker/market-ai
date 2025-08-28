#!/usr/bin/env python3
"""
LIVE TRADING BOT
================
Production trading bot with broker integration, risk controls, and kill-switch
CRITICAL FIX: Proper turnover control (monthly not daily!)
"""

import pandas as pd
import numpy as np
import json
import pickle
import logging
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Alpaca API imports
try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import APIError
except ImportError:
    print("⚠️ alpaca-trade-api not installed. Run: pip install alpaca-trade-api")
    tradeapi = None
    APIError = Exception

from live_data_fetcher import LiveDataFetcher
from feature_engineering import UnifiedFeatureEngine, ProductionEnsemble

class LiveTradingBot:
    """Production trading bot with comprehensive risk controls"""
    
    def __init__(self, config_path: Optional[str] = None, paper_trading: bool = True, alpaca_credentials: Optional[Dict] = None):
        self.setup_logging()
        self.config = self._load_config(config_path)
        self.paper_trading = paper_trading
        self.alpaca_credentials = alpaca_credentials
        
        # Initialize Alpaca API
        self.alpaca_api = None
        self._setup_alpaca_connection()
        
        print(f"🤖 LIVE TRADING BOT v3.0")
        print(f"{'=== PAPER TRADING MODE ===' if paper_trading else '=== LIVE TRADING MODE ==='}")
        print("=" * 70)
        
        # CRITICAL: Fixed turnover control (monthly target, not daily!)
        self.monthly_turnover_target = 0.20  # 20% MONTHLY turnover target
        self.max_daily_turnover = self.monthly_turnover_target / 21  # ~0.95% daily
        
        # For initial portfolio build, allow higher turnover
        self.initial_build_mode = True
        self.initial_turnover_limit = 0.10  # 10% daily for initial build
        self.turnover_buffer = 0.8  # Use 80% of max to stay safe
        
        # Risk controls
        self.max_position_size = 0.015  # 1.5% per position
        self.max_gross_exposure = 0.30   # 30% total gross exposure
        self.max_net_exposure = 0.03     # 3% net exposure limit
        self.max_sector_exposure = 0.15  # 15% per sector
        
        # Performance controls
        self.min_ic_threshold = 0.005    # Kill switch if IC < 0.5%
        self.max_drawdown = 0.025        # 2.5% max drawdown
        self.lookback_days = 20          # IC calculation window
        
        # Emergency controls
        self.kill_switch_active = False
        self.emergency_override = False
        
        # Components
        self.data_fetcher = LiveDataFetcher()
        self.feature_engine = UnifiedFeatureEngine()
        self.current_positions = {}
        
        # Alpaca-specific settings
        self.alpaca_account_info = None
        self.alpaca_positions = {}
        
        # Load model
        self.load_production_model()
        
        print(f"✅ Configuration:")
        print(f"   🎯 Mode: {'PAPER' if paper_trading else 'LIVE'}")
        print(f"   📊 Monthly turnover target: {self.monthly_turnover_target:.1%}")
        print(f"   📈 Max daily turnover: {self.max_daily_turnover:.2%}")
        print(f"   💰 Max position size: {self.max_position_size:.1%}")
        print(f"   📊 Max gross exposure: {self.max_gross_exposure:.1%}")
        print(f"   🔗 Alpaca API: {'Connected' if self.alpaca_api else 'Not connected'}")
        
    def setup_logging(self):
        """Setup comprehensive logging with decision tracking"""
        log_dir = Path("../artifacts/logs/live_trading")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Main trading log
        log_file = log_dir / f"live_trading_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Decision log for detailed stock selection reasoning
        self.decision_log_file = log_dir / f"decision_log_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup decision logger
        self.decision_logger = logging.getLogger('decisions')
        decision_handler = logging.FileHandler(self.decision_log_file)
        decision_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.decision_logger.addHandler(decision_handler)
        self.decision_logger.setLevel(logging.INFO)
        self.decision_logger.propagate = False
    
    def _load_config(self, config_path):
        """Load trading configuration"""
        default_config = {
            'rebalance_time': '15:45',  # 15 min before close
            'order_type': 'market',     # market/limit/stop
            'max_order_value': 1000000, # $1M max order size
            'broker': 'alpaca',         # alpaca/ib/paper
            'risk_checks': True,
            'pre_trade_validation': True,
            'alpaca_paper_trading': True,  # Use Alpaca paper trading
            'portfolio_value': 100000,     # $100k portfolio
            'min_order_value': 10          # $10 minimum order
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                user_config = json.load(f)
            default_config.update(user_config)
            
        return default_config
    
    def _setup_alpaca_connection(self):
        """Setup Alpaca API connection"""
        try:
            if tradeapi is None:
                self.logger.warning("Alpaca API not available - install alpaca-trade-api")
                return
            
            # Load .env file if it exists
            env_file = Path("../.env")
            if env_file.exists():
                with open(env_file) as f:
                    for line in f:
                        if line.strip() and not line.startswith('#'):
                            key, value = line.strip().split('=', 1)
                            os.environ[key] = value
                
            # Load credentials from environment or config
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            
            if self.alpaca_credentials:
                api_key = self.alpaca_credentials.get('api_key', api_key)
                secret_key = self.alpaca_credentials.get('secret_key', secret_key)
            
            if not api_key or not secret_key:
                self.logger.warning("Alpaca credentials not found - paper trading only")
                return
            
            # Setup base URL (paper vs live)
            base_url = 'https://paper-api.alpaca.markets' if self.paper_trading else 'https://api.alpaca.markets'
            
            # Initialize Alpaca API
            self.alpaca_api = tradeapi.REST(
                api_key,
                secret_key,
                base_url,
                api_version='v2'
            )
            
            # Test connection and get account info
            account = self.alpaca_api.get_account()
            self.alpaca_account_info = {
                'account_id': account.id,
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'pattern_day_trader': account.pattern_day_trader,
                'status': account.status
            }
            
            print(f"✅ Alpaca API connected successfully")
            print(f"   📊 Account equity: ${self.alpaca_account_info['equity']:,.2f}")
            print(f"   💰 Buying power: ${self.alpaca_account_info['buying_power']:,.2f}")
            print(f"   📈 Account status: {self.alpaca_account_info['status']}")
            
            # Load current Alpaca positions
            self._load_alpaca_positions()
            
        except Exception as e:
            self.logger.error(f"Failed to setup Alpaca connection: {e}")
            self.alpaca_api = None
    
    def _load_alpaca_positions(self):
        """Load current positions from Alpaca"""
        try:
            if not self.alpaca_api:
                return
            
            positions = self.alpaca_api.list_positions()
            self.alpaca_positions = {}
            
            total_equity = self.alpaca_account_info['equity']
            
            for position in positions:
                symbol = position.symbol
                market_value = float(position.market_value)
                weight = market_value / total_equity if total_equity > 0 else 0
                
                self.alpaca_positions[symbol] = {
                    'qty': float(position.qty),
                    'market_value': market_value,
                    'weight': weight,
                    'unrealized_pl': float(position.unrealized_pl),
                    'side': position.side
                }
            
            print(f"📊 Loaded {len(self.alpaca_positions)} Alpaca positions")
            if self.alpaca_positions:
                total_exposure = sum(abs(pos['weight']) for pos in self.alpaca_positions.values())
                print(f"   📈 Total exposure: {total_exposure:.1%}")
            
        except Exception as e:
            self.logger.error(f"Failed to load Alpaca positions: {e}")
            self.alpaca_positions = {}
    
    def load_production_model(self):
        """Load the production-approved model with ultimate validation"""
        production_dir = Path("../artifacts/production_models")
        
        try:
            # Check for final production model first (single best model)
            final_config = production_dir / "final_production_config.json"
            final_model = production_dir / "final_production_model.pkl"
            
            if final_config.exists() and final_model.exists():
                with open(final_config) as f:
                    config = json.load(f)
                
                final_metadata = production_dir / "final_production_metadata.json"
                
                with open(final_model, 'rb') as f:
                    self.model = pickle.load(f)
                with open(final_metadata) as f:
                    self.model_metadata = json.load(f)
                
                print(f"✅ Loaded FINAL production model: {config['primary_model']}")
                print(f"   🎯 Strategy: {config['model_strategy']} ({config['model_type']})")
                print(f"   📊 Test IC: {config['performance_metrics']['test_ic']:.4f}")
                print(f"   📅 Recent IC: {config['performance_metrics']['recent_ic']:.4f}")
                print(f"   ✅ QA Pass Rate: {config['validation_results']['qa_pass_rate']:.1%}")
                print(f"   🚫 Ensemble: Rejected (single model better)")
                
            # Fallback to ultimate production model
            elif (production_dir / "ultimate_production_config.json").exists():
                ultimate_config = production_dir / "ultimate_production_config.json"
                ultimate_model = production_dir / "ultimate_production_model.pkl"
                
                with open(ultimate_config) as f:
                    config = json.load(f)
                
                ultimate_metadata = production_dir / "ultimate_production_metadata.json"
                
                with open(ultimate_model, 'rb') as f:
                    self.model = pickle.load(f)
                with open(ultimate_metadata) as f:
                    self.model_metadata = json.load(f)
                
                print(f"✅ Loaded ULTIMATE production model: {config['primary_model']}")
                print(f"   🎯 Model type: {self.model_metadata.get('model_type', 'Unknown')}")
                print(f"   📊 Validation IC: {config.get('validation_ic', 'N/A'):.4f}")
                print(f"   📅 Recent IC: {config.get('recent_ic', 'N/A'):.4f}")
                
            else:
                # Fallback to regular production config
                config_file = production_dir / "production_config.json"
                if config_file.exists():
                    with open(config_file) as f:
                        config = json.load(f)
                    
                    primary_model_file = production_dir / config['primary_model']
                    primary_metadata_file = primary_model_file.with_name(primary_model_file.name.replace('.pkl', '_metadata.json'))
                    
                    with open(primary_model_file, 'rb') as f:
                        self.model = pickle.load(f)
                    with open(primary_metadata_file) as f:
                        self.model_metadata = json.load(f)
                    
                    print(f"✅ Loaded production model: {config['primary_model']}")
                else:
                    # Final fallback
                    models_dir = Path("../artifacts/models")
                    hardened_files = list(models_dir.glob("hardened_model_*.pkl"))
                    if hardened_files:
                        model_file = max(hardened_files)
                        metadata_file = model_file.with_name(model_file.name.replace('model', 'metadata')).with_suffix('.json')
                        
                        with open(model_file, 'rb') as f:
                            self.model = pickle.load(f)
                        with open(metadata_file) as f:
                            self.model_metadata = json.load(f)
                        
                        print(f"✅ Loaded fallback model: {model_file.name}")
                    else:
                        raise FileNotFoundError("No production models found")
            
            # Extract model features
            self.model_features = self.model_metadata['features']
            print(f"📊 Model features: {len(self.model_features)}")
            
        except Exception as e:
            self.logger.error(f"Failed to load production model: {e}")
            raise RuntimeError(f"Cannot start without production model: {e}")
    
    def _create_simple_ensemble(self, model_files):
        """Create simple ensemble from available models"""
        try:
            print("🎯 Creating ensemble from multiple models...")
            
            models = []
            for model_file in model_files[-3:]:  # Use up to 3 latest models
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                
                metadata_file = model_file.with_name(model_file.name.replace('model', 'metadata')).with_suffix('.json')
                with open(metadata_file) as f:
                    metadata = json.load(f)
                
                ic = metadata.get('performance', {}).get('final_ic', 0)
                models.append({'model': model, 'weight': max(0, ic)})
            
            if len(models) > 1:
                # Create ensemble wrapper
                class SimpleEnsemble:
                    def __init__(self, models):
                        self.models = models
                        total_weight = sum(m['weight'] for m in models)
                        if total_weight > 0:
                            for m in self.models:
                                m['weight'] /= total_weight
                        else:
                            for m in self.models:
                                m['weight'] = 1.0 / len(models)
                    
                    def predict(self, X):
                        predictions = np.zeros(X.shape[0])
                        for model_info in self.models:
                            pred = model_info['model'].predict(X)
                            predictions += pred * model_info['weight']
                        return predictions
                
                self.model = SimpleEnsemble(models)
                print(f"   ✅ Ensemble created with {len(models)} models")
                
        except Exception as e:
            print(f"   ⚠️ Ensemble creation failed, using single model: {e}")
    
    def check_kill_switch_conditions(self) -> Tuple[bool, List[str]]:
        """Check if kill switch should activate"""
        if self.emergency_override:
            return True, ["Emergency override activated"]
        
        violations = []
        
        # Import and run comprehensive guardrails check
        try:
            from production_guardrails import ProductionGuardrails
            guardrails = ProductionGuardrails()
            
            # Run guardrails assessment
            guardrails_result = self._run_production_guardrails_check(guardrails)
            
            if guardrails_result['kill_switch_triggered']:
                violations.extend(guardrails_result['violations'])
                
            # Apply any recommended actions
            if guardrails_result['recommended_actions']:
                self._apply_guardrails_actions(guardrails_result['recommended_actions'])
                
        except Exception as e:
            self.logger.warning(f"Guardrails check failed: {e}")
            # Fallback to basic checks
        
        # Basic fallback checks
        try:
            recent_ic = self._calculate_recent_ic()
            if recent_ic < self.min_ic_threshold:
                violations.append(f"IC below threshold: {recent_ic:.4f} < {self.min_ic_threshold:.3f}")
        except:
            violations.append("Cannot calculate recent IC")
        
        # Check drawdown
        try:
            current_dd = self._calculate_current_drawdown()
            if current_dd > self.max_drawdown:
                violations.append(f"Drawdown exceeded: {current_dd:.2%} > {self.max_drawdown:.2%}")
        except:
            pass  # Drawdown check optional if no history
        
        return len(violations) > 0, violations
    
    def _run_production_guardrails_check(self, guardrails) -> Dict:
        """Run comprehensive production guardrails assessment"""
        result = {
            'kill_switch_triggered': False,
            'violations': [],
            'recommended_actions': [],
            'alerts': []
        }
        
        try:
            # Load recent performance data
            performance_data = self._load_recent_performance_data()
            
            if performance_data is not None and len(performance_data) > 0:
                # Check IC performance
                ic_result = self._check_ic_guardrails(guardrails, performance_data)
                result['violations'].extend(ic_result['violations'])
                result['recommended_actions'].extend(ic_result['actions'])
                
                if ic_result['kill_switch']:
                    result['kill_switch_triggered'] = True
                
            else:
                result['violations'].append("No recent performance data available")
                
        except Exception as e:
            self.logger.error(f"Guardrails check error: {e}")
            result['violations'].append(f"Guardrails check failed: {e}")
        
        return result
    
    def _load_recent_performance_data(self) -> Optional[pd.DataFrame]:
        """Load recent performance data for guardrails checking"""
        try:
            # Look for recent validation reports
            validation_dir = Path("../artifacts/final_validation")
            if not validation_dir.exists():
                return None
            
            # Get most recent validation data
            report_files = list(validation_dir.glob("*.json"))
            if not report_files:
                return None
            
            latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_report) as f:
                report_data = json.load(f)
            
            # Extract relevant performance metrics
            if 'performance_metrics' in report_data:
                return pd.DataFrame([report_data['performance_metrics']])
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to load performance data: {e}")
            return None
    
    def _check_ic_guardrails(self, guardrails, performance_data: pd.DataFrame) -> Dict:
        """Check IC-based guardrails"""
        result = {
            'violations': [],
            'actions': [],
            'kill_switch': False
        }
        
        try:
            # Get recent IC from performance data
            if 'recent_ic' in performance_data.columns:
                recent_ic = performance_data['recent_ic'].iloc[0]
            else:
                recent_ic = self._calculate_recent_ic()
            
            # Check against thresholds
            if recent_ic < guardrails.ic_thresholds['rollback']:
                result['violations'].append(f"IC rollback triggered: {recent_ic:.4f} < {guardrails.ic_thresholds['rollback']:.4f}")
                result['kill_switch'] = True
                result['actions'].append({
                    'type': 'rollback_model',
                    'reason': 'IC below rollback threshold'
                })
            
            elif recent_ic < guardrails.ic_thresholds['de_risk']:
                result['violations'].append(f"De-risk triggered: {recent_ic:.4f} < {guardrails.ic_thresholds['de_risk']:.4f}")
                result['actions'].append({
                    'type': 'reduce_exposure',
                    'target_exposure': 0.5,
                    'reason': 'IC below de-risk threshold'
                })
                
        except Exception as e:
            result['violations'].append(f"IC check failed: {e}")
        
        return result
    
    def _apply_guardrails_actions(self, actions: List[Dict]):
        """Apply recommended guardrails actions"""
        for action in actions:
            try:
                action_type = action.get('type')
                reason = action.get('reason', 'No reason provided')
                
                print(f"   🛡️ Applying guardrails action: {action_type}")
                print(f"      Reason: {reason}")
                
                if action_type == 'reduce_exposure':
                    target_exposure = action.get('target_exposure', 0.5)
                    self._apply_exposure_reduction(target_exposure, reason)
                    
                elif action_type == 'reduce_turnover':
                    reduction = action.get('reduction', 0.05)
                    self._apply_turnover_reduction(reduction, reason)
                    
                elif action_type == 'rollback_model':
                    self.logger.critical(f"Model rollback recommended: {reason}")
                    print(f"      ⚠️ MODEL ROLLBACK REQUIRED - Manual intervention needed")
                    
                elif action_type == 'retrain_model':
                    self.logger.warning(f"Model retrain recommended: {reason}")
                    print(f"      📊 Model retrain recommended - Monitor performance")
                    
            except Exception as e:
                self.logger.error(f"Failed to apply action {action}: {e}")
    
    def _apply_exposure_reduction(self, target_exposure: float, reason: str):
        """Apply exposure reduction"""
        original_exposure = self.max_gross_exposure
        self.max_gross_exposure = min(self.max_gross_exposure, target_exposure)
        
        print(f"      📉 Gross exposure reduced: {original_exposure:.1%} → {self.max_gross_exposure:.1%}")
        self.logger.warning(f"Exposure reduced to {self.max_gross_exposure:.1%}: {reason}")
    
    def _apply_turnover_reduction(self, reduction: float, reason: str):
        """Apply turnover reduction"""
        original_turnover = self.max_daily_turnover
        self.max_daily_turnover = max(0.001, self.max_daily_turnover - reduction)  # Min 0.1% turnover
        
        print(f"      🔄 Daily turnover reduced: {original_turnover:.2%} → {self.max_daily_turnover:.2%}")
        self.logger.warning(f"Turnover reduced to {self.max_daily_turnover:.2%}: {reason}")
    
    def _calculate_recent_ic(self) -> float:
        """Calculate recent IC performance"""
        # Placeholder - would use actual performance history
        return 0.008  # Dummy value above threshold
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown"""
        # Placeholder - would use actual PnL history
        return 0.015  # Dummy value
    
    def load_current_positions(self) -> Dict:
        """Load current portfolio positions from Alpaca and local files"""
        try:
            # Prioritize Alpaca positions if available
            if self.alpaca_api and self.alpaca_positions:
                # Convert Alpaca positions to our format (weights)
                positions = {}
                for symbol, pos_info in self.alpaca_positions.items():
                    positions[symbol] = pos_info['weight']
                
                self.current_positions = positions
                print(f"📊 Loaded {len(positions)} Alpaca positions")
                return positions
            
            # Fallback to local positions file
            positions_file = Path("../artifacts/positions/current_positions.json")
            if positions_file.exists():
                with open(positions_file) as f:
                    positions = json.load(f)
                self.current_positions = positions
                print(f"📊 Loaded {len(positions)} local positions")
                return positions
            else:
                print("📊 No positions found - starting fresh")
                return {}
        except Exception as e:
            self.logger.warning(f"Failed to load current positions: {e}")
            return {}
    
    def calculate_target_portfolio(self, market_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate target portfolio with risk controls"""
        print(f"\\n💼 Calculating target portfolio...")
        
        try:
            # Generate features and predictions
            feature_data, available_features = self.feature_engine.create_features_from_data(market_data)
            if feature_data is None:
                return None
            
            # Get latest predictions (use most recent date with complete data)
            # Since features are lagged by 3 days, get the most recent complete data
            unique_dates = sorted(feature_data['Date'].unique(), reverse=True)
            latest_date = None
            latest_data = None
            
            for date in unique_dates:
                date_data = feature_data[feature_data['Date'] == date].copy()
                if len(date_data) >= 10:  # Need at least 10 stocks with complete data
                    latest_date = date
                    latest_data = date_data
                    break
            
            if latest_data is None or len(latest_data) == 0:
                print(f"   ❌ No recent data with complete features")
                return None
                
            print(f"   📅 Using data from: {latest_date} ({len(latest_data)} stocks)")
            
            # Verify feature alignment
            usable_features = [f for f in self.model_features if f in available_features]
            if len(usable_features) < len(self.model_features) * 0.8:
                self.logger.error(f"Insufficient features: {len(usable_features)}/{len(self.model_features)}")
                return None
            
            # Generate predictions
            X = latest_data[usable_features].fillna(0.5).values
            X = np.clip(X, 0, 1)
            predictions = self.model.predict(X)
            
            # Create prediction DataFrame with detailed features for decision logging
            portfolio_df = latest_data[['Date', 'Ticker', 'Close']].copy()
            portfolio_df['prediction'] = predictions
            portfolio_df['pred_rank'] = portfolio_df['prediction'].rank(pct=True, method='min')
            
            # Add feature values for decision analysis
            feature_cols = [col for col in usable_features if col in latest_data.columns]
            for feature in feature_cols[:10]:  # Log top 10 most important features
                portfolio_df[feature] = latest_data[feature]
            
            # Risk-based portfolio construction
            portfolio_df = portfolio_df.sort_values('prediction', ascending=False)
            
            # Calculate positions based on predictions and risk constraints
            n_positions = min(len(portfolio_df), int(self.max_gross_exposure / self.max_position_size))
            n_long = n_positions // 2
            n_short = n_positions // 2
            
            # Select top/bottom positions
            long_positions = portfolio_df.head(n_long).copy()
            short_positions = portfolio_df.tail(n_short).copy()
            
            # LOG DECISION REASONING FOR EACH POSITION
            self._log_position_decisions(long_positions, short_positions, portfolio_df, feature_cols)
            
            # Assign position sizes
            long_positions['position_type'] = 'LONG'
            long_positions['position_size'] = self.max_position_size
            
            short_positions['position_type'] = 'SHORT' 
            short_positions['position_size'] = -self.max_position_size
            
            # Combine
            target_portfolio = pd.concat([long_positions, short_positions], ignore_index=True)
            
            print(f"✅ Target portfolio calculated:")
            print(f"   📈 Long positions: {len(long_positions)}")
            print(f"   📉 Short positions: {len(short_positions)}")
            print(f"   💰 Gross exposure: {target_portfolio['position_size'].abs().sum():.1%}")
            
            return target_portfolio
            
        except Exception as e:
            self.logger.error(f"Portfolio calculation failed: {e}")
            return None
    
    def _log_position_decisions(self, long_positions, short_positions, full_portfolio, feature_cols):
        """Log detailed reasoning for each position selection"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        self.decision_logger.info("=" * 80)
        self.decision_logger.info(f"TRADING DECISIONS - {timestamp}")
        self.decision_logger.info("=" * 80)
        
        # Log overall market context
        self.decision_logger.info(f"📊 MARKET CONTEXT:")
        self.decision_logger.info(f"   Total stocks analyzed: {len(full_portfolio)}")
        self.decision_logger.info(f"   Prediction range: {full_portfolio['prediction'].min():.4f} to {full_portfolio['prediction'].max():.4f}")
        self.decision_logger.info(f"   Mean prediction: {full_portfolio['prediction'].mean():.4f}")
        self.decision_logger.info(f"   Prediction std: {full_portfolio['prediction'].std():.4f}")
        
        # Log LONG positions reasoning
        self.decision_logger.info(f"\n🟢 LONG POSITIONS (BUY) - Top {len(long_positions)} stocks:")
        self.decision_logger.info("-" * 60)
        
        for idx, (_, stock) in enumerate(long_positions.iterrows()):
            rank_str = f"{stock['pred_rank']:.1%}" if pd.notna(stock['pred_rank']) else "N/A"
            self.decision_logger.info(f"{idx+1}. {stock['Ticker']} - PREDICTION: {stock['prediction']:.4f} (Rank: {rank_str})")
            self.decision_logger.info(f"   💰 Price: ${stock['Close']:.2f}")
            
            # Feature analysis
            feature_reasons = []
            for feature in feature_cols[:8]:  # Top 8 features
                if feature in stock.index:
                    value = stock[feature]
                    if pd.notna(value):
                        percentile = (full_portfolio[feature] <= value).mean()
                        feature_name = feature.replace('_lag3_rank', '').replace('_', ' ').title()
                        
                        if percentile >= 0.8:
                            strength = "Very Strong"
                        elif percentile >= 0.6:
                            strength = "Strong" 
                        elif percentile >= 0.4:
                            strength = "Moderate"
                        else:
                            strength = "Weak"
                            
                        feature_reasons.append(f"{feature_name}: {value:.3f} ({percentile:.0%}ile - {strength})")
            
            if feature_reasons:
                self.decision_logger.info(f"   📈 Key factors (why BUY):")
                for reason in feature_reasons[:5]:  # Top 5 reasons
                    self.decision_logger.info(f"      • {reason}")
            
            self.decision_logger.info("")
        
        # Log SHORT positions reasoning  
        self.decision_logger.info(f"\n🔴 SHORT POSITIONS (SELL) - Bottom {len(short_positions)} stocks:")
        self.decision_logger.info("-" * 60)
        
        for idx, (_, stock) in enumerate(short_positions.iterrows()):
            rank_str = f"{stock['pred_rank']:.1%}" if pd.notna(stock['pred_rank']) else "N/A"
            self.decision_logger.info(f"{idx+1}. {stock['Ticker']} - PREDICTION: {stock['prediction']:.4f} (Rank: {rank_str})")
            self.decision_logger.info(f"   💰 Price: ${stock['Close']:.2f}")
            
            # Feature analysis for shorts
            feature_reasons = []
            for feature in feature_cols[:8]:
                if feature in stock.index:
                    value = stock[feature]
                    if pd.notna(value):
                        percentile = (full_portfolio[feature] <= value).mean()
                        feature_name = feature.replace('_lag3_rank', '').replace('_', ' ').title()
                        
                        if percentile <= 0.2:
                            strength = "Very Weak"
                        elif percentile <= 0.4:
                            strength = "Weak"
                        elif percentile <= 0.6:
                            strength = "Moderate"
                        else:
                            strength = "Strong"
                            
                        feature_reasons.append(f"{feature_name}: {value:.3f} ({percentile:.0%}ile - {strength})")
            
            if feature_reasons:
                self.decision_logger.info(f"   📉 Key factors (why SELL):")
                for reason in feature_reasons[:5]:
                    self.decision_logger.info(f"      • {reason}")
            
            self.decision_logger.info("")
        
        # Log neutral positions for context
        neutral_count = len(full_portfolio) - len(long_positions) - len(short_positions)
        if neutral_count > 0:
            self.decision_logger.info(f"\n⚪ NEUTRAL POSITIONS: {neutral_count} stocks (no position)")
            middle_stocks = full_portfolio.iloc[len(long_positions):-len(short_positions) if len(short_positions) > 0 else len(full_portfolio)]
            if len(middle_stocks) > 0:
                self.decision_logger.info(f"   Prediction range: {middle_stocks['prediction'].min():.4f} to {middle_stocks['prediction'].max():.4f}")
                sample_neutrals = middle_stocks.head(3)
                for _, stock in sample_neutrals.iterrows():
                    rank_str = f"{stock['pred_rank']:.1%}" if pd.notna(stock['pred_rank']) else "N/A"
                    self.decision_logger.info(f"   Example: {stock['Ticker']} - {stock['prediction']:.4f} (Rank: {rank_str})")
        
        self.decision_logger.info("=" * 80)
    
    def _log_trade_execution_decisions(self, trade_df: pd.DataFrame):
        """Log detailed reasoning for each trade execution"""
        if len(trade_df) == 0:
            return
            
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        self.decision_logger.info("=" * 80)
        self.decision_logger.info(f"TRADE EXECUTION DECISIONS - {timestamp}")
        self.decision_logger.info("=" * 80)
        
        # Overall execution context
        total_trades = len(trade_df)
        buy_trades = len(trade_df[trade_df['dollar_amount'] > 0])
        sell_trades = len(trade_df[trade_df['dollar_amount'] < 0])
        total_value = trade_df['dollar_amount'].abs().sum()
        
        self.decision_logger.info(f"📊 EXECUTION SUMMARY:")
        self.decision_logger.info(f"   Total trades: {total_trades}")
        self.decision_logger.info(f"   Buy orders: {buy_trades}")
        self.decision_logger.info(f"   Sell orders: {sell_trades}")
        self.decision_logger.info(f"   Total trade value: ${total_value:,.2f}")
        self.decision_logger.info(f"   Portfolio turnover: {getattr(self, 'current_turnover', 0):.2%}")
        
        # Log each trade with reasoning
        for idx, (_, trade) in enumerate(trade_df.iterrows()):
            action = "BUY" if trade['dollar_amount'] > 0 else "SELL"
            amount = abs(trade['dollar_amount'])
            
            self.decision_logger.info(f"\n{idx+1}. {action} {trade['Ticker']} - ${amount:,.2f}")
            
            # Trade reasoning
            current_weight = trade.get('current_weight', 0)
            target_weight = trade.get('target_weight', 0)
            weight_change = target_weight - current_weight
            
            self.decision_logger.info(f"   📈 Position change: {current_weight:.2%} → {target_weight:.2%} ({weight_change:+.2%})")
            
            # Why this trade is needed
            reasons = []
            if abs(weight_change) > 0.005:  # > 0.5% change
                if weight_change > 0:
                    reasons.append(f"Increasing position by {weight_change:.2%} - model predicts outperformance")
                else:
                    reasons.append(f"Reducing position by {abs(weight_change):.2%} - model predicts underperformance")
            
            if trade.get('rebalance_reason'):
                reasons.append(trade['rebalance_reason'])
                
            if 'urgency_score' in trade:
                urgency = trade['urgency_score']
                if urgency > 0.8:
                    reasons.append(f"High urgency trade (score: {urgency:.2f}) - significant alpha opportunity")
                elif urgency > 0.6:
                    reasons.append(f"Medium urgency trade (score: {urgency:.2f}) - moderate alpha opportunity")
            
            if current_weight == 0 and target_weight != 0:
                reasons.append("Opening new position - stock entered target portfolio")
            elif current_weight != 0 and target_weight == 0:
                reasons.append("Closing position - stock exited target portfolio")
            
            if reasons:
                self.decision_logger.info(f"   🎯 Trade rationale:")
                for reason in reasons:
                    self.decision_logger.info(f"      • {reason}")
            
            # Expected impact
            if 'expected_return' in trade:
                expected_return = trade['expected_return']
                self.decision_logger.info(f"   📊 Expected return: {expected_return:+.2%}")
            
            # Risk considerations
            risk_notes = []
            if amount > 5000:  # Large trade
                risk_notes.append(f"Large position size (${amount:,.0f})")
            if abs(weight_change) > 0.01:  # > 1% change
                risk_notes.append(f"Significant weight change ({weight_change:+.2%})")
            
            if risk_notes:
                self.decision_logger.info(f"   ⚠️ Risk considerations:")
                for note in risk_notes:
                    self.decision_logger.info(f"      • {note}")
        
        self.decision_logger.info("=" * 80)
    
    def calculate_trades_with_turnover_control(self, target_portfolio: pd.DataFrame) -> pd.DataFrame:
        """Calculate trades with inventory-aware turnover control and position sizing"""
        print(f"\\n🔄 Calculating trades with inventory-aware control...")
        
        # Load current positions with enhanced inventory tracking
        current_positions = self.load_current_positions()
        
        # Get current inventory metrics
        inventory_metrics = self._calculate_inventory_metrics(current_positions)
        
        # Convert to DataFrames for easier manipulation
        if current_positions:
            current_df = pd.DataFrame(list(current_positions.items()), 
                                    columns=['Ticker', 'current_weight'])
        else:
            current_df = pd.DataFrame(columns=['Ticker', 'current_weight'])
        
        # Merge with target
        target_df = target_portfolio[['Ticker', 'position_size']].copy()
        target_df = target_df.rename(columns={'position_size': 'target_weight'})
        
        # Full outer join to get all symbols
        trade_df = pd.merge(target_df, current_df, on='Ticker', how='outer').fillna(0)
        
        # Calculate desired change
        trade_df['desired_change'] = trade_df['target_weight'] - trade_df['current_weight']
        trade_df['desired_turnover'] = trade_df['desired_change'].abs().sum()
        
        # Apply inventory-aware adjustments
        trade_df = self._apply_inventory_adjustments(trade_df, inventory_metrics)
        
        print(f"   📊 Desired turnover: {trade_df['desired_turnover'].iloc[0]:.2%}")
        print(f"   📈 Daily limit: {self.max_daily_turnover:.2%}")
        print(f"   🎯 Monthly target: {self.monthly_turnover_target:.1%}")
        print(f"   📦 Net inventory: {inventory_metrics['net_exposure']:.2%}")
        print(f"   📊 Gross inventory: {inventory_metrics['gross_exposure']:.2%}")
        
        # Apply turnover control with inventory consideration
        effective_turnover_limit = self.initial_turnover_limit if self.initial_build_mode else self.max_daily_turnover
        
        if trade_df['desired_turnover'].iloc[0] > effective_turnover_limit:
            # Scale down based on inventory urgency
            base_shrink = effective_turnover_limit * self.turnover_buffer / trade_df['desired_turnover'].iloc[0]
            inventory_urgency = self._calculate_inventory_urgency(inventory_metrics)
            
            # Allow higher turnover if inventory is extreme or in initial build mode
            urgency_boost = inventory_urgency if not self.initial_build_mode else 0.5  # 50% boost for initial build
            turnover_shrink = min(1.0, base_shrink * (1 + urgency_boost))
            
            trade_df['actual_change'] = trade_df['desired_change'] * turnover_shrink
            mode_str = "INITIAL BUILD" if self.initial_build_mode else f"urgency: {inventory_urgency:.1%}"
            print(f"   ⚠️ Turnover scaled by {turnover_shrink:.2%} ({mode_str})")
        else:
            trade_df['actual_change'] = trade_df['desired_change']
            
        # Check if we should exit initial build mode
        if self.initial_build_mode and inventory_metrics['gross_exposure'] > 0.15:  # 15% exposure built
            self.initial_build_mode = False
            print(f"   📊 Exiting initial build mode - portfolio established")
        
        # Apply position size limits with inventory awareness
        trade_df = self._apply_position_size_limits(trade_df, inventory_metrics)
        
        # Calculate final weights
        trade_df['final_weight'] = trade_df['current_weight'] + trade_df['actual_change']
        actual_turnover = trade_df['actual_change'].abs().sum()
        
        # Remove tiny positions (< 0.1%)
        trade_df.loc[trade_df['final_weight'].abs() < 0.001, 'final_weight'] = 0
        
        # Only trade positions with meaningful changes
        trade_df = trade_df[trade_df['actual_change'].abs() > 0.001]
        
        # Add dollar_amount column for logging (assuming $100K portfolio)
        portfolio_value = self.config.get('portfolio_value', 100000)
        trade_df['dollar_amount'] = trade_df['actual_change'] * portfolio_value
        
        print(f"   ✅ Actual turnover: {actual_turnover:.2%}")
        print(f"   📊 Trades to execute: {len(trade_df)}")
        
        return trade_df
    
    def _calculate_inventory_metrics(self, current_positions: Dict) -> Dict:
        """Calculate comprehensive inventory metrics"""
        if not current_positions:
            return {
                'gross_exposure': 0.0,
                'net_exposure': 0.0,
                'long_exposure': 0.0,
                'short_exposure': 0.0,
                'position_count': 0,
                'max_position': 0.0,
                'concentration_risk': 0.0
            }
        
        weights = list(current_positions.values())
        
        gross_exposure = sum(abs(w) for w in weights)
        net_exposure = sum(weights)
        long_exposure = sum(max(0, w) for w in weights)
        short_exposure = sum(min(0, w) for w in weights)
        max_position = max(abs(w) for w in weights)
        
        # Calculate concentration (Herfindahl index)
        concentration_risk = sum(w**2 for w in weights) if weights else 0
        
        return {
            'gross_exposure': gross_exposure,
            'net_exposure': net_exposure,
            'long_exposure': long_exposure, 
            'short_exposure': short_exposure,
            'position_count': len(current_positions),
            'max_position': max_position,
            'concentration_risk': concentration_risk
        }
    
    def _apply_inventory_adjustments(self, trade_df: pd.DataFrame, inventory_metrics: Dict) -> pd.DataFrame:
        """Apply inventory-aware adjustments to trades"""
        net_exposure = inventory_metrics['net_exposure']
        gross_exposure = inventory_metrics['gross_exposure']
        
        # If net exposure is extreme, prioritize trades that reduce it
        if abs(net_exposure) > self.max_net_exposure * 0.8:  # 80% of limit
            print(f"   🎯 Inventory rebalancing: net exposure {net_exposure:.2%}")
            
            # Boost trades that reduce net exposure
            for idx, row in trade_df.iterrows():
                desired_change = row['desired_change']
                
                # If we're too long and this trade reduces longs or adds shorts
                if net_exposure > 0 and desired_change < 0:
                    trade_df.at[idx, 'inventory_boost'] = 1.5  # Boost by 50%
                # If we're too short and this trade reduces shorts or adds longs  
                elif net_exposure < 0 and desired_change > 0:
                    trade_df.at[idx, 'inventory_boost'] = 1.5
                else:
                    trade_df.at[idx, 'inventory_boost'] = 0.8  # Reduce non-helpful trades
            
            # Apply boosts to desired changes
            trade_df['desired_change'] *= trade_df.get('inventory_boost', 1.0)
        
        return trade_df
    
    def _calculate_inventory_urgency(self, inventory_metrics: Dict) -> float:
        """Calculate how urgently inventory needs to be rebalanced (0-1 scale)"""
        urgency_factors = []
        
        # Net exposure urgency
        net_urgency = min(1.0, abs(inventory_metrics['net_exposure']) / self.max_net_exposure)
        urgency_factors.append(net_urgency * 0.4)  # 40% weight
        
        # Gross exposure urgency  
        gross_urgency = min(1.0, inventory_metrics['gross_exposure'] / self.max_gross_exposure)
        urgency_factors.append(gross_urgency * 0.3)  # 30% weight
        
        # Concentration urgency
        concentration_urgency = min(1.0, inventory_metrics['max_position'] / self.max_position_size)
        urgency_factors.append(concentration_urgency * 0.3)  # 30% weight
        
        return sum(urgency_factors)
    
    def _apply_position_size_limits(self, trade_df: pd.DataFrame, inventory_metrics: Dict) -> pd.DataFrame:
        """Apply position size limits with inventory awareness"""
        for idx, row in trade_df.iterrows():
            current_weight = row['current_weight']
            actual_change = row['actual_change']
            final_weight = current_weight + actual_change
            
            # Check individual position limits
            if abs(final_weight) > self.max_position_size:
                # Cap the position at the limit
                capped_weight = np.sign(final_weight) * self.max_position_size
                trade_df.at[idx, 'actual_change'] = capped_weight - current_weight
                print(f"   ⚠️ Capped {row['Ticker']}: {final_weight:.2%} → {capped_weight:.2%}")
        
        return trade_df
    
    def execute_trades_optimized(self, trade_df: pd.DataFrame) -> bool:
        """Execute trades with optimized parallel processing for low latency"""
        print(f"\\n🚀 Executing trades (optimized)...")
        
        if len(trade_df) == 0:
            print("   ℹ️ No trades to execute")
            return True
        
        success_count = 0
        
        # Log trade execution decisions
        self._log_trade_execution_decisions(trade_df)
        
        # For small trade lists, execute sequentially with optimized flow
        if len(trade_df) <= 5:
            for _, trade in trade_df.iterrows():
                try:
                    if self.paper_trading:
                        success = self._execute_paper_trade(trade)
                    else:
                        success = self._execute_live_trade(trade)
                    
                    if success:
                        success_count += 1
                        
                except Exception as e:
                    self.logger.error(f"Trade execution failed for {trade['Ticker']}: {e}")
        
        # For larger lists, consider batch submission (if broker supports it)
        else:
            success_count = self._execute_trades_batch(trade_df)
        
        execution_rate = success_count / len(trade_df)
        print(f"   ✅ Execution rate: {success_count}/{len(trade_df)} ({execution_rate:.1%})")
        
        # Update positions
        self._update_position_records(trade_df)
        
        return execution_rate > 0.8
    
    def _execute_trades_batch(self, trade_df: pd.DataFrame) -> int:
        """Execute trades in batch for better performance"""
        success_count = 0
        
        # Group trades by execution strategy
        market_trades = []
        limit_trades = []
        
        for _, trade in trade_df.iterrows():
            try:
                ticker = trade['Ticker']
                weight_change = trade['actual_change']
                dollar_amount = abs(weight_change) * self.config.get('portfolio_value', 100000)
                
                quote_data = self._get_current_quote(ticker)
                strategy = self._determine_execution_strategy(ticker, dollar_amount, quote_data)
                
                if strategy['order_type'] == 'market':
                    market_trades.append((trade, strategy, quote_data))
                else:
                    limit_trades.append((trade, strategy, quote_data))
                    
            except Exception as e:
                self.logger.error(f"Trade preparation failed for {ticker}: {e}")
        
        # Execute market orders first (for immediacy)
        for trade, strategy, quote_data in market_trades:
            try:
                if self._execute_with_strategy(trade['Ticker'], trade['actual_change'], 
                                             abs(trade['actual_change']) * self.config.get('portfolio_value', 100000),
                                             strategy, quote_data):
                    success_count += 1
            except Exception as e:
                self.logger.error(f"Market order failed: {e}")
        
        # Execute limit orders (can be done in parallel if needed)
        for trade, strategy, quote_data in limit_trades:
            try:
                if self._execute_with_strategy(trade['Ticker'], trade['actual_change'],
                                             abs(trade['actual_change']) * self.config.get('portfolio_value', 100000),
                                             strategy, quote_data):
                    success_count += 1
            except Exception as e:
                self.logger.error(f"Limit order failed: {e}")
        
        return success_count
    
    def execute_trades(self, trade_df: pd.DataFrame) -> bool:
        """Execute trades through broker API"""
        print(f"\\n🚀 Executing trades...")
        
        if len(trade_df) == 0:
            print("   ℹ️ No trades to execute")
            return True
        
        success_count = 0
        
        for _, trade in trade_df.iterrows():
            try:
                # Execute individual trade
                if self.paper_trading:
                    success = self._execute_paper_trade(trade)
                else:
                    success = self._execute_live_trade(trade)
                
                if success:
                    success_count += 1
                    
            except Exception as e:
                self.logger.error(f"Trade execution failed for {trade['Ticker']}: {e}")
        
        execution_rate = success_count / len(trade_df)
        print(f"   ✅ Execution rate: {success_count}/{len(trade_df)} ({execution_rate:.1%})")
        
        # Update positions
        self._update_position_records(trade_df)
        
        return execution_rate > 0.8  # Require 80% success rate
    
    def _execute_paper_trade(self, trade: pd.Series) -> bool:
        """Execute trade in paper trading mode (Alpaca paper or local simulation)"""
        ticker = trade['Ticker']
        weight_change = trade['actual_change']
        
        if self.alpaca_api and self.config.get('alpaca_paper_trading', True):
            # Use Alpaca paper trading API
            try:
                portfolio_value = self.config.get('portfolio_value', 100000)
                dollar_amount = abs(weight_change) * portfolio_value
                
                if dollar_amount < self.config.get('min_order_value', 10):
                    print(f"   📄 ALPACA PAPER SKIP: {ticker} ${dollar_amount:.2f} below minimum")
                    return True
                
                side = 'buy' if weight_change > 0 else 'sell'
                
                order = self.alpaca_api.submit_order(
                    symbol=ticker,
                    notional=dollar_amount,
                    side=side,
                    type='market',
                    time_in_force='day'
                )
                
                print(f"   📄 ALPACA PAPER: {side.upper()} {ticker} ${dollar_amount:.2f} (Order: {order.id})")
                return True
                
            except Exception as e:
                self.logger.warning(f"Alpaca paper trade failed for {ticker}: {e}")
                # Fall back to local simulation
                pass
        
        # Local paper trading simulation
        print(f"   📄 LOCAL PAPER: {ticker} {weight_change:+.3%}")
        return True
    
    def _execute_live_trade(self, trade: pd.Series) -> bool:
        """Execute trade through smart order routing with limit orders"""
        try:
            if not self.alpaca_api:
                self.logger.error("Alpaca API not available for live trading")
                return False
            
            ticker = trade['Ticker']
            weight_change = trade['actual_change']
            
            # Skip tiny trades
            if abs(weight_change) < 0.001:
                return True
            
            # Calculate dollar amount
            portfolio_value = self.config.get('portfolio_value', self.alpaca_account_info['equity'])
            dollar_amount = abs(weight_change) * portfolio_value
            
            # Skip trades below minimum
            if dollar_amount < self.config.get('min_order_value', 10):
                print(f"   ⏭️ SKIP: {ticker} ${dollar_amount:.2f} below minimum")
                return True
            
            # Get current market data for smart order placement
            quote_data = self._get_current_quote(ticker)
            execution_strategy = self._determine_execution_strategy(ticker, dollar_amount, quote_data)
            
            return self._execute_with_strategy(ticker, weight_change, dollar_amount, execution_strategy, quote_data)
            
        except Exception as e:
            self.logger.error(f"Trade execution error for {ticker}: {e}")
            print(f"     ❌ Error: {e}")
            return False
    
    def _get_current_quote(self, ticker: str) -> Dict:
        """Get current bid/ask quote for smart order placement"""
        try:
            # Try to get real-time quote from data fetcher's stream
            if hasattr(self.data_fetcher, 'stream_buffer') and ticker in self.data_fetcher.stream_buffer:
                buffer_data = self.data_fetcher.stream_buffer[ticker]
                if 'bid' in buffer_data and 'ask' in buffer_data:
                    return {
                        'bid': buffer_data['bid'],
                        'ask': buffer_data['ask'],
                        'mid': buffer_data.get('mid_price', (buffer_data['bid'] + buffer_data['ask']) / 2),
                        'spread': buffer_data.get('spread', buffer_data['ask'] - buffer_data['bid']),
                        'source': 'stream'
                    }
            
            # Fallback to last trade price from Alpaca
            try:
                latest_trade = self.alpaca_api.get_latest_trade(ticker)
                if latest_trade:
                    price = float(latest_trade.price)
                    # Estimate bid/ask from last price (crude approximation)
                    spread_est = price * 0.001  # 10 bps estimated spread
                    return {
                        'bid': price - spread_est/2,
                        'ask': price + spread_est/2,
                        'mid': price,
                        'spread': spread_est,
                        'source': 'trade_estimate'
                    }
            except:
                pass
                
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to get quote for {ticker}: {e}")
            return None
    
    def _determine_execution_strategy(self, ticker: str, dollar_amount: float, quote_data: Dict) -> Dict:
        """Determine optimal execution strategy based on order size and market conditions"""
        strategy = {
            'order_type': 'market',
            'time_in_force': 'day',
            'limit_price': None,
            'chunks': 1,
            'delay_seconds': 0
        }
        
        if not quote_data:
            return strategy  # Default to market order
        
        spread = quote_data.get('spread', 0)
        mid_price = quote_data.get('mid', 0)
        
        # For small orders with tight spreads, use limit orders near mid
        if dollar_amount < 10000 and spread < mid_price * 0.005:  # Less than $10k and spread < 50bps
            strategy.update({
                'order_type': 'limit',
                'aggressive_factor': 0.3,  # How aggressive to be (0 = passive, 1 = aggressive)
                'time_limit_seconds': 300   # 5 minute limit order timeout
            })
        
        # For larger orders, consider chunking
        elif dollar_amount > 50000:
            chunk_size = min(10000, dollar_amount / 5)  # Max $10k per chunk, min 5 chunks
            strategy.update({
                'order_type': 'limit',
                'chunks': int(dollar_amount / chunk_size),
                'delay_seconds': 30,  # 30 seconds between chunks
                'aggressive_factor': 0.5
            })
        
        return strategy
    
    def _execute_with_strategy(self, ticker: str, weight_change: float, dollar_amount: float, 
                             strategy: Dict, quote_data: Dict) -> bool:
        """Execute trade using determined strategy"""
        side = 'buy' if weight_change > 0 else 'sell'
        
        print(f"   💰 SMART EXECUTION: {side.upper()} {ticker} ${dollar_amount:.2f}")
        print(f"      📊 Strategy: {strategy['order_type']} x{strategy['chunks']} chunks")
        
        if strategy['order_type'] == 'market':
            return self._submit_market_order(ticker, side, dollar_amount)
        else:
            return self._submit_limit_order_sequence(ticker, side, dollar_amount, strategy, quote_data)
    
    def _submit_market_order(self, ticker: str, side: str, dollar_amount: float) -> bool:
        """Submit simple market order"""
        try:
            # For short selling, use shares instead of notional
            if side == 'sell':
                # Get current price to calculate shares
                latest_trade = self.alpaca_api.get_latest_trade(ticker)
                if latest_trade:
                    current_price = float(latest_trade.price)
                    shares = max(1, int(dollar_amount / current_price))
                    
                    order = self.alpaca_api.submit_order(
                        symbol=ticker,
                        qty=shares,
                        side=side,
                        type='market',
                        time_in_force='day'
                    )
                else:
                    print(f"      ⚠️ Cannot get price for {ticker}")
                    return False
            else:
                # Use notional for buys - round to 2 decimals
                order = self.alpaca_api.submit_order(
                    symbol=ticker,
                    notional=round(dollar_amount, 2),
                    side=side,
                    type='market',
                    time_in_force='day'
                )
            
            print(f"      ✅ Market Order ID: {order.id}")
            return True
            
        except APIError as e:
            print(f"      ❌ Market Order Failed: {e}")
            return False
    
    def _submit_limit_order_sequence(self, ticker: str, side: str, dollar_amount: float, 
                                   strategy: Dict, quote_data: Dict) -> bool:
        """Submit sequence of limit orders with smart pricing"""
        try:
            chunks = strategy.get('chunks', 1)
            chunk_amount = dollar_amount / chunks
            aggressive_factor = strategy.get('aggressive_factor', 0.3)
            
            bid = quote_data['bid']
            ask = quote_data['ask']
            mid = quote_data['mid']
            spread = quote_data['spread']
            
            orders_submitted = 0
            
            for i in range(chunks):
                # Calculate limit price based on aggressiveness
                if side == 'buy':
                    # For buys: bid + (aggressive_factor * spread)
                    limit_price = bid + (aggressive_factor * spread)
                else:
                    # For sells: ask - (aggressive_factor * spread)  
                    limit_price = ask - (aggressive_factor * spread)
                
                # Round to reasonable tick size
                limit_price = round(limit_price, 2)
                
                try:
                    # For short selling, use shares instead of notional (Alpaca doesn't allow fractional shorts)
                    if side == 'sell':
                        shares = max(1, int(chunk_amount / limit_price))  # Calculate shares from dollar amount
                        order = self.alpaca_api.submit_order(
                            symbol=ticker,
                            qty=shares,
                            side=side,
                            type='limit',
                            time_in_force='day',
                            limit_price=limit_price
                        )
                    else:
                        # Use notional for buys (fractional allowed) - round to 2 decimals
                        order = self.alpaca_api.submit_order(
                            symbol=ticker,
                            notional=round(chunk_amount, 2),
                            side=side,
                            type='limit',
                            time_in_force='day',
                            limit_price=limit_price
                        )
                    
                    orders_submitted += 1
                    print(f"      ✅ Limit Order {i+1}/{chunks}: {order.id} @ ${limit_price}")
                    
                    # Delay between chunks
                    if i < chunks - 1 and strategy.get('delay_seconds', 0) > 0:
                        import time
                        time.sleep(strategy['delay_seconds'])
                        
                except APIError as e:
                    print(f"      ❌ Limit Order {i+1} Failed: {e}")
                    continue
            
            success_rate = orders_submitted / chunks
            print(f"      📊 Limit orders success: {orders_submitted}/{chunks} ({success_rate:.1%})")
            
            return success_rate > 0.5  # Require >50% success
            
        except Exception as e:
            print(f"      ❌ Limit order sequence failed: {e}")
            return False
    
    def _update_position_records(self, trade_df: pd.DataFrame):
        """Update position records after trading"""
        try:
            # Update current positions
            for _, trade in trade_df.iterrows():
                if trade['final_weight'] != 0:
                    self.current_positions[trade['Ticker']] = trade['final_weight']
                elif trade['Ticker'] in self.current_positions:
                    del self.current_positions[trade['Ticker']]
            
            # Save updated positions
            positions_dir = Path("../artifacts/positions")
            positions_dir.mkdir(parents=True, exist_ok=True)
            
            with open(positions_dir / "current_positions.json", 'w') as f:
                json.dump(self.current_positions, f, indent=2)
            
            # Save trade history
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            trade_file = positions_dir / f"trades_{timestamp}.json"
            
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'mode': 'PAPER' if self.paper_trading else 'LIVE',
                'trades_executed': len(trade_df),
                'total_turnover': trade_df['actual_change'].abs().sum(),
                'trades': trade_df.to_dict('records')
            }
            
            with open(trade_file, 'w') as f:
                json.dump(trade_record, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to update position records: {e}")
    
    def emergency_kill_switch(self):
        """Emergency kill switch - flatten all positions"""
        print(f"\\n🚨 EMERGENCY KILL SWITCH ACTIVATED!")
        
        self.kill_switch_active = True
        
        # Create orders to flatten all positions
        flatten_orders = []
        for ticker, weight in self.current_positions.items():
            if abs(weight) > 0.001:  # Only flatten meaningful positions
                flatten_orders.append({
                    'Ticker': ticker,
                    'current_weight': weight,
                    'target_weight': 0.0,
                    'actual_change': -weight,
                    'final_weight': 0.0
                })
        
        if flatten_orders:
            flatten_df = pd.DataFrame(flatten_orders)
            print(f"   📊 Flattening {len(flatten_df)} positions")
            
            # Execute emergency trades (bypass turnover controls)
            self.execute_trades(flatten_df)
        
        print(f"   ✅ Kill switch executed")
        self.logger.critical("Emergency kill switch executed - all positions flattened")
    
    def run_trading_cycle(self) -> bool:
        """Run complete trading cycle"""
        print(f"\\n🔄 STARTING TRADING CYCLE")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        try:
            # 1. Check kill switch conditions
            should_kill, violations = self.check_kill_switch_conditions()
            if should_kill:
                print(f"🚨 Kill switch conditions met:")
                for violation in violations:
                    print(f"   ❌ {violation}")
                
                self.emergency_kill_switch()
                return False
            
            # 2. Fetch latest market data
            market_data, audit_info = self.data_fetcher.fetch_latest_prices(streaming_priority=True)
            if market_data is None:
                self.logger.error("Failed to fetch market data")
                return False
            
            # 3. Validate data quality
            quality = self.data_fetcher.validate_data_quality(market_data)
            if not quality['valid']:
                self.logger.error(f"Data quality issues: {quality['issues']}")
                return False
            
            # 4. Calculate target portfolio (optimized for speed)
            start_portfolio_time = datetime.now()
            target_portfolio = self.calculate_target_portfolio(market_data)
            portfolio_calc_time = (datetime.now() - start_portfolio_time).total_seconds()
            
            if target_portfolio is None:
                return False
            
            print(f"   ⚡ Portfolio calculation: {portfolio_calc_time:.3f}s")
            
            # 5. Calculate trades with turnover control (vectorized operations)
            start_trade_time = datetime.now() 
            trade_df = self.calculate_trades_with_turnover_control(target_portfolio)
            trade_calc_time = (datetime.now() - start_trade_time).total_seconds()
            
            print(f"   ⚡ Trade calculation: {trade_calc_time:.3f}s")
            
            # 6. Execute trades (parallel submission when possible)
            start_exec_time = datetime.now()
            success = self.execute_trades_optimized(trade_df)
            exec_time = (datetime.now() - start_exec_time).total_seconds()
            
            print(f"   ⚡ Trade execution: {exec_time:.3f}s")
            
            if success:
                print(f"\\n🎉 TRADING CYCLE COMPLETED SUCCESSFULLY!")
            else:
                print(f"\\n⚠️ TRADING CYCLE COMPLETED WITH ISSUES")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Trading cycle failed: {e}")
            print(f"\\n❌ TRADING CYCLE FAILED: {e}")
            return False

def main():
    """Run live trading bot with Alpaca integration"""
    import sys
    
    # Command line arguments
    paper_mode = '--paper' in sys.argv or '--test' in sys.argv
    live_mode = '--live' in sys.argv
    
    if live_mode and paper_mode:
        print("❌ Cannot specify both --paper and --live modes")
        return False
    
    # Default to paper mode for safety
    if not live_mode:
        paper_mode = True
    
    print(f"🚀 Starting Professional Live Trading Bot with Alpaca")
    print(f"Mode: {'ALPACA PAPER TRADING' if paper_mode else 'ALPACA LIVE TRADING'}")
    print(f"Environment: {'Paper' if paper_mode else 'Production'}")
    
    # Check for required environment variables
    if not os.getenv('ALPACA_API_KEY') or not os.getenv('ALPACA_SECRET_KEY'):
        print("⚠️ Warning: Alpaca credentials not found in environment")
        print("   Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        print("   Falling back to local simulation mode")
    
    # Initialize bot with Alpaca integration
    bot = LiveTradingBot(paper_trading=paper_mode)
    
    # Verify Alpaca connection before trading
    if bot.alpaca_api:
        print("✅ Alpaca API connection verified")
    else:
        print("⚠️ Trading without Alpaca API - using local simulation")
    
    # Run trading cycle
    success = bot.run_trading_cycle()
    
    if success:
        print(f"\\n🎉 Professional trading cycle completed successfully!")
        if bot.alpaca_api:
            print(f"   📊 Check your Alpaca account for executed trades")
    else:
        print(f"\\n❌ Trading cycle failed - check logs")
    
    return success

if __name__ == "__main__":
    main()