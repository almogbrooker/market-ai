#!/usr/bin/env python3
"""
Configuration management for AI Trading Bot
"""

import os
from typing import Optional
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Legacy config for backward compatibility
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
TICKERS = ["AAPL","MSFT","GOOG","AMZN","NVDA","TSLA","META","AMD","INTC","QCOM"]
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "YOUR_NEWSAPI_KEY_HERE")

# Universe & windows
TOP_N = 30
TIME_WINDOW = 30
SEQ_LEN = 30

# Training / runtime
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
BATCH_SIZE = 8
EPOCHS = 60
LR = 3e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
PATIENCE = 8
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = f"{CHECKPOINT_DIR}/best_model.pth"
DATA_DIR = "data"
LOG_DIR = "runs"

# Model hyperparams
D_MODEL = 64
NHEAD = 4
TRANSFORMER_LAYERS = 6
GAT_HIDDEN = 64
GAT_OUT = 32
DROPOUT = 0.3

# ensure dirs
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

class TradingConfig:
    """Centralized configuration management"""
    
    # Alpaca API
    ALPACA_API_KEY: str = os.getenv('ALPACA_API_KEY', '')
    ALPACA_SECRET_KEY: str = os.getenv('ALPACA_SECRET_KEY', '')
    ALPACA_ENV: str = os.getenv('ALPACA_ENV', 'paper')  # paper or live
    
    # Risk Management
    MAX_GROSS: float = float(os.getenv('MAX_GROSS', '0.95'))
    DAILY_DD_KILLSWITCH: float = float(os.getenv('DAILY_DD_KILLSWITCH', '-0.03'))
    MAX_SINGLE_POSITION: float = float(os.getenv('MAX_SINGLE_POSITION', '0.20'))
    
    # Trading Parameters
    MIN_CONFIDENCE: float = float(os.getenv('MIN_CONFIDENCE', '0.95'))
    MIN_TRADE_VALUE: float = float(os.getenv('MIN_TRADE_VALUE', '100.0'))
    MAX_OPEN_ORDERS: int = int(os.getenv('MAX_OPEN_ORDERS', '20'))
    
    # External APIs (Optional)
    FRED_API_KEY: str = os.getenv('FRED_API_KEY', '')
    STOCKTWITS_TOKEN: str = os.getenv('STOCKTWITS_TOKEN', '')
    
    # Logging
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE: str = os.getenv('LOG_FILE', 'logs/trading_system.log')
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate critical configuration"""
        errors = []
        
        if not cls.ALPACA_API_KEY:
            errors.append("ALPACA_API_KEY is required")
        
        if not cls.ALPACA_SECRET_KEY:
            errors.append("ALPACA_SECRET_KEY is required")
        
        if cls.ALPACA_ENV not in ['paper', 'live']:
            errors.append("ALPACA_ENV must be 'paper' or 'live'")
        
        if not 0 < cls.MAX_GROSS <= 1.0:
            errors.append("MAX_GROSS must be between 0 and 1")
        
        if not -1.0 < cls.DAILY_DD_KILLSWITCH < 0:
            errors.append("DAILY_DD_KILLSWITCH must be negative (e.g., -0.03)")
        
        if errors:
            for error in errors:
                logging.error(f"❌ Config error: {error}")
            return False
        
        logging.info("✅ Configuration validated successfully")
        return True
    
    @classmethod
    def get_base_url(cls) -> str:
        """Get Alpaca API base URL"""
        return 'https://paper-api.alpaca.markets' if cls.ALPACA_ENV == 'paper' else 'https://api.alpaca.markets'
    
    @classmethod
    def is_paper_trading(cls) -> bool:
        """Check if using paper trading"""
        return cls.ALPACA_ENV == 'paper'
    
    @classmethod
    def setup_logging(cls):
        """Setup logging configuration"""
        log_dir = os.path.dirname(cls.LOG_FILE)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(cls.LOG_FILE),
                logging.StreamHandler()
            ]
        )
    
    @classmethod
    def print_config(cls):
        """Print current configuration (hiding secrets)"""
        print("🔧 AI Trading Bot Configuration")
        print("=" * 40)
        print(f"Environment: {cls.ALPACA_ENV.upper()}")
        print(f"API Key: {cls.ALPACA_API_KEY[:8]}...")
        print(f"Max Gross Exposure: {cls.MAX_GROSS:.1%}")
        print(f"Daily Kill Switch: {cls.DAILY_DD_KILLSWITCH:.1%}")
        print(f"Max Single Position: {cls.MAX_SINGLE_POSITION:.1%}")
        print(f"Min Confidence: {cls.MIN_CONFIDENCE:.1%}")
        print(f"Min Trade Value: ${cls.MIN_TRADE_VALUE}")
        print(f"Max Open Orders: {cls.MAX_OPEN_ORDERS}")
        print(f"Log Level: {cls.LOG_LEVEL}")
        print("=" * 40)

# Global config instance
config = TradingConfig()

# Validation on import
if __name__ != "__main__":
    if config.ALPACA_API_KEY:  # Only validate if keys are present
        config.validate_config()
        config.setup_logging()

if __name__ == "__main__":
    # Demo configuration
    config.print_config()
    is_valid = config.validate_config()
    print(f"\n✅ Configuration valid: {is_valid}")
