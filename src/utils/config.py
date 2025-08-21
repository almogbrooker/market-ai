#!/usr/bin/env python3
"""
Configuration management for AI Trading Bot
"""

import os
import json
import logging
from dataclasses import dataclass, asdict, replace
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class TradingSettings:
    """Model and training settings with reasonable defaults."""

    # Legacy config for backward compatibility
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    tickers: List[str] = ("AAPL,MSFT,GOOG,AMZN,NVDA,TSLA,META,AMD,INTC,QCOM".split(","))
    news_api_key: str = os.environ.get("NEWS_API_KEY", "YOUR_NEWSAPI_KEY_HERE")

    # Universe & windows
    top_n: int = 30
    time_window: int = 30
    seq_len: int = 30

    # Training / runtime
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    batch_size: int = 8
    epochs: int = 60
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    patience: int = 8
    checkpoint_dir: str = "checkpoints"
    best_model_path: str = "checkpoints/best_model.pth"
    data_dir: str = "data"
    log_dir: str = "runs"

    # Model hyperparams
    d_model: int = 64
    nhead: int = 4
    transformer_layers: int = 6
    gat_hidden: int = 64
    gat_out: int = 32
    dropout: float = 0.3

    def to_dict(self) -> dict:
        """Serialize settings to a dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize settings to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_env(cls) -> "TradingSettings":
        """Load settings with optional environment overrides."""
        base = cls()
        tickers = os.getenv("TICKERS", ",".join(base.tickers)).split(",")
        return replace(
            base,
            start_date=os.getenv("START_DATE", base.start_date),
            end_date=os.getenv("END_DATE", base.end_date),
            tickers=[t.strip() for t in tickers if t.strip()],
            news_api_key=os.getenv("NEWS_API_KEY", base.news_api_key),
            top_n=int(os.getenv("TOP_N", base.top_n)),
            time_window=int(os.getenv("TIME_WINDOW", base.time_window)),
            seq_len=int(os.getenv("SEQ_LEN", base.seq_len)),
            device=os.getenv("DEVICE", base.device),
            batch_size=int(os.getenv("BATCH_SIZE", base.batch_size)),
            epochs=int(os.getenv("EPOCHS", base.epochs)),
            lr=float(os.getenv("LR", base.lr)),
            weight_decay=float(os.getenv("WEIGHT_DECAY", base.weight_decay)),
            grad_clip=float(os.getenv("GRAD_CLIP", base.grad_clip)),
            patience=int(os.getenv("PATIENCE", base.patience)),
            checkpoint_dir=os.getenv("CHECKPOINT_DIR", base.checkpoint_dir),
            best_model_path=os.getenv("BEST_MODEL_PATH", base.best_model_path),
            data_dir=os.getenv("DATA_DIR", base.data_dir),
            log_dir=os.getenv("LOG_DIR", base.log_dir),
            d_model=int(os.getenv("D_MODEL", base.d_model)),
            nhead=int(os.getenv("NHEAD", base.nhead)),
            transformer_layers=int(os.getenv("TRANSFORMER_LAYERS", base.transformer_layers)),
            gat_hidden=int(os.getenv("GAT_HIDDEN", base.gat_hidden)),
            gat_out=int(os.getenv("GAT_OUT", base.gat_out)),
            dropout=float(os.getenv("DROPOUT", base.dropout)),
        )


# Instantiate settings and ensure directories
trading_settings = TradingSettings.from_env()
os.makedirs(trading_settings.checkpoint_dir, exist_ok=True)
os.makedirs(trading_settings.data_dir, exist_ok=True)
os.makedirs(trading_settings.log_dir, exist_ok=True)
# Backward compatible alias
settings = trading_settings

class TradingConfig:
    """Centralized configuration management"""

    # Dataclass-based settings
    settings: TradingSettings = trading_settings

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
    def serialize_settings(cls) -> str:
        """Return current dataclass settings as JSON."""
        return cls.settings.to_json()
    
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
