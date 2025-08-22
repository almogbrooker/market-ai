from importlib import reload
import types
import sys


def test_validate_config(monkeypatch):
    dummy = types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None)
    sys.modules["dotenv"] = dummy
    monkeypatch.setenv("ALPACA_API_KEY", "dummy")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "dummy")
    monkeypatch.setenv("ALPACA_ENV", "paper")
    from src.utils import config as config_module

    reload(config_module)
    assert config_module.TradingConfig.validate_config() is True
