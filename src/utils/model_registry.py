import json
from pathlib import Path

REGISTRY_FILE = Path('model_registry.json')

def evaluate_shadow_tests() -> bool:
    """Placeholder for shadow testing logic."""
    return True

def register_promotion(run_id: str, ic: float, threshold: float = 0.02) -> str:
    """Register model stage based on IC threshold and shadow tests."""
    shadow_pass = evaluate_shadow_tests()
    stage = 'production' if ic >= threshold and shadow_pass else 'staging'

    registry = {}
    if REGISTRY_FILE.exists():
        registry = json.loads(REGISTRY_FILE.read_text())
    registry[run_id] = {'ic': ic, 'shadow_pass': shadow_pass, 'stage': stage}
    REGISTRY_FILE.write_text(json.dumps(registry, indent=2))
    return stage
