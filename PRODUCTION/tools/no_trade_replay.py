#!/usr/bin/env python3
"""Lightweight no-trade replay that verifies model artifacts."""

from pathlib import Path
import json
import sys


MODEL_DIR = Path("PRODUCTION/models/best_institutional_model")
REQUIRED_FILES = ["config.json", "features.json", "gate.json", "model.pt"]


def main() -> None:
    """Validate that production artifacts can be read without error."""
    try:
        # Ensure all required files exist
        missing = [name for name in REQUIRED_FILES if not (MODEL_DIR / name).exists()]
        if missing:
            print(f"Missing artifacts: {', '.join(missing)}")
            sys.exit(1)

        # Validate JSON files can be parsed
        for name in ["config.json", "features.json", "gate.json"]:
            with open(MODEL_DIR / name, "r", encoding="utf-8") as handle:
                json.load(handle)

        print("Artifacts validated; no trades executed.")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Exception during no-trade replay: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
