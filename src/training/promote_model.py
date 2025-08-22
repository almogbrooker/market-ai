#!/usr/bin/env python3
"""Model promotion utility

This script promotes a trained model from staging to production. It performs
several safety checks before promotion:

1. Reads validation IC from a tracking log.
2. Ensures the validation IC meets a configurable threshold.
3. Runs a light shadow test using prediction logs to compute an out-of-sample
   IC on historical data.
4. On success, moves the model directory to the production models folder.
5. Records promotion details in a persistent log for auditability.

The script is intentionally lightweight. It assumes the candidate model
directory contains:

- ``metrics.json`` – Tracking metrics with at least ``validation_ic`` and
  optionally ``dataset_version``.
- ``predictions.csv`` – Two columns: ``prediction`` and ``actual`` used for a
  quick shadow IC calculation.

Example
-------
```bash
python src/training/promote_model.py \
    --candidate artifacts/candidate_model \
    --production PRODUCTION/models/best_candidate
```
"""
from __future__ import annotations

import argparse
import getpass
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

import csv
import math

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = REPO_ROOT / "PRODUCTION" / "promotion_log.jsonl"


def read_validation_ic(metrics_path: Path) -> Dict[str, Optional[float]]:
    """Read validation IC and dataset version from a metrics JSON file."""
    with metrics_path.open() as f:
        metrics = json.load(f)
    return {
        "validation_ic": metrics.get("validation_ic"),
        "dataset_version": metrics.get("dataset_version"),
    }


def _rank(values: List[float]) -> List[int]:
    """Return ranks starting at 1 for a sequence of values."""
    sorted_idx = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0] * len(values)
    for rank, idx in enumerate(sorted_idx, start=1):
        ranks[idx] = rank
    return ranks


def compute_ic_from_predictions(pred_path: Path) -> float:
    """Compute Spearman IC from a predictions CSV file.

    The CSV must contain columns ``prediction`` and ``actual``.
    """
    predictions: List[float] = []
    actuals: List[float] = []
    with pred_path.open() as f:
        reader = csv.DictReader(f)
        if not {"prediction", "actual"} <= set(reader.fieldnames or []):
            raise ValueError("predictions.csv must contain 'prediction' and 'actual' columns")
        for row in reader:
            predictions.append(float(row["prediction"]))
            actuals.append(float(row["actual"]))

    pred_rank = _rank(predictions)
    act_rank = _rank(actuals)

    mean_pred = sum(pred_rank) / len(pred_rank)
    mean_act = sum(act_rank) / len(act_rank)
    numerator = sum((p - mean_pred) * (a - mean_act) for p, a in zip(pred_rank, act_rank))
    denom = math.sqrt(
        sum((p - mean_pred) ** 2 for p in pred_rank)
            * sum((a - mean_act) ** 2 for a in act_rank)
    )
    return numerator / denom if denom else 0.0


def run_shadow_tests(candidate_dir: Path) -> float:
    """Run a light shadow test using stored predictions.

    Returns the computed IC.
    """
    pred_file = candidate_dir / "predictions.csv"
    if not pred_file.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_file}")
    return compute_ic_from_predictions(pred_file)


def get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def log_promotion(event: Dict, log_path: Path = DEFAULT_LOG) -> None:
    """Append a promotion event to the log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(event) + "\n")


def promote_model(
    candidate_dir: Path,
    production_dir: Path,
    ic_threshold: float = 0.02,
    log_path: Path = DEFAULT_LOG,
) -> Dict:
    """Promote a model from staging to production.

    Parameters
    ----------
    candidate_dir: Path
        Directory containing the candidate model artifacts.
    production_dir: Path
        Destination directory for production models.
    ic_threshold: float
        Minimum required IC for promotion.
    log_path: Path
        Where to record promotion events.
    """

    if not candidate_dir.exists():
        raise FileNotFoundError(f"Candidate directory not found: {candidate_dir}")

    metrics = read_validation_ic(candidate_dir / "metrics.json")
    validation_ic = metrics["validation_ic"]
    if validation_ic is None:
        raise ValueError("Validation IC not found in metrics.json")
    if validation_ic < ic_threshold:
        raise ValueError(
            f"Validation IC {validation_ic:.4f} below threshold {ic_threshold:.4f}"
        )

    shadow_ic = run_shadow_tests(candidate_dir)
    if shadow_ic < ic_threshold:
        raise ValueError(
            f"Shadow test IC {shadow_ic:.4f} below threshold {ic_threshold:.4f}"
        )

    production_dir = production_dir.resolve()
    production_dir.parent.mkdir(parents=True, exist_ok=True)

    if production_dir.exists():
        shutil.rmtree(production_dir)
    shutil.copytree(candidate_dir, production_dir)

    event = {
        "model_name": production_dir.name,
        "timestamp": datetime.utcnow().isoformat(),
        "user": getpass.getuser(),
        "commit_hash": get_git_commit(),
        "dataset_version": metrics.get("dataset_version"),
        "validation_ic": validation_ic,
        "shadow_ic": shadow_ic,
        "source": str(candidate_dir),
        "destination": str(production_dir),
    }
    log_promotion(event, log_path)
    return event


def main():
    parser = argparse.ArgumentParser(description="Promote model to production")
    parser.add_argument("--candidate", type=Path, required=True, help="Staging model directory")
    parser.add_argument(
        "--production", type=Path, required=True, help="Destination production directory"
    )
    parser.add_argument(
        "--ic-threshold",
        type=float,
        default=0.02,
        help="Minimum IC required for promotion",
    )
    parser.add_argument(
        "--log", type=Path, default=DEFAULT_LOG, help="Promotion log file"
    )
    args = parser.parse_args()

    event = promote_model(
        candidate_dir=args.candidate,
        production_dir=args.production,
        ic_threshold=args.ic_threshold,
        log_path=args.log,
    )
    print(json.dumps(event, indent=2))


if __name__ == "__main__":
    main()
