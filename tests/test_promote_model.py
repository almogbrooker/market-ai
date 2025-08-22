import json
from pathlib import Path

import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.training.promote_model import promote_model, compute_ic_from_predictions


def test_compute_ic_from_predictions(tmp_path):
    csv_path = tmp_path / "predictions.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prediction", "actual"])
        writer.writerows([[0.1, 1], [0.2, 2], [0.3, 3]])
    ic = compute_ic_from_predictions(csv_path)
    assert ic == 1.0


def test_promote_model(tmp_path):
    candidate = tmp_path / "staging_model"
    candidate.mkdir()
    metrics = {"validation_ic": 0.05, "dataset_version": "test-v1"}
    (candidate / "metrics.json").write_text(json.dumps(metrics))
    with (candidate / "predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prediction", "actual"])
        writer.writerows([[0.1, 1], [0.2, 2], [0.3, 3]])

    production = tmp_path / "prod" / "model"
    log_path = tmp_path / "promotion_log.jsonl"

    event = promote_model(candidate, production, ic_threshold=0.02, log_path=log_path)

    assert production.exists()
    assert json.loads(log_path.read_text().strip()) == event
    assert event["validation_ic"] == metrics["validation_ic"]
    assert event["shadow_ic"] == 1.0
