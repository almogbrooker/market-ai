import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from utils.intent_hash import compute_intent_hash


def test_hash_identical_rounding():
    payload1 = {"symbol": "AAPL", "prediction": round(0.123456, 4)}
    payload2 = {"symbol": "AAPL", "prediction": round(0.123459, 4)}
    assert compute_intent_hash(payload1) == compute_intent_hash(payload2)


def test_hash_differs_when_rounded_values_differ():
    payload1 = {"symbol": "AAPL", "prediction": round(0.123456, 4)}
    payload2 = {"symbol": "AAPL", "prediction": round(0.123556, 4)}
    assert compute_intent_hash(payload1) != compute_intent_hash(payload2)
