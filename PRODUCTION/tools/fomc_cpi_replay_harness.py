#!/usr/bin/env python3
"""Replay harness for FOMC and CPI event days.

Simulates historical market data to stress the execution stack, measuring
latency and fill quality under load while logging failure points. Results are
used to tune timeout and retry logic.
"""

import argparse
import csv
import json
import logging
import random
import statistics
import time
from pathlib import Path
from typing import Dict, List

LOG_FILE = "replay_failures.log"


class ReplayHarness:
    """Replay historical event data and analyse execution performance."""

    def __init__(self, data_dir: Path, timeout: float, retries: int) -> None:
        self.data_dir = Path(data_dir)
        self.timeout = timeout
        self.retries = retries
        self.latencies: List[float] = []
        self.fills: List[float] = []
        self.failures: List[Dict] = []

        logging.basicConfig(
            filename=LOG_FILE,
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
        )

    def _load_event_files(self) -> List[Path]:
        """Return CSV files with replay data."""
        files = list(self.data_dir.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV event files found in {self.data_dir}")
        return files

    def _simulate_order(self, expected_price: float) -> bool:
        """Simulate a single order fill and record metrics."""
        start = time.perf_counter()
        time.sleep(random.uniform(0, self.timeout * 1.5))
        latency = time.perf_counter() - start

        executed_price = expected_price * (1 + random.gauss(0, 0.001))
        fill_quality = abs(executed_price - expected_price)

        self.latencies.append(latency)
        self.fills.append(fill_quality)

        if latency > self.timeout or fill_quality > 0.005:
            failure = {
                "expected": expected_price,
                "executed": executed_price,
                "latency": latency,
                "fill_quality": fill_quality,
            }
            self.failures.append(failure)
            logging.warning("execution failure %s", json.dumps(failure))
            return False
        return True

    def run(self) -> Dict[str, float]:
        """Replay all events and return tuning results."""
        files = self._load_event_files()
        for file in files:
            with file.open() as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    price = float(row.get("price", 0.0))
                    attempt = 0
                    success = False
                    while attempt <= self.retries and not success:
                        attempt += 1
                        success = self._simulate_order(price)
                        if not success and attempt <= self.retries:
                            time.sleep(self.timeout / 2)
        return self._tune()

    def _tune(self) -> Dict[str, float]:
        """Tune timeout and retry logic based on metrics."""
        if not self.latencies:
            raise RuntimeError("No executions were simulated.")
        p95 = statistics.quantiles(self.latencies, n=100)[94]
        failure_rate = len(self.failures) / len(self.latencies)
        if p95 > self.timeout:
            self.timeout = p95 * 1.1
        if failure_rate > 0.02:
            self.retries += 1
        results = {
            "latency_p95": p95,
            "failure_rate": failure_rate,
            "timeout": self.timeout,
            "retries": self.retries,
        }
        Path("replay_results.json").write_text(json.dumps(results, indent=2))
        logging.info("tuning results %s", json.dumps(results))
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replay FOMC/CPI day market data and tune execution logic"
    )
    parser.add_argument(
        "--data-dir",
        default="data/event_replays",
        help="Directory with historical FOMC/CPI CSV files",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.5,
        help="Initial timeout threshold in seconds",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Initial number of retries for failed executions",
    )
    args = parser.parse_args()
    harness = ReplayHarness(Path(args.data_dir), args.timeout, args.retries)
    summary = harness.run()
    print(json.dumps(summary, indent=2))
