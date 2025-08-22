import subprocess
from pathlib import Path
from typing import Dict

try:
    import mlflow  # type: ignore
except Exception:  # pragma: no cover - mlflow optional
    mlflow = None  # fallback when mlflow isn't installed


def _git_commit() -> str:
    """Return current git commit hash."""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    except Exception:
        return 'unknown'


def _data_version(dataset_path: str) -> str:
    """Infer dataset version from a sibling .dvc file or git tree hash."""
    try:
        root = Path(dataset_path).parts[0]
        dvc_file = Path(f"{root}.dvc")
        if dvc_file.exists():
            import yaml
            with open(dvc_file) as f:  # type: ignore
                info = yaml.safe_load(f)
            return info.get('outs', [{}])[0].get('md5', 'unknown')
        return subprocess.check_output(['git', 'rev-parse', f'HEAD:{root}']).decode().strip()
    except Exception:
        return 'unknown'


class MLflowTracker:
    """Minimal MLflow tracker that logs commit hash and dataset version."""

    def __init__(self, dataset_path: str, experiment: str = 'training') -> None:
        self.dataset_path = dataset_path
        self.experiment = experiment
        self.run_id = None

    def __enter__(self):  # -> "MLflowTracker":
        if mlflow is None:
            print('MLflow not installed; tracking disabled.')
            return self
        mlflow.set_experiment(self.experiment)
        mlflow.start_run()
        mlflow.log_param('git_commit', _git_commit())
        mlflow.log_param('data_version', _data_version(self.dataset_path))
        self.run_id = mlflow.active_run().info.run_id
        return self

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        if mlflow is None or self.run_id is None:
            return
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v))

    def __exit__(self, exc_type, exc, tb) -> None:
        if mlflow is not None and self.run_id is not None:
            mlflow.end_run()
