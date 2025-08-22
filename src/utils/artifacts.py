from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import joblib
import torch


def save_model_artifacts(
    model_dir: Path,
    model: torch.nn.Module,
    scaler: Any,
    feature_list: Iterable[str],
    gate_config: Dict[str, Any],
    training_meta: Dict[str, Any],
) -> Optional[Path]:
    """Atomically save model artifacts and metadata.

    Parameters
    ----------
    model_dir : Path
        Destination directory for the model artifacts.
    model : torch.nn.Module
        Trained model to persist.
    scaler : Any
        Preprocessing object saved with joblib.
    feature_list : Iterable[str]
        Sequence of model features.
    gate_config : Dict[str, Any]
        Configuration for gating mechanism.
    training_meta : Dict[str, Any]
        Metadata including data ranges, random seeds, and training params.

    Returns
    -------
    Optional[Path]
        Path to backup of previous model version if it existed, otherwise ``None``.
    """
    model_dir = Path(model_dir)
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(dir=model_dir.parent, prefix=model_dir.name + "_tmp_"))

    try:
        torch.save(model.state_dict(), tmp_dir / "model.pt")
        joblib.dump(scaler, tmp_dir / "scaler.joblib")
        with open(tmp_dir / "feature_list.json", "w") as f:
            json.dump(list(feature_list), f, indent=2)
        with open(tmp_dir / "gate.json", "w") as f:
            json.dump(gate_config, f, indent=2)

        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=model_dir.parent)
            .decode()
            .strip()
        )
        model_card = {
            "commit_hash": commit_hash,
            "data_range": training_meta.get("data_range"),
            "random_seed": training_meta.get("random_seed"),
            "training_params": training_meta.get("training_params"),
        }
        with open(tmp_dir / "model_card.json", "w") as f:
            json.dump(model_card, f, indent=2)

        backup_dir: Optional[Path] = None
        if model_dir.exists():
            backup_dir = model_dir.parent / f"{model_dir.name}_backup_{datetime.now():%Y%m%d_%H%M%S}"
            shutil.move(str(model_dir), str(backup_dir))
        shutil.move(str(tmp_dir), str(model_dir))
        return backup_dir
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
