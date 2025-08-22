import hashlib
import json
from typing import Any, Dict


def compute_intent_hash(payload: Dict[str, Any]) -> str:
    """Compute a deterministic hash for a trading intent payload.

    The caller should round or quantize any floating point values (e.g.,
    ``prediction``) before calling this function. The provided payload is
    serialized with sorted keys to ensure stable hashes.
    """
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
