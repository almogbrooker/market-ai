import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.super_ensemble import SuperEnsemble, ModelInfo

def test_zero_sum_accuracy_weights():
    # Create instance without invoking __init__
    ensemble = SuperEnsemble.__new__(SuperEnsemble)
    ensemble.model_info = {
        'model_a': ModelInfo('model_a', 'dummy', {}, '', 0.0, 1.0),
        'model_b': ModelInfo('model_b', 'dummy', {}, '', 0.0, 2.0),
    }
    SuperEnsemble.calculate_weights(ensemble)
    weights = np.array([info.weight for info in ensemble.model_info.values()])
    expected_loss_weights = np.array([1.0, 0.5])
    expected_loss_weights = expected_loss_weights / expected_loss_weights.sum()
    expected = 0.7 * np.array([0.5, 0.5]) + 0.3 * expected_loss_weights
    assert np.allclose(weights, expected)
