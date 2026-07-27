"""Smoke tests that the environment is sane, independent of any model download."""

import numpy as np
import torch
import transformers


def test_torch_cpu_tensor_ops():
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert torch.allclose(x.sum(dim=1), torch.tensor([3.0, 12.0]))


def test_numpy_available():
    assert np.array([1, 2, 3]).sum() == 6


def test_transformers_importable():
    assert transformers.__version__ == "4.51.3"


def test_sense_neural_package_importable():
    import sense_neural

    assert sense_neural is not None
