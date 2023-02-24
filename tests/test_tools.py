#!/usr/bin/env python3
"""Collection of tests for the utils."""
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest
import torch


def test_getsize() -> None:
    from pyapes.tools.utils import getsize

    @dataclass
    class Dummy:
        id: int
        desc: str
        data: npt.NDArray[np.float64]
        torch_tensor: torch.Tensor

    A = Dummy(0, "dummy data", np.random.rand(100, 100), torch.rand(10))

    print(getsize(A))

    # Check blacklist.
    # Get size cannot calculate type, module, or function
    with pytest.raises(TypeError):
        dummy_func = lambda a: a + 10
        getsize(dummy_func)
