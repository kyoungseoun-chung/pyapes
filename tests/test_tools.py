#!/usr/bin/env python3
"""Collection of tests for the utils."""
import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest
import torch


def test_container(report: bool = True) -> None:

    from pyABC.tools.utils import Container

    A = Container(test=10)

    # Empty container
    B = Container()

    assert A.test == 10

    if report:
        print(A)
        print(B)
        print(dir(A))

    with pytest.raises(AttributeError):
        print(A.dummy)

    delattr(A, "test")


def test_getsize() -> None:

    from pyABC.tools.utils import getsize

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


def test_progress_bar() -> None:

    from pyABC.tools.progress import RichPBar

    test_itrator = np.arange(10)

    pbar = RichPBar(test_itrator, desc="TEST", mininterval=0)

    for _ in pbar:
        pass


def test_max_iter() -> None:

    from pyABC.tools.errors import MaximunIterationReachedWarning

    with pytest.warns(MaximunIterationReachedWarning):
        msg = f"0/0"
        warnings.warn(msg, MaximunIterationReachedWarning)


def test_solution_diverge() -> None:

    from pyABC.tools.errors import SolutionDoesNotConverged

    with pytest.raises(SolutionDoesNotConverged):
        raise SolutionDoesNotConverged("Solution diverged!")


def test_wrong_input() -> None:

    from pyABC.tools.errors import WrongInputError, WrongInputWarning

    with pytest.warns(WrongInputWarning):
        msg = f"0/0"
        warnings.warn(msg, WrongInputWarning)

    with pytest.raises(WrongInputError):
        raise WrongInputError("Received wrong input!")


def test_size_does_not_match() -> None:

    from pyABC.tools.errors import SizeDoesNotMatchError

    with pytest.raises(SizeDoesNotMatchError):
        raise SizeDoesNotMatchError(f"{10} is not {11}")


def test_value_does_not_match() -> None:

    from pyABC.tools.errors import ValueDoesNotMatchError

    with pytest.raises(ValueDoesNotMatchError):
        raise ValueDoesNotMatchError(f"{10} is not {11}")


def test_feature_not_implemented() -> None:

    from pyABC.tools.errors import FeatureNotImplementedError

    with pytest.raises(FeatureNotImplementedError):
        raise FeatureNotImplementedError("This feature is not implemented!")


if __name__ == "__main__":
    test_container()
    test_getsize()
