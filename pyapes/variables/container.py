#!/usr/bin/env python3
"""Variable container for Jacobian and Hessian."""
from dataclasses import dataclass
import torch
from torch import Tensor


class Derivatives:
    """Base class for Jacobian and Hessian. Intention is to use generic indices (x, y, z for example) to access each derivative group.

    Example:
        >>> jac = Jac(x=...)
        >>> jac.x
        Tensor(...)
        >>> hess = Hess(xx=...)
        >>> hess.xx
        Tensor(...)
    """

    def __init__(self):
        self.max = 0

        total_var = len(vars(self).items()) - 1

        for idx, (_, v) in enumerate(vars(self).items()):
            if idx == total_var:
                # Exclude self.max for counting
                break

            if v.shape[0] == 0:
                pass
            else:
                self.max += 1

        self.keys = [
            k
            for idx, (k, v) in enumerate(vars(self).items())
            if idx < total_var and v.shape[0] != 0
        ]

    def __getitem__(self, key: str) -> Tensor:
        """Return the derivative group by a key. If the key is given for the Hessian, the key is always sorted in alphabetical order.

        Example:
            >>> hess["xz"]
            hess.xz
            >>> hess["zx"]
            hess.xz
            >>> hess["yx"]
            hess.xy
        """

        item = getattr(self, "".join(sorted(key.lower())))
        if item.shape[0] == 0:
            raise KeyError(f"Derivative: key {key} not found.")
        else:
            return item

    def __len__(self) -> int:
        return self.max

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.max:
            res = getattr(self, self.keys[self.n])
            self.n += 1
            return res
        else:
            raise StopIteration


@dataclass
class Jac(Derivatives):
    x: Tensor = torch.tensor([])
    y: Tensor = torch.tensor([])
    z: Tensor = torch.tensor([])
    r: Tensor = torch.tensor([])

    def __post_init__(self):
        super().__init__()


@dataclass
class Hess(Derivatives):
    xx: Tensor = torch.tensor([])
    xy: Tensor = torch.tensor([])
    xz: Tensor = torch.tensor([])
    yy: Tensor = torch.tensor([])
    yz: Tensor = torch.tensor([])
    zz: Tensor = torch.tensor([])
    rr: Tensor = torch.tensor([])
    rz: Tensor = torch.tensor([])
    zz: Tensor = torch.tensor([])

    def __post_init__(self):
        super().__init__()
