#!/usr/bin/env python3
"""Handle boundaries.
"""
from dataclasses import dataclass
from typing import Any
from typing import Optional

import numpy as np
import numpy.typing as npt
import rich.repr
import torch

from pyABC.solver.fluxes import DimOrder


# Dimension of the compuational domain
NDIM = 3

# Cell unit vector
UVEC = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
)

# West-East-Back-Front-South-North
# order in -1 -> 1
FACE_DIR = [["w", "e"], ["b", "f"], ["s", "n"]]

DIM_ORDER = [2, 2, 1, 1, 0, 0]
DIR_ORDER = [1, -1, 1, -1, 1, -1]


def get_basic_domain(
    LX: npt.NDArray[np.float64], bc_types: list[str], bc_vals: Optional[list]
) -> tuple[list, dict]:
    """Bet basic domain object and bcs config.

    Note:
        - Domain is defined as following manner:

        .. code-block:: text

                z
                |
                .---- x
               /
              y

    Args:
        LX: domain size
        bc_types: BC types. One of [dirichlet, neumann, periodic, symmetric]
        bc_vals: boundary values if required
            - if bc_vals is None, set all boundaries to dirichlet type and assign zero.
    """

    # Left - Right - Down - Up - Back- Front
    name_set = ["yzL", "yzR", "xyD", "xyU", "xzB", "xzF"]
    ex_set = [
        [0.0, LX[1], LX[2]],
        [0.0, LX[1], LX[2]],
        [LX[0], LX[1], 0.0],
        [LX[0], LX[1], 0.0],
        [LX[0], 0.0, LX[2]],
        [LX[0], 0.0, LX[2]],
    ]

    xp_set = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    nvec_set = [
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    obj_config = []

    for n, e, x, v in zip(name_set, ex_set, xp_set, nvec_set):

        obj_config.append(
            {
                "name": n,
                "type": "patch",
                "geometry": {
                    "e_x": e,
                    "x_p": x,
                    "n_vec": np.asarray(v, dtype=np.int64),
                },
            }
        )

    bcs_config = dict()
    bc_ids = [str(i) for i in range(6)]

    if bc_vals is None:
        for i in bc_ids:
            bcs_config[i] = ["patch", "dirichlet", [0.0]]
    else:
        for i, t, v in zip(bc_ids, bc_types, bc_vals):
            bcs_config[i] = ["patch", t, v]

    return (obj_config, bcs_config)


@dataclass
class Patch:
    """2D patch element for the boundaries.

    Args:
        config (dict): dictionary for Patch configuration
            - It should contain "name", "type", and "geometry"
            - "geometry" should contain "e_x" and "x_p"
        id (int): id for the Patch
    """

    config: dict
    id: int

    def __post_init__(self):

        if len(self.config) != 3:
            from pyABC.tools.errors import SizeDoesNotMatchError

            msg = (
                "Seems like you are missing some configurations.\n"
                "It should have name, type, geometry.\n"
                f"Given config: {self.config}"
            )
            raise SizeDoesNotMatchError(msg)

        # Get name and type
        self.name = self.config["name"].lower()
        self.type = self.config["type"].lower()

        # Get the geometry information
        self.n_vec = self.config["geometry"]["n_vec"]
        # patch normal vector
        self.e_x = self.config["geometry"]["e_x"]
        self.x_p = self.config["geometry"]["x_p"]

    @property
    def size(self):

        return torch.prod(self.e_x)

    def __rich_repr__(self) -> rich.repr.Result:

        yield f"{self.name: >12} | (id: {self.id})"
        yield "n_vec", self.n_vec
        yield "e_x", self.e_x
        yield "x_p", self.x_p


@dataclass
class ImmersedBody:

    config: dict
    id: int

    def __post_init__(self):
        # Get name and type
        self.name = self.config["name"].lower()
        self.type = self.config["type"].lower()

        from pyABC.tools.errors import FeatureNotImplementedError

        msg = "ImmersedBody is not implemented!!"
        raise FeatureNotImplementedError(msg)


BC_OBJECT_FACTORY = {
    "patch": Patch,
    "immersed_body": ImmersedBody,
}


@dataclass
class BC:
    """Base class of the boundary condition object.

    Note:
        - :py:attr:`bc_face`, :py:attr:`bc_dir_dim`, :py:attr:`n_vec`
          will not be initialized at the beginning.
          It stores dummy data and will be actually initialized when
          :py:func:`create_patch_mask` function called.

    Args:
        obj_type: BC object type (patch, inner body)
        bc_type: BC type either (Neumann or Dirichlet)
        bc_id: BC id
        bc_val: values for the boundary condition
        bc_var_name: name of the variable
        device: torch.device. Either cpu or cuda
        bc_face: boundary face name. One of [w, e, n, s, f, b]. Defaults to ""
        bc_dir_dim: bc normal vector direction
        n_vec: bc normal vector
    """

    obj_type: str
    bc_type: str
    bc_id: str
    bc_val: Any  # This can be either torch.Tensor or None (for symmetric and periodic) << Not sure yet...
    bc_var_name: str
    device: torch.device

    # Belows are dummies at initialization.
    # Can be specified once self.set_n_vec is called.
    bc_face: str = ""
    bc_dir_dim: int = -1
    n_vec: torch.Tensor = torch.zeros(3)

    def set_n_vec(
        self, n_vec: npt.NDArray[np.float64], dx: npt.NDArray[np.float64]
    ):
        r"""Set :py:attr:`n_vec` which stored in objs.

        Args:
            n_vec: normal vector. $\pm$ :py:data:`UVEC`
            dx: grid spacing
        """

        for i, uvec in enumerate(UVEC):
            check_dir = n_vec @ uvec

            if check_dir != 0:
                if check_dir < 0:
                    self.bc_face = FACE_DIR[i][0]
                else:
                    self.bc_face = FACE_DIR[i][1]

                self.bc_flux_dir = int(check_dir)

                # Since the mesh data is in the order of z-y-x, we reverse
                # the indices
                self.bc_dir_dim = getattr(DimOrder, self.bc_face)
                self.bc_dx_dim = i

        # Convert to torch.Tensor
        self.n_vec = torch.tensor(
            n_vec, dtype=torch.float64, device=self.device
        )
        self.DX = dx

    def __rich_repr__(self) -> rich.repr.Result:

        yield f"{self.bc_var_name}: {self.obj_type} | id: {self.bc_id}"
        yield f"\ttype", self.bc_type
        yield "\tval", self.bc_val
        yield "\tn_vec", self.n_vec
        yield "\tface_dir", self.bc_face


class Wall(BC):
    """Wall boundary condition. Wall mass flux is zero."""

    def apply(self, mask: torch.Tensor, flux: torch.Tensor) -> torch.Tensor:
        """At wall, gradient is zero (flux is zero)."""

        flux[mask] = 0

        return flux

    def apply_div(
        self, mask: torch.Tensor, flux: torch.Tensor
    ) -> torch.Tensor:

        flux[mask] = 0

        return flux


class Dirichlet(BC):
    r"""Apply Dirichlet boundary conditions.

    .. math::

        f^{BC}_i = \left( \vec{u}_{BC} \cdot \vec{n}_{BC} \right)
        \Phi^{BC} \frac{S_{BC}}{V_i}

    Note:
        - If this is wall, $\vec{u}_{BC} = 0$, therefore, $f^{BC}_i=0$.

    """

    def apply_fdm(self, mask: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """Dirichlet BC for finite different method.
        We do not use flux, rather directly use Variables.VAR and
        assign self.bc_val using the mask defined.

        Note:
            - This is only used for the scalar field since it appears in
              the Poisson equation.

        Args:
            mask: mask of bc object
            var: Variable where BC is applied
        """
        var[mask] = self.bc_val[0]

        return var

    def apply(
        self,
        mask: torch.Tensor,
        flux: torch.Tensor,
        surface: float,
        vol: float,
        var: torch.Tensor,
        type: Optional[str] = None,
    ) -> torch.Tensor:
        """Apply BC.

        Args:
            mask: boundary mask
            flux: face of flux at BC
            surface: surface area of BC
            vol: cell volume at BC
            idx: index over vector dimension. Defaults to 0 (for scalar).
        """

        if self.bc_val.size()[0] == 1:
            # Scalar
            if type == "laplacian":
                flux[mask] = (
                    (self.bc_val[0] - var[mask])
                    / (0.5 * self.DX[self.bc_dx_dim])
                    * self.bc_flux_dir
                    * surface
                    / vol
                )
            else:
                flux[mask] = self.bc_val[0] * self.bc_flux_dir * surface / vol

        elif len(self.bc_val) == 3:
            # Vector
            # Since we are working at the structured grid,
            # torch.matmul is enough to calculate face value
            if type == "laplacian":

                at_bc = self.bc_val @ self.n_vec
                bc_var = var[self.bc_dir_dim]

                flux[mask] = (
                    (at_bc - bc_var[mask])
                    / (0.5 * self.DX[self.bc_dx_dim])
                    * surface
                    / vol
                    * self.bc_flux_dir
                )

            else:
                flux[mask] = (
                    (self.bc_val @ self.n_vec)
                    * surface
                    / vol
                    * self.bc_flux_dir
                )
        else:
            from pyABC.tools.errors import SizeDoesNotMatchError

            msg = f"Invalid amount of BCs given! len(self.bc_val) = {len(self.bc_val)}"
            raise SizeDoesNotMatchError(msg)

        return flux


class Neumann(BC):
    r"""Apply Neumann boundary conditions.

    .. math::

        f^{BC}_i = \vec{q}_{BC} \cdot \vec{n}_{BC}
    """

    def apply_fdm(self, mask: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        r"""Neumann boundary condition for the FDM. Use first-order finite
        differences.

        .. math::

            frac{\Phi^{BC} - \Phi^{BC \pm 1}}{\Delta x_i} = f^{BC}

        """

        mask_prev = torch.roll(mask, -self.bc_flux_dir, self.bc_dir_dim)  # type: ignore

        var[mask] = self.DX[self.bc_dir_dim] * self.bc_val[0] + var[mask_prev]

        return var

    def apply(
        self,
        mask: torch.Tensor,
        flux: torch.Tensor,
        surface: float,
        vol: float,
        var: torch.Tensor,
        type: Optional[str] = None,
    ) -> torch.Tensor:

        if self.bc_val.size()[0] == 1:
            # Scalar
            if type == "grad":

                var_c = var[mask]
                flux[mask] = (
                    (0.5 * self.bc_val[0] * self.DX[self.bc_dx_dim] + var_c)
                    * self.bc_flux_dir
                    * surface
                    / vol
                )

            else:
                flux[mask] = self.bc_val[0] * self.bc_flux_dir * surface / vol

        elif len(self.bc_val) == 3:
            # Vector
            if type == "grad":

                var_c = var[self.bc_dir_dim][mask]
                at_bc = self.bc_val @ self.n_vec
                flux[mask] = (
                    (0.5 * at_bc * self.DX[self.bc_dx_dim] + var_c)
                    * self.bc_flux_dir
                    * surface
                    / vol
                )
            else:

                flux[mask] = (
                    (self.bc_val @ self.n_vec)
                    * self.bc_flux_dir
                    * surface
                    / vol
                )
        else:
            from pyABC.tools.errors import SizeDoesNotMatchError

            msg = f"Invalid amount of BCs given! len(self.bc_val) = {len(self.bc_val)}"
            raise SizeDoesNotMatchError(msg)

        return flux


class Symmetry(BC):
    r"""Symmetry boundary condition.

    .. math::

        \Phi^{N} = \Phi^{N-1}, \quad \Phi^{1} = \Phi^{2}

    Note:
        - Since we use torch.roll to compute fluxes, we don't need to do
          something for the symmetry BC. Flux calculation itself already
          imposes symmetry BCs.
    """

    def apply_fdm(self, mask: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        # Mast at the previous
        mask_prev = torch.roll(mask, self.bc_flux_dir, self.bc_dir_dim)  # type: ignore

        var[mask] = var[mask_prev]

        return var

    def apply(self, *args) -> torch.Tensor:
        # Do nothing
        # args[0]: mask, args[1]: flux
        return args[1]

    def apply_laplacian(self, *args) -> torch.Tensor:
        # Do nothing <<< NOT SURE YET!!
        # args[0]: mask, args[1]: flux
        return args[1]


class Periodic(BC):
    r"""Periodic boundary condition.

    .. math::

            \Phi^{N} = \Phi^{1}, \quad \Phi^{1} = \Phi^{N}

    Note:
        Since we are using torch.roll when we calculate cell surface value,
        we do nothing for periodic boundary condition.

    """

    def apply_fdm(self, mask: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        r"""Periodic boundary condition used in the FDM."""

        # Mast at the next period
        mask_np = torch.roll(mask, -self.bc_flux_dir, self.bc_dir_dim)  # type: ignore

        var[mask] = var[mask_np]

        return var

    def apply(self, *args) -> torch.Tensor:

        return args[-1]

    def apply_div(self, *args) -> torch.Tensor:

        return args[-1]

    def apply_laplacian(self, *args) -> torch.Tensor:

        return args[-1]


def apply_bc_fdm(
    var: torch.Tensor,
    bcs: list,
    masks: dict,
) -> torch.Tensor:
    """Apply BCs of variables to be discretized.
    Since it is for FDM, we directly assign BC to Variables.VAR
    """

    # Apply Bcs
    for id, bc in enumerate(bcs):

        mask_at_bc = masks[id]

        var = bc.apply_fdm(mask_at_bc, var)

    # Update variable
    return var


def create_patch_mask(
    x: list[npt.NDArray[np.float64]],
    dx: npt.NDArray[np.float64],
    nx: npt.NDArray[np.int64],
    objs: list,
    bcs: list,
    device: torch.device,
) -> dict:
    """Create a mask from the objects (self.mesh.objs).
    This function is separated from Boundaries class to have better control.

    Note:
        - Only works for the Patch. For the :py:class:`ImmersedBody`,
          need separate treatment

    Todos:
        - Native torch function. (Currently we are using numpy to
          calculate the mask)

    Args:
        x: coordinates
        dx: grid spacing
        nx: number of grid points
        objs: boundary objects
        bcs: configuration for the boundary conditions
        device: torch device

    Returns
        Created mask dictionary. Dictionary key is obj.id.

    Raise:
        If `len(bcs) != len(objs)`, raise SizeDoesNotMatchError
    """

    # Loop over bcs dictionary. Key represents the Variables in use
    # bcs["var"]["each patch"]

    len_bc = len(bcs)
    len_obj = len(objs)

    if len_bc != len_obj:

        from pyABC.tools.errors import SizeDoesNotMatchError

        msg = (
            f"A number of objects ({len_obj}) does not match with "
            f"A number of BC defined ({len_bc})!"
        )
        raise SizeDoesNotMatchError(msg)

    mask_to_save = dict()

    # Loop over patch objects
    for obj in objs:

        mask = np.zeros((nx[2], nx[1], nx[0]), dtype=np.bool8)

        if obj.type == "patch":
            _nx = np.zeros(NDIM, dtype=np.int64)
            _ix = np.zeros_like(_nx)

            x_p = obj.x_p.copy()
            e_x = obj.e_x.copy()

            bcs[obj.id].set_n_vec(obj.n_vec, dx)

            for i in range(NDIM):

                x_p[i] = x[i][np.argmin(abs(x[i] - x_p[i]))]
                _nx[i] = np.ceil(e_x[i] / dx[0]).astype(np.int64) + 1
                _ix[i] = np.argmin(abs(x[i] - x_p[i]))

            mask[
                _ix[2] : _ix[2] + _nx[2],
                _ix[1] : _ix[1] + _nx[1],
                _ix[0] : _ix[0] + _nx[0],
            ] = True

            # Convert to torch tensor
            mask = torch.from_numpy(mask).to(device)
        else:
            # <<< FUTURE PLAN
            from pyABC.tools.errors import FeatureNotImplementedError

            raise FeatureNotImplementedError("ImmersedBody BC")

        # Save as sub dictionary
        # Should id be face dir?
        mask_to_save[obj.id] = mask

    return mask_to_save


BOUNDARY_FACTORY = {
    "dirichlet": Dirichlet,
    "neumann": Neumann,
    "symmetry": Symmetry,
    "periodic": Periodic,
}
