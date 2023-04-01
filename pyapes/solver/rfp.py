#!/usr/bin/env python3
"""FDC discretization especially for the Rosenbluth Fokker-Planck equation"""
import torch
from torch import Tensor

from pyapes.variables import Field
from pyapes.variables.container import Hess
from pyapes.variables.container import Jac


# TODO: NEED TO REVISE BCS
class Friction:
    """Friction term

    Warnings:
        - Currently only supports rz coordinate system
    """

    @staticmethod
    def __call__(jacH: Jac, var: Field) -> Tensor:
        if var.mesh.coord_sys != "rz":
            raise NotImplementedError(
                "FP: Friction is only implemented for rz coordinate system."
            )

        Hr = jacH.r
        Hz = jacH.z

        pdf = var[0]
        dx = var.mesh.dx

        Arp = (torch.roll(Hr, -1, 0) + Hr) / 2.0
        Arm = (Hr + torch.roll(Hr, 1, 0)) / 2.0

        Azp = (torch.roll(Hz, -1, 1) + Hz) / 2.0
        Azm = (Hz + torch.roll(Hz, 1, 1)) / 2.0

        Prp = (torch.roll(pdf, -1, 0) + pdf) / 2.0
        Prm = (pdf + torch.roll(pdf, 1, 0)) / 2.0

        Pzp = (torch.roll(pdf, -1, 1) + pdf) / 2.0
        Pzm = (pdf + torch.roll(pdf, 1, 1)) / 2.0

        r_p = (torch.roll(var.mesh.R, -1, 0) + var.mesh.R) / 2
        r_m = (var.mesh.R + torch.roll(var.mesh.R, 1, 0)) / 2
        r = var.mesh.R

        friction = (Azp * Pzp - Azm * Pzm) / dx[1] + (
            r_p * Arp * Prp - r_m * Arm * Prm
        ) / (r * dx[0])

        # BC: the normal component of the flux to zero
        # r = 0, Arm = 0
        friction[0, :] = (Azp * Pzp - Azm * Pzm)[0, :] / (dx[1])
        # r = R, Arp = 0
        friction[-1, :] = (Azp * Pzp - Azm * Pzm)[-1, :] / (dx[1]) + 2.0 * (
            (-r_m * Arm * Prm) / (r * dx[0])
        )[-1, :]

        # z = 0
        friction[:, 0] = (
            2.0 * (Azp * Pzp)[:, 0] / (dx[1])
            + torch.nan_to_num(
                (r_p * Arp * Prp - r_m * Arm * Prm) / (r * dx[0]),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )[:, 0]
        )

        # z = Z
        friction[:, -1] = (
            2.0 * (-Azm * Pzm)[:, -1] / (dx[1])
            + torch.nan_to_num(
                (r_p * Arp * Prp - r_m * Arm * Prm) / (r * dx[0]),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )[:, -1]
        )

        return friction


class Diffusion:
    r"""Divergence of an anisotropic diffusion tensor using the symmetric finite difference discretization.

    Note:
        - D_rz term is evaluated using the bilinear interpolation of the values at the cell center.

    Warnings:
        - Currently only supports rz coordinate system

    .. math::

        \nabla \cdot (\mathbf{D} \cdot \nabla \Phi)
    """

    @staticmethod
    def __call__(hessG: Hess, var: Field) -> Tensor:
        if var.mesh.coord_sys != "rz":
            raise NotImplementedError(
                "FP: Diffusion is only implemented for rz coordinate system."
            )

        Drr = hessG.rr
        Dzz = hessG.zz
        Drz = hessG.rz

        pdf = var[0]
        dx = var.mesh.dx

        # (D+ + D-)/2 * (P+ - P-)/dx -> (D+ + D-) * (P+ - P-) / dx
        Drr_Pr_rpz = (
            (torch.roll(Drr, -1, 0) + Drr)
            * (torch.roll(pdf, -1, 0) - pdf)
            / (2.0 * dx[0])
        )

        Drr_Pr_rmz = (
            (torch.roll(Drr, 1, 0) + Drr)
            * (pdf - torch.roll(pdf, 1, 0))
            / (2.0 * dx[0])
        )

        Dzz_Pz_rzp = (
            (torch.roll(Dzz, -1, 1) + Dzz)
            * (torch.roll(pdf, -1, 1) - pdf)
            / (2.0 * dx[1])
        )

        Dzz_Pz_rzm = (
            (torch.roll(Dzz, 1, 1) + Dzz)
            * (pdf - torch.roll(pdf, 1, 1))
            / (2.0 * dx[1])
        )

        Drz_pp = _c_interp(Drz, 1, 1)
        Drz_pm = _c_interp(Drz, 1, 0)
        Drz_mp = _c_interp(Drz, 0, 1)
        Drz_mm = _c_interp(Drz, 0, 0)

        Drz_Pr_rzp = 0.25 * Drz_pp * (
            _flux(pdf, (1, 0), (0, 0), dx[0]) + _flux(pdf, (1, 1), (0, 1), dx[0])
        ) + 0.25 * Drz_mp * (
            _flux(pdf, (0, 0), (-1, 0), dx[0]) + _flux(pdf, (0, 1), (-1, 1), dx[0])
        )

        Drz_Pr_rzm = 0.25 * Drz_pm * (
            _flux(pdf, (1, -1), (0, -1), dx[0]) + _flux(pdf, (1, 0), (0, 0), dx[0])
        ) + 0.25 * Drz_mm * (
            _flux(pdf, (0, -1), (-1, -1), dx[0]) + _flux(pdf, (0, 0), (-1, 0), dx[0])
        )

        Drz_Pz_rpz = 0.25 * Drz_pp * (
            _flux(pdf, (0, 1), (0, 0), dx[1]) + _flux(pdf, (1, 1), (1, 0), dx[1])
        ) + 0.25 * Drz_mp * (
            _flux(pdf, (0, 0), (0, -1), dx[1]) + _flux(pdf, (1, 0), (1, -1), dx[1])
        )

        Drz_Pz_rmz = 0.25 * Drz_pm * (
            _flux(pdf, (-1, 1), (-1, 0), dx[1]) + _flux(pdf, (0, 1), (0, 0), dx[1])
        ) + 0.25 * Drz_mm * (
            _flux(pdf, (-1, 0), (-1, -1), dx[1]) + _flux(pdf, (0, 0), (0, -1), dx[1])
        )

        r_p = (torch.roll(var.mesh.grid[0], -1, 0) + var.mesh.grid[0]) / 2
        r_m = (var.mesh.grid[0] + torch.roll(var.mesh.grid[0], 1, 0)) / 2
        r = var.mesh.grid[0]

        diffusion = (
            (Dzz_Pz_rzp - Dzz_Pz_rzm) / dx[1]
            + (Drz_Pr_rzp - Drz_Pr_rzm) / dx[1]
            + (r_p * Drz_Pz_rpz - r_m * Drz_Pz_rmz) / (r * dx[0])
            + (r_p * Drr_Pr_rpz - r_m * Drr_Pr_rmz) / (r * dx[0])
        )

        # BC?
        # r = 0
        diffusion[0, :] = (Dzz_Pz_rzp - Dzz_Pz_rzm)[0, :] / dx[1] + 2.0 * (
            Drz_Pr_rzp - Drz_Pr_rzm
        )[0, :] / dx[1]

        # r = R
        diffusion[-1, :] = (
            ((Dzz_Pz_rzp - Dzz_Pz_rzm) / dx[1] + (Drz_Pr_rzp - Drz_Pr_rzm) / dx[1])[
                -1, :
            ]
            + 2.0 * ((-r_m * Drz_Pz_rmz) / (r * dx[0]))[-1, :]
            + 2.0 * ((-r_m * Drr_Pr_rmz) / (r * dx[0]))[-1, :]
        )

        # diffusion[-1, :] = 0.0

        # z = 0
        diffusion[:, 0] = (
            2.0 * ((Dzz_Pz_rzp) / dx[1] + (Drz_Pr_rzp) / dx[1])[:, 0]
            + torch.nan_to_num(
                (r_p * Drz_Pz_rpz - r_m * Drz_Pz_rmz) / (r * dx[0])
                + (r_p * Drr_Pr_rpz - r_m * Drr_Pr_rmz) / (r * dx[0]),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )[:, 0]
        )

        # z = Z
        diffusion[:, -1] = (
            2.0 * ((-Dzz_Pz_rzm) / dx[1] + (-Drz_Pr_rzm) / dx[1])[:, -1]
            + torch.nan_to_num(
                (r_p * Drz_Pz_rpz - r_m * Drz_Pz_rmz) / (r * dx[0])
                + (r_p * Drr_Pr_rpz - r_m * Drr_Pr_rmz) / (r * dx[0]),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )[:, -1]
        )

        return diffusion


def _flux(
    var: Tensor, idx_p: tuple[int, int], idx_m: tuple[int, int], dx: Tensor
) -> Tensor:
    """Gradient at the cell surface."""

    ip = (-idx_p[0], -idx_p[1])
    im = (-idx_m[0], -idx_m[1])

    return (torch.roll(var, ip, (0, 1)) - torch.roll(var, im, (0, 1))) / dx


def _c_interp(var: Tensor, upper_i: int, upper_j: int) -> Tensor:
    """Compute bi-linear interpolation of the values at the cell center.

    Note:
        - `upper_i` and `upper_j` indicates the upper right corner of the cell.

    Args:
        var (Tensor): Variable to be interpolated
        upper_i (int): Upper index in the i direction
        upper_j (int): Upper index in the j direction
    """

    return (
        torch.roll(var, (-upper_i, -upper_j), (0, 1))  # i+1, j+1 (upper right)
        + torch.roll(var, (-upper_i, -upper_j + 1), (0, 1))  # i+1, j
        + torch.roll(var, (-upper_i + 1, -upper_j), (0, 1))  # i, j+1
        + torch.roll(var, (-upper_i + 1, -upper_j + 1), (0, 1))  # i, j
    ) / 4


class RFP:
    """Simple Fokker-Planck operator. All operators return a Tensor not Field"""

    friction: Friction = Friction()
    diffusion: Diffusion = Diffusion()


def mc_limiter(a: Tensor, b: Tensor) -> Tensor:
    """Monotonized-central flux limitting scheme"""

    return minmod(2.0 * minmod(a, b), (a + b) / 2.0)


def minmod(a: Tensor, b: Tensor) -> Tensor:
    """Min-mod function."""
    val = torch.zeros_like(a)

    mask = torch.logical_and(a.ge(0.0), b.ge(0.0))

    val[mask] = torch.min(a[mask], b[mask])

    mask = torch.logical_and(a.lt(0.0), b.lt(0.0))

    val[mask] = torch.max(a[mask], b[mask])

    mask = (a * b).le(0.0)

    val[mask] = 0.0

    return val
