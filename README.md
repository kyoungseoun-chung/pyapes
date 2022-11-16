
# THIS LIBRARY IS CURRENTLY BROKEN!! IT IS PRE-ALPHA STAGE

# pyapes

**PY**thon **A**wesome **P**artial differential **E**quation **S**olver (general purpose finite volume PDE solver)

The code was inspired by [airborne_covid19](https://gitlab.ethz.ch/ifd-pdf/airborne-transmission/airborne_covid19).

![python](http://ForTheBadge.com/images/badges/made-with-python.svg)

## Description

`pyapes` is designed to solve various engineering problems in rectangular grid.

`pyapes` (should be/have) is/has

- Cross-platform
  - Both tested on Mac and Linux (Arch)
  - Windows support is under testing
- GPU acceleration in a structured grid with [PyTorch](https://pytorch.org)
  - Use of `torch.Tensor`. User can choose either `torch.device("cpu")` or `torch.device("cuda")`.
- Generically expressed (OpenFOAM-like, human-readable formulation)

```python3
>>> var = Field(...)
>>> solver = Solver(config)
>>> fvm = solver.fvm  # discretization for the implicit field
>>> fvc = solver.fvc  # discretization for the explicit field
>>> solver.set_eq(
fvm.Ddt(var) + fvm.Div(var, u) + fvm.Laplacian(var) \
== fvc.Grad(var) + fvc.Source(var)
)
>>> solver()
```

## Installation

We recommend to use `poetry` to manage all dependencies.

```bash
git clone git@gitlab.ethz.ch:kchung/pyapes.git
cd pyapes
poetry install
```

> Later, `pip` install via pypi.org will be supported.

## Dependencies

- Core dependency
  - `python >= 3.10`
  - `numpy >= 1.21.0`
  - `torch >= 1.10.0`
  - `pyevtk >= 1.2.0`
  - `tensorboard >= 2.7.0`
- Misc dependency (Aesthetic)
  - `tqdm >= 4.62.3`
  - `rich >= 10.12.0`

## Features (Planning)

- GPU enabled computation (using `torch`)
- Finite Volume Method:
  - Support spatial discretization in `FVC` and `FVM` manner
    - `Grad`, `Laplacian`, `Div` (`FVM` only)
  - Implicit time integration in `Ddt`
    - The backward Euler and the Crank-Nicolson
  - Flux limiters
  - Immersed body boundary conditions
  - Linear system solvers
    - Supports Jacobi, Conjugated gradient (CG), CG-Newton, and Gauss-Seidel (need verification) method.

## Todos

- Refactoring the code:
  - Better data structure
    - [x] Flux class
    - [ ] `FVC` for explicit discretization
    - [ ] `FVM` for implicit discretization
  - BCs
    - [x] Dirichlet
    - [ ] Neumann
    - [ ] Symmetry
    - [ ] Periodic
    - [ ] Inflow/Outflow
  - Discretization
    - [x] `Grad`
    - [x] `Laplacian`
    - [ ] `Div`
    - [ ] `Ddt`
- Need different derivative order at the cell face
  - Additional features
    - [ ] High order time discretization
    - [ ] Immersed body BC
    - [ ] Flux limiters
- Testing and validation
  - Create test files
    - [x] `test_mesh.py`
    - [x] `test_variables.py`
    - [ ] `test_fvc.py`
    - [ ] `test_fvm.py`
  - PDF solver examples
    - [ ] The diffusion equation
    - [ ] The Euler equation
    - [ ] The Navier-Stokes equation at low Reynolds numbers
    - [ ] The Black-Scholes equation
- Publish to pypi.org
