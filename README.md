
# THIS LIBRARY IS CURRENTLY BROKEN!! IT IS PRE-ALPHA STAGE

# pyapes

**PY**thon **A**wesome **P**artial differential **E**quation **S**olver (general purpose finite difference PDE solver)

![python](http://ForTheBadge.com/images/badges/made-with-python.svg)

## Description

`pyapes` is designed to solve various engineering problems in rectangular grid.

The goal of `pyapes` (should be/have) is

- Cross-platform
  - Both tested on Mac and Linux (Arch)
  - Windows support is under testing
- GPU acceleration in a structured grid with [PyTorch](https://pytorch.org)
  - Use of `torch.Tensor`. User can choose either `torch.device("cpu")` or `torch.device("cuda")`.
- Generically expressed (OpenFOAM-like, human-readable formulation)

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

## Implemented Features

- CPU/GPU computation using `torch`
- FDM Discretizations
  - Spatial: `Grad`, `Laplacian`, `Div`
  - Temporal: `Ddt`
- Boundary conditions:
  - `Dirichlet`
- Testing and demonstration
  - `Mesh`, `Field`, `FDM`
  - `Solver`
    - for the Poisson equation.

## Examples

### The 1D Poisson equation

For the clarity, below code snippet is vastly simplified. See `./demos/poisson.py` file for more details

```python
# Only shows relevant modules
from pyapes.core.geometry import Box
from pyapes.core.mesh import Mesh
from pyapes.core.solver.fdm import FDM
from pyapes.core.solver.ops import Solver
from pyapes.core.variables import Field
...

# Construct mesh
mesh = Mesh(Box[0:1], None, [0.02])

# Construct scalar field to be solved
var = Field("p", 1, mesh, {"domain": f_bc, "obstacle": None})
# Set RHS of PDE
rhs = torch.Tensor(...)

# Set solver and FDM discretizer
solver = Solver({"fdm": {"method": "cg", "tol": 1e-6, "max_it": 1000, "report" True}})
fdm = FDM()

# ∇^2 p = r
solver.set_eq(fdm.laplacian(1.0, var) == fdm.rhs(rhs))
# Solve for var
report = solver.solve() # report contains information regarding solver convergence
```

Resulting in

![1d-poisson-result](./assets/demo_figs/poisson_1d.png)

## Todos

- Boundary conditions
  - [ ] Neumann
  - [ ] Symmetry
  - [ ] Periodic
  - [ ] Inflow/Outflow
- Need different derivative order at the cell face
  - Additional features
    - [ ] High order time discretization
    - [ ] Immersed body BC
    - [x] Flux limiters
      - High-order flux limiter (`quick`) is WIP.
- Testing and validation
  - Create test files
    - `test_solver.py`
      - [ ] The advection-diffusion equation
      - [ ] The Euler equation
      - [ ] The Navier-Stokes equation at low Reynolds numbers
      - [ ] The Black-Scholes equation
- Publish to pypi.org
