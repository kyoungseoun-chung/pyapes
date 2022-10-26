# THIS LIBRARY IS CURRENTLY BROKEN!! IT IS PRE-ALPHA STAGE!
# Python Awesome Partial differential Equation Solver: PyAPES

<img src="PyAPES/assets/logo.png" width="150"/>

The code was inspired by https://gitlab.ethz.ch/ifd-pdf/airborne-transmission/airborne_covid19.

![python](http://ForTheBadge.com/images/badges/made-with-python.svg)


## Description

`pyapes` is designed to solve various engineering problems.

`pyapes` is/has
- Cross-platform
	- Both tested on Mac and Linux (Arch)
	- Windows support is under testing
- GPU acceleration in a structured grid with [PyTorch](https://pytorch.org)
	- Use of `torch.Tensor`. User can choose either `torch.device("cpu")` or `torch.device("cuda")`
- Generically expressed (OpenFOAM-like, human-readable formulation)
	```python
	# FDM
	fvc.solve(fvc.laplacian(var) == rhs)
	# FVM
	fvm.set_eq(fvm.Ddt(var) + fvm.Div(var, u) + fvm.Laplacian(var))
	fvm.solve(fvm.eq == rhs)
	```
## Installation

We recommend to use `poetry` to manage all dependencies.

```bash
git clone git@gitlab.ethz.ch:kchung/pyapes.git
cd pyapes
poetry install
```

## Dependencies

- Core dependency
	- python >= 3.9
	- numpy >= 1.21.0
	- torch >= 1.10.0
	- pyevtk >= 1.2.0
	- tensorboard >= 2.7.0
- Misc dependency (Aesthetic)
	- tqdm >= 4.62.3
	- rich >= 10.12.0

## Features
- PDE solvers
	- [ ] Navier-Stokes equation at low Reynolds numbers
		- [ ] Convection equation
		- [ ] Diffusion equation
    	- [x] Poisson equation (FDM manner. Working on FVM)
		- Supports Jacobi, Conjugated gradient (CG), CG-Newton, and Gauss-Seidel (need verification) method.
- Finite Volume Mesh:
	- [x] Easy mesh creation and definition of boundary conditions
	- [ ] Spatial discretization
		- `Grad`, `Laplacian`, `Div`
	- [ ] Time discretization
		- Forward/backward Euler and Crank-Nicolson
	- [ ] Flux limiters
	- [ ] Immersed body

## TO-DOs

- Slimmer dependency
	- [x] Remove unnecessary dependencies
- Refactoring the code:
	- [ ] Make it more concise and intuitive
	- [ ] Better data structure
	- [ ] Add new features: High order time discretization, Immersed body BC, and Flux limiters
- Testing and validation
	- [ ] Create test files
