# THIS LIBRARY IS CURRENTLY BROKEN!! IT IS PRE-ALPHA STAGE!
# Python Awesome Basic Calculus: pyABC

<img src="pyABC/assets/logo.png" width="150"/>

The code was inspired by https://gitlab.ethz.ch/ifd-pdf/airborne-transmission/airborne_covid19.

![python](http://ForTheBadge.com/images/badges/made-with-python.svg)


[![pipeline](https://gitlab.ethz.ch/kchung/pyabc/badges/main/pipeline.svg)](https://gitlab.ethz.ch/kchung/pyabc/commits/main)
[![coverage](https://gitlab.ethz.ch/kchung/pyabc/badges/main/coverage.svg)](https://gitlab.ethz.ch/kchung/pyabc/commits/main)
[![python3.9](https://img.shields.io/badge/python-3.9-blue)](https://www.python.org/downloads/release/python-390/)

## Description

PyABC is designed to solve various engineering problems.

PyABC is/has
- Cross-platform
	- Both tested on Mac and Linux (Arch)
	- Windows support is under testing
- GPU acceleration in a structured grid with [PyTorch](https://pytorch.org)
	- Use of `torch.Tensor`. User can choose either `torch.device("cpu")` or `torch.device("cuda")`
- Generically expressed (OpenFoam-like, human-readable formulation)
	```python
	# FDM
	fvc.solve(fvc.laplacian(var) == rhs)
	# FVM
	fvm.set_eq(fvm.Ddt(var) + fvm.Div(var, u) + fvm.Laplacian(var))
	fvm.solve(fvm.eq == rhs)
	```
## Installation

```bash
git clone git@gitlab.ethz.ch:kchung/pyabc.git
cd pyabc
python3 -m pip install -e ./
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
