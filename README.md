# KolSol

KolSol is a pseudospectral Kolmogorov flow solver, using a Fourier-Galerkin approach as described by Canuto [[1]](#R1). 
This library contains both `NumPy` and `PyTorch` implementations to allow for autograd-compatible workflows. Please note 
that the implementation of the FFT within `NumPy` is much more efficient so for general usage this version is preferable.

The solver provides numerical solutions to the divergence-free Navier-Stokes equations:

$$
\begin{aligned}
  \nabla \cdot u &= 0 \\
  \partial_t u + u \cdot \nabla u &= - \nabla p + \nu \Delta u + f
\end{aligned}
$$

As a spectral method is employed, using a larger quantity of wavenumbers will reduce the numerical error.
<br>**Note:** Highly accurate simulations can be attained even with relatively few wavenumbers.

- [x] **License:** MIT
- [x] **Python Versions:** 3.8.10+

## **Flowfield:**
<p align='center'>
<img src="media/flowfield.png"/>
</p>

## **Installation:**
To install KolSol, please clone the repository and then run the following command:

```shell
$ pip install poetry
$ poetry install
```

## **Solver Example:**

```python
import numpy as np
from kolsol.numpy.solver import KolSol

# instantiate solver
ks = KolSol(nk=8, nf=4, re=30.0, ndim=2)
dt = 0.01

# define initial conditions
u_hat = ks.random_field(magnitude=10.0, sigma=2.0, k_offset=[0, 3])

# simulate :: run over transients
for _ in np.arange(0.0, 180.0, dt):
  u_hat += dt * ks.dynamics(u_hat)

# simulate :: generate results
for _ in np.arange(0.0, 300.0, dt):
  u_hat += dt * ks.dynamics(u_hat)

# generate physical field
u_field = ks.fourier_to_phys(u_hat, nref=256)
```

## **Data Generation:**

The given `generate.py` script can be used to generate simulation data for a Kolmogorov flow field, for example:

```bash
$ python generate.py --data-path .data/RE30/results.h5 --resolution 256 --re 30.0 --time-simulation 60.0
```

Running the above command results in the following file structure:

```
data
├── RE30
│   └── results.h5
├── README.md
└── generate.py
```

**Note:** Additional command-line arguments are available for customising simulations.

## **References:**
<a id="R1">**[1]**</a> Canuto, C. (1988) Spectral methods in fluid dynamics. New York: Springer-Verlag.

