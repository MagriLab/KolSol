# KolSol

KolSol is a pseudospectral Kolmogorov flow solver, using the a Fourier-Galerkin approach as described by Canuto [[1]](#R1). This library contains both `NumPy` and `PyTorch` implementations to allow for autograd-compatible workflows. Please note that the implementation of the FFT within `NumPy` is much more efficient so for general usage this version is preferable.

The solver provides numerical solutions to the divergence-free Navier-Stokes equations:

<p align='center'>
<img src="https://latex.codecogs.com/png.image?\dpi{110}&space;\begin{aligned}\nabla&space;\cdot&space;\textbf{u}&space;&=&space;0\\\\\partial&space;\textbf{u}_t&space;&plus;&space;\textbf{u}&space;\cdot&space;\nabla&space;\textbf{u}&space;&=&space;-\nabla&space;p&space;&plus;&space;\nu&space;\Delta&space;\textbf{u}&space;&plus;&space;\textit{\textbf{f}}\end{aligned}&space;" title="\begin{aligned}\nabla \cdot \textbf{u} &= 0\\\\\partial \textbf{u}_t + \textbf{u} \cdot \nabla \textbf{u} &= -\nabla p + \nu \Delta \textbf{u} + \textit{\textbf{f}}\end{aligned} " />
</p>

As a spectral method is employed, using a larger quantity of wavenumbers will reduce the numerical error.
<br>**Note:** Highly accurate simulations can be attained even with relatively few wavenumbers.

## **Flowfield:**
<p align='center'>
<img src="media/flowfield.png"/>
</p>

## **Installation:**
To install KolSol, please clone the repository and then run the following command:

```shell
$ python setup.py install
```

## **Solver Example:**

```python
import numpy as np
from kolsol.numpy.solver import KolSol

# instantiate solver
ks = KolSol(nk=8, nf=4, re=30.0, ndim=2)

# define initial conditions
u_hat = ks.random_field(1.0, 0.001, [0, 3])

# run simulation
for t in np.arange(0.0, 10.0, 0.01):
  u_hat += dt * ks.dynamics(u_hat)
```

## **References:**
<a id="R1">**[1]**</a> Canuto, C. (1988) Spectral methods in fluid dynamics. New York: Springer-Verlag.

