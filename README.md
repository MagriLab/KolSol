# KolSol

KolSol is a pseudospectral Kolmogorov flow solver, using the a Fourier-Galerkin approach as described by Canuto [1]. This library contains both `NumPy` and `PyTorch` implementations to allow for autograd-compatible workflows. Please note that the implementation of the FFT within `NumPy` is much more efficient so for general usage this version is preferable.

- **License:** MIT
- **Python Version:** 3.6

## **Example:**

```python
import numpy as np
from kolsol.numpy.solver import KolSol

# instantiate solver
ks = KolSol(nk=8, nf=4, re=30.0, ndim=2)

# define initial conditions
u_hat = (1.0 + 1.0j) * np.random.uniform(size=(17, 17))

# run simulation
for t in np.arange(0.0, 10.0, 0.01):
  u_hat += dt * ks.dynamics(u_hat)
```

#### References:
[1] Canuto, C. (1988) Spectral methods in fluid dynamics. New York: Springer-Verlag.

