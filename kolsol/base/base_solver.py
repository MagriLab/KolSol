from typing import Protocol

import numpy as np
import torch

from ..utils.types import TypeTensor


class Solver(Protocol[TypeTensor]):

    def dynamics(self, u_hat: TypeTensor) -> TypeTensor:
        ...


class BaseNumpyKolSol:

    def __init__(self, nk: int, nf: int, re: float, ndim: int = 2) -> None:

        self.nk = nk
        self.nf = nf
        self.re = re
        self.ndim = ndim

        self.nk_grid = 2 * self.nk
        self.mk_grid = 4 * self.nk

    def dynamics(self, u_hat: np.ndarray) -> np.ndarray:

        r"""Computes \partial u_hat / \partial t

        Parameters
        ----------
        u_hat: np.ndarray
            Velocity field in the Fourier domain.

        Returns
        -------
        np.ndarray
            Computed time-derivative.
        """

        raise NotImplementedError('BaseNumpyKolSol::dynamics()')


class BaseTorchKolSol:

    def __init__(self, nk: int, nf: int, re: float, ndim: int = 2) -> None:

        self.nk = nk
        self.nf = nf
        self.re = re
        self.ndim = ndim

        self.nk_grid = 2 * self.nk
        self.mk_grid = 4 * self.nk

    def dynamics(self, u_hat: torch.Tensor) -> torch.Tensor:

        r"""Computes \partial u_hat / \partial t

        Parameters
        ----------
        u_hat: torch.Tensor
            Velocity field in the Fourier domain.

        Returns
        -------
        torch.Tensor
            Computed time-derivative.
        """

        raise NotImplementedError('BaseTorchKolSol::dynamics()')

