from typing import List, Optional

from ..utils.types import TypeTensor


class BaseKolSol:

    def __init__(self, nk: int, nf: int, re: float, ndim: int = 2) -> None:

        self.nk = nk
        self.nf = nf
        self.re = re
        self.ndim = ndim

        self.nk_grid = 2 * self.nk + 1
        self.mk_grid = 4 * self.nk + 1

    def dynamics(self, u_hat: TypeTensor) -> TypeTensor:

        r"""Computes \partial u_hat / \partial t

        Parameters
        ----------
        u_hat: torch.Tensor
            Velocity field in the Fourier domain.

        Returns
        -------
        TypeTensor
            Computed time-derivative.
        """

        raise NotImplementedError('BaseKolSol::dynamics()')

    def aap(self, f1: TypeTensor, f2: TypeTensor) -> TypeTensor:

        """Computes anti-aliased product between two tensors.

        Parameters
        ----------
        f1: torch.Tensor
            Tensor one.
        f2: torch.Tensor
            Tensor two.

        Returns
        -------
        TypeTensor
            Anti-aliased product of tensor one, tensor two.
        """

        raise NotImplementedError('BaseKolSol::aap()')

    def dissip(self, u_hat: TypeTensor) -> TypeTensor:

        """Computes dissipation from a given flow-field.

        Parameters
        ----------
        u_hat: torch.Tensor
            Velocity field in the Fourier domain.

        Returns
        -------
        TypeTensor
            Computed dissipation.
        """

        raise NotImplementedError('BaseKolSol::dissip()')

    def vorticity(self, u_hat: TypeTensor) -> TypeTensor:

        """Computes vorticity of the given velocity field.

        Parameters
        ----------
        u_hat: torch.Tensor
            Velocity field in the Fourier domain.

        Returns
        -------
        TypeTensor
            Computed vorticity field in the Fourier domain.
        """

        raise NotImplementedError('BaseKolSol::vorticity()')

    def fourier_to_phys(self, t_hat: TypeTensor) -> TypeTensor:

        """Functional mapping from Fourier domain to physical domain.

        Parameters
        ----------
        t_hat: torch.Tensor
            Tensor in the Fourier domain to map to the physical domain.

        Returns
        -------
        TypeTensor
            Tensor in the physical domain.
        """

        raise NotImplementedError('BaseKolSol::fourier_to_phys()')

    def phys_to_fourier(self, t: TypeTensor) -> TypeTensor:

        """Functional mapping from the physical domain to the Fourier domain.

        Parameters
        ----------
        t: torch.Tensor
            Tensor in the physical domain to map to the Fourier domain.

        Returns
        -------
        TypeTensor
            Tensor in the Fourier domain.
        """

        raise NotImplementedError('BaseKolSol::phys_to_fourier()')

    def random_field(self, magnitude: float, sigma: float, k_offset: Optional[List[int]] = None) -> TypeTensor:

        """Generate random field in the Fourier domain based on k distribution.

        Parameters
        ----------
        magnitude: float
            Magnitude to use to compute distribution.
        sigma: float
            Sigma to use to compute distribution.
        k_offset: Optional[List[int]]
            Magnitude of wavenumber offset in each dimension.

        Returns
        -------
        TypeTensor
            Generated random field in the Fourier domain.
        """

        raise NotImplementedError('BaseKolSol::random_field()')
