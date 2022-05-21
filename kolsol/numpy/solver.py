import itertools as it
import string
from typing import List, Optional

import einops
import numpy as np
import opt_einsum as oe

from ..base.base_solver import BaseKolSol
from ..utils.enums import eDirection


class KolSol(BaseKolSol):

    def __init__(self, nk: int, nf: int, re: float, ndim: int = 2) -> None:

        """Kolmogorov Flow Solver Class.

        Implementation of a Fourier-Galerkin pseudospectral solver for the
        incompressible navier-stokes equations as described by Canuto, CH [7.2].

        Parameters
        ----------
        nk: int
            Number of symmetric wavenumbers to use.
        nf: int
            Prescribed forcing frequency.
        re: float
            Reynolds number of the flow.
        ndim: int, default=2
            Number of dimensions to solve for.
        """

        super().__init__(nk, nf, re, ndim)

        x = np.linspace(0.0, 2.0 * np.pi, 2 * self.nk + 2)[:-1]
        self.xt = np.stack(np.meshgrid(*(x for _ in range(self.ndim)), indexing='ij'), axis=-1)

        k = np.fft.fftshift(np.fft.fftfreq(self.nk_grid, 1 / self.nk_grid))
        self.kt = np.stack(np.meshgrid(*(k for _ in range(self.ndim)), indexing='ij'), axis=-1)
        self.kk = np.sum(np.power(self.kt, 2), axis=-1)

        self.nabla = 1j * self.kt
        self.f = np.zeros((*(self.nk_grid for _ in range(self.ndim)), self.ndim), dtype=np.complex128)
        self.f[..., eDirection.i] = np.fft.fftshift(np.fft.fftn(np.sin(self.nf * self.xt[..., eDirection.j])))

    def dynamics(self, u_hat: np.ndarray) -> np.ndarray:

        """Calculate time-derivative of the velocity field.

        Parameters
        ----------
        u_hat: np.ndarray
            Velocity field in the Fourier domain.

        Returns
        -------
        du_hat_dt: np.ndarray
            Time-derivative of the velocity field in the Fourier domain.
        """

        # Canuto EQ [7.2.12]
        aapt = np.array([
            [self.aap(u_hat[..., u_j], u_hat[..., u_i]) for u_j in range(self.ndim)] for u_i in range(self.ndim)
        ])

        f_hat = oe.contract('...t, ut... -> ...u', -self.nabla, aapt)

        with np.errstate(divide='ignore', invalid='ignore'):
            k_dot_f = oe.contract('...u, ...u -> ...', self.kt, f_hat + self.f) / self.kk
            k_dot_f[tuple([...]) + tuple(self.nk for _ in range(self.ndim))] = 0.0

        kk_fk = oe.contract('...u, ... -> ...u', self.kt, k_dot_f)

        # Canuto EQ [7.2.11]
        du_hat_dt = (f_hat + self.f) - kk_fk - (1.0 / self.re) * oe.contract('...u, ... -> ...u', u_hat, self.kk)

        return du_hat_dt

    def aap(self, f1: np.ndarray, f2: np.ndarray) -> np.ndarray:

        """Anti-aliased product using padding.

        See Canuto, CH [3.2.2].

        Parameters
        ----------
        f1: np.ndarray
            Array one.
        f2: np.ndarray
            Array two.

        Returns
        -------
        f1f2_hat: np.ndarray
            Anti-aliased product of the two arrays.
        """

        n_leading_dims = f1.ndim - self.ndim
        leading_dims = f1.shape[:n_leading_dims]

        lb, ub = self.nk, 3 * self.nk + 1
        scaling = (self.mk_grid / self.nk_grid) ** self.ndim

        fhat = np.stack((f1, f2), axis=-1)

        fhat_padded = np.zeros(([*leading_dims] + [self.mk_grid for _ in range(self.ndim)] + [2]), dtype=np.complex128)
        fhat_padded[tuple([...]) + tuple(slice(lb, ub) for _ in range(self.ndim)) + tuple([slice(None)])] = fhat

        # define axes to work between
        axs_lb, axs_ub = n_leading_dims, self.ndim + n_leading_dims

        f_phys = np.fft.irfftn(
            np.fft.ifftshift(fhat_padded, axes=range(axs_lb, axs_ub)),
            s=fhat_padded.shape[slice(axs_lb, axs_ub)],
            axes=range(axs_lb, axs_ub)
        )

        f1f2_hat_padded = np.fft.fftn(
            np.prod(f_phys, axis=-1),
            s=f_phys.shape[slice(axs_lb, axs_ub)],
            axes=range(axs_lb, axs_ub)
        )

        f1f2_hat_padded = scaling * np.fft.fftshift(f1f2_hat_padded, axes=range(axs_lb, axs_ub))
        f1f2_hat = f1f2_hat_padded[tuple([...]) + tuple(slice(lb, ub) for _ in range(self.ndim))]

        return f1f2_hat

    def dissip(self, u_hat: np.ndarray) -> np.ndarray:

        """Calculate dissipation of the velocity field.

        Parameters
        ----------
        u_hat: np.ndarray
            Velocity field in the Fourier domain.

        Returns
        -------
        np.ndarray
            Dissipation of the velocity field.
        """

        w_hat = self.vorticity(u_hat)

        # generate indices for each dimension
        g = iter(string.ascii_letters)
        idx = ''.join(next(g) for _ in range(self.ndim))

        dissipation = oe.contract(f'...{idx} -> ...',  w_hat * np.conj(w_hat))
        dissipation /= (self.re * self.nk_grid ** (2 * self.ndim))

        return dissipation.real

    def vorticity(self, u_hat: np.ndarray) -> np.ndarray:

        """Calculate the vorticity of the flow field.

        Parameters
        ----------
        u_hat: np.ndarray
            Velocity field in the Fourier domain.

        Returns
        -------
        np.ndarray
            Vorticity of the flow field.
        """

        if self.ndim == 3:
            raise ValueError('Vorticity not currently implemented correctly for ndim=3')

        return np.cross(self.nabla, u_hat)

    def pressure(self, u_hat: np.ndarray) -> np.ndarray:

        """Calculates the pressure field.

        Parameters
        ----------
        u_hat: np.ndarray
            Velocity field to calculate pressure field from.

        Returns
        -------
        p_hat: np.ndarray
            Pressure field in the Fourier domain.
        """

        # Canuto EQ [7.2.12]
        aapt = np.array([
            [self.aap(u_hat[..., u_j], u_hat[..., u_i]) for u_j in range(self.ndim)] for u_i in range(self.ndim)
        ])

        f_hat = oe.contract('...t, ut... -> ...u', -self.nabla, aapt)

        with np.errstate(divide='ignore', invalid='ignore'):
            p_hat = oe.contract('...u, ...u -> ...', -self.nabla, f_hat) / self.kk
            p_hat[tuple([...]) + tuple(self.nk for _ in range(self.ndim))] = 0.0

        return p_hat

    def fourier_to_phys(self, t_hat: np.ndarray, nref: Optional[int] = None) -> np.ndarray:

        """Transform Fourier domain to physical domain.

        Parameters
        ----------
        t_hat: np.ndarray
            Field tensor in the Fourier domain.
        nref: Optional[int]
            Number of reference points in the physical domain.

        Returns
        -------
        phys: np.ndarray
            Field tensor in the physical domain.
        """

        if not len(t_hat.shape) > self.ndim:
            raise ValueError('Please ensure that a field of the correct shape is passed in.')

        n_leading_dims = t_hat.ndim - (self.ndim + 1)
        leading_dims = t_hat.shape[:n_leading_dims]

        if not nref:
            t_hat_aug = t_hat
            scaling = 1.0
        else:
            ishift = (nref - 2 * self.nk) // 2
            scaling = (nref / self.nk_grid) ** self.ndim

            t_hat_aug = np.zeros(
                ([*leading_dims] + [nref for _ in range(self.ndim)] + [t_hat.shape[-1]]), dtype=np.complex128
            )

            ishift_slice = [slice(ishift, ishift + self.nk_grid) for _ in range(self.ndim)]
            t_hat_aug[tuple([...]) + tuple(ishift_slice) + tuple([slice(None)])] = t_hat

        # define axes to work between
        axs_lb, axs_ub = n_leading_dims, n_leading_dims + self.ndim

        phys = scaling * np.fft.irfftn(
            np.fft.ifftshift(t_hat_aug, axes=range(axs_lb, axs_ub)),
            s=t_hat_aug.shape[slice(axs_lb, axs_ub)],
            axes=range(axs_lb, axs_ub)
        )

        return phys

    def phys_to_fourier(self, t: np.ndarray) -> np.ndarray:

        """Transform physical domain to Fourier domain.

        Parameters
        ----------
        t: np.ndarray
            Field tensor in the physical domain.

        Returns
        -------
        t_hat: np.ndarray
            Field tensor in the Fourier domain.
        """

        n_leading_dims = t.ndim - (self.ndim + 1)

        nref = t.shape[n_leading_dims]
        ishift = (nref - 2 * self.nk) // 2
        scaling = (self.nk_grid / nref) ** self.ndim

        axs_lb, axs_ub = n_leading_dims, n_leading_dims + self.ndim
        t_hat_padded = scaling * np.fft.fftshift(
            np.fft.fftn(t, s=t.shape[slice(axs_lb, axs_ub)], axes=range(axs_lb, axs_ub)),
            axes=range(axs_lb, axs_ub)
        )

        ishift_slice = [slice(ishift, ishift + self.nk_grid) for _ in range(self.ndim)]
        t_hat = t_hat_padded[tuple([...]) + tuple(ishift_slice) + tuple([slice(None)])]

        return t_hat

    def random_field(self, magnitude: float, sigma: float, k_offset: Optional[List[int]] = None) -> np.array:

        """Generate random velocity field in the Fourier domain.

        Parameters
        ----------
        magnitude: float
            Magnitude of the field.
        sigma: float
            Standard deviation of the field.
        k_offset: Optional[List[int]]
            Wavenumber offsets for each dimension.

        Returns
        -------
        u_hat: np.ndarray
            Random field in the Fourier domain.
        """

        if k_offset and len(k_offset) != self.ndim:
            raise ValueError('Must provide offsets for each dimension.')

        random_field = np.random.uniform(size=(*(self.nk_grid for _ in range(self.ndim)), self.ndim))
        delta = np.zeros_like(random_field)

        if k_offset:
            for idx, k in enumerate(k_offset):
                delta[..., idx] -= k

        mag = np.exp(-0.5 * (np.sqrt(np.sum(np.square(self.kt - delta), axis=-1)) / sigma ** 2))
        mag = einops.repeat(mag, '... -> ... b', b=self.ndim)
        mag = magnitude * mag / np.sqrt(2.0 * np.pi * sigma ** 2)

        u_hat = mag * np.exp(2.0j * np.pi * random_field)

        u_hat = np.fft.irfftn(
            np.fft.ifftshift(
                u_hat, axes=range(self.ndim)
            ), s=u_hat.shape[:self.ndim], axes=range(self.ndim)
        )

        u_hat = np.fft.fftshift(
            np.fft.fftn(
                u_hat, s=u_hat.shape[:self.ndim], axes=range(self.ndim)
            ), axes=range(self.ndim)
        )

        return u_hat

    def energy_spectrum(self, u_hat: np.ndarray, agg: bool = False) -> np.ndarray:

        """Calculates energy spectrum of flow field.

        Parameters
        ----------
        u_hat: np.ndarray
            Flow-field to calculate energy spectrum of.
        agg: bool
            Determines whether to take a mean.

        Returns
        -------
        ek: np.ndarray
            Energy spectrum of given field.
        """

        n_leading_dims = u_hat.ndim - (self.ndim + 1)
        leading_dims = u_hat.shape[:n_leading_dims]

        uu = 0.5 * oe.contract('...u -> ...', u_hat * np.conj(u_hat)).real
        intk = np.sqrt(self.kk).astype(int)

        ek = np.zeros((tuple([*leading_dims]) + tuple([np.max(intk) + 1])))
        for combo in it.product(*map(range, intk.shape)):
            ek[tuple([...]) + tuple([intk[combo]])] += uu[tuple([...]) + tuple(combo)] / self.nk_grid ** self.ndim

        if agg:
            ek = einops.reduce(ek, '... k -> k', np.mean)

        return ek
