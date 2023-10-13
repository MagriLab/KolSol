import itertools as it
import string
from typing import List, Optional, Union

import einops
import numpy as np
import opt_einsum as oe
import torch

from ..base.base_solver import BaseTorchKolSol
from ..utils.enums import eDirection


class KolSol(BaseTorchKolSol):

    def __init__(self,
                 nk: int,
                 nf: int,
                 re: float,
                 ndim: int = 2,
                 device: Union[torch.device, str] = torch.device('cpu')) -> None:

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

        self.device = device

        x = torch.linspace(0.0, 2.0 * np.pi, self.nk_grid + 1)[:-1].to(self.device)
        self.xt = torch.stack(torch.meshgrid(*(x for _ in range(self.ndim)), indexing='ij'), dim=-1)

        k = torch.fft.fftshift(torch.fft.fftfreq(self.nk_grid, 1 / self.nk_grid)).to(self.device)
        self.kt = torch.stack(torch.meshgrid(*(k for _ in range(self.ndim)), indexing='ij'), dim=-1)
        self.kk = torch.sum(torch.pow(self.kt, 2), dim=-1)

        self.kk_div = torch.sum(torch.pow(self.kt, 2), dim=-1)
        self.kk_div[tuple(self.nk for _ in range(self.ndim)) + tuple([...])] = 1.0

        self.nabla = 1j * self.kt

        self.f = torch.zeros(
            (*(self.nk_grid for _ in range(self.ndim)), self.ndim), dtype=torch.complex128
        ).to(self.device)

        self.f[..., eDirection.i] = torch.fft.fftshift(torch.fft.fftn(torch.sin(self.nf * self.xt[..., eDirection.j])))

        # converting relevant attributes to complex
        self.xt = self.xt.to(torch.complex128)

        self.kt = self.kt.to(torch.complex128)
        self.kk = self.kk.to(torch.complex128)
        self.kk_div = self.kk_div.to(torch.complex128)

        self.nabla = self.nabla.to(torch.complex128)

    def dynamics(self, u_hat: torch.Tensor) -> torch.Tensor:

        """Calculate time-derivative of the velocity field.

        Parameters
        ----------
        u_hat: torch.Tensor
            Velocity field in the Fourier domain.

        Returns
        -------
        du_hat_dt: torch.Tensor
            Time-derivative of the velocity field in the Fourier domain.
        """

        uij_aapt = []
        for u_i in range(self.ndim):

            uj_aapt = []
            for u_j in range(self.ndim):
                uj_aapt.append(self.aap(u_hat[..., u_j], u_hat[..., u_i]))

            uij_aapt.append(torch.stack(uj_aapt, dim=0))

        # Canuto EQ [7.2.12]
        aapt = torch.stack(uij_aapt, dim=0)
        f_hat = oe.contract('...t, ut... -> ...u', -self.nabla, aapt)

        k_dot_f = oe.contract('...u, ...u -> ...', self.kt, f_hat + self.f) / self.kk_div
        k_dot_f[tuple([...]) + tuple(self.nk for _ in range(self.ndim))] = 0.0

        kk_fk = oe.contract('...u, ... -> ...u', self.kt, k_dot_f)

        # Canuto EQ [7.2.11]
        du_hat_dt = (f_hat + self.f) - kk_fk - (1.0 / self.re) * oe.contract('...u, ... -> ...u', u_hat, self.kk)

        return du_hat_dt

    def aap(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:

        """Anti-aliased product using padding.

        See Canuto, CH [3.2.2].

        Parameters
        ----------
        f1: torch.Tensor
            Array one.
        f2: torch.Tensor
            Array two.

        Returns
        -------
        f1f2_hat: torch.Tensor
            Anti-aliased product of the two arrays.
        """

        n_leading_dims = f1.ndim - self.ndim
        leading_dims = f1.shape[:n_leading_dims]

        lb, ub = self.nk, 3 * self.nk
        scaling = (self.mk_grid / self.nk_grid) ** self.ndim

        fhat = torch.stack((f1, f2), dim=-1)

        fhat_padded = torch.zeros(
            size=([*leading_dims] + [self.mk_grid for _ in range(self.ndim)] + [2]),
            dtype=torch.complex128,
            device=self.device
        )

        fhat_padded[tuple([...]) + tuple(slice(lb, ub) for _ in range(self.ndim)) + tuple([slice(None)])] = fhat

        # define axes to work between
        axs_lb, axs_ub = n_leading_dims, self.ndim + n_leading_dims

        f_phys = torch.fft.irfftn(
            torch.fft.ifftshift(fhat_padded, dim=tuple(range(axs_lb, axs_ub))),
            s=fhat_padded.shape[slice(axs_lb, axs_ub)],
            dim=tuple(range(axs_lb, axs_ub))
        )

        f1f2_hat_padded = torch.fft.fftn(
            torch.prod(f_phys, dim=-1),
            s=f_phys.shape[slice(axs_lb, axs_ub)],
            dim=tuple(range(axs_lb, axs_ub))
        )

        f1f2_hat_padded = scaling * torch.fft.fftshift(f1f2_hat_padded, dim=tuple(range(axs_lb, axs_ub)))
        f1f2_hat = f1f2_hat_padded[tuple([...]) + tuple(slice(lb, ub) for _ in range(self.ndim))]

        return f1f2_hat

    def dissip(self, u_hat: torch.Tensor) -> torch.Tensor:

        """Calculate dissipation of the velocity field.

       Parameters
       ----------
       u_hat: np.ndarray
           Velocity field in the Fourier domain.

       Returns
       -------
       dissipation: np.ndarray
           Dissipation of the velocity field.
       """

        w_hat = self.vorticity(u_hat)

        # generate indices for each dimension
        g = iter(string.ascii_letters)
        idx = ''.join(next(g) for _ in range(self.ndim))

        dissipation = oe.contract(f'...{idx} -> ...', w_hat * torch.conj(w_hat))
        dissipation /= (self.re * self.nk_grid ** (2 * self.ndim))

        return dissipation.real

    def vorticity(self, u_hat: torch.Tensor) -> torch.Tensor:

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

        if self.ndim == 2:

            x_dy = self.nabla[..., eDirection.j] * u_hat[..., eDirection.i]
            y_dx = self.nabla[..., eDirection.i] * u_hat[..., eDirection.j]

            return x_dy - y_dx

        raise ValueError('Vorticity not currently implemented correctly for ndim=3')

    def pressure(self, u_hat: torch.Tensor) -> torch.Tensor:

        """Calculates the pressure field.

        Parameters
        ----------
        u_hat: torch.Tensor
            Velocity field to calculate pressure field from.

        Returns
        -------
        p_hat: torch.Tensor
            Pressure field in the Fourier domain.
        """

        # Canuto EQ [7.2.12]
        uij_aapt = []
        for u_i in range(self.ndim):

            uj_aapt = []
            for u_j in range(self.ndim):
                uj_aapt.append(self.aap(u_hat[..., u_j], u_hat[..., u_i]))

            uij_aapt.append(torch.stack(uj_aapt, dim=0))

        # Canuto EQ [7.2.12]
        aapt = torch.stack(uij_aapt, dim=0)
        f_hat = oe.contract('...t, ut... -> ...u', -self.nabla, aapt)

        p_hat = oe.contract('...u, ...u -> ...', -self.nabla, f_hat) / self.kk_div
        p_hat[tuple([...]) + tuple(self.nk for _ in range(self.ndim))] = 0.0

        return p_hat

    def fourier_to_phys(self, t_hat: torch.Tensor, nref: Optional[int] = None) -> torch.Tensor:

        """Transform Fourier domain to physical domain.

        Parameters
        ----------
        t_hat: torch.Tensor
            Field tensor in the Fourier domain.
        nref: Optional[int]
            Number of reference points in the physical domain.

        Returns
        -------
        phys: torch.Tensor
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

            t_hat_aug = torch.zeros(
                size=([*leading_dims] + [nref for _ in range(self.ndim)] + [t_hat.shape[-1]]),
                dtype=torch.complex128,
                device=self.device
            )

            ishift_slice = [slice(ishift, ishift + self.nk_grid) for _ in range(self.ndim)]
            t_hat_aug[tuple([...]) + tuple(ishift_slice) + tuple([slice(None)])] = t_hat

        # define axes to work between
        axs_lb, axs_ub = n_leading_dims, n_leading_dims + self.ndim

        phys = scaling * torch.fft.irfftn(
            torch.fft.ifftshift(
                t_hat_aug, dim=tuple(range(axs_lb, axs_ub))
            ), s=t_hat_aug.shape[slice(axs_lb, axs_ub)], dim=tuple(range(axs_lb, axs_ub))
        )

        return phys

    def phys_to_fourier(self, t: torch.Tensor) -> torch.Tensor:

        """Transform physical domain to Fourier domain.

        Parameters
        ----------
        t: torch.Tensor
            Field tensor in the physical domain.

        Returns
        -------
        t_hat: torch.Tensor
            Field tensor in the Fourier domain.
        """

        n_leading_dims = t.ndim - (self.ndim + 1)

        nref = t.shape[n_leading_dims]
        ishift = (nref - 2 * self.nk) // 2
        scaling = (self.nk_grid / nref) ** self.ndim

        # define axes to work between
        axs_lb, axs_ub = n_leading_dims, n_leading_dims + self.ndim

        t_hat_padded = scaling * torch.fft.fftshift(
            torch.fft.fftn(
                t, s=t.shape[slice(axs_lb, axs_ub)], dim=tuple(range(axs_lb, axs_ub))
            ), dim=tuple(range(axs_lb, axs_ub))
        )

        ishift_slice = [slice(ishift, ishift + self.nk_grid) for _ in range(self.ndim)]
        t_hat = t_hat_padded[tuple([...]) + tuple(ishift_slice) + tuple([slice(None)])]

        return t_hat

    def random_field(self, magnitude: float, sigma: float, k_offset: Optional[List[int]] = None) -> torch.Tensor:

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
        torch.Tensor
            Random field in the Fourier domain.
        """

        if k_offset and len(k_offset) != self.ndim:
            raise ValueError('Must provide offsets for each dimension.')

        random_field = torch.from_numpy(
            np.random.uniform(size=(*(self.nk_grid for _ in range(self.ndim)), self.ndim))
        ).to(self.device)

        delta = torch.zeros_like(random_field)

        if k_offset:
            for idx, k in enumerate(k_offset):
                delta[..., idx] -= k

        mag = torch.exp(-0.5 * (torch.sqrt(torch.sum(torch.square(self.kt - delta), dim=-1)) / sigma ** 2))
        mag = einops.repeat(mag, '... -> ... b', b=self.ndim)
        mag = magnitude * mag / np.sqrt(2.0 * np.pi * sigma ** 2)

        u_hat = mag * torch.exp(2.0j * np.pi * random_field)

        u_hat = torch.fft.irfftn(
            torch.fft.ifftshift(
                u_hat, dim=tuple(range(self.ndim))
            ), s=u_hat.shape[:self.ndim], dim=tuple(range(self.ndim))
        )

        u_hat = torch.fft.fftshift(
            torch.fft.fftn(
                u_hat, s=u_hat.shape[:self.ndim], dim=tuple(range(self.ndim))
            ), dim=tuple(range(self.ndim))
        )

        return u_hat

    def energy_spectrum(self, u_hat: torch.Tensor, agg: bool = False) -> torch.Tensor:

        """Calculates energy spectrum of flow field.

        Parameters
        ----------
        u_hat: torch.Tensor
            Flow-field to calculate energy spectrum of.
        agg: bool
            Determines whether to take a mean.

        Returns
        -------
        ek: torch.Tensor
            Energy spectrum of given field.
        """

        n_leading_dims = u_hat.ndim - (self.ndim + 1)
        leading_dims = u_hat.shape[:n_leading_dims]

        uu = 0.5 * oe.contract('...u -> ...', u_hat * torch.conj(u_hat)).real
        intk = torch.sqrt(self.kk).to(torch.int)

        ek = torch.zeros((tuple([*leading_dims]) + tuple([torch.max(intk) + 1])), device=self.device)
        for combo in it.product(*map(range, intk.shape)):
            ek[tuple([...]) + tuple([intk[combo]])] += uu[tuple([...]) + tuple(combo)] / self.nk_grid ** self.ndim

        if agg:
            ek = einops.reduce(ek, '... k -> k', torch.mean)

        return ek
