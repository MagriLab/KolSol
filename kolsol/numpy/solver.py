from typing import Optional

import einops
import numpy as np

from kolsol.utils.enums import eDirection


class KolSol:

    def __init__(self, nk: int, nf: int, re: float, ndim: int = 2) -> None:

        """Kolmogorov Flow Solver Class.

        Implementation of a Fourier-Galerkin pseudospectral solver for the
        incompressible navier-stokes equations as described by Canuto.

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

        if not nk % 2 == 0:
            raise ValueError('Should pass an even number of wavenumbers, k.')

        self.nk = nk
        self.nf = nf
        self.re = re
        self.ndim = ndim

        self.nk_grid = 2 * self.nk + 1
        self.mk_grid = 4 * self.nk + 1

        x = np.linspace(0, 2.0 * np.pi, 2 * self.nk + 2)[:-1]
        self.xt = np.stack(np.meshgrid(*(x for _ in range(self.ndim)), indexing='xy'), axis=-1)

        k = np.fft.fftshift(np.fft.fftfreq(self.nk_grid, 1 / self.nk_grid))
        self.kt = np.stack(np.meshgrid(*(k for _ in range(self.ndim)), indexing='xy'), axis=-1)
        self.kk = np.sum(np.power(self.kt, 2), axis=-1)

        self.nabla = 1j * self.kt
        self.f = 1j * np.zeros((*(self.nk_grid for _ in range(self.ndim)), self.ndim))
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

        # TODO :: Add equation numbers for reference to the Canuto book.
        aapt = np.array([[self.aap(u_hat[..., u_j], u_hat[..., u_i])for u_j in range(self.ndim)] for u_i in range(self.ndim)])
        f_hat = np.einsum('...t, ut... -> ...u', -self.nabla, aapt)

        k_dot_f = np.einsum('...u, ...u -> ...', self.kt, f_hat + self.f) / self.kk
        k_dot_f[tuple(self.nk for _ in range(self.ndim))] = 0.0

        kk_fk = self.kt * einops.repeat(k_dot_f, '... -> ... b', b=self.ndim)

        du_hat_dt = (f_hat + self.f) - kk_fk - (1.0 / self.re) * einops.repeat(self.kk, '... -> ... b', b=self.ndim) * u_hat
        return du_hat_dt

    def aap(self, f1: np.ndarray, f2: np.ndarray) -> np.ndarray:

        """Anti-aliased product using padding.

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

        lb, ub = self.nk, 3 * self.nk + 1
        scaling = (self.mk_grid / self.nk_grid) ** self.ndim

        fhat = np.stack((f1, f2), axis=-1)
        fhat_padded = np.zeros(([self.mk_grid for _ in range(self.ndim)] + [2]), dtype=np.complex128)
        fhat_padded[tuple(slice(lb, ub) for _ in range(self.ndim))] = fhat

        f_phys = np.fft.irfftn(
            np.fft.ifftshift(fhat_padded, axes=range(self.ndim)),
            s=fhat_padded.shape[:self.ndim],
            axes=range(self.ndim)
        )

        f1f2_hat_padded = np.fft.fftn(
            np.prod(f_phys, axis=-1),
            s=f_phys.shape[:self.ndim],
            axes=range(self.ndim)
        )

        f1f2_hat_padded = scaling * np.fft.fftshift(f1f2_hat_padded, axes=range(self.ndim))
        f1f2_hat = f1f2_hat_padded[tuple(slice(lb, ub) for _ in range(self.ndim))]

        return f1f2_hat

    def dissip(self, u_hat: np.ndarray) -> np.ndarray:

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
        dissipation = np.sum(w_hat * w_hat.conjugate())
        dissipation = np.squeeze(dissipation) / self.re / self.nk_grid ** 4
        dissipation = dissipation.real

        return dissipation

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

        return np.cross(self.nabla, u_hat)

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
        np.ndarray
            Field tensor in the physical domain.
        """

        if not nref:
            t_hat_aug = t_hat
            scaling = 1.0
        else:
            ishift = (nref - 2 * self.nk) // 2
            scaling = (nref / self.nk_grid) ** self.ndim

            t_hat_aug = np.zeros(([nref for _ in range(self.ndim)] + [self.ndim]), dtype=np.complex128)
            t_hat_aug[tuple(slice(ishift, ishift + self.nk_grid) for _ in range(self.ndim)) + tuple([...])] = t_hat

        return scaling * np.fft.irfftn(np.fft.ifftshift(t_hat_aug, axes=range(self.ndim)), s=t_hat_aug.shape[:self.ndim], axes=range(self.ndim))

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

        if not t.ndim == self.ndim + 1:
            raise ValueError('Invalid dimensions...')

        nref = t.shape[0]
        ishift = (nref - 2 * self.nk) // 2
        scaling = (self.nk_grid / nref) ** self.ndim

        t_hat_padded = scaling * np.fft.fftshift(np.fft.fftn(t, s=t.shape[:self.ndim], axes=range(self.ndim)), axes=range(self.ndim))
        t_hat = t_hat_padded[tuple(slice(ishift, ishift + self.nk_grid) for _ in range(self.ndim)) + tuple([...])]

        return t_hat
