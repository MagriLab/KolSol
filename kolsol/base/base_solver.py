from ..utils.types import TypeTensor
from typing import List, Optional


class BaseKolSol:

    def __init__(self, nk: int, nf: int, re: float, ndim: int = 2) -> None:

        self.nk = nk
        self.nf = nf
        self.re = re
        self.ndim = ndim

        self.nk_grid = 2 * self.nk + 1
        self.mk_grid = 4 * self.nk + 1

    def dynamics(self, u_hat: TypeTensor) -> TypeTensor:
        raise NotImplementedError('BaseKolSol::dynamics()')

    def aap(self, f1: TypeTensor, f2: TypeTensor) -> TypeTensor:
        raise NotImplementedError('BaseKolSol::aap()')

    def dissip(self, u_hat: TypeTensor) -> TypeTensor:
        raise NotImplementedError('BaseKolSol::dissip()')

    def vorticity(self, u_hat: TypeTensor) -> TypeTensor:
        raise NotImplementedError('BaseKolSol::vorticity()')

    def fourier_to_phys(self, u_hat: TypeTensor) -> TypeTensor:
        raise NotImplementedError('BaseKolSol::fourier_to_phys()')

    def phys_to_fourier(self, u: TypeTensor) -> TypeTensor:
        raise NotImplementedError('BaseKolSol::phys_to_fourier()')

    def random_field(self, magnitude: float, sigma: float, k_offset: Optional[List[int]] = None) -> TypeTensor:
        raise NotImplementedError('BaseKolSol::random_field()')
