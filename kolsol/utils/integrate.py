from typing import Callable

from .types import TypeTensor


def euler_step(fn: Callable[[TypeTensor], TypeTensor], dt: float) -> Callable[[TypeTensor], TypeTensor]:

    """Calculate integration step using Euler's method.

    Parameters
    ----------
    fn: Callable[[TypeTensor], TypeTensor]
        Function to calculate the derivative.
    dt: float
        Time-step size.

    Returns
    -------
    Callable[[TypeTensor], TypeTensor]
        Function to calculate integration step.
    """

    def _euler(field: TypeTensor) -> TypeTensor:
        return dt * fn(field)

    return _euler


def rk4_step(fn: Callable[[TypeTensor], TypeTensor], dt: float) -> Callable[[TypeTensor], TypeTensor]:

    """Calculate integration step using RK4 method.

    Parameters
    ----------
    fn: Callable[[TypeTensor], TypeTensor]
        Function to calculate the derivative.
    dt: float
        Time-step size.

    Returns
    -------
    Callable[[TypeTensor], TypeTensor]
        Function to calculate integration step.
    """

    def _rk4(field: TypeTensor) -> TypeTensor:

        k1 = fn(field)
        k2 = fn(field + 0.5 * k1 * dt)
        k3 = fn(field + 0.5 * k2 * dt)
        k4 = fn(field + 1.0 * k3 * dt)

        return dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    return _rk4
