import argparse
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import tqdm
from kolsol.numpy.solver import KolSol


def setup_directory(path: Path) -> None:

    """Sets up the relevant simulation directory.

    Parameters
    ----------
    path: Path
        Path to .h5 file to write results to.
    """

    if not path.suffix == '.h5':
        raise ValueError('setup_directory() :: Must pass .h5 path.')

    if path.exists():
        raise FileExistsError(f'setup_directory() :: {path} already exists.')

    path.parent.mkdir(parents=True, exist_ok=True)


def write_h5(path: Path, data: Dict[str, Any]) -> None:

    """Writes results dictionary to .h5 file.

    Parameters
    ----------
    path: Path
        Corresponding .h5 file to write results to.
    data: Dict[str, Any]
        Data to write to file.
    """

    hf = h5py.File(path, 'w')

    for k, v in data.items():
        hf.create_dataset(k, data=v)

    hf.close()


def main(args: argparse.Namespace) -> None:

    """Run Kolmogorov flow solver.

    Parameters
    ----------
    args: argparse.Namespace
        Arguments from the command line.
    """

    np.seterr(divide='ignore')

    print('Initialising Kolmogorov Flow Solver...')

    setup_directory(args.data_path)

    ks = KolSol(nk=args.nk, nf=args.nf, re=args.re, ndim=args.ndim)
    field_hat = ks.random_field(magnitude=1.0, sigma=2.0)

    # define time-arrays for simulation run
    t_arange = np.arange(0.0, args.time_simulation, args.dt)
    transients_arange = np.arange(0.0, args.time_transient, args.dt)

    nt = t_arange.shape[0]
    nt_transients = transients_arange.shape[0]

    # setup recording arrays
    velocity_arr = np.zeros(shape=(nt, args.resolution, args.resolution, args.ndim))
    dissipation_arr = np.zeros(shape=(nt, 1))

    # integrate over transients
    msg = '01 :: Integrating over transients.'
    for _ in tqdm.trange(nt_transients, desc=msg):
        field_hat += args.dt * ks.dynamics(field_hat)

    # integrate over simulation domain
    msg = '02 :: Integrating over simulation domain.'
    for t in tqdm.trange(nt, desc=msg):

        field_hat += args.dt * ks.dynamics(field_hat)

        velocity_arr[t, ...] = ks.fourier_to_phys(field_hat, nref=args.resolution)
        dissipation_arr[t, ...] = ks.dissip(field_hat)

    data_dict = {
        're': args.re,
        'dt': args.dt,
        'nk': args.nk,
        'nf': args.nf,
        'ndim': args.ndim,
        'time': t_arange,
        'resolution': args.resolution,
        'velocity_field': velocity_arr,
        'dissipation': dissipation_arr
    }

    print('03 :: Writing results to file.')
    write_h5(args.data_path, data_dict)

    print('Simulation Done.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate Kolmogorov flow data')

    # arguments to define output
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--resolution', type=int, required=True)

    # arguments to define simulation
    parser.add_argument('--re', type=float, required=True)
    parser.add_argument('--dt', type=float, default=0.01)

    parser.add_argument('--time-simulation', type=float, required=True)
    parser.add_argument('--time-transient', type=float, default=180.0)

    # arguments for kolsol
    parser.add_argument('--nk', type=int, default=8)
    parser.add_argument('--nf', type=int, default=4)
    parser.add_argument('--ndim', type=int, default=2)

    parsed_args = parser.parse_args()

    main(parsed_args)

