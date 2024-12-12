from typing import Optional
import warnings

import numpy as np
import qmt
from scipy.optimize import minimize
from scipy.signal import butter
from scipy.signal import sosfiltfilt

from .._base import Method
from .._base import MethodWrapper


def _constraint(acc1, gyr1, gyrdot1, r1, acc2, gyr2, gyrdot2, r2):
    def gamma(gyr, gyrdot, r):
        return np.cross(gyr, np.cross(gyr, r)) + np.cross(gyrdot, r)

    return np.linalg.norm(acc1 - gamma(gyr1, gyrdot1, r1), axis=-1) - np.linalg.norm(
        acc2 - gamma(gyr2, gyrdot2, r2), axis=-1
    )


def _lpf(x, hz, cutoff):
    return np.stack(
        [
            sosfiltfilt(butter(4, cutoff, output="sos", fs=hz), x[:, i])
            for i in range(x.shape[-1])
        ]
    ).T


def _dot(x, hz: float, lpf_freq: float | None = 10.0):
    if lpf_freq is not None:
        x = _lpf(x, hz, lpf_freq)
    xdot = (x[2:] - x[:-2]) / (2 * (1 / hz))
    return np.vstack((xdot[0][None], xdot, xdot[-1][None]))


def _jpos_solve(
    acc1: np.ndarray,
    gyr1: np.ndarray,
    acc2: np.ndarray,
    gyr2: np.ndarray,
    hz: float,
    verbose: bool = False,
    seed: Optional[int] = None,
    opt_kwargs: dict = dict(method="BFGS"),
):
    """Joint Translation Estimation. Estimates the vector from joint center to IMU 1
    and IMU2.
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6402423

    Args:
        acc1 (np.ndarray): Nx3, m/s**2, with or without gravity
        gyr1 (np.ndarray): Nx3, rad/s
        acc2 (np.ndarray): Nx3, m/s**2 with or without gravity
        gyr2 (np.ndarray): Nx3, rad/s
        hz (float): Sampling rate, Hz
        verbose (bool, optional): Print information to stdout. Defaults to False.
        seed (int, optional): Seed used for initilization of the optimization. By
            default this function is non-deterministic. Fixing the seed makes this
            function deterministic.
        opt_kwargs (dict, optional): Kwargs for `scipy.optimize.minimze`

    Returns:
        tuple: Array (3,), Array (3,), dict
        which are joint-to-imu1-vector, joint-to-imu2-vector, and additional infos
    """
    if seed is not None:
        np.random.seed(seed)

    T = acc1.shape[0]
    for arr in [acc1, gyr1, acc2, gyr2]:
        assert arr.shape == (
            T,
            3,
        ), (
            "All IMU data must be given as a Nx3 array with consistent number of "
            + f"samples `N` but found {arr.shape}"
        )

    for arr in [gyr1, gyr2]:
        max_val = np.max(np.abs(arr))
        if max_val > 10:
            warnings.warn(
                f"Found very large gyroscope or phi value of {max_val}. Are you sure "
                "you have provided Gyroscope and `phi` value in radians?"
            )

    acc1 = _lpf(acc1, hz, 25 if hz > 50 else 15)
    acc2 = _lpf(acc2, hz, 25 if hz > 50 else 15)

    gyrdot1 = _dot(gyr1, hz)
    gyrdot2 = _dot(gyr2, hz)

    def residual(x, acc1, gyr1, gyrdot1, acc2, gyr2, gyrdot2):
        r1 = x[:3]
        r2 = x[3:]

        return _constraint(acc1, gyr1, gyrdot1, r1, acc2, gyr2, gyrdot2, r2)

    def mean_squared_residual(x):
        e = residual(
            x,
            acc1,
            gyr1,
            gyrdot1,
            acc2,
            gyr2,
            gyrdot2,
        )
        return np.mean(e**2)

    initial_params = np.random.normal(size=(6,)) * 0.2
    res = minimize(mean_squared_residual, initial_params, **opt_kwargs)
    final_residual = mean_squared_residual(res.x)

    if verbose:
        print(f"Final residual={final_residual} m/s**2")

    return (
        res.x[:3],
        res.x[3:],
        {"final residual m/s**2": final_residual, "scipy_minimize_result": res},
    )


class JointPosition(MethodWrapper):
    def __init__(self, method: Method, dof_is_1d: bool = False, verbose: bool = False):
        super().__init__(method)
        self.dof_is_1d = dof_is_1d
        self.verbose = verbose

    def apply(self, T, acc1, acc2, gyr1, gyr2, mag1, mag2):
        qhat, extras = super().apply(
            T=T, acc1=acc1, acc2=acc2, gyr1=gyr1, gyr2=gyr2, mag1=mag1, mag2=mag2
        )

        n_initival_opt_values = 3
        j1 = j2 = None
        res = 1e16
        for _ in range(n_initival_opt_values):
            _j1, _j2, infos = _jpos_solve(
                acc1,
                gyr1,
                acc2,
                gyr2,
                1 / self.getTs(),
                verbose=self.verbose,
            )
            _res = infos["final residual m/s**2"]

            if _res < res:
                j1, j2 = _j1, _j2

        if self.dof_is_1d:
            axis_imu1, axis_imu2 = qmt.jointAxisEstHingeOlsson(
                acc1,
                acc2,
                gyr1,
                gyr2,
                estSettings=dict(quiet=True),
            )
            j1 = self._project_out_axis(j1, axis_imu1[:, 0])
            j2 = self._project_out_axis(j2, axis_imu2[:, 0])

        extras.update(
            {
                "joint-center-to-body1": j1,
                "joint-center-to-body2": j2,
            }
        )
        return qhat, extras

    @staticmethod
    def _project_out_axis(r, axis):
        return r - axis * (axis @ r)
