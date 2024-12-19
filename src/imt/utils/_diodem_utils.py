import math
from typing import Optional, TypeVar
import warnings

import numpy as np
from qmt import nanInterp
from qmt import quatInterp
from qmt import vecInterp
from scipy.interpolate import CubicSpline
import tree

PyTree = TypeVar("PyTree")


def crop_tail(
    signal: PyTree,
    hz: Optional[int | float | PyTree] = None,
    strict: bool = True,
    verbose: bool = True,
):
    "Crop all signals to length of shortest signal."
    verbose_msg_index = False
    if hz is None:
        hz = 1.0
        verbose_msg_index = True

    if isinstance(hz, (int, float)):
        hz = tree.map_structure(lambda _: hz, signal)

    # int -> float
    hz = tree.map_structure(float, hz)

    def length_in_seconds(arr, hz):
        assert arr.ndim < 3
        return len(arr) / hz

    signal_lengths_seconds = tree.map_structure(length_in_seconds, signal, hz)
    shortest_length_seconds = min(tree.flatten(signal_lengths_seconds))
    hz_of_shortest_length = tree.flatten(hz)[
        np.argmin(tree.flatten(signal_lengths_seconds))
    ]

    if strict:
        # reduce shortest_length until it becomes a clearn crop for all other
        # frequencies
        flat_hz = tree.flatten(hz)
        dt = 1 / hz_of_shortest_length
        iter = 100
        for _ in range(iter):
            for each_hz in flat_hz:
                if (round(shortest_length_seconds * each_hz, 10) % 1) != 0.0:
                    break  # cleancrop not possible
            else:
                break  # cleancrop possible
            shortest_length_seconds -= dt
        else:
            warnings.warn(
                "No cleancrop possible, tried to reduce shortest signal by "
                f"{iter * dt} seconds"
            )

    if verbose:
        if verbose_msg_index:
            print(
                f"`crop_tail`: Crop off at index i="
                f"{int(shortest_length_seconds * hz_of_shortest_length)}"
            )
        else:
            print(f"`crop_tail`: Crop off at t={shortest_length_seconds}s")

    def crop(arr, hz):
        if strict:
            crop_tail = np.round(shortest_length_seconds * hz, decimals=10)
            err_msg = (
                "No clean crop possible: shortest_length_seconds="
                + f"{shortest_length_seconds}; hz={hz}"
            )
            assert (crop_tail % 1) == 0.0, err_msg
            crop_tail = int(crop_tail)
        else:
            crop_tail = math.ceil(shortest_length_seconds * hz)
        return arr[:crop_tail]

    return tree.map_structure(crop, signal, hz)


def hz_helper(
    segments: list[str],
    imus: list[str] = ["imu_rigid", "imu_flex"],
    markers: list[int] = [1, 2, 3, 4],
    hz_imu: float = 40.0,
    hz_omc: float = 120.0,
):
    hz_in = {}
    imu_dict = dict(acc=hz_imu, mag=hz_imu, gyr=hz_imu)
    for seg in segments:
        hz_in[seg] = {}
        for imu in imus:
            hz_in[seg][imu] = imu_dict
        for marker in markers:
            hz_in[seg][f"marker{marker}"] = hz_omc
        hz_in[seg]["quat"] = hz_omc

    return hz_in


def resample(
    signal: PyTree,
    hz_in: int | float | PyTree,
    hz_out: int | float | PyTree,
    quatdetect: bool = True,
    vecinterp_method: str = "linear",
) -> PyTree:
    # int -> float
    hz_in, hz_out = tree.map_structure(float, (hz_in, hz_out))

    if isinstance(hz_in, float):
        hz_in = tree.map_structure(lambda _: hz_in, signal)
    if isinstance(hz_out, float):
        hz_out = tree.map_structure(lambda _: hz_out, signal)

    def resample_array(signal: np.ndarray, hz_in, hz_out):
        is1D = False
        if signal.ndim == 1:
            is1D = True
            signal = signal[:, None]
        assert signal.ndim == 2

        N = signal.shape[0]
        ts_out = np.arange(N, step=hz_in / hz_out)
        signal = nanInterp(signal)
        if quatdetect and signal.shape[1] == 4:
            signal = quatInterp(signal, ts_out)
        else:
            if vecinterp_method == "linear":
                signal = vecInterp(signal, ts_out)
            elif vecinterp_method == "cubic":
                signal = _cubic_interpolation(signal, ts_out)
            else:
                raise NotImplementedError(
                    "`vecinterp_method` must be one of ['linear', 'cubic']"
                )
        if is1D:
            signal = signal[:, 0]
        return signal

    return tree.map_structure(resample_array, signal, hz_in, hz_out)


def _cubic_interpolation(signal: np.ndarray, ts_out: np.ndarray):
    ts_in = np.arange(len(signal))
    interp_1D = lambda arr: (CubicSpline(ts_in, arr)(ts_out))
    return np.array([interp_1D(signal[:, i]) for i in range(signal.shape[1])]).T
