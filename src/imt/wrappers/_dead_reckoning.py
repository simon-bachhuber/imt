import numpy as np
import qmt
from vqf import offlineVQF

from .._base import Method
from .._base import MethodWrapper


def _find_false_sequences(boolean_array):
    false_sequences = []
    in_sequence = False
    start_index = -1

    for i, value in enumerate(boolean_array):
        if not value and not in_sequence:
            # Start of a new sequence
            start_index = i
            in_sequence = True
        elif value and in_sequence:
            # End of a sequence
            false_sequences.append((start_index, i - 1))
            in_sequence = False

    # Handle the case where the array ends with a sequence of 0s
    if in_sequence:
        false_sequences.append((start_index, len(boolean_array) - 1))

    return false_sequences


class DeadReckoning(MethodWrapper):
    def __init__(self, method: Method, zvu: bool = False):
        super().__init__(method)
        self.zvu = zvu

    def apply(self, T, **kwargs):
        assert T is not None, "`DeadReckoning` requires offline application"
        qhat, extras = super().apply(T, **kwargs)

        gyr, acc, mag = kwargs["gyr2"], kwargs["acc2"], kwargs["mag2"]
        Ts = self.getTs()
        out = offlineVQF(
            gyr.copy(), acc.copy(), mag if mag is None else mag.copy(), Ts, dict()
        )
        quat = out["quat6D"] if mag is None else out["quat9D"]
        restDetected = out["restDetected"]

        # if there is rest set velocity to zero, in-between rest phases
        # perform zero velocity updates
        acc = qmt.rotate(quat, acc) - np.array([0, 0, 9.81])

        if self.zvu:
            motion_phases = _find_false_sequences(restDetected)
            for start, stop in motion_phases:
                r = slice(max(start - 1, 0), stop + 1)
                acc[r] = acc[r] - np.mean(acc[r], axis=0)

        vel = np.cumsum(acc, axis=0) * Ts
        vel[restDetected] = 0.0
        pos = np.cumsum(vel, axis=0) * Ts
        extras["dead-reckoning-position"] = pos
        extras["restDetected"] = restDetected
        return qhat, extras
