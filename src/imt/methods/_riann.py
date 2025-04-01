"""RIANN method from publication https://www.mdpi.com/2673-2688/2/3/28"""

import numpy as np
import qmt
from riann.riann import RIANN as _RIANN

from .._base import Method


class RIANN(Method):
    def __init__(self):
        self._riann1 = _RIANN()
        self._riann2 = _RIANN()

    @classmethod
    def copy(cls):
        return cls()

    def reset(self):
        Ts = self.getTs()
        for riann in [self._riann1, self._riann2]:
            riann.set_sampling_rate(1 / Ts)
            riann.reset_state()

    def apply(self, T: int | None, acc1, acc2, gyr1, gyr2, mag1, mag2):
        riann_method = "predict_step" if T is None else "predict"

        if acc1 is not None and gyr1 is not None:
            q1 = getattr(self._riann1, riann_method)(acc1, gyr1)
        else:
            q1 = np.array([1.0, 0, 0, 0])

        q2 = getattr(self._riann2, riann_method)(acc2, gyr2)

        return qmt.qmult(qmt.qinv(q1), q2), {}
