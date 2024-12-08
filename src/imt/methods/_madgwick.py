import numpy as np
import qmt

from ._method import Method


class Madgwick(Method):
    def __init__(self, use_mag_only_if_both: bool = False):
        self.strict = use_mag_only_if_both

    def apply(self, T: int | float, acc1, acc2, gyr1, gyr2, mag1, mag2):
        mag1, mag2 = self._process_mag(mag1, mag2)

        assert T is not None, "`Madgwick` only supports `offline` application"

        if acc1 is not None and gyr1 is not None:
            q1 = qmt.oriEstMadgwick(gyr1, acc1, mag1, params=dict(Ts=self.Ts))
        else:
            q1 = np.array([1.0, 0, 0, 0])
        q2 = qmt.oriEstMadgwick(gyr2, acc2, mag2, params=dict(Ts=self.Ts))
        return qmt.qmult(qmt.qinv(q1), q2), {}

    def _process_mag(self, *mags):
        if self.strict and any([m is None for m in mags]):
            return len(mags) * [None]
        return mags
