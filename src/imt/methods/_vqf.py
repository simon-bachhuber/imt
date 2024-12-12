"""
VQF method from https://www.sciencedirect.com/science/article/pii/S156625352200183X
"""

import warnings

import numpy as np
import qmt
from vqf import VQF as _VQF

from .._base import Method


class VQF(Method):
    def __init__(self, offline: bool = False, use_mag_only_if_both: bool = False):
        """
        Initializes a VQF Method for a pair of bodies. 9D VQF is magnetic data is
        provided, else 6D VQF.

        Args:
            offline (bool): If True, enables offline mode, which performs orientation
                estimation for entire time-series data in a batch. If False, operates
                in online mode, processing data step by step.
            use_mag_only_if_both (bool): If True, magnetic data will only be used if
                magnetometer inputs are available for both the body and its parent. If
                False, uses alaways 9D VQF if magnetic data is provided.
        """
        self.offline = offline
        self.strict = use_mag_only_if_both

    def apply(self, T: int | float, acc1, acc2, gyr1, gyr2, mag1, mag2):
        mag1, mag2 = self._process_mag(mag1, mag2)

        if T is None:
            if self.offline:
                warnings.warn(
                    "`offline` is enabled but no time-series provided, "
                    "switching to `offline`=`False`"
                )
            return self._apply_timestep(acc1, acc2, gyr1, gyr2, mag1, mag2)

        if self.offline:
            warnlimit = 5
            Ts = self.getTs()
            duration = T * Ts
            if duration < warnlimit:
                warnings.warn(
                    "`offline` is enabled but timeseries is shorter "
                    f"than the warning limit, {duration}s < {warnlimit}s"
                )
            if acc1 is not None and gyr1 is not None:
                q1 = qmt.oriEstOfflineVQF(gyr1, acc1, mag1, params=dict(Ts=Ts))
            else:
                q1 = np.array([1.0, 0, 0, 0])
            q2 = qmt.oriEstOfflineVQF(gyr2, acc2, mag2, params=dict(Ts=Ts))
            return qmt.qmult(qmt.qinv(q1), q2), {}
        else:
            return super().apply(
                T=T, acc1=acc1, acc2=acc2, gyr1=gyr1, gyr2=gyr2, mag1=mag1, mag2=mag2
            )

    def _apply_timestep(self, acc1, acc2, gyr1, gyr2, mag1, mag2):
        if acc1 is not None and gyr1 is not None:
            q1 = self._update_and_get(self.vqf1, acc1, gyr1, mag1)
        else:
            q1 = np.array([1.0, 0, 0, 0])
        q2 = self._update_and_get(self.vqf2, acc2, gyr2, mag2)
        return qmt.qmult(qmt.qinv(q1), q2), {}

    @staticmethod
    def _update_and_get(vqf, a, g, m):
        vqf.update(g.copy(), a.copy(), m.copy() if m is not None else None)
        if m is None:
            return vqf.getQuat6D()
        else:
            return vqf.getQuat9D()

    def reset(self):
        Ts = self.getTs()
        self.vqf1 = _VQF(Ts)
        self.vqf2 = _VQF(Ts)

    def _process_mag(self, *mags):
        if self.strict and any([m is None for m in mags]):
            return len(mags) * [None]
        return mags
