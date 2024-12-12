from typing import Optional

import numpy as np
import qmt
from vqf import PyVQF

from .._base import Method
from .._base import MethodWrapper


class LPF(MethodWrapper):
    def __init__(
        self,
        method: Method,
        acc_f_cutoff: Optional[float] = None,
        gyr_f_cutoff: Optional[float] = None,
        mag_f_cutoff: Optional[float] = None,
        quat_f_cutoff: Optional[float] = None,
    ):
        super().__init__(method)
        self.acc_f_cutoff = acc_f_cutoff
        self.gyr_f_cutoff = gyr_f_cutoff
        self.mag_f_cutoff = mag_f_cutoff
        self.quat_f_cutoff = quat_f_cutoff

    def _setup(self, suffix: str, f: float | None):
        name = f"_lpfs_{suffix}"
        _LPF = _Quat_LPF if suffix == "quat" else _Vec_LPF
        if f is None:
            setattr(self, name, [])
        else:
            N = 1 if suffix == "quat" else 2
            Ts = self.getTs()
            setattr(self, name, [_LPF(f, Ts) for _ in range(N)])

    def apply(self, T, acc1, acc2, gyr1, gyr2, mag1, mag2):
        if self.acc_f_cutoff is not None:
            acc1 = self._lpfs_acc[0].step(acc1)
            acc2 = self._lpfs_acc[1].step(acc2)
        if self.gyr_f_cutoff is not None:
            gyr1 = self._lpfs_gyr[0].step(gyr1)
            gyr2 = self._lpfs_gyr[1].step(gyr2)
        if self.mag_f_cutoff is not None:
            if mag1 is not None:
                mag1 = self._lpfs_mag[0].step(mag1)
            if mag2 is not None:
                mag2 = self._lpfs_mag[1].step(mag2)

        qhat, extras = super().apply(
            T, acc1=acc1, acc2=acc2, gyr1=gyr1, gyr2=gyr2, mag1=mag1, mag2=mag2
        )

        if self.quat_f_cutoff is not None:
            qhat = self._lpfs_quat[0].step(qhat)
        return qhat, extras

    def reset(self):
        self._setup("acc", self.acc_f_cutoff)
        self._setup("gyr", self.gyr_f_cutoff)
        self._setup("mag", self.mag_f_cutoff)
        self._setup("quat", self.quat_f_cutoff)
        return super().reset()


class _Mixin:

    def step(self, xq: np.ndarray):
        if xq.ndim == 1:
            return self._step(xq)
        xq_filtered = np.zeros_like(xq)
        for t in range(xq.shape[0]):
            xq_filtered[t] = self._step(xq[t])
        return xq_filtered


class _Quat_LPF(_Mixin):
    def __init__(self, f_cutoff: float, Ts: float):
        tau = 1 / (2 * np.pi * f_cutoff)
        self.k = PyVQF.gainFromTau(tau, Ts)
        self.reset()

    def reset(self):
        self.state = None

    def _step(self, q: np.ndarray):
        if self.state is None:
            self.state = q
            return q
        q_err = qmt.qrel(self.state, q)
        angle, axis = qmt.quatAngle(q_err), qmt.quatAxis(q_err)
        q_err = qmt.quatFromAngleAxis(angle * self.k, axis)
        self.state = qmt.qmult(self.state, q_err)
        return self.state


class _Vec_LPF(_Mixin):

    def __init__(self, f_cutoff: float, Ts: float):
        self.tau = np.sqrt(2) / (2 * np.pi * f_cutoff)
        self.Ts = Ts
        self.b, self.a = PyVQF.filterCoeffs(self.tau, self.Ts)
        self.reset()

    def reset(self):
        self.state = None

    def _step(self, x: np.ndarray):
        if self.state is None:
            self.state = PyVQF.filterInitialState(x, self.b, self.a)
        return PyVQF.filterVec(x, self.tau, self.Ts, self.b, self.a, self.state)
