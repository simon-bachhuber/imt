import copy
from pathlib import Path
import warnings

import numpy as np
import onnxruntime as ort
import qmt
import tree
from vqf import VQF


def _select_t(kwargs, t):
    def f(a):
        if a is None:
            return a
        return a[t]

    return tree.map_structure(f, kwargs)


class Solution:
    def apply(self, T: int | None, **kwargs):
        "Returns from child-to-body orientation"
        if T is None:
            return self._apply_timestep(**kwargs)

        quats = np.zeros((T, 4))
        for t in range(T):
            quats[t] = self._apply_timestep(**_select_t(kwargs, t))
        return quats

    def _apply_timestep(self, **kwargs):
        raise NotImplementedError

    def reset(self) -> None:
        "Guaranteed to be called before first usage of `apply`"
        pass

    def setTs(self, Ts: float) -> None:
        self.Ts = Ts

    def copy(self):
        return copy.deepcopy(self)


class ONNX_Solution(Solution):
    hidden_dim: int
    filename: str

    def __init__(self):
        self.session = ort.InferenceSession(
            Path(__file__).parent.joinpath(f"onnx/{self.filename}.onnx")
        )

    def reset(self):
        self.state = np.zeros((self.hidden_dim,), dtype=np.float32)

    @classmethod
    def copy(cls):
        return cls()


class Online_RelOri_1D2D3D_Solution(ONNX_Solution):
    hidden_dim = 2400
    filename = "relOri-1D2D3D-100Hz-v0"

    def _apply_timestep(self, acc1, acc2, gyr1, gyr2, mag1, mag2):
        qhat, self.state = self.session.run(
            None,
            {
                "acc1 (3,) [m/s^2]": acc1.astype(np.float32),
                "acc2 (3,) [m/s^2]": acc2.astype(np.float32),
                "gyr1 (3,) [rad/s]": gyr1.astype(np.float32),
                "gyr2 (3,) [rad/s]": gyr2.astype(np.float32),
                "previous_state (2400,); init with zeros": self.state,
            },
        )
        return qhat


class _VQF_Mixin_MagConsistency:
    "A mixin class to enforce consistency checks on magnetic data inputs"

    def __init__(self, strict: bool):
        self.strict = strict

    def _process_mag(self, *mags):
        if self.strict and any([m is None for m in mags]):
            return len(mags) * [None]
        return mags


class VQF_Solution(_VQF_Mixin_MagConsistency, Solution):
    def __init__(self, offline: bool = False, use_mag_only_if_both: bool = False):
        """
        Initializes a VQF Solution for a pair of bodies. 9D VQF is magnetic data is
        provided, else 6D VQF.

        Args:
            offline (bool): If True, enables offline mode, which performs orientation
                estimation for entire time-series data in a batch. If False, operates
                in online mode, processing data step by step.
            use_mag_only_if_both (bool): If True, magnetic data will only be used if
                magnetometer inputs are available for both the body and its parent. If
                False, uses alaways 9D VQF if magnetic data is provided.
        """
        super().__init__(use_mag_only_if_both)
        self.offline = offline

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
            duration = T * self.Ts
            if duration < warnlimit:
                warnings.warn(
                    "`offline` is enabled but timeseries is shorter "
                    f"than the warning limit, {duration}s < {warnlimit}s"
                )
            if acc1 is not None and gyr1 is not None:
                q1 = qmt.oriEstOfflineVQF(gyr1, acc1, mag1, params=dict(Ts=self.Ts))
            else:
                q1 = np.array([1.0, 0, 0, 0])
            q2 = qmt.oriEstOfflineVQF(gyr2, acc2, mag2, params=dict(Ts=self.Ts))
            return qmt.qmult(qmt.qinv(q1), q2)
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
        return qmt.qmult(qmt.qinv(q1), q2)

    @staticmethod
    def _update_and_get(vqf, a, g, m):
        vqf.update(g.copy(), a.copy(), m.copy() if m is not None else None)
        if m is None:
            return vqf.getQuat6D()
        else:
            return vqf.getQuat9D()

    def reset(self):
        self.vqf1 = VQF(self.Ts)
        self.vqf2 = VQF(self.Ts)


class QMT_HeadingConstraintSolution(_VQF_Mixin_MagConsistency, Solution):
    def __init__(
        self,
        dof: int,
        axes_directions: np.ndarray | None = None,
        method_1d: str = "1d_corr",
        method_2d: str = "gyro",
        use_mag_only_if_both: bool = False,
    ):
        super().__init__(use_mag_only_if_both)

        if axes_directions is not None:
            self.axes_directions = np.atleast_2d(axes_directions)
            self.axes_directions /= np.linalg.norm(
                self.axes_directions, axis=1, keepdims=True
            )
        else:
            self.axes_directions = None

        assert dof in [
            1,
            2,
        ], "Currently only 1D or 2D joints supported with this method"
        self.dof = dof

        if dof == 2 and axes_directions is None:
            raise Exception("For 2D joints this methods needs joint axes information")

        self.method_1d, self.method_2d = method_1d, method_2d

    def apply(self, acc1, acc2, gyr1, gyr2, mag1, mag2, T: int | None):
        if T is None:
            raise NotImplementedError(
                "qmt-based heading correction does not allow for online application; "
                "please time-batch your imu data"
            )

        if acc1 is None or gyr1 is None:
            raise Exception("Can not be used for bodies with parent=-1")

        mag1, mag2 = self._process_mag(mag1, mag2)

        q1 = qmt.oriEstOfflineVQF(gyr1, acc1, mag1, params=dict(Ts=self.Ts))
        q2 = qmt.oriEstOfflineVQF(gyr2, acc2, mag2, params=dict(Ts=self.Ts))

        ts = np.arange(T * self.Ts, step=self.Ts)
        if self.dof == 1:
            if self.axes_directions is None:
                axis_imu1, axis_imu2 = qmt.jointAxisEstHingeOlsson(
                    acc1,
                    acc2,
                    gyr1,
                    gyr2,
                    estSettings=dict(quiet=True),
                )
                axis = axis_imu1[:, 0]
            else:
                axis = self.axes_directions[0]

            q2 = qmt.headingCorrection(
                gyr1,
                gyr2,
                q1,
                q2,
                ts,
                axis,
                None,
                estSettings=dict(
                    constraint=self.method_1d, windowTime=min(8.0, ts[-1])
                ),
            )[0]
        else:
            q2 = qmt.headingCorrection(
                gyr1,
                gyr2,
                q1,
                q2,
                ts,
                self.axes_directions,
                None,
                estSettings=dict(
                    constraint=self.method_2d, windowTime=min(8.0, ts[-1])
                ),
            )[0]
        return qmt.qmult(qmt.qinv(q1), q2)
