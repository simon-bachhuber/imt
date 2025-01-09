"""
Heading correction methods from https://github.com/dlaidig/qmt
"""

import numpy as np
import qmt

from .._base import Method


class HeadCor(Method):
    def __init__(
        self,
        dof: int,
        axes_directions: np.ndarray | None,
        method_1d: str = "1d_corr",
        method_2d: str = "euler",
        use_mag_only_if_both: bool = False,
        offline_vqf: bool = False,
    ):
        """Uses `VQF` + `qmt.headingCorrection`

        Args:
            dof (int): The DOF of the joint between two segments
            axes_directions (np.ndarray | None): The joint axes directions of this joint
            method_1d (str, optional): Method to use for 1D joint. Defaults to "1d_corr"
            method_2d (str, optional): Method to use for 2D joint. Defaults to "euler".
            use_mag_only_if_both (bool, optional): Use magnetometer measurements for VQF
                only if both IMUs have magnetometer measurements. Defaults to False.
            offline_vqf (bool, optional): If `True` uses offlineVQF else onlineVQF.
                Defaults to False.

        Raises:
            Exception: If `dof` == 3
        """
        self.strict = use_mag_only_if_both
        self.offline_vqf = offline_vqf

        if axes_directions is not None:
            self.axes_directions = np.atleast_2d(axes_directions)
            self.axes_directions /= np.linalg.norm(
                self.axes_directions, axis=1, keepdims=True
            )
            assert dof == self.axes_directions.shape[0]
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

        Ts = self.getTs()
        _vqf = qmt.oriEstOfflineVQF if self.offline_vqf else qmt.oriEstVQF
        q1 = _vqf(gyr1, acc1, mag1, params=dict(Ts=Ts))
        q2 = _vqf(gyr2, acc2, mag2, params=dict(Ts=Ts))

        ts = np.arange(T * Ts, step=Ts)
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
        return qmt.qmult(qmt.qinv(q1), q2), {}

    def _process_mag(self, *mags):
        if self.strict and any([m is None for m in mags]):
            return len(mags) * [None]
        return mags
