"""
Heading correction methods from https://github.com/dlaidig/qmt
"""

from typing import Optional

import numpy as np
import qmt

from .._base import Method
from ..wrappers._jpos import _compute_j1_j2


class HeadCor(Method):
    def __init__(
        self,
        dof: int,
        axes_directions: np.ndarray | None,
        use_mag_only_if_both: bool = False,
        offline_vqf: bool = False,
        heading_correction_estSettings: Optional[dict] = None,
    ):
        """Uses `VQF` + `qmt.headingCorrection`

        Args:
            dof (int): The DOF of the joint between two segments
            axes_directions (np.ndarray | None): The joint axes directions of this joint
            use_mag_only_if_both (bool, optional): Use magnetometer measurements for VQF
                only if both IMUs have magnetometer measurements. Defaults to False.
            offline_vqf (bool, optional): If `True` uses offlineVQF else onlineVQF.
                Defaults to False.
            heading_correction_estSettings(dict, optional): `estSettings` dictionary
                that is forward to `qmt.headingCorrection`; refer to docstring there
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

        self.dof = dof
        assert dof in [1, 2, 3]

        if dof == 2 and axes_directions is None:
            raise Exception("For 2D joints this methods needs joint axes information")

        default_constraint = {1: "1d_corr", 2: "euler", 3: "conn"}
        self._heading_correction_estSettings = {
            # set default constraint depending on `dof`
            "constraint": default_constraint[dof]
        }
        if heading_correction_estSettings is not None:
            self._heading_correction_estSettings.update(
                heading_correction_estSettings.copy()
            )

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

        ts = np.linspace(0, (T - 1) * Ts, T)

        estSettings = self._heading_correction_estSettings.copy()
        if "windowTime" not in estSettings:
            estSettings["windowTime"] = min(8.0, ts[-1])

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
                acc1,
                acc2,
                q1,
                q2,
                ts,
                axis,
                None,
                estSettings=estSettings,
            )[0]
        elif self.dof == 2:
            q2 = qmt.headingCorrection(
                gyr1,
                gyr2,
                acc1,
                acc2,
                q1,
                q2,
                ts,
                self.axes_directions,
                None,
                estSettings=estSettings,
            )[0]
        elif self.dof == 3:
            r1, r2 = _compute_j1_j2(Ts, False, False, acc1, acc2, gyr1, gyr2)
            q2 = qmt.headingCorrection(
                gyr1,
                gyr2,
                acc1,
                acc2,
                q1,
                q2,
                ts,
                # we only do this such that internally `headingCorrection` sets `dof`
                # to 3; only the shape of the array is used in this case
                np.ones((3, 3)),
                {"r1": -r1, "r2": -r2},  # minus to convert from jc-to-IMU to IMU-to-jc
                estSettings=estSettings,
            )[0]
        else:
            raise NotImplementedError

        return qmt.qmult(qmt.qinv(q1), q2), {}

    def _process_mag(self, *mags):
        if self.strict and any([m is None for m in mags]):
            return len(mags) * [None]
        return mags
