import numpy as np
import qmt

from .._base import MethodWrapper


class JointTracker1D(MethodWrapper):
    def apply(self, T, **kwargs):
        assert T is not None, "`JointTracker1D` requires offline application"
        # `qhat` is from 2-to-1
        qhat, extras = super().apply(T, **kwargs)
        q_cor, j2 = self._q_cor(
            kwargs["acc1"], kwargs["acc2"], kwargs["gyr1"], kwargs["gyr2"]
        )
        # compute 2-to-1'
        qhat_cor = qmt.qmult(q_cor, qhat)
        # project onto axis
        joint_angle = qmt.quatProject(qhat_cor, j2)["projAngle"]
        # the joint angle value always starts at zero with no wrapping effects
        joint_angle = np.unwrap(joint_angle)
        joint_angle = joint_angle - joint_angle[0]

        extras["joint_axis_direction"] = j2
        extras["joint_angle_rad"] = joint_angle
        return qhat, extras

    @staticmethod
    def _q_cor(acc1, acc2, gyr1, gyr2):
        "`q_cor` rotates from 1-to-1' such that 1'-to-2 is around `j2`"
        axis_imu1, axis_imu2 = qmt.jointAxisEstHingeOlsson(
            acc1,
            acc2,
            gyr1,
            gyr2,
            estSettings=dict(quiet=True),
        )
        j1, j2 = axis_imu1[:, 0], axis_imu2[:, 0]
        q_cor = _quat_rotate_x1_to_x2(j1, j2)
        return q_cor, j2


def _quat_rotate_x1_to_x2(x1, x2):
    "Determines `q` such that x2 = rotate(q, x1)"
    return qmt.quatFromAngleAxis(np.arccos(x1 @ x2), np.cross(x1, x2))
