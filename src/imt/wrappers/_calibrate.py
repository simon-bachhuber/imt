import qmt

from .._base import MethodWrapper
from ._joint_tracker import JointTracker1D


class CalibrateMag(MethodWrapper):
    def apply(self, T, acc1, acc2, gyr1, gyr2, mag1, mag2):
        assert T is not None, "`CalibrateMag` requires offline application"

        Ts = self.getTs()
        if mag1 is not None:
            mag1 = _calibrate_mag(gyr1, acc1, mag1, Ts)
        if mag2 is not None:
            mag2 = _calibrate_mag(gyr2, acc2, mag2, Ts)

        return super().apply(
            T=T, acc1=acc1, acc2=acc2, gyr1=gyr1, gyr2=gyr2, mag1=mag1, mag2=mag2
        )


def _calibrate_mag(gyr, acc, mag, Ts):
    gain, bias, *_ = qmt.calibrateMagnetometerSimple(gyr, acc, mag, Ts)
    return gain * mag - bias


class SenToSeg1DCal(MethodWrapper):
    def apply(self, T, acc1, acc2, gyr1, gyr2, mag1, mag2):
        assert T is not None, "`SenToSeg1DCal` requires offline application"
        # `q_cor` is rotation from 1 to 1'
        q_cor, j2 = JointTracker1D._q_cor(acc1, acc2, gyr1, gyr2)
        acc1 = qmt.rotate(q_cor, acc1)
        gyr1 = qmt.rotate(q_cor, gyr1)
        mag1 = qmt.rotate(q_cor, mag1) if mag1 is not None else mag1
        qhat, extras = super().apply(
            T=T, acc1=acc1, acc2=acc2, gyr1=gyr1, gyr2=gyr2, mag1=mag1, mag2=mag2
        )

        # undo the correction, we have estimated now 2-to-1', we want 2-to-1
        qhat = qmt.qmult(qmt.qinv(q_cor), qhat)

        extras["sen_to_seg_1d_cal_joint_axis_direction"] = j2
        return qhat, extras
