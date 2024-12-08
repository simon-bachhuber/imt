import qmt

from ._method import MethodWrapper


class CalibrateMag(MethodWrapper):
    def apply(self, T, acc1, acc2, gyr1, gyr2, mag1, mag2):
        assert T is not None, "`CalibrateMag` requires offline application"

        if mag1 is not None:
            mag1 = _calibrate_mag(gyr1, acc1, mag1, self.unwrapped.Ts)
        if mag2 is not None:
            mag2 = _calibrate_mag(gyr2, acc2, mag2, self.unwrapped.Ts)

        return super().apply(
            T=T, acc1=acc1, acc2=acc2, gyr1=gyr1, gyr2=gyr2, mag1=mag1, mag2=mag2
        )


def _calibrate_mag(gyr, acc, mag, Ts):
    gain, bias, *_ = qmt.calibrateMagnetometerSimple(gyr, acc, mag, Ts)
    return gain * mag - bias
