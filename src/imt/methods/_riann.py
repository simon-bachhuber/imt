"""RIANN method from publication https://www.mdpi.com/2673-2688/2/3/28"""

from pathlib import Path

import numpy as np
import onnxruntime as rt
import qmt

from .._base import Method

_onnx_path = Path(__file__).parent.joinpath("onnx/riann.onnx")


class RIANN(Method):
    def __init__(self):

        self.session = rt.InferenceSession(_onnx_path)
        self.h0 = np.zeros((2, 1, 1, 200), dtype=np.float32)

    def predict(self, acc, gyr, fs):
        """
        Update plot with external x-values.
        Parameters
        ----------
        acc: numpy-array [sequence_length x 3]
            Acceleration data of the IMU. The axis order is x,y,z.
        gyr: numpy-array [sequence_length x 3]
            Gyroscope data of the IMU. The axis order is x,y,z.
        fs: float
            Samplingrate of the provided IMU data

        Returns
        -------
        attitude unit-quaternions [sequence_length x 4]
        """
        # prepare minibatch for runtime execution
        np_inp = np.concatenate(
            [acc, gyr, np.tile(1 / fs, (acc.shape[0], 1))], axis=-1
        ).astype(np.float32)[None, ...]

        return self.session.run([], {"input": np_inp, "h0": self.h0})[0][0]

    def copy(self):
        return self

    def apply(self, T: int | float, acc1, acc2, gyr1, gyr2, mag1, mag2):
        if T is None:
            raise Exception("`RIANN` does not support `online` application")

        Ts = self.getTs()
        if acc1 is not None and gyr1 is not None:
            q1 = self.predict(acc1, gyr1, 1 / Ts)
        else:
            q1 = np.array([1.0, 0, 0, 0])
        q2 = self.predict(acc2, gyr2, 1 / Ts)
        return qmt.qmult(qmt.qinv(q1), q2), {}
