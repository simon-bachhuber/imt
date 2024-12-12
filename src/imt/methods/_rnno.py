"""
RNNO method from
1) https://ieeexplore.ieee.org/document/9841375, and
2) https://ieeexplore.ieee.org/document/10225275
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort

from .._base import Method


class _ONNX(Method):
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


class RNNO_rO(_ONNX):
    "https://wandb.ai/simipixel/RING_2D/runs/y0umf1ho/overview"
    hidden_dim = 2400
    filename = "rnno-rO-100Hz-v0"

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
        return qhat, {}

    def reset(self):
        super().reset()
        assert (
            self.getTs() == 0.01
        ), "Currently `RNNO_rO` only supports 100Hz; Resample using eg `qmt.nanInterp`"


class RNNO(_ONNX):
    "https://wandb.ai/simipixel/RING_2D/runs/b0ga1rx9/overview"
    hidden_dim = 2400
    filename = "rnno-100Hz-v0"

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
        return qhat[1], {"incl-body1-to-eps": qhat[0]}

    def reset(self):
        super().reset()
        assert (
            self.getTs() == 0.01
        ), "Currently `RNNO` only supports 100Hz; Resample using eg `qmt.nanInterp`"
