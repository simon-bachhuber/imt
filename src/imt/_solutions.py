from pathlib import Path

import numpy as np
import onnxruntime as ort
from vqf import VQF


class Solution:
    def apply(self, *args):
        pass

    def reset(self) -> None:
        pass


class ONNX_Solution(Solution):
    hidden_dim: int
    filename: str

    def __init__(self):
        self.session = ort.InferenceSession(
            Path(__file__).parent.joinpath(f"onnx/{self.filename}.onnx")
        )

    def apply(self, *args):
        pass

    def reset(self):
        self.state = np.zeros((self.hidden_dim,), dtype=np.float32)


class Online_RelOri_1D2D3D(ONNX_Solution):
    hidden_dim = 2400
    filename = "relOri-1D2D3D-100Hz-v0"

    def apply(self, acc1, acc2, gyr1, gyr2, **kwargs):
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


class VQF_Solution(Solution):
    def __init__(self, Ts: float):
        self.vqf = VQF(Ts)

    def apply(self, acc2, gyr2, mag2, **kwargs):
        self.vqf.update(gyr2, acc2, mag2)
        if mag2 is None:
            return self.vqf.getQuat6D()
        else:
            return self.vqf.getQuat9D()

    def reset(self):
        self.vqf.resetState()
