"""
RING method from https://openreview.net/forum?id=h2C3rkn0zR
"""

import warnings

import numpy as np
import qmt

from .._base import Method


class RING_2S(Method):
    N = 2
    F = 6

    def apply(self, T, acc1, acc2, gyr1, gyr2, mag1, mag2):
        X = np.zeros((1 if T is None else T, self.N, self.F))
        assert acc1 is not None, "`RING_2S` only supports relative orientation"

        X[:, 0, :3] = acc1 / 9.81
        X[:, -1, :3] = acc2 / 9.81
        X[:, 0, 3:6] = gyr1 / 2.2
        X[:, -1, 3:6] = gyr2 / 2.2

        qhat, self.state = self.ringnet.apply(X=X, state=self.state)
        qhat = qhat[:, 1]

        if T is None:
            return qhat[0], {}
        return qhat, {}

    def setTs(self, Ts):
        assert Ts == 0.01, (
            "For sampling rates != 100Hz, "
            + "please use `imt.wrappers.FractualStepping(imt.methods.RING())` instead"
        )
        return super().setTs(Ts)

    def reset(self):
        import ring

        self.ringnet = ring.ml.RING(
            params="/Users/simon/Downloads/0xd87430067580b75.pickle",
            lam=(-1, 0),
            hidden_state_dim=600,
            stack_rnn_cells=2,
            message_dim=400,
            send_message_n_layers=0,
            layernorm=True,
        )
        X = np.zeros((1, self.N, self.F))
        _, self.state = self.ringnet.init(X=X)


class RING(Method):
    def __init__(
        self,
        dof: int = 1,
        axes_directions: np.ndarray | None = None,
    ):
        self.dof = dof

        if dof > 1:
            warnings.warn(
                "Setting `dof` > 1 is an experimental feature that will likely produces"
                + " incorrect results; you might be better of setting to `dof`=1 even"
                + " for higher DOF joints"
            )

        if axes_directions is not None:
            assert self.dof != 6
            self.axes_directions = np.atleast_2d(axes_directions).astype(np.float64)
            assert self.axes_directions.shape[0] == dof
        else:
            self.axes_directions = None

        self.N = {6: 1, 1: 2, 2: 3, 3: 4}[dof]

    def apply(self, T, acc1, acc2, gyr1, gyr2, mag1, mag2):
        X = np.zeros((1 if T is None else T, self.N, 9))
        if acc1 is None and gyr1 is None:
            assert self.dof == 6
            X[:, 0, :3] = acc2
            X[:, 0, 3:6] = gyr2
        else:
            X[:, 0, :3] = acc1
            X[:, -1, :3] = acc2
            X[:, 0, 3:6] = gyr1
            X[:, -1, 3:6] = gyr2
            if self.axes_directions is not None:
                for i in range(self.dof):
                    X[:, i + 1, 6:9] = self.axes_directions[i]

        qhat, self.state = self.ringnet.apply(X=X, state=self.state)

        if self.N == 1:
            qhat = qmt.qinv(qhat[:, 0])
        elif self.N == 2:
            qhat = qhat[:, 1]
        elif self.N == 3:
            qhat = qmt.qmult(qhat[:, 1], qhat[:, 2])
        else:
            qhat = qmt.qmult(qhat[:, 1], qmt.qmult(qhat[:, 2], qhat[:, 3]))

        if T is None:
            return qhat[0], {}
        return qhat, {}

    def reset(self):
        import ring

        self.ringnet = ring.RING(list(range(-1, self.N - 1)), self.getTs(), jit=True)
        X = np.zeros((1, self.N, 9))
        _, self.state = self.ringnet.init(X=X)

    def copy(self):
        return RING(self.dof, self.axes_directions)
