"""
RING method from https://openreview.net/forum?id=h2C3rkn0zR
"""

import numpy as np
import qmt

from .._base import Method


class RING(Method):
    def __init__(
        self,
        dof: int = 1,
        axes_directions: np.ndarray | None = None,
    ):
        self.dof = dof

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
