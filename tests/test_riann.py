import numpy as np

import imt


def test_riann():
    np.random.seed(1)
    acc1, acc2, gyr1, gyr2 = np.random.normal(size=(4, 100, 3))

    solver = imt.Solver([-1, 0], methods=2 * [imt.methods.RIANN()], Ts=0.01)

    qhats = []
    for t in range(100):
        qhat, _ = solver.step(
            {0: dict(acc=acc1[t], gyr=gyr1[t]), 1: dict(acc=acc2[t], gyr=gyr2[t])}
        )
        qhats.append(qhat[1])

    qhats1 = np.stack(qhats)

    solver.reset()
    qhats = []
    for t in range(100):
        qhat, _ = solver.step(
            {0: dict(acc=acc1[t], gyr=gyr1[t]), 1: dict(acc=acc2[t], gyr=gyr2[t])}
        )
        qhats.append(qhat[1])
    qhats2 = np.stack(qhats)

    solver.reset()
    qhats, _ = solver.step({0: dict(acc=acc1, gyr=gyr1), 1: dict(acc=acc2, gyr=gyr2)})
    qhats3 = qhats[1]

    np.testing.assert_allclose(qhats1, qhats2)
    np.testing.assert_allclose(qhats1, qhats3)
