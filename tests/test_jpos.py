from diodem import load_data
from jpos import solve

import imt


def test_jpos():
    data = load_data(1, motion_start="slow1")
    acc1, gyr1, acc2, gyr2 = (
        data["seg1"]["imu_rigid"]["acc"],
        data["seg1"]["imu_rigid"]["gyr"],
        data["seg2"]["imu_rigid"]["acc"],
        data["seg2"]["imu_rigid"]["gyr"],
    )
    hz = 100
    r1, r2, _ = solve(acc1, gyr1, acc2, gyr2, hz=hz, phi=None, order=0, verbose=True)

    _, extras = imt.Solver(
        [-1, 0],
        [
            imt.methods.NoOpMethod(),
            imt.wrappers.JointPosition(imt.methods.NoOpMethod(), verbose=True),
        ],
        1 / hz,
    ).step({0: dict(acc=acc1, gyr=gyr1), 1: dict(acc=acc2, gyr=gyr2)})

    print(r1, r2)
    print(extras)


if __name__ == "__main__":
    test_jpos()
