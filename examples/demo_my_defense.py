import time

import jax
import jax.numpy as jnp
from movella_dot import MovellaDotIMUs
import numpy as np
import qmt
import ring
from ring import System
from ring.extras.interactive_viewer import InteractiveViewer
from ring.rendering import base_render

import imt
from imt.wrappers._lpf import _Quat_LPF

sys_str_wood = """
<x_xy model="demo_wood">
  <options dt="0.01" gravity="0.0 0.0 9.81"/>
  <defaults>
    <geom color="self"/>
  </defaults>
  <worldbody>
    <geom type="xyz" dim="0.5"/>
    <body joint="spherical" name="seg1" pos="0.4 0.0 0.0" damping="5.0 5.0 5.0">
      <geom pos="0.1 0.0 0.0" mass="1.0" type="box" dim="0.2 0.05 0.02"/>
      <body joint="frozen" name="imu1" pos="0.08 0.0 0.02">
        <geom mass="0.1" color="black" type="box" dim="0.05 0.03 0.02"/>
        <geom mass="0.1" color="orange" pos="0 0 0.001" type="box" dim="0.03 0.02 0.02"/>
        <geom type="xyz" dim="0.04"/>
      </body>
      <geom pos="0.2 0 0" type="capsule" dim="0.01 0.02" euler="0 90 0" color="gray" mass="1"/>
      <geom pos="0.23 0 0" type="sphere" dim="0.02" color="gray" mass="1"/>
      <body joint="spherical" name="seg2" pos="0.26 0.0 0.0" damping="3.0 3.0 3.0">
        <geom pos="0.2 0 0" type="capsule" dim="0.01 0.02" euler="0 90 0" color="gray" mass="1"/>
        <geom pos="0 0 0" type="capsule" dim="0.01 0.02" euler="0 90 0" color="gray" mass="1"/>
        <geom pos="0.1 0.0 0.0" mass="1.0" type="box" dim="0.2 0.05 0.02"/>
        <geom pos="0.23 0 0" type="sphere" dim="0.02" color="gray" mass="1"/>
        <body joint="frozen" name="imu2" pos="0.12 0.0 0.02">
          <geom mass="0.1" color="black" type="box" dim="0.05 0.03 0.02"/>
          <geom mass="0.1" color="orange" pos="0 0 0.001" type="box" dim="0.03 0.02 0.02"/>
          <geom type="xyz" dim="0.04"/>
        </body>
        <body joint="spherical" name="seg3" pos="0.26 0.0 0.0" damping="3.0 3.0 3.0">
            <geom pos="0 0 0" type="capsule" dim="0.01 0.02" euler="0 90 0" color="gray" mass="1"/>
          <geom pos="0.1 0.0 0.0" mass="1.0" type="box" dim="0.2 0.02 0.05"/>
          <body joint="frozen" name="imu3" pos="0.12 -0.02 0" euler="90 0 0">
            <geom mass="0.1" color="black" type="box" dim="0.05 0.03 0.02"/>
            <geom mass="0.1" color="orange" pos="0 0 0.001" type="box" dim="0.03 0.02 0.02"/>
            <geom type="xyz" dim="0.04"/>
        </body>
        </body>
      </body>
    </body>
  </worldbody>
</x_xy>
"""  # noqa: E501

sys_str_3dprinted = """
<x_xy model="demo_3dprinted">
  <options dt="0.01" gravity="0.0 0.0 9.81"/>
  <defaults>
    <geom color="self"/>
  </defaults>
  <worldbody>
    <geom type="xyz" dim="0.5"/>
    <body joint="spherical" name="seg1" pos="0.4 0.0 0.0" damping="5.0 5.0 5.0">
      <geom type="box" mass="1" pos="0.1 0 0" dim="0.2 0.05 0.05" color="dustin_exp_white"/>
      <geom type="box" mass="0.1" pos="0.03 -0.05 0" dim="0.01 0.1 0.01" color="black"/>
      <geom type="box" mass="0.1" pos="0.17 -0.05 0" dim="0.01 0.1 0.01" color="black"/>
      <body joint="frozen" name="imu1" pos="0.08 0.0 0.03">
        <geom mass="0.1" color="black" type="box" dim="0.05 0.03 0.02"/>
        <geom mass="0.1" color="orange" pos="0 0 0.001" type="box" dim="0.03 0.02 0.02"/>
        <geom type="xyz" dim="0.04"/>
      </body>
      <geom pos="0.2 0 0" type="capsule" dim="0.01 0.02" euler="0 90 0" color="gray" mass="1"/>
      <geom pos="0.23 0 0" type="sphere" dim="0.02" color="gray" mass="1"/>
      <body joint="spherical" name="seg2" pos="0.26 0.0 0.0" damping="3.0 3.0 3.0">
        <geom pos="0.2 0 0" type="capsule" dim="0.01 0.02" euler="0 90 0" color="gray" mass="1"/>
        <geom pos="0 0 0" type="capsule" dim="0.01 0.02" euler="0 90 0" color="gray" mass="1"/>
        <geom type="box" mass="1" pos="0.1 0 0" dim="0.2 0.05 0.05" color="dustin_exp_blue"/>
        <geom type="box" mass="0.1" pos="0.03 -0.05 0" dim="0.01 0.1 0.01" color="dustin_exp_white"/>
        <geom type="box" mass="0.1" pos="0.17 0.05 0" dim="0.01 0.1 0.01" color="dustin_exp_white"/>
        <geom pos="0.23 0 0" type="sphere" dim="0.02" color="gray" mass="1"/>
        <body joint="frozen" name="imu2" pos="0.12 0.0 0.03">
          <geom mass="0.1" color="black" type="box" dim="0.05 0.03 0.02"/>
          <geom mass="0.1" color="orange" pos="0 0 0.001" type="box" dim="0.03 0.02 0.02"/>
          <geom type="xyz" dim="0.04"/>
        </body>
        <body joint="spherical" name="seg3" pos="0.26 0.0 0.0" damping="3.0 3.0 3.0">
          <geom pos="0 0 0" type="capsule" dim="0.01 0.02" euler="0 90 0" color="gray" mass="1"/>
          <geom type="box" mass="1" pos="0.1 0 0" dim="0.2 0.05 0.05" color="dustin_exp_white"/>
          <geom type="box" mass="0.1" pos="0.1 0.05 0" dim="0.01 0.1 0.01" color="black"/>
          <geom type="box" mass="0.1" pos="0.15 -0.05 0" dim="0.01 0.1 0.01" color="black"/>
          <body joint="frozen" name="imu3" pos="0.09 0.0 0.03">
            <geom mass="0.1" color="black" type="box" dim="0.05 0.03 0.02"/>
            <geom mass="0.1" color="orange" pos="0 0 0.001" type="box" dim="0.03 0.02 0.02"/>
            <geom type="xyz" dim="0.04"/>
        </body>
        </body>
      </body>
    </body>
  </worldbody>
</x_xy>
"""  # noqa: E501


_made_invisible = False


def _make_middle_imu_invisible(viewer: InteractiveViewer):
    global _made_invisible
    if _made_invisible:
        return
    _made_invisible = True
    for i in range(5):
        viewer.make_geometry_transparent(3, i)


WOOD_DEMO = False


if __name__ == "__main__":

    Ts = 1 / 100
    solver = imt.Solver(
        [-1],
        methods=[imt.methods.VQF()],
        Ts=Ts,
    )
    imus = MovellaDotIMUs(60)
    sys = System.create(sys_str_wood if WOOD_DEMO else sys_str_3dprinted)
    viewer = InteractiveViewer(
        sys,
        height=1080,
        width=1920,
        floor_material="gray",
    )
    ringnet = ring.RING([-1, 0, 1], Ts=Ts, jit=False, use_lpf=False)
    _X = np.zeros((1, 3, 9))
    _X[0, 1, 7] = 1.0
    _X[0, 2, 8] = 1.0
    _, state = ringnet.init(bs=None, X=jnp.array(_X))
    # warmup
    jit_ringnet_apply = jax.jit(ringnet.apply)
    jit_ringnet_apply(X=jnp.array(_X), state=state)

    lpf1 = _Quat_LPF(5, Ts)
    lpf2 = _Quat_LPF(5, Ts)

    dt_target = Ts - 0.001

    try:
        dt = dt_target
        last_t = time.time()
        initial_time = time.time()
        t = -1
        while True:
            t += 1

            # if enough time has passed to achieve framerate
            while dt < dt_target:
                time.sleep(0.001)
                dt = time.time() - last_t

            last_t = time.time()
            measurements = imus.get_latest_measurements(short_address=True)

            imu1, imu2, imu3 = (
                measurements["8C"],
                measurements["8B"],
                measurements["63"],
            )
            acc1, acc2, acc3 = imu1["acc"], imu2["acc"], imu3["acc"]
            gyr1, gyr2, gyr3 = (
                np.deg2rad(imu1["gyr"]),
                np.deg2rad(imu2["gyr"]),
                np.deg2rad(imu3["gyr"]),
            )
            qhat, _ = solver.step(
                {
                    0: dict(acc=acc1, gyr=gyr1),
                }
            )

            if WOOD_DEMO:
                # rotate imu3 by 90° around x-axis
                q_imu2_s2s = qmt.quatFromAngleAxis(np.deg2rad(90), [1, 0, 0])
                acc3, gyr3 = qmt.rotate(q_imu2_s2s, acc3), qmt.rotate(q_imu2_s2s, gyr3)

            _X[0, 0, :3] = acc1
            _X[0, 0, 3:6] = gyr1
            _X[0, 1, :3] = acc2
            _X[0, 1, 3:6] = gyr2
            _X[0, 2, :3] = acc3
            _X[0, 2, 3:6] = gyr3
            yhat, state = jit_ringnet_apply(X=jnp.array(_X), state=state)
            qrel1 = qmt.qinv(yhat[0, 1])
            qrel2 = qmt.qinv(yhat[0, 2])

            qrel1 = lpf1._step(qrel1)
            qrel2 = lpf2._step(qrel2)

            q = np.concatenate((qmt.qinv(qhat[0]), qrel1, qrel2))
            viewer.update_q(q)

            if not imu2["is_alive"]:
                _make_middle_imu_invisible(viewer)

            dt = time.time() - last_t

            if (t % 10) == 0:
                print(
                    f"t = {round(time.time() - initial_time, 2)}, Achieved FPS at most: {round(1 / dt)}"  # noqa: E501
                )

            if not viewer.process.is_alive():
                break
    finally:
        imus.close()
        if base_render._scene is not None:
            base_render._scene.close()
        solver.close()
        viewer.close()
