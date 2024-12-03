import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import ring

X = jnp.zeros((1, 1, 12))

net = ring.ml.RNNO(
    4,
    return_quats=True,
    eval=False,
    v1=True,
    rnn_layers=[800] * 3,
    linear_layers=[400] * 1,
    act_fn_rnn=lambda X: X,
    params="/Users/simon/Downloads/0x643a580673fc3f4.pickle",
    celltype="gru",
    scale_X=False,
).unwrapped

_, state0 = net.init(X=X)
state_flat, unflatten = jax.flatten_util.ravel_pytree(state0)


def timestep(a1, a2, g1, g2, state_tm1):
    grav, pi = jnp.array(9.81), jnp.array(2.2)
    X = jnp.concatenate((a1 / grav, a2 / grav, g1 / pi, g2 / pi))[None, None]
    yhat, state = net.apply(X=X, state=unflatten(state_tm1))
    return yhat[0, 0], jax.flatten_util.ravel_pytree(state)[0]


a = jnp.ones((3,), dtype=np.float32)

ring.ml.ml_utils.to_onnx(
    timestep,
    "relOri-1D2D3D-100Hz-v0.onnx",
    a,
    a,
    a,
    a,
    state_flat,
    in_args_names=[
        "acc1 (3,) [m/s^2]",
        "acc2 (3,) [m/s^2]",
        "gyr1 (3,) [rad/s]",
        "gyr2 (3,) [rad/s]",
        "previous_state (2400,); init with zeros",
    ],
    out_args_names=["quat; rel-ori imu2-to-imu1", "next_state"],
    validate=True,
)
