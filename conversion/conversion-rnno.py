import os
import time

import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import onnxruntime as ort
import ring

X = jnp.zeros((1, 2, 6))


net = ring.ml.RING(
    params="/Users/simon/Documents/PYTHON/imt/to_onnx/0xe58c20067500b83.pickle",
    celltype="gru",
    lam=(-1, 0),
    layernorm=True,
    forward_factory=ring.ml.rnno_v1.rnno_v1_forward_factory,
    rnn_layers=[800] * 3,
    linear_layers=[400] * 1,
    act_fn_rnn=lambda X: X,
    output_dim=8,
)
net = ring.ml.base.NoGraph_FilterWrapper(net, quat_normalize=True)


_, state0 = net.init(X=X)
state_flat, unflatten = jax.flatten_util.ravel_pytree(state0)


def timestep(a1, a2, g1, g2, state_tm1):
    grav, pi = jnp.array(9.81), jnp.array(2.2)
    X = jnp.concatenate(
        (
            jnp.concatenate((a1 / grav, g1 / pi))[None],
            jnp.concatenate((a2 / grav, g2 / pi))[None],
        )
    )[None]
    yhat, state = net.apply(X=X, state=unflatten(state_tm1))
    return yhat[0], jax.flatten_util.ravel_pytree(state)[0]


a = jnp.ones((3,), dtype=np.float32)

filename = "rnno-100Hz-v0.onnx"

if not os.path.exists(filename):
    ring.ml.ml_utils.to_onnx(
        timestep,
        filename,
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
        out_args_names=[
            "quats (2, 4); incl-body1-to-eps, body2-to-body1",
            "next_state",
        ],
        validate=True,
    )


# TIMING TESTING #
def time_f(f, N: int = 1000):
    # maybe JIT
    f()
    t1 = time.time()
    for _ in range(N):
        f()
    print(f"One executation of `f` took: {((time.time() - t1) / N) * 1000}ms")


jit_timestep = jax.jit(timestep)
print("JAX version")
time_f(lambda: jit_timestep(a, a, a, a, state_flat))

a = np.array(a)
state_flat = np.array(state_flat)
session = ort.InferenceSession(filename)


def onnx_timestep():
    session.run(
        None,
        {
            "acc1 (3,) [m/s^2]": a,
            "acc2 (3,) [m/s^2]": a,
            "gyr1 (3,) [rad/s]": a,
            "gyr2 (3,) [rad/s]": a,
            "previous_state (2400,); init with zeros": state_flat,
        },
    )


print("ONNX version")
time_f(onnx_timestep)
