import os
import time

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import onnxruntime as ort
import ring
import tree_utils


@hk.without_apply_rng
@hk.transform_with_state
def f(X):
    X, prev_message_p, prev_mailbox_i = X
    prev_state = hk.get_state("inner_cell_state", [2, 400])
    X = tree_utils.batch_concat_acme((X, prev_message_p, prev_mailbox_i), 0)
    output, next_state = ring.ml.ringnet.StackedRNNCell("gru", 400, 2, True)(
        X, prev_state
    )
    hk.set_state("inner_cell_state", next_state)
    next_message_i = hk.nets.MLP([400, 200])(next_state[-1])
    output = hk.nets.MLP([400, 4])(output)
    output = output / jnp.linalg.norm(output, axis=-1, keepdims=True)
    return output, next_message_i


params = ring.utils.pickle_load(
    "~/Documents/PYTHON/ring/src/ring/ml/params/0x13e3518065c21cd8.pickle"
)
state0_flat = jnp.ones((2, 400))
messages = jnp.ones((200,))
a = jnp.ones((3,))
dt = jnp.ones((1,))


def timestep(a, g, ja, dt, mp_tm1, mc_tm1, state_tm1):
    X = (jnp.concat((a / 9.81, dt / 0.1, g / 2.2, ja / 0.57)), mp_tm1, mc_tm1)
    (qhat, m_t), state_t = f.apply(params, {"~": {"inner_cell_state": state_tm1}}, X)
    return qhat, m_t, state_t["~"]["inner_cell_state"]


filename = "ring-node-v0.onnx"
overwrite = True

if overwrite or not os.path.exists(filename):
    ring.ml.ml_utils.to_onnx(
        timestep,
        filename,
        a,
        a,
        a,
        dt,
        messages,
        messages,
        state0_flat,
        in_args_names=[
            "acc (3,) [m/s^2]",
            "gyr (3,) [rad/s]",
            "joint-axis (3,)",
            "dt (1,) [s]",
            "previous message parent (200,); init with zeros",
            "previous message children (200,); init with zeros",
            "previous state (2, 400,); init with zeros",
        ],
        out_args_names=["quat (4,); child-to-parent", "next message", "next state"],
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
time_f(lambda: jit_timestep(a, a, a, dt, messages, messages, state0_flat))

a = np.array(a)
state0_flat = np.array(state0_flat)
messages = np.array(messages)
dt = np.array(dt)
session = ort.InferenceSession(filename)


def onnx_timestep():
    session.run(
        None,
        {
            "acc (3,) [m/s^2]": a,
            "gyr (3,) [rad/s]": a,
            "joint-axis (3,)": a,
            "dt (1,) [s]": dt,
            "previous message parent (200,); init with zeros": messages,
            "previous message children (200,); init with zeros": messages,
            "previous state (2, 400,); init with zeros": state0_flat,
        },
    )


print("ONNX version")
time_f(onnx_timestep)
