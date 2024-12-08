import haiku as hk
import jax.numpy as jnp
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
X = (jnp.zeros((10,)), jnp.zeros((200,)), jnp.zeros((200,)))
f.apply(params, {"~": {"inner_cell_state": jnp.zeros((2, 400))}}, X)
