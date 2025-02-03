from diodem import load_data
import jax
import numpy as np
import pandas as pd
import qmt
import ring
import tqdm

import imt

np.random.seed(1)


def _simulate_imu(
    data_of_seg: dict,
    pos_offset=np.array([0.0, 0, 0]),
    dt: float = 0.01,
):
    pos = sum(data_of_seg[f"marker{i}"] for i in range(1, 5)) / 4
    quat = data_of_seg["quat"]
    xs = ring.Transform.create(pos=pos, rot=qmt.qinv(quat))
    xs = ring.algebra.transform_mul(ring.Transform.create(pos=pos_offset), xs)
    return jax.tree.map(
        lambda a: np.array(a, dtype=np.float64),
        ring.algorithms.imu(
            xs,
            np.array([0, 0, 9.81]),
            dt,
            low_pass_filter_pos_f_cutoff=13.5,
            low_pass_filter_rot_cutoff=16.0,
        ),
    )


pmms = [0, 0.05, 0.1, 0.15]
results = {pmm: {i: [] for i in [1, 2, 3]} for pmm in pmms}
for exp_id in tqdm.tqdm(range(1, 12)):
    data = load_data(exp_id, 1, -1)

    chain = (
        ["seg1", "seg2", "seg3", "seg4", "seg5"]
        if exp_id < 6
        else ["seg5", "seg1", "seg2", "seg3", "seg4"]
    )
    dofs = [3, 1, 1, 1] if exp_id < 6 else [1, 2, 2, 1]
    for i, (seg1, seg2) in enumerate(zip(chain, chain[1:])):
        for pmm in pmms:
            random_offset = np.random.uniform(
                -pmm,
                pmm,
                size=(
                    2,
                    3,
                ),
            )
            q_pred, _ = imt.Solver(
                [-1, 0], methods=[imt.methods.VQF(), imt.methods.RNNO()], Ts=0.01
            ).step(
                {
                    i: _simulate_imu(data[seg], random_offset[i])
                    for i, seg in enumerate([seg1, seg2])
                }
            )
            qrel_pred = qmt.qrel(q_pred[0], q_pred[1])
            qrel_truth = qmt.qrel(data[seg1]["quat"], data[seg2]["quat"])
            error_deg = np.rad2deg(
                np.abs(qmt.quatAngle(qmt.qrel(qrel_truth, qrel_pred)))
            )

            results[pmm][dofs[i]].append(
                (
                    f"exp_id={exp_id}",
                    f"{seg1}-{seg2}",
                    qrel_truth,
                    qrel_pred,
                    error_deg,
                    random_offset,
                )
            )

np.save(ring.utils.parse_path(__file__, extension="npy"), results)


def mean_std_deg(result_ele: list[tuple], warmup: int = 500):
    errors = [np.mean(ele[-1][warmup:]) for ele in result_ele]
    return np.mean(errors), np.std(errors)


# Define methods and DOFs
dofs = [1, 2, 3]

# Create a data structure for the table
data = []
for pmm in pmms:
    row = []
    for dof in dofs:
        mean, std = mean_std_deg(results[pmm][dof], warmup=1000)
        row.append(f"{mean:.2f} ± {std:.2f}")
    data.append(row)

# Create a DataFrame
df = pd.DataFrame(
    data,
    index=[f"Pos-Min-Max {pmm}" for pmm in pmms],
    columns=[f"DOF {d}" for d in dofs],
)

# Display the table
print(df)
