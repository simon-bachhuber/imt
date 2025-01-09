from diodem import load_data
from diodem._src import _load_timings
import numpy as np
import pandas as pd
import qmt
from tqdm import tqdm

import imt


def filter_motions(motions: list[str]) -> list[tuple[str]]:
    out = []
    start = stop = None
    for motion in motions:
        if not exclude_motion(motion):
            motion_postfix = motion[(len(motion.split("_")[0]) + 1) :]
            if start is None:
                start = motion_postfix
            stop = motion_postfix
        else:
            if start is not None:
                out.append((start, stop))
            start = stop = None
    return out


def exclude_motion(motion: str) -> bool:
    excludes = ["pause", "canonical", "shaking", "dangle"]
    motion = motion.split("_")[1]
    for exclude in excludes:
        N = len(exclude)
        if motion[:N] == exclude:
            return True
    return False


def mean_std_deg(result_ele: list[tuple], warmup: int = 500):
    errors = [np.mean(ele[-1][warmup:]) for ele in result_ele]
    return np.mean(errors), np.std(errors)


# Simulated function for demonstration; replace with your `results_mean_std`
def results_mean_std(method, dof):
    return mean_std_deg(results[method][dof])


def _to_1d(ja):
    ja = np.array(ja)
    if ja.ndim == 1:
        return ja
    return ja[0]


def _to_2d(ja):
    ja = np.array(ja)
    if ja.ndim == 2:
        return ja
    return np.stack([ja, ja], axis=0)


methods = {
    "qmt_proj": lambda ja: imt.methods.HeadCor(1, _to_1d(ja), method_1d="proj"),
    "qmt_euler_1d": lambda ja: imt.methods.HeadCor(1, _to_1d(ja), method_1d="euler_1d"),
    "qmt_euler_2d": lambda ja: imt.methods.HeadCor(1, _to_1d(ja), method_1d="euler_2d"),
    "qmt_1d_corr": lambda ja: imt.methods.HeadCor(1, _to_1d(ja), method_1d="1d_corr"),
    "qmt_euler": lambda ja: imt.methods.HeadCor(2, _to_2d(ja), method_2d="euler"),
    "ring": lambda ja: imt.methods.RING(),
    "ours": lambda ja: imt.methods.RNNO(),
}

results = {m: {i: [] for i in [1, 2, 3]} for m in methods}

for exp_id in tqdm(range(1, 11)):
    for motion_start, motion_stop in tqdm(filter_motions(_load_timings(exp_id))):
        data = load_data(exp_id, motion_start, motion_stop)

        chain = (
            ["seg1", "seg2", "seg3", "seg4", "seg5"]
            if exp_id < 6
            else ["seg5", "seg1", "seg2", "seg3", "seg4"]
        )
        ja_1d = (
            [None, [1.0, 0, 0], [0.0, 1, 0], [0.0, 0, 1]]
            if exp_id < 6
            else [[0.0, 1, 0], None, None, [0.0, 1, 0]]
        )
        dofs = [3, 1, 1, 1] if exp_id < 6 else [1, 2, 2, 1]
        for i, (seg1, seg2) in tqdm(enumerate(zip(chain, chain[1:]))):
            dof = dofs[i]
            if dof == 1:
                ja = ja_1d[i]
            elif dof == 2:
                ja = [[0.0, 1, 0], [0.0, 0, 1]]
            else:  # dof == 3
                ja = [[0.0, 1, 0], [0.0, 0, 1]]

            for method_str, method in methods.items():
                q_pred, _ = imt.Solver(
                    [-1, 0], methods=[imt.methods.VQF(), method(ja)], Ts=0.01
                ).step(
                    {
                        i: {
                            "acc": data[seg]["imu_rigid"]["acc"],
                            "gyr": data[seg]["imu_rigid"]["gyr"],
                        }
                        for i, seg in enumerate([seg1, seg2])
                    }
                )
                qrel_pred = qmt.qrel(q_pred[0], q_pred[1])
                qrel_truth = qmt.qrel(data[seg1]["quat"], data[seg2]["quat"])
                error_deg = np.rad2deg(
                    np.abs(qmt.quatAngle(qmt.qrel(qrel_truth, qrel_pred)))
                )

                results[method_str][dof].append(
                    (
                        f"exp_id={exp_id}",
                        f"{motion_start} -> {motion_stop}",
                        f"{seg1}-{seg2}",
                        qrel_truth,
                        qrel_pred,
                        error_deg,
                    )
                )


np.save("embc_results", results)

# Define methods and DOFs
dofs = [1, 2, 3]

# Create a data structure for the table
data = []
for method in methods:
    row = []
    for dof in dofs:
        mean, std = results_mean_std(method, dof)
        row.append(f"{mean:.2f} ± {std:.2f}")
    data.append(row)

# Create a DataFrame
df = pd.DataFrame(
    data, index=[f"Method {m}" for m in methods], columns=[f"DOF {d}" for d in dofs]
)

# Display the table
print(df)
