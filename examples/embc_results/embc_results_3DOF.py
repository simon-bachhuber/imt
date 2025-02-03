from diodem import load_data
import fire
import jax
import numpy as np
import pandas as pd
import qmt
import ring
from ring.maths import quat_random
from tqdm import tqdm
import tree

import imt


def _simulate_imu(
    pos,
    quat,
    pos_offset,
    dt: float,
):
    xs = ring.Transform.create(pos=pos, rot=qmt.qinv(quat))
    xs = ring.algebra.transform_mul(ring.Transform.create(pos=pos_offset), xs)
    return ring.algorithms.imu(
        xs,
        np.array([0, 0, 9.81]),
        dt,
        low_pass_filter_pos_f_cutoff=13.5,
        low_pass_filter_rot_cutoff=16.0,
    )


def filter_motions(motions: list[str], excludes) -> list[tuple[str]]:

    def exclude_motion(motion: str) -> bool:
        motion = motion.split("_")[1]
        for exclude in excludes:
            N = len(exclude)
            if motion[:N] == exclude:
                return True
        return False

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


def _to_1d(ja):
    if ja.ndim == 1:
        return ja
    return ja[0]


def _to_2d(ja):
    if ja.ndim == 2:
        return ja
    return np.stack([ja, ja], axis=0)


class RandomQuat:
    def __init__(self, max_ang_deg: float = 180.0, seed: int = 1):
        self.seed = seed
        self.max_ang_rad = np.deg2rad(max_ang_deg)

    def sample(self, batch_shape=()) -> np.ndarray:
        self.seed += 1
        return np.array(
            quat_random(
                jax.random.PRNGKey(self.seed),
                maxval=self.max_ang_rad,
                batch_shape=batch_shape,
            )
        )


methods = {
    "ours": lambda ja: imt.methods.RNNO(),
}

methods = {"ours": lambda ja: imt.methods.RNNO()}


def main(
    max_ang_deg: float = 0.0,
    rand_sensor_ori: str = "random",
    output_path=None,
    disable_tqdm=False,
    nonrigid: bool = False,
):
    randomQuatGenerator = RandomQuat(max_ang_deg=max_ang_deg, seed=100)
    results = {m: {i: [] for i in [1, 2, 3]} for m in methods}
    imu_key = "imu_nonrigid" if nonrigid else "imu_rigid"

    for exp_id in tqdm(range(1, 6), disable=disable_tqdm):
        motions = [(1, -1)] if exp_id > 2 else [(3, -1)]

        for motion_start, motion_stop in motions:
            chain = (
                ["seg1", "seg2"]
                if exp_id < 6
                else ["seg5", "seg1", "seg2", "seg3", "seg4"]
            )
            ja_1d = (
                [None, [1.0, 0, 0], [0.0, 1, 0], [0.0, 0, 1]]
                if exp_id < 6
                else [[0.0, 1, 0], None, None, [0.0, 1, 0]]
            )
            dofs = [3, 1, 1, 1] if exp_id < 6 else [1, 2, 2, 1]
            for i, (seg1, seg2) in tqdm(
                enumerate(zip(chain, chain[1:])), disable=disable_tqdm
            ):
                dof = dofs[i]
                if dof == 1:
                    ja = ja_1d[i]
                elif dof == 2:
                    ja = [[0.0, 1, 0], [0.0, 0, 1]]
                else:  # dof == 3
                    ja = [[0.0, 1, 0], [0.0, 0, 1]]
                ja = np.array(ja)

                data = load_data(exp_id, motion_start, motion_stop)
                # throw aways first 20 seconds
                data = tree.map_structure(lambda a: a[2000:], data)

                # maybe apply virtual rotation; let this be from 2 -> 2'
                qrand = randomQuatGenerator.sample(batch_shape=(2,))

                if rand_sensor_ori == "random":
                    pass
                elif rand_sensor_ori == "aligned":
                    qrand[1] = qrand[0]
                else:
                    raise Exception()

                rotate_segments = [seg1, seg2]
                for i, rotate_seg in enumerate(rotate_segments):
                    data[rotate_seg][imu_key]["acc"] = qmt.rotate(
                        qrand[i], data[rotate_seg][imu_key]["acc"]
                    )
                    data[rotate_seg][imu_key]["gyr"] = qmt.rotate(
                        qrand[i], data[rotate_seg][imu_key]["gyr"]
                    )
                    data[rotate_seg]["quat"] = qmt.qmult(
                        data[rotate_seg]["quat"], qmt.qinv(qrand[i])
                    )
                if len(rotate_segments) > 0:
                    ja = qmt.rotate(qrand[1], ja)

                for method_str, method in methods.items():
                    q_pred, _ = imt.Solver(
                        [-1, 0], methods=[imt.methods.VQF(), method(ja)], Ts=0.01
                    ).step(
                        {
                            i: {
                                "acc": data[seg][imu_key]["acc"],
                                "gyr": data[seg][imu_key]["gyr"],
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
                            qrand,
                        )
                    )

    if output_path is not None:
        np.save(output_path, results)

    def mean_std_deg(result_ele: list[tuple], warmup: int = 500):
        errors = [np.mean(ele[5][warmup:]) for ele in result_ele]
        return np.mean(errors), np.std(errors)

    # Define methods and DOFs
    dofs = [1, 2, 3]

    # Create a data structure for the table
    data = []
    for method in methods:
        row = []
        for dof in dofs:
            mean, std = mean_std_deg(results[method][dof], warmup=1000)
            row.append(f"{mean:.2f} ± {std:.2f}")
        data.append(row)

    # Create a DataFrame
    df = pd.DataFrame(
        data, index=[f"Method {m}" for m in methods], columns=[f"DOF {d}" for d in dofs]
    )

    # Display the table
    print(df)


if __name__ == "__main__":
    fire.Fire(main)
