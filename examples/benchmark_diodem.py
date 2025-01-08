from typing import Optional

from diodem import load_data
import fire
import numpy as np
import qmt

import imt


def main(
    segments: list[str] = ["seg1"],
    exp_id: int = 1,
    motion_start: str | int = 1,
    # by default uses the next motion sequence as stop,
    # if set to -1 goes from motion_start to the end of the trial
    motion_stop: Optional[str | int] = None,
    graph: list[int] = None,
    Ts: float = 0.01,
    warmup: float = 5,
):
    N = len(segments)
    graph = graph if graph else list(range(-1, N - 1))
    data = load_data(
        exp_id,
        motion_start=motion_start,
        motion_stop=motion_stop,
        resample_to_hz=1 / Ts,
    )
    qhat, _ = imt.Solver(graph, methods=None, Ts=Ts, body_names=segments).step(
        {
            seg: {
                "acc": data[seg]["imu_rigid"]["acc"],
                "gyr": data[seg]["imu_rigid"]["gyr"],
            }
            for seg in segments
        }
    )

    warmup = int(warmup / Ts)
    for i, seg_i in enumerate(segments):
        p = graph[i]
        if p == -1:
            q_p_truth = q_p_pred = np.array([1.0, 0, 0, 0])
        else:
            seg_p = segments[p]
            q_p_truth = data[seg_p]["quat"]
            q_p_pred = qhat[seg_p]

        q_i_truth = data[seg_i]["quat"]
        q_i_pred = qhat[seg_i]
        qrel_truth = qmt.qrel(q_p_truth, q_i_truth)
        qrel_pred = qmt.qrel(q_p_pred, q_i_pred)
        error_deg = np.rad2deg(
            np.abs(qmt.quatAngle(qmt.qrel(qrel_truth, qrel_pred)))[warmup:]
        )

        print(
            f"{seg_i}: {np.round(np.mean(error_deg), 2)} ± {np.round(np.std(error_deg), 2)}"  # noqa: E501
        )


if __name__ == "__main__":
    fire.Fire(main)
