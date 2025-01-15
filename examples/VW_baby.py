import numpy as np
import qmt

import imt
from imt.utils.view import view
from imt.utils.view import VisOptions

# LOAD DATA
file = "/Users/simon/Documents/data/vivian_waldheim_baby_data/Datenaustausch_IMU/npy/S0131_dict_frame.npy"  # noqa: E501
sensors = ["S0333", "S1094", "S0593", "S0994", "S0477"]
data = np.load(file, allow_pickle=True).item()
Hz = 52

# PREPARE DATA
imu_data = {
    i: dict(acc=data[sensors[i]]["acc"], gyr=data[sensors[i]]["gyr_rad"])
    for i in range(5)
}
imu_data = imt.utils.resample(imt.utils.crop_tail(imu_data, Hz), Hz, 100.0)
imu_data[0] = dict(
    acc=qmt.rotate(qmt.quatFromAngleAxis(-np.pi, [0, 0, 1]), imu_data[0]["acc"]),
    gyr=qmt.rotate(qmt.quatFromAngleAxis(-np.pi, [0, 0, 1]), imu_data[0]["gyr"]),
)

# ESTIMATE ORIENTATIONS
rel_method = imt.methods.RING(axes_directions=np.array([1.0, 0, 0]))
graph = [-1, 0, 1, 0, 3]
qhat, extras = imt.Solver(graph, [imt.methods.VQF(True)] + [rel_method] * 4, 0.01).step(
    imu_data
)

# VISUALISATION
extras[1]["joint-center-to-body1"] = np.array([-0.05, 0.15, 0])
extras[3]["joint-center-to-body1"] = np.array([0.05, 0.15, 0])
extras[1]["joint-center-to-body2"] = np.array([0, -0.05, -0.025])
extras[3]["joint-center-to-body2"] = np.array([0, -0.05, -0.025])
extras[2]["joint-center-to-body1"] = np.array([0, 0.05, 0])
extras[2]["joint-center-to-body2"] = np.array([0, -0.05, 0])
extras[4]["joint-center-to-body1"] = np.array([0, 0.05, 0])
extras[4]["joint-center-to-body2"] = np.array([0, -0.05, 0])

pos = np.zeros((qhat[0].shape[0], 3))
pos[:, 2] = 0.05
vis_options = VisOptions(
    show_floor=True,
    show_stars=False,
    body_xyz_dim=0.03,
    imu_xyz_dim=0.02,
    imu_type="cylinder",
    imu_dim="0.01 0.005",
    imu_color="black",
    floor_material="beige",
    joint_to_imu_color=[0.5, 0.5, 0.5, 1],
    imu_offset=[0, 0, -0.01],
    show_imu_xyz=False,
)
view(
    graph,
    qhat,
    extras,
    hz=100,
    show_every_nth_frame=4,
    global_translation=pos,
    height=720,
    width=1280,
    vis_options=vis_options,
)
