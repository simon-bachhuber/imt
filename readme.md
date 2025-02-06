![gait-example](media/gait_demo.gif)

# Plug-and-Play Inertial Motion Tracking

This python package combines many well-established methods to provide a unified interface applicable to diverse inertial motion tracking tasks.

> **ℹ️ Installation:**
> 
> You can install with `pip install git+https://github.com/simon-bachhuber/imt.git`.

## Quickstart
```python
import imt

solver = imt.Solver(graph=[-1, 0], methods=None, Ts=0.01)
imu_data = {0: dict(acc=acc1, gyr=gyr1), 1: dict(acc=acc2, gyr=gyr2)}
quaternions, extras = solver.step(imu_data)
```

In this package, `imt.methods` are the core algorithms that estimate orientations. They are resposible for the first output `quaternions` of `quaternions, extras = imt.Solver.step(...)`. `imt.wrappers` are used to wrap `imt.methods` and estimate additional quantities or augment the wrapped method. 
Most methods can be applied both online, allowing for real-time motion tracking, as well as offline. The following algorithms are available:

| Class    | Publication/Author             | $a_p$ | $g_p$ | $m_p$ | $a_i$ | $g_i$ | $m_i$ | Online |
|----------|-------------------------|-------|-------|-------|-------|-------|-------|--------|
| `imt.methods.RIANN`        | Weber et al. (2021), https://www.mdpi.com/2673-2688/2/3/28      | ✘     | ✘     | ✘     | ✔     | ✔     | ✘     | ✘      |
| `imt.methods.VQF`        | Laidig et al. (2022), https://arxiv.org/abs/2203.17024    | ✘     | ✘     | ✘     | ✔     | ✔     | ◯     | ✔      |
| `imt.methods.HeadCor(method_1d="1d_corr")`        | Laidig et al. (2017), https://pubmed.ncbi.nlm.nih.gov/28813947/         | ✔      | ✔     | ✘     | ✔     | ✔      | ✘     | ✘      |
| `imt.methods.HeadCor(method_1d="euler_1d")`        | Lehmann et al. (2020), https://api.semanticscholar.org/CorpusID:214710126         | ✔      | ✔     | ✘     | ✔     | ✔      | ✘     | ✘      |
| `imt.methods.HeadCor(method_1d="euler_2d")`        | Laidig et al. (2019), https://ieeexplore.ieee.org/document/8857535         | ✔      | ✔     | ✘     | ✔     | ✔      | ✘     | ✘      |
| `imt.methods.RING`        | Bachhuber et al. (2024), https://openreview.net/forum?id=h2C3rkn0zR        | ✔      | ✔     | ✘     | ✔     | ✔      | ✘     | ✔      |
| `imt.methods.RNNO`        | EMBC 2025        | ✔      | ✔     | ✘     | ✔     | ✔      | ✘     | ✔      |
| `imt.wrappers.CalibrateMag`        | Laidig        | ✔/◯      | ✔/◯     | ✔/◯     | ✔/◯     | ✔/◯      | ✔/◯     | ✘      |
| `imt.wrappers.SenToSeg1DCal`        | Bachhuber        | ✔      | ✔     | ✘     | ✔     | ✔      | ✘     | ✘      |
| `imt.wrappers.DeadReckoning`        | Bachhuber        | ✘      | ✘     | ✘     | ✔     | ✔      |   ◯   | ✘      |
| `imt.wrappers.JointTracker1D`        | Bachhuber        | ✔      | ✔     | ✘     | ✔     | ✔      | ✘     | ✘      |
| `imt.wrappers.JointPosition`        | Seel et al. (2012), https://ieeexplore.ieee.org/document/6402423        | ✔      | ✔     | ✘     | ✔     | ✔      | ✘     | ✘      |
| `imt.wrappers.LPF`        | Bachhuber        | ◯      | ◯     | ◯     | ◯     | ◯      | ◯     |  ✔      |

In the table $a_i$ is the acceleterometer data of the body $i$. Let $p$ be the body index of the $i$-th body, then $a_p$ is the parent-body's accelereometer data. $g_{i/p}$ and $m_{i/p}$ are gyroscope and magnetometer, respectively. ✔ means that the algorithms uses this information, ✘ means that it doesn't, and ◯ means that it can use that information.

Plug-and-play solutions for standard use-cases are provided, such as:
- Knee Joint Angle Tracking (see `examples/knee_angle_tracking.ipynb`)
- Shoulder Joint Tracking (see `examples/shoulder_tracking.ipynb`)
- Gait Tracking (see `/examples/lower_extremities_*.ipynb`)
- Head Tracking (see `examples/head_tracking.ipynb`)

## Knee Joint Angle Tracking
```python
import imt

solver = imt.Solver(
    graph=[-1, 0], 
    methods=None, 
    Ts=0.01, 
    body_name=["thigh", "shank"]
)
imu_data = {"thigh": dict(acc=acc1, gyr=gyr1), "shank": dict(acc=acc2, gyr=gyr2)}
quaternions, extras = solver.step(imu_data)
```
![knee-angle-tracking-example](media/knee_tracking.gif)

## Shoulder Joint Tracking
```python
import imt

solver = imt.Solver(
    graph=[-1, 0], 
    methods=None, 
    Ts=0.01, 
    body_name=["chest", "upperarm"]
)
imu_data = {"chest": dict(acc=acc1, gyr=gyr1), "upperarm": dict(acc=acc2, gyr=gyr2)}
quaternions, extras = solver.step(imu_data)
```
![shoulder-joint-tracking-example](media/shoulder_tracking.gif)

## Advanced Usage Example

```python
import imt
import numpy as np

# Let's consider a *three-segment KC*

# First, define a graph with one body connecting to the world/earth (0) and two child bodies (1 and 2)
graph = [-1, 0, 0]

# Define the methods that are used for solving the relative orientation subproblems in the graph
methods = [
    # use `vqf` because this body connects to earth, so there is no constraint to exploit
    imt.methods.VQF(),
    # let's assume there is a 1-DOF joint between body '1' and body '0'
    imt.methods.HeadCor(dof=1),
    # let's assume we don't know how many DOFs there are between body '2' and body '0', so we
    # will use a general-purpose method (but which will be less accurate)
    imt.methods.RNNO()
]
# We can also let methods be `None` then a set of default methods will be determined auto-
# matically based on the graph
methods = None

# The sampling rate of the IMU data
Ts = 0.01  # Sampling time (100 Hz)

# Initialize the solver
solver = imt.Solver(graph, methods, Ts)

# Define IMU data for the bodies (non-batched)
imu_data = {
    0: {"acc": np.array([0.0, 0.0, 9.81]), "gyr": np.array([0.1, 0.2, 0.3])},
    1: {"acc": np.array([0.0, 0.0, 9.81]), "gyr": np.array([0.2, 0.3, 0.4])},
    2: {"acc": np.array([0.0, 0.0, 9.81]), "gyr": np.array([0.3, 0.4, 0.5])},
}

# Process the IMU data to compute body-to-world orientations
quaternions, extras = solver.step(imu_data)
print("Quaternions (non-batched):", quaternions)
# so the '0' entry is the quaternion from body '0' to body '-1' (earth)
# similarly, the '1' entry is the quaterion from body '1' to body '0'
# finally, the '2' entry is the quaterion from body '2' to body '0'
>>> {0: array([...]), 1: array([...]), 2: array([...])}

# Reset the solver afterwards
solver.reset()

# Define time-batched IMU data for the bodies
imu_data_batched = {
    0: {
        "acc": np.array([[0.0, 0.0, 9.81], [0.0, 0.0, 9.81], [0.0, 0.0, 9.81]]),
        "gyr": np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]])
    },
    1: {
        "acc": np.array([[0.0, 0.0, 9.81], [0.0, 0.0, 9.81], [0.0, 0.0, 9.81]]),
        "gyr": np.array([[0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]])
    },
    2: {
        "acc": np.array([[0.0, 0.0, 9.81], [0.0, 0.0, 9.81], [0.0, 0.0, 9.81]]),
        "gyr": np.array([[0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]])
    },
}

# Process the time-batched IMU data to compute body-to-world orientations
quaternions_batched, extras = solver.step(imu_data_batched)
print("Quaternions (time-batched):", quaternions_batched)
>>> {0: array([[...]]), 1: array([[...]]), 2: array([[...]])}
```