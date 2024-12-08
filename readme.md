![gait-example](media/gait_demo.gif)

# Plug-and-Play Inertial Motion Tracking

This python package combines many well-established methods to provide a unified interface applicable to diverse inertial motion tracking tasks.

Plug-and-play solutions for standard use-cases are provided, such as
- knee joint angle tracking 
- hip joint angle tracking
- arm tracking
- lower extremities / gait tracking (see `/examples/lower_extremities.ipynb`)
- full-body motion capture

Most methods can be applied both online, allowing for real-time motion tracking, as well as offline.

## Installation

```pip install git+https://github.com/simon-bachhuber/imt.git```

## Example for three-segment KC

```python
import imt
import numpy as np

# Define a graph with one body connecting to the world/earth (0) and two child bodies (1 and 2)
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
quaternions = solver.step(imu_data)
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
quaternions_batched = solver.step(imu_data_batched)
print("Quaternions (time-batched):", quaternions_batched)
>>> {0: array([[...]]), 1: array([[...]]), 2: array([[...]])}
```