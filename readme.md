# High-level Interface for Inertial Motion Tracking

## Installation

```pip install git+https://github.com/simon-bachhuber/imt.git```

# Example for three-segment KC

```python
import imt
import numpy as np

# Define a graph with one body connecting to the world/earth (0) and two child bodies (1 and 2)
graph = [-1, 0, 0]
# Make no assumption on the type of joint, so it could be either 1D, 2D, or 3D rotational
# joint that connects two bodies (excluding the bodies that connect to world, 
# those are always free joints)
problem = "1D2D3D"
# The sampling rate of the IMU data
Ts = 0.01  # Sampling time (100 Hz)

# Initialize the solver
solver = imt.Solver(graph, problem, Ts)

# Define IMU data for the bodies (non-batched)
imu_data = {
    0: {"acc": np.array([0.0, 0.0, 9.81]), "gyr": np.array([0.1, 0.2, 0.3])},
    1: {"acc": np.array([0.0, 0.0, 9.81]), "gyr": np.array([0.2, 0.3, 0.4])},
    2: {"acc": np.array([0.0, 0.0, 9.81]), "gyr": np.array([0.3, 0.4, 0.5])},
}

# Process the IMU data to compute body-to-world orientations
quaternions = solver.step(imu_data)
print("Quaternions (non-batched):", quaternions)

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
```