import numpy as np

from imt._graph import Graph
from imt._solutions import Online_RelOri_1D2D3D_Solution
from imt._solutions import Solution

# body number <integer> -> {'acc': ..., 'gyr': ..., 'mag': ... (optional)}
D = dict[int, dict[str, np.ndarray]]


class Solver:

    def __init__(self, graph: list[int], solutions: list[Solution], Ts: float):
        """
        Initializes the `Solver` with a graph of bodies, problem type, and sampling time.

        Args:
            graph (list[int]): A list representing the parent-child relationships between
                bodies. Each index corresponds to a body, and the value at that index
                specifies the parent body. The special integer `-1` represents the
                "world body" (i.e., a root node with no parent).
            solutions (list[Solution]): A list of solutions that are used for each two-
                segment sub-kinematic-chain in the problem to solve for the relative
                orientation between the two. Note that the connection to earth is also
                considered a two-segment chain, so for bodies that connect to `-1` it
                is recommended to use `VQF_Solution`
            Ts (float): The sampling time in seconds. Currently, only `0.01` (100 Hz)
                is supported. Resampling should be done externally if the data is at
                a different frequency.

        Example:
            >>> # Define a graph with one body (body 0) connecting to world (-1) and
            >>> # one body (body 1) connecting to body 0
            >>> graph = [-1, 0]
            >>> solutions = [imt.solutions.VQF_Solution()] * 2
            >>> Ts = 0.01  # Sampling time (100 Hz)
            >>> solver = Solver(graph, solutions, Ts)
            >>>
            >>> # Define IMU data for the bodies, here we provide 'mag' data as well because
            >>> # we use `VQF` for everything and without magnetometer the heading would be
            >>> # incorrect
            >>> imu_data = {
            ...     0: {"acc": np.array([0.0, 0.0, 9.81]), "gyr": np.array([0.1, 0.2, 0.3]),
            ...         "mag": np.array([1.0, 0, 0])},
            ...     1: {"acc": np.array([0.0, 0.0, 9.81]), "gyr": np.array([0.2, 0.3, 0.4]),
            ...         "mag": np.array([0.5, 0.4, 0])},
            ... }
            >>> # Process the IMU data to compute body-to-world orientations
            >>> quaternions = solver.step(imu_data)
            >>> print(quaternions)
            >>> # so the '0' entry is the quaternion from body '0' to body '-1' (earth)
            >>> # similarly, the '1' entry is the quaterion from body '1' to body '0'
            {0: array([...]), 1: array([...])}
            >>> # reset the solver afterwards
            >>> solver.reset()
            >>> # `.step` also accepts time-batched data
            >>> imu_data = {
            ...     0: {
            ...         "acc": np.array([[0.0, 0.0, 9.81], [0.0, 0.0, 9.81], [0.0, 0.0, 9.81]]),
            ...         "gyr": np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]]),
            ...         "mag": ...
            ...     },
            ...     1: {
            ...         "acc": np.array([[0.0, 0.0, 9.81], [0.0, 0.0, 9.81], [0.0, 0.0, 9.81]]),
            ...         "gyr": np.array([[0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]]),
            ...         "mag": ...
            ...     }
            >>> }
            >>> quaternions = solver.step(imu_data)
            >>> print(quaternions)
            {0: array([[...]]), 1: array([[...]])}
        """  # noqa: E501
        self._graph = Graph(graph)
        self._graph.assert_valid()

        for sub_solver in solutions:
            if isinstance(sub_solver, Online_RelOri_1D2D3D_Solution):
                assert (
                    Ts == 0.01
                ), "Currently `Online_RelOri_1D2D3D` only supports 100Hz; "
                "Resample using e.g. `qmt.nanInterp`"
            sub_solver.setTs(Ts)

        self._sub_solvers = solutions
        self.reset()

    def step(self, imu_data: D):
        """
        Processes IMU data to compute body-to-earth (body-to-world) orientations.

        Args:
            imu_data (D): A dictionary where keys are body indices (integers),
                and values are dictionaries with sensor data. Each sensor data dictionary
                must contain the following keys:
                - "acc" (np.ndarray): Accelerometer data, either 1D or 2D (time-batched).
                - "gyr" (np.ndarray): Gyroscope data, either 1D or 2D (time-batched).
                - "mag" (np.ndarray, optional): Magnetometer data, either 1D or 2D
                (time-batched).

        Returns:
            dict[int, np.ndarray]: A dictionary mapping body indices to quaternion arrays,
                representing body-to-world orientations. If the input is batched, the
                quaternions will have shape `(T, 4)` for each body, where `T` is the batch
                size. For non-batched input, the shape will be `(4,)` for each body.

        Notes:
            - The function supports both batched and non-batched inputs. Batched inputs
            are identified if the accelerometer data has more than one dimension.
            - The forward kinematics are applied to combine the orientations of all bodies
            in the system.
        """  # noqa: E501
        quats = self._step(imu_data)
        quats = self._graph.forward_kinematics([q for q in quats])
        # returns 1D or 2D array of body-to-eps quaterions
        return {i: q for i, q in enumerate(quats)}

    def _step(self, imu_data: D) -> np.ndarray:
        """Returns array of body-to-parent orientations with shape (n_bodies, T, 4) or
        (n_bodies, 4)"""
        shape = imu_data[0]["acc"].shape
        batched = len(shape) > 1
        if batched:
            T = shape[0]
            quats = np.zeros((self._graph.n_bodies(), T, 4))
        else:
            T = None
            quats = np.zeros((self._graph.n_bodies(), 4))

        for i, sub_solver in enumerate(self._sub_solvers):
            p = self._graph.parent(i)

            if p in imu_data:
                imu_data_p = imu_data[p]
                acc1, gyr1 = imu_data_p["acc"], imu_data_p["gyr"]
                mag1 = imu_data_p["mag"] if "mag" in imu_data_p else None
            else:
                assert p == -1
                acc1 = gyr1 = mag1 = None

            imu_data_i = imu_data[i]
            acc2, gyr2 = imu_data_i["acc"], imu_data_i["gyr"]
            mag2 = imu_data_i["mag"] if "mag" in imu_data_i else None

            quats[i] = sub_solver.apply(
                acc1=acc1, gyr1=gyr1, mag1=mag1, acc2=acc2, gyr2=gyr2, mag2=mag2, T=T
            )

        return quats

    def reset(self):
        "Resets all sub-solvers to their initial states."
        for sub_solver in self._sub_solvers:
            sub_solver.reset()
