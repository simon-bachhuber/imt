import numpy as np
import tree

from imt._graph import Graph
from imt._solutions import Online_RelOri_1D2D3D
from imt._solutions import Solution
from imt._solutions import VQF_Solution

# body number <integer> -> {'acc': ..., 'gyr': ..., 'mag': ... (optional)}
D = dict[int, dict[str, np.ndarray]]


_valid_problem_strings = ["1D2D3D"]


class Solver:

    def __init__(self, graph: list[int], problem: str, Ts: float):
        """
        Initializes the `Solver` with a graph of bodies, problem type, and sampling time.

        Args:
            graph (list[int]): A list representing the parent-child relationships between
                bodies. Each index corresponds to a body, and the value at that index
                specifies the parent body. The special integer `-1` represents the
                "world body" (i.e., a root node with no parent).
            problem (str): A string specifying the problem type. Must be one of the
                predefined valid problem types (e.g., "1D2D3D").
            Ts (float): The sampling time in seconds. Currently, only `0.01` (100 Hz)
                is supported. Resampling should be done externally if the data is at
                a different frequency.

        Raises:
            AssertionError: If `problem` is not in the list of valid problem strings.
            AssertionError: If `Ts` is not `0.01`.

        Notes:
            - The `Solver` creates sub-solvers for each body in the graph.
            For world bodies (`parent == -1`), the sub-solver uses a default
            orientation solution (`VQF_Solution`). For other bodies, it uses a
            relative orientation solution (`Online_RelOri_1D2D3D`).


        Example:
            >>> # Define a graph with one body connecting to world (0) and
            >>> # two child bodies (1 and 2)
            >>> graph = [-1, 0, 0]
            >>> problem = "1D2D3D"
            >>> Ts = 0.01  # Sampling time (100 Hz)
            >>> solver = Solver(graph, problem, Ts)
            >>>
            >>> # Define IMU data for the bodies
            >>> imu_data = {
            ...     0: {"acc": np.array([0.0, 0.0, 9.81]), "gyr": np.array([0.1, 0.2, 0.3])},
            ...     1: {"acc": np.array([0.0, 0.0, 9.81]), "gyr": np.array([0.2, 0.3, 0.4])},
            ...     2: {"acc": np.array([0.0, 0.0, 9.81]), "gyr": np.array([0.3, 0.4, 0.5])},
            ... }
            >>> # Process the IMU data to compute body-to-world orientations
            >>> quaternions = solver.step(imu_data)
            >>> print(quaternions)
            {0: array([...]), 1: array([...]), 2: array([...])}
            >>> # reset the solver afterwards
            >>> solver.reset()
            >>> # `.step` also accepts time-batched data
            >>> imu_data = {
            ...     0: {
            ...         "acc": np.array([[0.0, 0.0, 9.81], [0.0, 0.0, 9.81], [0.0, 0.0, 9.81]]),
            ...         "gyr": np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]])
            ...     },
            ...     1: {
            ...         "acc": np.array([[0.0, 0.0, 9.81], [0.0, 0.0, 9.81], [0.0, 0.0, 9.81]]),
            ...         "gyr": np.array([[0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]])
            ...     }
            ...     2: {
            ...         "acc": np.array([[0.0, 0.0, 9.81], [0.0, 0.0, 9.81], [0.0, 0.0, 9.81]]),
            ...         "gyr": np.array([[0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]])
            ...     }
            >>> }
            >>> quaternions = solver.step(imu_data)
            >>> print(quaternions)
            {0: array([[...]]), 1: array([[...]]), 2: array([[...]])}
        """  # noqa: E501
        self._graph = Graph(graph)
        self._graph.assert_valid()

        self._problem = problem
        assert problem in _valid_problem_strings

        assert (
            Ts == 0.01
        ), "Currently, only 100Hz supported; Resample using e.g. `qmt.nanInterp`"

        self._sub_solvers: list[Solution] = [
            VQF_Solution(Ts) if self._graph.parent(i) == -1 else Online_RelOri_1D2D3D()
            for i in range(self._graph.n_bodies())
        ]

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
        # imu_data must be a dictionary that contains arrays that are either 1D or 2D
        shape = imu_data[0]["acc"].shape
        batched = len(shape) > 1

        if batched:
            T = shape[0]
            quats = np.zeros((T, self._graph.n_bodies(), 4))
            for t in range(T):
                quats[t] = self._step(tree.map_structure(lambda a: a[t], imu_data))
            quats = quats.transpose((1, 0, 2))
        else:
            quats = self._step(imu_data)

        quats = self._graph.forward_kinematics([q for q in quats])
        # returns 1D or 2D array of body-to-eps quaterions
        return {i: q for i, q in enumerate(quats)}

    def _step(self, imu_data: D) -> np.ndarray:
        "Returns array of body-to-parent orientations with shape (n_bodies, 4)"
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
                acc1=acc1, gyr1=gyr1, mag1=mag1, acc2=acc2, gyr2=gyr2, mag2=mag2
            )

        return quats

    def reset(self):
        "Resets all sub-solvers to their initial states."
        for sub_solver in self._sub_solvers:
            sub_solver.reset()
