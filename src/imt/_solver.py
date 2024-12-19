from typing import Any, Optional

import numpy as np

import imt
from imt._graph import Graph


class Solver:

    def __init__(
        self,
        graph: list[int | str],
        methods: list[imt.Method] | None,
        Ts: float,
        body_names: Optional[list[str]] = None,
    ):
        """
        Initializes the `Solver` with a graph of bodies, problem type, and sampling time.

        Args:
            graph (list[int | str]): A list representing the parent-child relationships between
                bodies. Each index corresponds to a body, and the value at that index
                specifies the parent body. The special integer `-1` represents the
                "world body" (i.e., a root node with no parent).
            methods (list[Method] | None): A list of methods that are used for each two-
                segment sub-kinematic-chain in the problem to solve for the relative
                orientation between the two. Note that the connection to earth is also
                considered a two-segment chain, so for bodies that connect to `-1` it
                is recommended to use `imt.methods.VQF`.
                If it is set to `None` a default set of methods will be used that is
                determined based on the graph structure.
            Ts (float): The sampling time in seconds. Currently, only `0.01` (100 Hz)
                is supported. Resampling should be done externally if the data is at
                a different frequency.
            body_names (list[string]): Can be used to give names to the bodies instead
                of the default integer 0..N naming scheme. If names are provided, then
                the graph array can be provided as a list of those body names.
                Defaults to None.

        Example:
            >>> # Define a graph with one body (body 0) connecting to world (-1) and
            >>> # one body (body 1) connecting to body 0
            >>> graph = [-1, 0]
            >>> methods = None
            >>> Ts = 0.01  # Sampling time (100 Hz)
            >>> solver = Solver(graph, methods, Ts)
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
            >>> quaternions, _ = solver.step(imu_data)
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
            >>> quaternions, _ = solver.step(imu_data)
            >>> print(quaternions)
            {0: array([[...]]), 1: array([[...]])}
        """  # noqa: E501
        if body_names is not None:
            assert len(body_names) == len(
                graph
            ), "Partially naming bodies is not supported, either name all bodies or "
            "none"
            assert len(set(body_names)) == len(body_names), "Duplicated names"
            # maps body names to body numbers
            graph = self._guarantuee_body_numbers(body_names, graph)

        # maps body numbers to body names
        self._body_names = body_names

        self._graph = Graph(graph)
        self._graph.assert_valid()

        # load default methods
        if methods is None:
            methods = [
                (
                    imt.methods.VQF()
                    if p == -1
                    else imt.wrappers.FractualStepping(imt.methods.RNNO(), 100.0)
                )
                for p in graph
            ]

        # create deep-copy because the user might have used one instance mutliple times
        methods = [m.copy() for m in methods]

        for sub_solver in methods:
            sub_solver.setTs(Ts)

        self._sub_solvers = methods
        self.reset()

    def step(
        self, imu_data: dict[int | str, dict[str, np.ndarray]]
    ) -> tuple[dict[int | str, np.ndarray], dict[int | str, dict[str, np.ndarray]]]:
        """
        Processes IMU data to compute body-to-earth (body-to-world) orientations.

        Args:
            imu_data (dict[int | str, dict]): A dictionary where keys are body indices or names,
                and values are dictionaries with sensor data. Each sensor data dictionary
                must contain the following keys:
                - "acc" (np.ndarray): Accelerometer data, either 1D or 2D (time-batched).
                - "gyr" (np.ndarray): Gyroscope data, either 1D or 2D (time-batched).
                - "mag" (np.ndarray, optional): Magnetometer data, either 1D or 2D
                (time-batched).

        Returns:
            dict[int | str, np.ndarray]: A dictionary mapping body indices to quaternion arrays,
                representing body-to-world orientations. If the input is batched, the
                quaternions will have shape `(T, 4)` for each body, where `T` is the batch
                size. For non-batched input, the shape will be `(4,)` for each body.

        Notes:
            - The function supports both batched and non-batched inputs. Batched inputs
            are identified if the accelerometer data has more than one dimension.
            - The forward kinematics are applied to combine the orientations of all bodies
            in the system.
        """  # noqa: E501
        imu_data = self._guarantuee_body_numbers(self._body_names, imu_data)
        quats, extras = self._step(imu_data)
        quats = self._graph.forward_kinematics([q for q in quats])

        # returns 1D or 2D array of body-to-eps quaterions
        return self._maybe_rename(
            {i: q for i, q in enumerate(quats)}
        ), self._maybe_rename(extras)

    @staticmethod
    def _guarantuee_body_numbers(
        body_names: list[str] | None, obj: dict[int | str, Any] | list[int | str]
    ) -> dict[int, Any] | list[int]:
        if body_names is not None:
            body_numbers = {name: i for i, name in enumerate(body_names)}
            _rename_str = lambda i: i if isinstance(i, int) else body_numbers[i]
            if isinstance(obj, dict):
                obj = {_rename_str(i): v for i, v in obj.items()}
            elif isinstance(obj, list):
                obj = [_rename_str(i) for i in obj]
            else:
                raise NotImplementedError
        else:
            assert all(
                [isinstance(i, int) for i in obj]
            ), "found named bodies but `body_names` was not specified"
        return obj

    def _maybe_rename(self, data: dict[int, Any]) -> dict[int | str, Any]:
        if self._body_names is None:
            return data
        return {self._body_names[i]: v for i, v in data.items()}

    def _step(
        self, imu_data: dict[int, dict[str, np.ndarray]]
    ) -> tuple[np.ndarray, dict[int, dict[str, np.ndarray]]]:
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

        extras = {}
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

            quat, extra = sub_solver.apply(
                acc1=acc1, gyr1=gyr1, mag1=mag1, acc2=acc2, gyr2=gyr2, mag2=mag2, T=T
            )

            extras[i] = extra
            quats[i] = quat

        return quats, extras

    def reset(self):
        "Resets all internal state"
        for sub_solver in self._sub_solvers:
            sub_solver.reset()

    def print_graph(self):
        self._graph.print(self._body_names)
