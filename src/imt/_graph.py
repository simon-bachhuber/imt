import numpy as np
import qmt
import tree


class Graph:
    def __init__(self, graph: list[int]):
        self.lam = graph

    def n_bodies(self) -> int:
        return len(self.lam)

    def bodies_to_world(self) -> list[int]:
        return [i for i in range(self.n_bodies()) if self.parent(i) == -1]

    def assert_valid(self) -> None:
        "Asssert that the provided graph is valid"
        assert len(self.bodies_to_world()) > 0

        for i in range(self.n_bodies()):
            assert self.parent(i) < i

    def parent(self, of: int) -> int:
        return self.lam[of]

    def children(self, of: int) -> list[int]:
        "List all direct children of body, does not include body itself"
        return [i for i in range(self.n_bodies()) if self.parent(i) == of]

    def ancestors(self, of: int) -> list[int]:
        "List all children and children's children; does not include body itself"
        children = self.children(of)
        grandchildren = [self.ancestors(n) for n in children]
        return tree.flatten([children, grandchildren])

    def forward_kinematics(self, qs: list[np.ndarray]) -> list[np.ndarray]:
        "Converts body-to-parent to body-to-eps"
        # convert from body-to-parent to parent-to-body
        qs = [qmt.qinv(q) for q in qs]
        # this dict will store eps-to-body/parent
        qs_eps = {-1: np.array([1.0, 0, 0, 0])}

        for i in range(self.n_bodies()):
            p = self.parent(i)
            qs_eps[i] = qmt.qmult(qs[i], qs_eps[p])

        return [qmt.qinv(qs_eps[i]) for i in range(self.n_bodies())]
