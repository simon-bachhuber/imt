from typing import Optional

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

    def print(self, body_names: Optional[list[str]] = None) -> None:
        _print_graph(self.lam, body_names=body_names)


# Function to build a tree structure
def _build_tree(body_names, parents):
    nodes = {name: [] for name in body_names}  # Initialize empty adjacency list
    for i, parent_index in enumerate(parents):
        if parent_index == -1:  # Root node
            root = body_names[i]
        else:
            parent_name = body_names[parent_index]
            nodes[parent_name].append(body_names[i])
    return root, nodes


# Recursive function to create the tree visualization
def _visualize_tree(root, nodes, prefix=""):
    lines = []
    children = nodes[root]
    for i, child in enumerate(children):
        is_last = i == len(children) - 1
        connector = "└── " if is_last else "├── "
        sub_prefix = "    " if is_last else "│   "
        lines.append(f"{prefix}{connector}{child}")
        lines.extend(_visualize_tree(child, nodes, prefix + sub_prefix))
    return lines


def _print_graph(
    graph: list[int | str], body_names: Optional[list[str]] = None
) -> None:
    if body_names is not None:
        body_numbers = {name: i for i, name in enumerate(body_names)}
        graph = [i if isinstance(i, int) else body_numbers[i] for i in graph]
    else:
        body_names = [str(i) for i in range(len(graph))]

    lines = _visualize_tree(*_build_tree(body_names, graph))
    header = []
    header.append("-1 (Earth)")
    header.append("|")
    header.append(body_names[0])

    for line in header + lines:
        print(line)
