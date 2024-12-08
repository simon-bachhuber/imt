from typing import Optional


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


def print_graph(graph: list[int | str], body_names: Optional[list[str]] = None) -> None:
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
