import copy
from typing import Sequence

import numpy as np
import tree


def _tree_batch(
    trees: Sequence,
):
    trees = tree.map_structure(lambda arr: arr[None], trees)

    if len(trees) == 0:
        return trees
    if len(trees) == 1:
        return trees[0]

    return tree.map_structure(lambda *arrs: np.concatenate(arrs, axis=0), *trees)


def _select_t(kwargs, t):
    def f(a):
        if a is None:
            return a
        return a[t]

    return tree.map_structure(f, kwargs)


class Method:
    def apply(
        self, T: int | None, **kwargs
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        "Returns from child-to-body orientation"
        if T is None:
            return self._apply_timestep(**kwargs)

        quats = np.zeros((T, 4))
        extras = []
        for t in range(T):
            quat, extra = self._apply_timestep(**_select_t(kwargs, t))
            quats[t] = quat
            extras.append(extra)
        return quats, _tree_batch(extras)

    def _apply_timestep(self, **kwargs) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        raise NotImplementedError

    def reset(self) -> None:
        "Guaranteed to be called before first usage of `apply`"
        pass

    def setTs(self, Ts: float) -> None:
        self._Ts = Ts

    def getTs(self) -> float:
        return self._Ts

    @property
    def Ts(self):
        raise Exception("Use `getTs()`")

    @Ts.setter
    def Ts(self, value):
        raise Exception("Use `setTs()`")

    def copy(self):
        return copy.deepcopy(self)

    @property
    def unwrapped(self):
        return self


class MethodWrapper(Method):
    def __init__(self, method: Method):
        self._method = method

    @property
    def unwrapped(self):
        return self._method.unwrapped

    def reset(self):
        self._method.reset()

    def apply(self, T: int | None, **kwargs):
        return self._method.apply(T=T, **kwargs)

    def _apply_unrolled(
        self, T: int | None, **kwargs
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if T is None:
            return self._apply_timestep(**kwargs)

        quats = np.zeros((T, 4))
        extras = []
        for t in range(T):
            quat, extra = self._apply_timestep(**_select_t(kwargs, t))
            quats[t] = quat
            extras.append(extra)
        return quats, _tree_batch(extras)

    def _apply_timestep(self, **kwargs) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        raise NotImplementedError

    def setTs(self, Ts: float):
        self._method.setTs(Ts)

    def getTs(self) -> float:
        return self._method.getTs()

    @property
    def Ts(self):
        raise Exception("Use `getTs()`")

    @Ts.setter
    def Ts(self, value):
        raise Exception("Use `setTs()`")

    def copy(self):
        _method = self._method
        self._method = None
        copy_self = copy.deepcopy(self)
        copy_self._method = _method.copy()
        self._method = _method
        return copy_self
