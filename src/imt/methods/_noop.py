import numpy as np

from .._base import Method


class NoOpMethod(Method):
    def _apply_timestep(self, **kwargs):
        return np.array([1.0, 0, 0, 0]), {}
