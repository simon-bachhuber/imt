from .._base import Method
from .._base import MethodWrapper


class FractualStepping(MethodWrapper):
    """Simple online resampler based on fractual stepping principle"""

    def __init__(self, method: Method, method_sampling_rate: float):
        super().__init__(method)
        self.internal_sampling_rate = method_sampling_rate

    def setTs(self, Ts: float):
        self._Ts = Ts
        self.external_sampling_rate = 1 / Ts
        return super().setTs(1 / self.internal_sampling_rate)

    def getTs(self) -> float:
        return self._Ts

    def apply(self, T, **kwargs):
        return self._apply_unrolled(T, **kwargs)

    def _apply_timestep(self, **kwargs):
        self.time_counter += 1
        # Check if enough time has passed to step the internal method
        while (self.time_counter >= self.step_interval) or (self.last_e is None):
            self.last_e = super().apply(T=None, **kwargs)
            self.time_counter -= self.step_interval
        return self.last_e

    def reset(self):
        self.step_interval = self.external_sampling_rate / self.internal_sampling_rate
        self.time_counter = 0  # Tracks the relative time for fractional stepping
        self.last_e = None
        return super().reset()
