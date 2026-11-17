"""Rule-based synchronisation policies used as baselines."""
from __future__ import annotations

import numpy as np

from ..env import SyncEnv


class FixedRate:
    """Push every ``period`` slots; phases are staggered so the fleet does not fire at once."""
    always_sense = False

    def __init__(self, period: int = 4, seed: int = 0):
        self.period = period
        self.seed = seed

    def __call__(self, env: SyncEnv):
        if not hasattr(self, "_phase") or len(self._phase) != env.S:
            self._phase = np.random.default_rng(self.seed).integers(0, self.period, env.S)
        return ((env.t + self._phase) % self.period) == 0


class RandomRate:
    always_sense = False

    def __init__(self, p: float = 0.25, seed: int = 0):
        self.p, self.rng = p, np.random.default_rng(seed)

    def __call__(self, env: SyncEnv):
        return self.rng.random(env.S) < self.p


class DeltaThreshold:
    """Send-on-delta: push when the twin deviates from the reading by >= delta * eps.

    With ``env.cfg.predictor == 'holt'`` this becomes the classic dual-prediction scheme.
    """
    always_sense = True

    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def __call__(self, env: SyncEnv):
        return np.abs(env.X[env.t] - env.mirror_pred()) >= self.delta * env.eps


class CongestionAdaptiveThreshold:
    """Heuristic congestion control: the trigger threshold grows with gateway backlog
    (similar in spirit to AIMD/credit back-off).  Used as a strong hand-crafted comparator."""
    always_sense = True

    def __init__(self, delta: float = 0.8, gain: float = 0.5, guard: float = 0.5):
        self.delta, self.gain, self.guard = delta, gain, guard

    def __call__(self, env: SyncEnv):
        cap = env.cfg.channel.capacity if env.cfg.channel.kind != "ideal" else 1e9
        backlog_slots = env.W / cap
        thr = self.delta * (1.0 + self.gain * backlog_slots)
        dev = np.abs(env.X[env.t] - env.mirror_pred()) / env.eps
        # never let a stream sit beyond the SLA bound without an update in flight
        need = (np.abs(env.X[env.t] - env.cloud_pred()) / env.eps > 1.0 + self.guard) & (env.inflight == 0)
        return (dev >= thr) | need
