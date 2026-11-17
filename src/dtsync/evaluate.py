"""Policy roll-out over a dataset split and metric collection."""
from __future__ import annotations

import copy
import time

import numpy as np

from .data import TelemetryDataset
from .env import EnvCfg, SyncEnv


def run_policy(ds: TelemetryDataset, cfg: EnvCfg, policy, split: str = "test", seed: int = 0,
               start: int | None = None, length: int | None = None, return_env: bool = False):
    cfg = copy.deepcopy(cfg)
    cfg.always_sense = getattr(policy, "always_sense", True)
    env = SyncEnv(ds, cfg, seed=seed)
    a, b = ds.splits[split]
    s = a if start is None else start
    L = (b - s) if length is None else length
    env.reset(s, L)
    if hasattr(policy, "reset"):
        policy.reset(env)
    t0 = time.perf_counter()
    n_dec = 0
    while not env.done:
        act = policy(env)
        env.step(act)
        n_dec += env.S
    wall = time.perf_counter() - t0
    m = env.metrics()
    m["decision_us"] = 1e6 * wall / max(n_dec, 1)
    if return_env:
        return m, env
    return m


def flat(m: dict) -> dict:
    out = {k: v for k, v in m.items() if not isinstance(v, dict)}
    for k, v in m.get("group_mae", {}).items():
        out[f"mae_{k}"] = v
    return out
