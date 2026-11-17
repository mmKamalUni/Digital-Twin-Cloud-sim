"""Shared experiment configuration and helpers."""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from dtsync.agents.dqn import DQNCfg, DQNPolicy, SLAController, train_dqn  # noqa: E402
from dtsync.agents.ppo import PPOCfg, train_ppo  # noqa: E402
from dtsync.data import load_dataset  # noqa: E402
from dtsync.env import ChannelCfg, EnvCfg, RewardCfg  # noqa: E402
from dtsync.evaluate import flat, run_policy  # noqa: E402
from dtsync.policies.heuristics import (CongestionAdaptiveThreshold, DeltaThreshold,  # noqa: E402
                                        FixedRate, RandomRate)

RESULTS = os.path.join(ROOT, "results")
MODELS = os.path.join(RESULTS, "models")
os.makedirs(MODELS, exist_ok=True)

SEEDS = [0, 1, 2]
PHI = 0.9
TRAIN_SLOTS = 20000
SLA_TARGET = 0.02            # 98 % of stream-time within tolerance
CONGESTION_PRICE = 0.3

IDEAL = ChannelCfg("ideal")
SHARED = ChannelCfg("shared", capacity=12, buffer=96)


def shared(capacity: float = 12) -> ChannelCfg:
    return ChannelCfg("shared", capacity=capacity, buffer=8 * capacity)


# ----------------------------------------------------------------------------- variants
# Each proposed component can be switched on independently:
#   A = mirrored predictive twin + predictive features
#   B = congestion awareness (gateway feedback features + congestion-priced cost)
#   C = SLA-constrained, lambda-conditioned decomposed Q with cloud-side lambda controller
def variant(A: bool, B: bool, C: bool, channel: ChannelCfg = SHARED) -> tuple[EnvCfg, str]:
    obs = {(False, False): "huang", (True, False): "pred", (False, True): "cong", (True, True): "full"}[(A, B)]
    reward = RewardCfg("lagrangian" if C else "huang", congestion_price=CONGESTION_PRICE if B else 0.0)
    return EnvCfg(predictor="holt" if A else "zoh", phi=PHI, channel=channel, reward=reward), obs


def variant_name(A, B, C):
    tags = [t for t, f in zip("ABC", (A, B, C)) if f]
    return "DQN-ref (retrained shared)" if not tags else "DQN-ref + " + "+".join(tags)


def dqn_cfg(constrained_or_shared: bool = True, slots: int = TRAIN_SLOTS) -> DQNCfg:
    # sparse exploration for congested training; the reference reproduction keeps the
    # uniform epsilon-greedy exploration of standard DQN
    return DQNCfg(total_slots=slots, warmup_slots=1000,
                  explore_send_p=0.2 if constrained_or_shared else 0.5)


def model_path(tag: str, seed: int) -> str:
    return os.path.join(MODELS, f"{tag}_s{seed}.pt")


def get_dqn(tag, ds, env_cfg, obs, cfg: DQNCfg, seed, log=None):
    """Train (or load a cached) DQN policy."""
    p = model_path(tag, seed)
    if os.path.exists(p):
        pol = DQNPolicy.load(p)
        info = json.load(open(p + ".json"))
        return pol, info
    t = time.time()
    pol, info = train_dqn(ds, env_cfg, obs, cfg, seed=seed, verbose=False)
    torch.save(pol.state(), p)
    info = {k: v for k, v in info.items() if k != "history"} | {"history": info["history"]}
    json.dump(info, open(p + ".json", "w"))
    if log:
        log(f"trained {tag} seed {seed} in {time.time() - t:.0f}s")
    return pol, info


def _train_job(job):
    torch.set_num_threads(2)
    kind, tag, ds_name, ds_kw, env_cfg, obs, cfg, seed = job
    ds = load_dataset(ds_name, root=os.path.join(ROOT, "data"), **ds_kw)
    p = model_path(tag, seed)
    if os.path.exists(p):
        return tag, seed, "cached"
    t = time.time()
    if kind == "dqn":
        pol, info = train_dqn(ds, env_cfg, obs, cfg, seed=seed, verbose=False)
        torch.save(pol.state(), p)
    else:
        pol, info = train_ppo(ds, env_cfg, obs, cfg, seed=seed, verbose=False)
        torch.save(dict(ac=pol.ac.state_dict(), obs_kind=obs, d=pol.ac.pi[0].in_features,
                        hidden=pol.ac.pi[0].out_features), p)
    json.dump(info, open(p + ".json", "w"))
    return tag, seed, f"{time.time() - t:.0f}s"


def run_jobs(jobs, workers: int = 6, log=print):
    """Train many agents in parallel processes (results are cached on disk)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    todo = [j for j in jobs if not os.path.exists(model_path(j[1], j[7]))]
    log(f"{len(jobs)} training jobs, {len(todo)} to run")
    if not todo:
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_train_job, j) for j in todo]
        for f in as_completed(futs):
            tag, seed, msg = f.result()
            log(f"  done {tag} seed {seed} ({msg})")


def load_agent(tag, seed):
    p = model_path(tag, seed)
    s = torch.load(p, weights_only=False)
    info = json.load(open(p + ".json"))
    if "ac" in s:
        from dtsync.agents.ppo import ActorCritic, PPOPolicy
        ac = ActorCritic(s["d"], s["hidden"])
        ac.load_state_dict(s["ac"])
        return PPOPolicy(ac, s["obs_kind"]), info
    return DQNPolicy.load(p), info


def agg(rows: list[dict], by: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    num = [c for c in df.columns if c not in by and pd.api.types.is_numeric_dtype(df[c])]
    g = df.groupby(by, sort=False)[num]
    mean, std = g.mean(), g.std().fillna(0.0)
    out = mean.copy()
    for c in num:
        out[c + "_std"] = std[c]
    return out.reset_index()


def save(df: pd.DataFrame, name: str):
    path = os.path.join(RESULTS, name)
    df.to_csv(path, index=False)
    print(f"saved {path}")


def logger(name):
    path = os.path.join(RESULTS, f"{name}.log")
    f = open(path, "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        f.write(line + "\n")
        f.flush()
    return log


# ----------------------------------------------------------------------------- helpers
RHO_TARGET = 0.38   # mean gateway utilisation of send-on-delta; gives 12 msg/slot on Intel


def capacity_rule(ds) -> float:
    """Provision the gateway so plain send-on-delta loads it to RHO_TARGET on average."""
    m = run_policy(ds, EnvCfg(predictor="zoh", channel=IDEAL), DeltaThreshold(1.0), "train", 0)
    load = m["send_rate_pct"] / 100.0 * ds.S
    return float(max(1.0, round(load / RHO_TARGET, 1)))


def proposed(seed: int, tag: str = "abl_ABC", target: float = SLA_TARGET):
    """Full method (A+B+C) with the cloud-side SLA controller."""
    pol, info = load_agent(tag, seed)
    cal = json.load(open(os.path.join(RESULTS, "exp3_lambda_cal.json")))
    lam_star = cal[f"{tag}_s{seed}"]["lam_star"]
    pol.controller = SLAController(target, lam0=min(2.0, lam_star), hi=lam_star)
    return pol, info
