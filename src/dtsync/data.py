"""Dataset loading and preprocessing.

Three telemetry workloads are supported:

* ``intel``     - Intel Berkeley Research Lab sensor data (54 Mica2Dot motes, 2004).
                  Streams = (mote, variable) pairs for temperature, humidity and light.
* ``nab``       - Numenta Anomaly Benchmark, realKnownCause + realAWSCloudwatch series
                  (machine temperature, EC2/RDS CPU, network, request counts).
* ``synthetic`` - Gaussian + Poisson burst generator (extension of the original
                  ``telemetry.py``) with a tunable cross-stream burst correlation.

Every loader returns a :class:`TelemetryDataset` holding a dense ``(T, S)`` matrix on a
regular slot grid plus the per-stream SLA tolerance ``eps``.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class TelemetryDataset:
    name: str
    X: np.ndarray                 # (T, S) float32 physical values
    eps: np.ndarray               # (S,) SLA tolerance per stream (physical units)
    stream_names: list
    stream_group: np.ndarray      # (S,) integer variable/group id
    group_names: list
    dt: float                     # slot length in seconds
    splits: dict = field(default_factory=dict)  # name -> (start, end) slot indices

    @property
    def T(self):
        return self.X.shape[0]

    @property
    def S(self):
        return self.X.shape[1]

    def split(self, name):
        a, b = self.splits[name]
        return self.X[a:b]

    def summary(self):
        rows = []
        for g, gname in enumerate(self.group_names):
            m = self.stream_group == g
            if not m.any():
                continue
            x = self.X[:, m]
            d = np.abs(np.diff(x, axis=0))
            rows.append(dict(group=gname, streams=int(m.sum()), mean=float(x.mean()),
                             std=float(x.std()), min=float(x.min()), max=float(x.max()),
                             eps_mean=float(self.eps[m].mean()),
                             p99_step=float(np.percentile(d, 99)),
                             frac_steps_gt_eps=float((d > self.eps[m]).mean())))
        return pd.DataFrame(rows)


def _hampel(x: np.ndarray, k: int = 2, n_sigma: float = 6.0) -> np.ndarray:
    """Remove isolated spikes (sensor glitches) column-wise; persistent steps are kept."""
    df = pd.DataFrame(x)
    med = df.rolling(2 * k + 1, center=True, min_periods=1).median()
    mad = (df - med).abs().rolling(2 * k + 1, center=True, min_periods=1).median()
    scale = 1.4826 * mad + 1e-6
    out = df.where((df - med).abs() <= n_sigma * scale + 1e-3 * med.abs().clip(lower=1.0), med)
    return out.to_numpy()


def _range_eps(X_train: np.ndarray, frac: float) -> np.ndarray:
    lo = np.percentile(X_train, 1, axis=0)
    hi = np.percentile(X_train, 99, axis=0)
    return np.maximum(frac * (hi - lo), 1e-6)


def _make_splits(T: int, fr=(0.6, 0.1, 0.3)):
    a = int(T * fr[0])
    b = int(T * (fr[0] + fr[1]))
    return {"train": (0, a), "val": (a, b), "test": (b, T)}


# --------------------------------------------------------------------------------------
# Intel Berkeley Lab
# --------------------------------------------------------------------------------------
INTEL_VARS = ["temp", "hum", "light"]
INTEL_VALID = {"temp": (5.0, 40.0), "hum": (5.0, 100.0), "light": (0.0, 2000.0)}


def load_intel(path: str, epoch_max: int = 30000, min_coverage: float = 0.5,
               eps_frac: float = 0.03, max_gap: int = 40, cache: str | None = None) -> TelemetryDataset:
    if cache and os.path.exists(cache):
        return load_npz(cache)
    cols = ["date", "time", "epoch", "mote", "temp", "hum", "light", "volt"]
    df = pd.read_csv(path, sep=r"\s+", header=None, names=cols, on_bad_lines="skip")
    df = df.dropna(subset=["mote", "temp", "hum", "light"])
    df = df[(df.mote >= 1) & (df.mote <= 54) & (df.epoch >= 0) & (df.epoch < epoch_max)]
    for v, (lo, hi) in INTEL_VALID.items():
        df = df[df[v].between(lo, hi)]
    df["mote"] = df.mote.astype(int)
    g = df.groupby(["epoch", "mote"])[INTEL_VARS].mean()
    coverage = g.groupby(level="mote").size() / epoch_max
    motes = sorted(coverage[coverage >= min_coverage].index.tolist())

    mats, names, groups = [], [], []
    grid = pd.RangeIndex(0, epoch_max)
    for vi, v in enumerate(INTEL_VARS):
        piv = g[v].unstack("mote").reindex(index=grid, columns=motes)
        piv = piv.interpolate(limit=max_gap, limit_area="inside").ffill().bfill()
        x = _hampel(piv.to_numpy(dtype=np.float64))
        mats.append(x)
        names += [f"mote{m:02d}_{v}" for m in motes]
        groups += [vi] * len(motes)
    X = np.concatenate(mats, axis=1).astype(np.float32)
    splits = _make_splits(len(X))
    a, b = splits["train"]
    ds = TelemetryDataset("intel", X, _range_eps(X[a:b], eps_frac).astype(np.float32), names,
                          np.array(groups), INTEL_VARS, dt=31.0, splits=splits)
    if cache:
        save_npz(ds, cache)
    return ds


# --------------------------------------------------------------------------------------
# Numenta Anomaly Benchmark
# --------------------------------------------------------------------------------------
NAB_GROUPS = ["cpu", "network", "requests", "temperature", "other"]


def _nab_group(fname: str) -> int:
    f = fname.lower()
    if "cpu" in f or "grok" in f:
        return 0
    if "network" in f or "disk" in f:
        return 1
    if "request" in f or "latency" in f or "taxi" in f:
        return 2
    if "temperature" in f:
        return 3
    return 4


def load_nab(folder: str, length: int = 4000, eps_frac: float = 0.03,
             cache: str | None = None) -> TelemetryDataset:
    if cache and os.path.exists(cache):
        return load_npz(cache)
    cols, names, groups = [], [], []
    for f in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        s = pd.read_csv(f)["value"].to_numpy(dtype=np.float64)
        if len(s) < length:
            continue
        s = s[:length]
        if np.std(s) < 1e-9:
            continue
        cols.append(s)
        base = os.path.splitext(os.path.basename(f))[0]
        names.append(base)
        groups.append(_nab_group(base))
    X = np.stack(cols, axis=1).astype(np.float32)
    splits = _make_splits(len(X))
    a, b = splits["train"]
    ds = TelemetryDataset("nab", X, _range_eps(X[a:b], eps_frac).astype(np.float32), names,
                          np.array(groups), NAB_GROUPS, dt=300.0, splits=splits)
    if cache:
        save_npz(ds, cache)
    return ds


# --------------------------------------------------------------------------------------
# Synthetic bursty workload (generalises the original TelemetryGenerator)
# --------------------------------------------------------------------------------------
def make_synthetic(S: int = 120, T: int = 20000, burst_corr: float = 0.6,
                   p_global: float = 0.004, p_local: float = 0.002, burst_mag: float = 5.0,
                   noise_std: float = 0.1, decay: float = 0.985, eps_frac: float = 0.03,
                   seed: int = 0) -> TelemetryDataset:
    """Mean-reverting baseline + Gaussian noise + Poisson bursts.

    ``burst_corr`` is the probability that a stream participates in a global (common-cause)
    burst, so it controls how strongly bursts are synchronised across the fleet.
    """
    rng = np.random.default_rng(seed)
    base = rng.uniform(50, 150, size=S)
    X = np.empty((T, S))
    level = np.zeros(S)
    burst = np.zeros(S)
    for t in range(T):
        level = 0.995 * level + rng.normal(0, noise_std, S)
        jump = np.zeros(S)
        if rng.random() < p_global:
            joins = rng.random(S) < burst_corr
            jump += joins * rng.choice([-1.0, 1.0]) * burst_mag * rng.uniform(0.5, 1.5, S)
        loc = rng.random(S) < p_local
        jump += loc * rng.choice([-1.0, 1.0], S) * burst_mag * rng.uniform(0.5, 1.5, S)
        burst = decay * burst + jump
        X[t] = base + level + burst
    splits = _make_splits(T)
    a, b = splits["train"]
    return TelemetryDataset(f"synthetic_rho{burst_corr:.2f}", X.astype(np.float32),
                            _range_eps(X[a:b], eps_frac).astype(np.float32),
                            [f"syn{i:03d}" for i in range(S)], np.zeros(S, dtype=int),
                            ["synthetic"], dt=1.0, splits=splits)


# --------------------------------------------------------------------------------------
def save_npz(ds: TelemetryDataset, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, X=ds.X, eps=ds.eps, stream_names=np.array(ds.stream_names),
                        stream_group=ds.stream_group, group_names=np.array(ds.group_names),
                        dt=ds.dt, name=ds.name,
                        splits=np.array([[k, *v] for k, v in ds.splits.items()], dtype=object))


def load_npz(path: str) -> TelemetryDataset:
    z = np.load(path, allow_pickle=True)
    splits = {str(r[0]): (int(r[1]), int(r[2])) for r in z["splits"]}
    return TelemetryDataset(str(z["name"]), z["X"], z["eps"], z["stream_names"].tolist(),
                            z["stream_group"], z["group_names"].tolist(), float(z["dt"]), splits)


def load_dataset(name: str, root: str = "data", **kw) -> TelemetryDataset:
    if name == "intel":
        return load_intel(os.path.join(root, "raw/intel/data.txt.gz"),
                          cache=os.path.join(root, "processed/intel.npz"), **kw)
    if name == "nab":
        tag = "_".join(f"{k}{v}" for k, v in sorted(kw.items()))
        return load_nab(os.path.join(root, "raw/nab"),
                        cache=os.path.join(root, f"processed/nab{('_' + tag) if tag else ''}.npz"), **kw)
    if name.startswith("synthetic"):
        return make_synthetic(**kw)
    raise ValueError(name)
