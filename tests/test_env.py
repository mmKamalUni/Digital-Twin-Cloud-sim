"""Sanity tests for the synchronisation environment (synthetic data, no downloads needed).

Run with:  python -m pytest tests -q
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from dtsync.data import make_synthetic  # noqa: E402
from dtsync.env import ChannelCfg, EnvCfg  # noqa: E402
from dtsync.evaluate import run_policy  # noqa: E402
from dtsync.policies.heuristics import DeltaThreshold, FixedRate  # noqa: E402

DS = make_synthetic(S=24, T=3000, burst_corr=0.5, seed=0)


def test_send_on_delta_ideal_never_violates():
    m = run_policy(DS, EnvCfg(predictor="zoh", channel=ChannelCfg("ideal")), DeltaThreshold(1.0), "test")
    assert m["sla_violation_pct"] == 0.0
    assert 0.0 < m["send_rate_pct"] < 100.0


def test_fixed_rate_send_rate():
    m = run_policy(DS, EnvCfg(predictor="zoh", channel=ChannelCfg("ideal")), FixedRate(4), "test")
    assert abs(m["send_rate_pct"] - 25.0) < 1.0


def test_shared_gateway_adds_delay_and_errors():
    ideal = run_policy(DS, EnvCfg(predictor="zoh", channel=ChannelCfg("ideal")), DeltaThreshold(1.0), "test")
    tight = run_policy(DS, EnvCfg(predictor="zoh", channel=ChannelCfg("shared", capacity=1, buffer=8)),
                       DeltaThreshold(1.0), "test")
    assert tight["latency_mean_s"] > ideal["latency_mean_s"]
    assert tight["sla_violation_pct"] >= ideal["sla_violation_pct"]


def test_metrics_in_range():
    m = run_policy(DS, EnvCfg(predictor="holt", phi=0.9, channel=ChannelCfg("shared", capacity=4, buffer=32)),
                   DeltaThreshold(1.0), "test")
    assert 0.0 <= m["data_quality"] <= 1.0
    assert 0.0 <= m["event_detection_pct"] <= 100.0
    assert np.isfinite(m["bandwidth_Bps"])
