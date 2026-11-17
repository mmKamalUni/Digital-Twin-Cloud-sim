"""Experiment 5 - generalisation to other workloads.

(a) NAB cloud/industrial telemetry (21 series, 5-min slots): zero-shot transfer of the
    Intel-trained agents and agents retrained on NAB.
(b) Synthetic bursty fleet with controlled cross-stream burst correlation rho in [0, 1]:
    the gateway is provisioned once (capacity rule at rho = 0.5) so only burst
    synchronisation changes along the sweep.
Outputs: results/exp5_nab.csv, results/exp5_synthetic.csv
"""
from common import *  # noqa: F401,F403

from dtsync.env import EnvCfg

NAB_KW = dict(eps_frac=0.1)
SYN_KW = dict(burst_corr=0.5)
RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]
LAMS = [0.3, 0.5, 0.8, 1.2, 1.8, 2.7, 4.0, 6.0, 9.0, 14.0, 20.0, 30.0]   # same grid as exp3


def evaluate_set(ds, ch, rows, extra, trained_tags):
    zoh, holt = EnvCfg(predictor="zoh", channel=ch), EnvCfg(predictor="holt", phi=PHI, channel=ch)
    for name, cfg, pol in [("Send-on-delta (1.0 eps)", zoh, DeltaThreshold(1.0)),
                           ("Dual-prediction DPS (1.0 eps)", holt, DeltaThreshold(1.0)),
                           ("Congestion-adaptive DPS", holt, CongestionAdaptiveThreshold())]:
        rows.append(dict(method=name, seed=0, **extra, **flat(run_policy(ds, cfg, pol, "test", 0))))
    ref = EnvCfg(predictor="zoh", channel=ch, reward=RewardCfg("huang"))
    full, _ = variant(True, True, True, ch)
    for s in SEEDS:
        pol, _ = load_agent("ref_dqn_ideal", s)
        rows.append(dict(method="DQN-ref (Intel-trained, zero-shot)", seed=s, **extra,
                         **flat(run_policy(ds, ref, pol, "test", s))))
        pol, _ = proposed(s)
        rows.append(dict(method="Proposed (Intel-trained, zero-shot)", seed=s, **extra,
                         **flat(run_policy(ds, full, pol, "test", s))))
        pol, _ = load_agent(trained_tags[0], s)
        rows.append(dict(method="DQN-ref (retrained)", seed=s, **extra,
                         **flat(run_policy(ds, ref, pol, "test", s))))
        pol, _ = load_agent(trained_tags[1], s)
        curve = []
        for lam in LAMS:
            pol.lam = lam
            m = run_policy(ds, full, pol, "val", s)
            curve.append((lam, m["sla_violation_pct"], m["send_rate_pct"]))
        lam_star = select_lambda(curve)
        pol.lam = lam_star
        rows.append(dict(method="Proposed (retrained)", seed=s, **extra, lam_star=lam_star,
                         **flat(run_policy(ds, full, pol, "test", s))))


def main():
    log = logger("exp5")
    root = os.path.join(ROOT, "data")
    nab = load_dataset("nab", root=root, **NAB_KW)
    cap = capacity_rule(nab)
    log(f"NAB: {nab.S} streams, capacity rule -> {cap}")
    syn = load_dataset("synthetic", **SYN_KW)
    cap_syn = capacity_rule(syn)
    log(f"synthetic: capacity rule -> {cap_syn}")
    ref_cfg = EnvCfg(predictor="zoh", channel=IDEAL, reward=RewardCfg("huang"))
    full_nab, _ = variant(True, True, True, shared(cap))
    full_syn, _ = variant(True, True, True, shared(cap_syn))
    jobs = []
    for s in SEEDS:
        jobs += [("dqn", "nab_ref", "nab", NAB_KW, ref_cfg, "huang",
                  DQNCfg(total_slots=12000, warmup_slots=1000, episode_len=600), s),
                 ("dqn", "nab_full", "nab", NAB_KW, full_nab, "full",
                  DQNCfg(total_slots=12000, warmup_slots=1000, episode_len=600, explore_send_p=0.2), s),
                 ("dqn", "syn_ref", "synthetic", SYN_KW, ref_cfg, "huang",
                  DQNCfg(total_slots=12000, warmup_slots=1000), s),
                 ("dqn", "syn_full", "synthetic", SYN_KW, full_syn, "full",
                  DQNCfg(total_slots=12000, warmup_slots=1000, explore_send_p=0.2), s)]
    run_jobs(jobs, workers=8, log=log)

    rows = []
    evaluate_set(nab, shared(cap), rows, dict(dataset="NAB", capacity=cap), ("nab_ref", "nab_full"))
    save(agg(rows, ["method", "dataset"]), "exp5_nab.csv")
    log("NAB done")
    rows = []
    for rho in RHOS:
        ds = load_dataset("synthetic", burst_corr=rho)
        evaluate_set(ds, shared(cap_syn), rows, dict(dataset="synthetic", rho=rho, capacity=cap_syn),
                     ("syn_ref", "syn_full"))
        log(f"rho {rho} done")
    save(agg(rows, ["method", "rho"]), "exp5_synthetic.csv")


if __name__ == "__main__":
    main()
