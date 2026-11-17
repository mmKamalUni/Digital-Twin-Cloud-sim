"""Experiment 1 - reproduction of the reference DQN adaptive-sampling study.

Setting of the reference paper: Intel Lab Data, ideal (instantaneous, lossless) delivery,
reward r = w1*I - w2*C - w3*D with (w1, w2, w3) = (0.4, 0.4, 0.2).

Outputs
  results/exp1_table1.csv          Table 1 analogue (fixed / random / threshold / PPO / DQN)
  results/exp1_weights.csv         reward-weight sensitivity (paper Fig. 2)
  results/exp1_interference.csv    robustness to sensor interference (paper Fig. 3)
  results/exp2_gap.csv             the same policies deployed behind a shared gateway
"""
from common import *  # noqa: F401,F403

from dtsync.env import EnvCfg

WEIGHTS = [(0.6, 0.2, 0.2), (0.4, 0.4, 0.2), (0.5, 0.2, 0.3), (0.3, 0.3, 0.4), (0.2, 0.5, 0.3)]
INTERF = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def ref_cfg(w=(0.4, 0.4, 0.2), channel=IDEAL):
    return EnvCfg(predictor="zoh", channel=channel, reward=RewardCfg("huang", w=w))


def heuristics():
    return {"Fixed-frequency (T=4)": FixedRate(4), "Fixed-frequency (T=16)": FixedRate(16),
            "Random (p=1/6)": RandomRate(1 / 6), "Threshold-trigger (1.0 eps)": DeltaThreshold(1.0)}


def main():
    log = logger("exp1")
    ds = load_dataset("intel", root=os.path.join(ROOT, "data"))
    jobs = []
    for s in SEEDS:
        jobs.append(("dqn", "ref_dqn_ideal", "intel", {}, ref_cfg(), "huang", dqn_cfg(False), s))
        jobs.append(("ppo", "ref_ppo_ideal", "intel", {}, ref_cfg(), "huang",
                     PPOCfg(total_slots=TRAIN_SLOTS), s))
    for w in WEIGHTS[:1] + WEIGHTS[2:]:
        for s in SEEDS[:2]:
            tag = "ref_dqn_w" + "_".join(f"{x:.1f}" for x in w)
            jobs.append(("dqn", tag, "intel", {}, ref_cfg(w), "huang", dqn_cfg(False), s))
    run_jobs(jobs, workers=6, log=log)

    # --- Table 1 analogue and gap (same policies, two channels) --------------------
    rows_ideal, rows_gap = [], []
    for name, pol in heuristics().items():
        for s in SEEDS:
            if hasattr(pol, "rng"):
                pol.rng = np.random.default_rng(s)
            rows_ideal.append(dict(method=name, seed=s, **flat(run_policy(ds, ref_cfg(), pol, "test", s))))
            rows_gap.append(dict(method=name, seed=s, **flat(run_policy(ds, ref_cfg(channel=SHARED), pol, "test", s))))
    for tag, name in [("ref_ppo_ideal", "PPO adaptive"), ("ref_dqn_ideal", "DQN adaptive (reference)")]:
        for s in SEEDS:
            pol, info = load_agent(tag, s)
            extra = dict(train_seconds=info["train_seconds"], n_params=info["n_params"])
            rows_ideal.append(dict(method=name, seed=s, **extra, **flat(run_policy(ds, ref_cfg(), pol, "test", s))))
            rows_gap.append(dict(method=name, seed=s, **extra,
                                 **flat(run_policy(ds, ref_cfg(channel=SHARED), pol, "test", s))))
    save(agg(rows_ideal, ["method"]), "exp1_table1.csv")
    save(pd.DataFrame(rows_ideal), "exp1_table1_raw.csv")
    save(agg(rows_gap, ["method"]), "exp2_gap.csv")
    log("table 1 + gap done")

    # --- reward weight sensitivity --------------------------------------------------
    rows = []
    for w in WEIGHTS:
        tag = "ref_dqn_ideal" if w == (0.4, 0.4, 0.2) else "ref_dqn_w" + "_".join(f"{x:.1f}" for x in w)
        for s in SEEDS[:2]:
            pol, _ = load_agent(tag, s)
            rows.append(dict(weights=str(w), seed=s, **flat(run_policy(ds, ref_cfg(w), pol, "test", s))))
    save(agg(rows, ["weights"]), "exp1_weights.csv")
    log("weights done")

    # --- interference robustness ------------------------------------------------------
    rows = []
    pols = dict(heuristics())
    for tag, name in [("ref_ppo_ideal", "PPO adaptive"), ("ref_dqn_ideal", "DQN adaptive (reference)")]:
        for s in SEEDS:
            pols[(name, s)] = load_agent(tag, s)[0]
    for lvl in INTERF:
        cfg = ref_cfg()
        cfg.obs_noise = lvl
        for key, pol in pols.items():
            name, seeds = (key, SEEDS) if isinstance(key, str) else (key[0], [key[1]])
            for s in seeds:
                rows.append(dict(method=name, interference=lvl, seed=s,
                                 **flat(run_policy(ds, cfg, pol, "test", s))))
    save(agg(rows, ["method", "interference"]), "exp1_interference.csv")
    log("interference done")


if __name__ == "__main__":
    main()
