"""Experiment 3 - proposed method and full component ablation (Intel Lab, shared gateway).

Rows
  * heuristics: fixed-rate, send-on-delta, dual-prediction (DPS), congestion-adaptive DPS
  * DQN-ref (as published: trained with ideal delivery)       -> deployed behind the gateway
  * DQN-ref retrained behind the gateway (no new component)   -> control for training env
  * every combination of the proposed components A, B, C (2^3 - 1 rows)

Variants with component C are evaluated at the congestion-optimal price lambda* found on the
validation split, and additionally with the online cloud-side SLA controller (upper bound
lambda*).  Lambda sweeps (no retraining) give the Pareto fronts.

Outputs: results/exp3_ablation.csv, results/exp3_pareto.csv, results/exp3_lambda_cal.json
"""
import itertools

from common import *  # noqa: F401,F403

from dtsync.env import EnvCfg

LAM_GRID = [0.3, 0.5, 0.8, 1.2, 1.8, 2.7, 4.0, 6.0, 9.0, 14.0, 20.0, 30.0]
DELTAS = [0.5, 0.65, 0.8, 1.0, 1.25, 1.6, 2.0]
COMBOS = [c for c in itertools.product([False, True], repeat=3)]


def tag_of(A, B, C):
    return "abl_" + "".join(t if f else "-" for t, f in zip("ABC", (A, B, C)))


def calibrate(ds, cfg, pol, seed):
    """Validation price sweep -> lambda* (cheapest price meeting the SLA target, otherwise the
    congestion-optimal price); lambda* is also the upper bound of the online controller."""
    curve = []
    for lam in LAM_GRID:
        pol.controller, pol.lam = None, lam
        m = run_policy(ds, cfg, pol, "val", seed)
        curve.append((lam, m["sla_violation_pct"], m["send_rate_pct"]))
    return select_lambda(curve), curve


def eval_variant(arg):
    """Evaluate one trained variant/seed on the test split (C variants: price lambda* fixed from
    validation, plus the online SLA controller and a lambda sweep for the Pareto front)."""
    A, B, C, s = arg
    torch.set_num_threads(1)
    ds = load_dataset("intel", root=os.path.join(ROOT, "data"))
    cfg, obs = variant(A, B, C)
    name = variant_name(A, B, C)
    pol, info = load_agent(tag_of(A, B, C), s)
    extra = dict(train_seconds=info["train_seconds"], n_params=info["n_params"], A=A, B=B, C=C)
    rows, pareto, cal = [], [], {}
    if not C:
        rows.append(dict(method=name, family="drl", seed=s, **extra, **flat(run_policy(ds, cfg, pol, "test", s))))
        return rows, pareto, cal
    lam_star, curve = calibrate(ds, cfg, pol, s)
    cal[f"{tag_of(A, B, C)}_s{s}"] = dict(lam_star=lam_star, val_curve=curve)
    for lam in LAM_GRID:
        pol.controller, pol.lam = None, lam
        pareto.append(dict(method=name, knob=lam, seed=s, **flat(run_policy(ds, cfg, pol, "test", s))))
    # main row: price fixed at the validation-calibrated lambda*
    pol.controller, pol.lam = None, lam_star
    rows.append(dict(method=name, family="drl", seed=s, **extra, lam_star=lam_star, lam_mean=lam_star,
                     **flat(run_policy(ds, cfg, pol, "test", s))))
    # online cloud-side SLA controller (target SLA_TARGET, upper bound lambda*)
    pol.controller = SLAController(SLA_TARGET, lam0=min(2.0, lam_star), hi=lam_star)
    m = run_policy(ds, cfg, pol, "test", s)
    rows.append(dict(method=name + " (online SLA controller)", family="drl-controller", seed=s, **extra,
                     lam_star=lam_star, lam_mean=float(np.mean(pol.lam_trace)), **flat(m)))
    return rows, pareto, cal


def main():
    log = logger("exp3")
    ds = load_dataset("intel", root=os.path.join(ROOT, "data"))
    jobs = []
    for (A, B, C) in COMBOS:
        cfg, obs = variant(A, B, C)
        for s in SEEDS:
            jobs.append(("dqn", tag_of(A, B, C), "intel", {}, cfg, obs, dqn_cfg(True), s))
    run_jobs(jobs, workers=8, log=log)

    rows, pareto, cal = [], [], {}
    # ---------------------------------------------------------------- heuristics
    zoh = EnvCfg(predictor="zoh", channel=SHARED)
    holt = EnvCfg(predictor="holt", phi=PHI, channel=SHARED)
    heur = [("Fixed-frequency (T=16)", zoh, FixedRate(16)),
            ("Send-on-delta (1.0 eps)", zoh, DeltaThreshold(1.0)),
            ("Dual-prediction DPS (1.0 eps)", holt, DeltaThreshold(1.0)),
            ("Congestion-adaptive DPS", holt, CongestionAdaptiveThreshold())]
    for name, cfg, pol in heur:
        for s in SEEDS:
            rows.append(dict(method=name, family="heuristic", seed=s, **flat(run_policy(ds, cfg, pol, "test", s))))
    for d in DELTAS:
        for name, cfg, pol in [("Send-on-delta", zoh, DeltaThreshold(d)), ("Dual-prediction DPS", holt, DeltaThreshold(d)),
                               ("Congestion-adaptive DPS", holt, CongestionAdaptiveThreshold(delta=0.8 * d))]:
            pareto.append(dict(method=name, knob=d, **flat(run_policy(ds, cfg, pol, "test", 0))))
    log("heuristics done")

    # ---------------------------------------------------------------- reference DQN
    while not all(os.path.exists(model_path("ref_dqn_ideal", s) + ".json") for s in SEEDS):
        time.sleep(20)   # produced by exp1_reproduce.py
    ref_cfg = EnvCfg(predictor="zoh", channel=SHARED, reward=RewardCfg("huang"))
    for s in SEEDS:
        pol, info = load_agent("ref_dqn_ideal", s)
        rows.append(dict(method="DQN-ref (published setting)", family="drl", seed=s,
                         train_seconds=info["train_seconds"], n_params=info["n_params"],
                         **flat(run_policy(ds, ref_cfg, pol, "test", s))))

    # ---------------------------------------------------------------- ablation grid
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(12) as ex:
        for r, pf, c in ex.map(eval_variant, [(A, B, C, s) for (A, B, C) in COMBOS for s in SEEDS]):
            rows += r
            pareto += pf
            cal |= c
    log("ablation grid evaluated")

    save(agg(rows, ["method", "family"]), "exp3_ablation.csv")
    save(pd.DataFrame(rows), "exp3_ablation_raw.csv")
    pf = pd.DataFrame(pareto)
    save(pf.groupby(["method", "knob"], sort=False).mean(numeric_only=True).reset_index(), "exp3_pareto.csv")
    json.dump(cal, open(os.path.join(RESULTS, "exp3_lambda_cal.json"), "w"), indent=1)
    log("exp3 done")


if __name__ == "__main__":
    main()
