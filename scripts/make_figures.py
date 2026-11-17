"""Figures (PDF + PNG) and LaTeX tables from the experiment CSVs in results/.

Each figure is produced only if its input file exists, so the script can be re-run as
experiments finish.  Output: figures/*.pdf, figures/*.png, results/tables/*.tex
"""
from common import *  # noqa: F401,F403

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = os.path.join(ROOT, "figures")
TAB = os.path.join(RESULTS, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# one fixed colour per method family, used in every figure
COL = {"send-on-delta": "#2a78d6", "threshold": "#2a78d6", "dual-prediction": "#1baf7a",
       "congestion-adaptive": "#eda100", "dqn": "#eb6834", "proposed": "#6250d6",
       "fixed": "#e87ba4", "random": "#008300", "ppo": "#e34948"}
MARK = {"send-on-delta": "o", "threshold": "o", "dual-prediction": "s", "congestion-adaptive": "D",
        "dqn": "^", "proposed": "P", "fixed": "v", "random": "X", "ppo": "<"}
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.6, "lines.markersize": 4.5,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 300,
})


def key(method: str) -> str:
    m = method.lower()
    if "proposed" in m or "+ a" in m or "+a" in m:
        return "proposed"
    for k in ["congestion-adaptive", "dual-prediction", "send-on-delta", "threshold", "fixed", "random", "ppo"]:
        if k in m:
            return k
    return "dqn"


def col(method):
    return COL[key(method)]


def csv(name):
    p = os.path.join(RESULTS, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def out(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)
    print("figure", name)


def target_line(ax, y=100 * SLA_TARGET, label="SLA target"):
    ax.axhline(y, color=MUTED, lw=0.9, ls="--", zorder=1)
    ax.annotate(label, (1.0, y), xycoords=("axes fraction", "data"), xytext=(-2, 3),
                textcoords="offset points", ha="right", fontsize=7, color=MUTED)


# ============================================================================ figures
def fig_gap():
    a, b = csv("exp1_table1.csv"), csv("exp2_gap.csv")
    if a is None or b is None:
        return
    order = list(a["method"])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for ax, metric, lab in [(axes[0], "event_detection_pct", "Critical events detected (%)"),
                            (axes[1], "sla_violation_pct", "SLA violations (% of stream-time)")]:
        for i, m in enumerate(order):
            x0 = float(a.loc[a.method == m, metric].iloc[0])
            x1 = float(b.loc[b.method == m, metric].iloc[0])
            ax.plot([x0, x1], [i, i], color=GRID, lw=2.2, zorder=1, solid_capstyle="round")
            ax.scatter([x0], [i], s=26, facecolor="white", edgecolor=col(m), lw=1.4, zorder=3)
            ax.scatter([x1], [i], s=26, color=col(m), edgecolor="white", lw=0.8, zorder=3)
        ax.set_xlabel(lab)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(order)), order)
    axes[0].invert_yaxis()
    h = [plt.Line2D([], [], ls="", marker="o", mfc="white", mec=MUTED, label="ideal delivery (as published)"),
         plt.Line2D([], [], ls="", marker="o", mfc=MUTED, mec="white", label="shared cloud gateway")]
    fig.legend(handles=h, loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.06))
    out(fig, "fig_gap")


def fig_interference():
    d = csv("exp1_interference.csv")
    if d is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for ax, metric, lab in [(axes[0], "data_quality", "Data quality"),
                            (axes[1], "event_detection_pct", "Critical events detected (%)")]:
        for m, g in d.groupby("method", sort=False):
            ax.plot(g["interference"], g[metric], color=col(m), marker=MARK[key(m)], label=m,
                    ls=":" if "T=16" in m else "-")
        ax.set_xlabel("Sensor interference level (noise std / tolerance)")
        ax.set_ylabel(lab)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    out(fig, "fig_interference")


def fig_weights():
    d = csv("exp1_weights.csv")
    if d is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
    labels = [w.replace("(", "").replace(")", "").replace(" ", "") for w in d["weights"]]
    x = np.arange(len(d))
    for ax, metric, lab in [(axes[0], "data_quality", "Data quality"),
                            (axes[1], "send_rate_pct", "Transmissions (% of slots)"),
                            (axes[2], "event_detection_pct", "Events detected (%)")]:
        ax.bar(x, d[metric], width=0.62, color=COL["dqn"], edgecolor="white", lw=1.0,
               yerr=d.get(metric + "_std"), error_kw=dict(lw=0.8, ecolor=MUTED, capsize=2))
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_title(lab)
        ax.grid(axis="x", visible=False)
    axes[0].set_xlabel(r"($w_1$, $w_2$, $w_3$)")
    out(fig, "fig_weights")


def fig_pareto():
    p, a = csv("exp3_pareto.csv"), csv("exp3_ablation.csv")
    if p is None:
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    show = ["Send-on-delta", "Dual-prediction DPS", "Congestion-adaptive DPS", "DQN-ref + A+B+C"]
    for m in show:
        g = p[p.method == m].sort_values("send_rate_pct")
        if g.empty:
            continue
        lab = "Proposed A+B+C ($\\lambda$ sweep)" if "A+B+C" in m else m + " ($\\delta$ sweep)"
        ax.plot(g["send_rate_pct"], g["sla_violation_pct"], color=col(m), marker=MARK[key(m)], label=lab)
    if a is not None:
        r = a[a.method == "DQN-ref (published setting)"]
        if not r.empty:
            ax.scatter(r["send_rate_pct"], r["sla_violation_pct"], color=COL["dqn"], marker="^", s=40,
                       zorder=4, label="DQN-ref (published setting)")
        r = a[a.method == "DQN-ref + A+B"]
        if not r.empty:
            ax.scatter(r["send_rate_pct"], r["sla_violation_pct"], s=40, marker="s", facecolor="white",
                       edgecolor=COL["proposed"], lw=1.2, zorder=4, label="A+B (no C)")
        r = a[a.method == "DQN-ref + A+B+C"]
        if not r.empty:
            ax.scatter(r["send_rate_pct"], r["sla_violation_pct"], s=80, marker="*", color=COL["proposed"],
                       edgecolor="white", lw=0.8, zorder=5, label=r"Proposed at validated $\lambda^\star$")
    target_line(ax)
    ax.set_xlabel("Transmissions (% of stream-slots)")
    ax.set_ylabel("SLA violations (%)")
    ax.set_yscale("log")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.8)
    out(fig, "fig_pareto")


def fig_ablation():
    a = csv("exp3_ablation.csv")
    if a is None:
        return
    d = a[a.family == "drl"].copy()
    order = ["DQN-ref (published setting)", "DQN-ref (retrained shared)", "DQN-ref + A", "DQN-ref + B",
             "DQN-ref + C", "DQN-ref + A+B", "DQN-ref + A+C", "DQN-ref + B+C", "DQN-ref + A+B+C"]
    d = d.set_index("method").reindex([o for o in order if o in set(d.method)]).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7), sharey=True)
    y = np.arange(len(d))
    colors = [COL["proposed"] if m == "DQN-ref + A+B+C" else ("#a9a2e6" if "+" in m else COL["dqn"])
              for m in d.method]
    for ax, metric, lab in [(axes[0], "sla_violation_pct", "SLA violations (%)"),
                            (axes[1], "event_detection_pct", "Events detected (%)"),
                            (axes[2], "bandwidth_Bps", "Uplink bandwidth (B/s)")]:
        ax.barh(y, d[metric], height=0.64, color=colors, edgecolor="white", lw=1.0,
                xerr=d.get(metric + "_std"), error_kw=dict(lw=0.8, ecolor=MUTED, capsize=2))
        ax.set_xlabel(lab)
        ax.grid(axis="y", visible=False)
        sd = d.get(metric + "_std", pd.Series(np.zeros(len(d)))).fillna(0).to_numpy()
        for yi, v, e in zip(y, d[metric], sd):
            ax.annotate(f"{v:.1f}", (v + e, yi), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=6.5, color=INK)
        ax.set_xlim(0, float((d[metric] + sd).max()) * 1.22)
    axes[0].axvline(100 * SLA_TARGET, color=MUTED, lw=0.9, ls="--")
    axes[0].set_yticks(y, [m.replace("DQN-ref ", "Ref. ").replace("(published setting)", "(published)")
                           for m in d.method])
    axes[0].invert_yaxis()
    out(fig, "fig_ablation")


def fig_capacity():
    d = csv("exp4_capacity.csv")
    if d is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for m, g in d.groupby("method", sort=False):
        axes[0].plot(g["capacity"], g["sla_violation_pct"], color=col(m), marker=MARK[key(m)], label=m)
        axes[1].plot(g["capacity"], g["latency_p95_s"], color=col(m), marker=MARK[key(m)], label=m)
    target_line(axes[0])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("SLA violations (%)")
    axes[1].set_ylabel("p95 delivery latency (s)")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("Gateway capacity (messages / slot)")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    out(fig, "fig_capacity")


def fig_rho():
    d = csv("exp5_synthetic.csv")
    if d is None:
        return
    keep = ["Send-on-delta (1.0 eps)", "Dual-prediction DPS (1.0 eps)", "Congestion-adaptive DPS",
            "DQN-ref (retrained)", "Proposed (retrained)"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for m in keep:
        g = d[d.method == m]
        if g.empty:
            continue
        axes[0].plot(g["rho"], g["sla_violation_pct"], color=col(m), marker=MARK[key(m)], label=m)
        axes[1].plot(g["rho"], g["event_detection_pct"], color=col(m), marker=MARK[key(m)], label=m)
    target_line(axes[0])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("SLA violations (%)")
    axes[1].set_ylabel("Critical events detected (%)")
    for ax in axes:
        ax.set_xlabel(r"Cross-stream burst correlation $\rho$")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    out(fig, "fig_rho")


def fig_timeline():
    """Gateway backlog and fleet-wide violations around the largest test-split burst."""
    if not os.path.exists(os.path.join(RESULTS, "exp3_lambda_cal.json")):
        return
    ds = load_dataset("intel", root=os.path.join(ROOT, "data"))
    zoh = EnvCfg(predictor="zoh", channel=SHARED)
    holt = EnvCfg(predictor="holt", phi=PHI, channel=SHARED)
    full, _ = variant(True, True, True)
    runs = [("Send-on-delta (1.0 eps)", zoh, DeltaThreshold(1.0)),
            ("Dual-prediction DPS (1.0 eps)", holt, DeltaThreshold(1.0)),
            ("Congestion-adaptive DPS", holt, CongestionAdaptiveThreshold()),
            ("Proposed (A+B+C)", full, proposed(0)[0])]
    logs = {}
    for name, cfg, pol in runs:
        _, env = run_policy(ds, cfg, pol, "test", 0, return_env=True)
        logs[name] = (np.asarray(env.backlog_log), (np.abs(np.asarray(env.err_log)) > 1).mean(1) * 100)
    b0 = logs["Send-on-delta (1.0 eps)"][0]
    c = int(np.argmax(np.convolve(b0, np.ones(30), "same")))
    lo, hi = max(0, c - 120), min(len(b0), c + 240)
    t = (np.arange(lo, hi) - lo) * ds.dt / 60
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.6), sharex=True)
    for name, (b, v) in logs.items():
        axes[0].plot(t, b[lo:hi], color=col(name), lw=1.2, label=name)
        axes[1].plot(t, v[lo:hi], color=col(name), lw=1.2)
    axes[0].set_ylabel("Backlog (msgs)")
    axes[1].set_ylabel("Out of SLA (%)")
    axes[1].set_xlabel("Time (min)")
    fig.align_ylabels(axes)
    axes[1].legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, -0.32),
                   ncol=2, fontsize=6.5)
    out(fig, "fig_timeline")


# ============================================================================ tables
def fmt(v, s=None, d=2):
    if pd.isna(v):
        return "--"
    return f"{v:.{d}f}" if s is None or pd.isna(s) else f"{v:.{d}f}$\\pm${s:.{d}f}"


def tex_table(df, cols, header, name, caption, label, digits=None):
    digits = digits or {}
    lines = [r"\begin{table*}[t]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\footnotesize", r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        cells = [str(r["method"]).replace("eps", r"$\varepsilon$")]
        for c in cols:
            cells.append(fmt(r[c], r.get(c + "_std"), digits.get(c, 2)))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(os.path.join(TAB, name), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("table", name)


MAIN_COLS = ["data_quality", "sla_violation_pct", "event_detection_pct", "send_rate_pct", "bandwidth_Bps",
             "redundancy_pct", "energy_mJ_per_stream_day", "latency_p95_s", "dropped_pct"]
MAIN_HEAD = ["Method", "DQ", "SLA viol. (\\%)", "Events (\\%)", "Tx (\\%)", "B/s", "Redund. (\\%)",
             "Energy (mJ/d)", "p95 lat. (s)", "Drops (\\%)"]
DIG = dict(data_quality=3, bandwidth_Bps=1, energy_mJ_per_stream_day=0, latency_p95_s=1)


def tables():
    a = csv("exp1_table1.csv")
    if a is not None:
        tex_table(a, MAIN_COLS, MAIN_HEAD, "tab_reproduction.tex",
                  "Reproduced Table~1 setting (Intel Lab test split, ideal delivery), mean$\\pm$std over seeds.",
                  "tab:repro", DIG)
    b = csv("exp2_gap.csv")
    if b is not None:
        tex_table(b, MAIN_COLS, MAIN_HEAD, "tab_gap.tex",
                  "The same policies deployed behind a shared cloud ingestion gateway (12 msg/slot).",
                  "tab:gap", DIG)
    c = csv("exp3_ablation.csv")
    if c is not None:
        tex_table(c, MAIN_COLS, MAIN_HEAD, "tab_ablation.tex",
                  "Proposed method and ablation behind the shared gateway (Intel Lab test split).",
                  "tab:ablation", DIG)
    n = csv("exp5_nab.csv")
    if n is not None:
        tex_table(n, MAIN_COLS, MAIN_HEAD, "tab_nab.tex", "Generalisation to NAB cloud telemetry.",
                  "tab:nab", DIG)
    s = csv("exp6_simpy.csv")
    if s is not None:
        s = s.copy()
        s["method"] = s["method"] + " / " + s["workers"].astype(str) + "w"
        tex_table(s, MAIN_COLS, MAIN_HEAD, "tab_simpy.tex",
                  "Event-level SimPy validation (exponential service, 1 or 4 ingestion workers).",
                  "tab:simpy", DIG)


if __name__ == "__main__":
    which = sys.argv[1:] or ["gap", "interference", "weights", "pareto", "ablation", "capacity", "rho",
                             "timeline", "tables"]
    for w in which:
        (tables if w == "tables" else globals()["fig_" + w])()
