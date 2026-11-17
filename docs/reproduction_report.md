# Reproduction Report

**Student:** Malik Murtaza Kamal (22i-2622), BS Software Engineering, FAST-NUCES Islamabad
**Course:** Cloud Computing — research project (Track 5: Digital Twins and Cloud Computing)

## 1. Paper selected

| Item | Details |
|---|---|
| Title | Context-Aware Adaptive Sampling for Intelligent Data Acquisition Systems Using DQN |
| Authors | Weiqiang Huang, Juecen Zhan, Yumeng Sun, Xu Han, Tai An, Nan Jiang |
| Venue | arXiv preprint arXiv:2504.09344 (April 2025); no peer-reviewed venue was found at the time of writing |
| Paper link | https://arxiv.org/abs/2504.09344 |
| Official code | None released. The method was re-implemented from the paper's description |
| Dataset | Intel Berkeley Research Lab sensor data — http://db.csail.mit.edu/labdata/labdata.html |
| This repository | https://github.com/mmKamalUni/Digital-Twin-Cloud-sim |

**Why this paper.** It is a recent (2025) DRL method for deciding *when* an IoT node should sample and
transmit, which is exactly the synchronisation decision a cloud-hosted digital twin depends on. It uses a
public dataset, reports a clear main table, and compares DQN with PPO, fixed-rate, random and threshold
policies. The lack of official code is a limitation; it was handled by re-implementing the method and
reporting every choice the paper does not specify (Section 4).

## 2. Method of the paper (as described)

* Each sensor is an agent that decides at every step whether to sample/transmit (discrete action).
* State: recent readings and their variation, deviation from the last reported value, time since the last
  sample, recent action history and the remaining energy budget.
* Reward `R = λ1·I(a) − λ2·C(a) − λ3·D(a)`: information gain, energy cost and redundancy.
  Default weights (0.4, 0.4, 0.2); a sensitivity study uses five weight vectors.
* DQN with experience replay, ε-greedy exploration and soft target-network updates; PPO as a comparison.
* Main results (Table 1 of the paper): data quality, energy (mJ), redundancy (%), critical-event
  detection (%), plus a reward-weight sensitivity figure and a sensor-interference robustness figure.

## 3. Environment

| Item | Value |
|---|---|
| OS / CPU | Windows 11, AMD Ryzen 9 8940HX (16 cores), 15 GB RAM |
| GPU | Not used (networks have ~19k parameters; environment stepping dominates run time) |
| Python | 3.12.7 |
| Libraries | PyTorch 2.15.0 (CPU), NumPy 2.4.4, pandas 3.0.2, SimPy 4.1.2, Matplotlib 3.10.9 |
| Seeds | 0, 1, 2 (results reported as mean ± std over seeds) |
| Run command | `python scripts/exp1_reproduce.py` |

## 4. Re-implementation choices (what the paper does not specify)

| Paper element | Not specified in the paper | Choice made here |
|---|---|---|
| Dataset subset | motes, variables, period | Epochs 0–30000; the 40 motes with ≥ 50 % coverage × temperature, humidity, light = 120 streams; 31 s slots |
| Cleaning | — | valid-range filter, linear interpolation of gaps ≤ 40 epochs, Hampel spike filter |
| Train / test split | — | chronological 60 / 10 / 30 % (train / validation / test) |
| Data quality | formula | mean over stream-slots of `1 − min(|x − x̂|/ε, 1)`; ε = 3 % of the training P1–P99 range of each stream |
| Information gain `I(a)` | formula | the per-slot data quality above |
| Energy | model, units | 2.0 mJ per transmission, 0.15 mJ per sensor reading |
| Redundancy `D(a)` | formula | a transmission is redundant if the twin was already within 0.5 ε |
| Critical event | definition | a change of ≥ 3 ε within two slots; detected if the twin is within ε inside 2 slots (10-slot refractory period) |
| Network / hyper-parameters | sizes, learning rate, γ, buffer | MLP 2×128, Adam 5·10⁻⁴, γ = 0.95, τ = 0.005, batch 1024, buffer 400k, ε-greedy 1 → 0.02, 20 000 training slots × 120 streams (parameter sharing) |
| Fixed / random baselines | period, probability | fixed period T = 4 (also T = 16); random p = 1/6 (gives the paper's random:fixed energy ratio of 0.67) |
| Threshold baseline | threshold | send-on-delta at 1.0 ε |
| Delivery | — | ideal: every transmission reaches the twin in the same slot |

## 5. Results — original vs reproduced (Intel Lab, test split, ideal delivery)

The paper reports energy in mJ without stating the horizon or number of nodes, so energy is compared as the
ratio to the fixed-rate policy. The other three metrics are compared directly.

| Method | Metric | Original | Reproduced | Difference |
|---|---|---|---|---|
| Fixed-rate | Data quality | 0.72 | 0.933 ± 0.000 | +0.213 |
| | Energy (× fixed) | 1.00 | 1.00 | 0 |
| | Redundancy (%) | 38.6 | 91.7 | +53.1 |
| | Event detection (%) | 82.1 | 85.4 | +3.3 |
| Random | Data quality | 0.65 | 0.869 ± 0.001 | +0.219 |
| | Energy (× fixed) | 0.67 | 0.67 | 0 |
| | Redundancy (%) | 27.4 | 89.4 | +62.0 |
| | Event detection (%) | 76.4 | 65.8 ± 0.7 | −10.6 |
| Threshold | Data quality | 0.78 | 0.636 | −0.144 |
| | Energy (× fixed) | 0.84 | 0.42 | −0.42 |
| | Redundancy (%) | 24.9 | 0.0 | −24.9 |
| | Event detection (%) | 85.6 | 100.0 | +14.4 |
| PPO | Data quality | 0.87 | 0.904 ± 0.001 | +0.034 |
| | Energy (× fixed) | 0.61 | 0.55 | −0.06 |
| | Redundancy (%) | 12.3 | 47.8 | +35.5 |
| | Event detection (%) | 91.8 | 100.0 ± 0.0 | +8.2 |
| **DQN** | Data quality | 0.83 | 0.894 ± 0.008 | +0.064 |
| | Energy (× fixed) | 0.64 | 0.52 | −0.12 |
| | Redundancy (%) | 15.1 | 43.5 | +28.4 |
| | Event detection (%) | 89.3 | 99.7 ± 0.4 | +10.4 |

Reward-weight sensitivity (DQN, 2 seeds each):

| (w1, w2, w3) | Data quality | Transmissions (%) | Redundancy (%) | Event detection (%) |
|---|---|---|---|---|
| (0.6, 0.2, 0.2) | 0.940 | 9.7 | 60.9 | 100.0 |
| (0.5, 0.2, 0.3) | 0.917 | 7.9 | 48.2 | 100.0 |
| (0.4, 0.4, 0.2) default | 0.890 | 6.4 | 41.9 | 99.6 |
| (0.3, 0.3, 0.4) | 0.858 | 5.5 | 23.7 | 99.0 |
| (0.2, 0.5, 0.3) | 0.790 | 3.2 | 19.6 | 62.0 |

Training time (clean run, single process, 2 CPU threads, 20 000 slots × 120 streams; `results/training_time.csv`):
DQN 55 s, PPO 43 s (19 074 parameters each).

## 6. What was reproduced and what was not

**Reproduced (qualitative claims of the paper):**

1. The adaptive DRL policies beat fixed-rate and random sampling on data quality and event detection while
   using less energy (DQN: 0.52× the fixed-rate energy; paper: 0.64×).
2. PPO is slightly better than DQN on data quality and event detection, as in the paper (0.904 vs 0.894;
   paper 0.87 vs 0.83), at slightly higher energy.
3. Random sampling is the weakest policy on event detection.
4. Weight sensitivity follows the paper's trend: more weight on information raises quality and energy;
   more weight on energy lowers both. With the energy weight at 0.5, event detection collapses (62 %).
5. Robustness to sensor interference (Gaussian noise, standard deviation up to 1.0 ε): the DRL policies
   keep event detection high (DQN 99.7 → 89.2 %, PPO 100 → 95.8 %) while fixed-rate and random sampling
   degrade (85 → 62 %, 66 → 50 %), as claimed in the paper (`results/exp1_interference.csv`,
   `figures/fig_interference.pdf`). The paper does not mention a side effect seen here: noise triggers more
   transmissions (DQN 6.6 → 53.6 % of slots), so robustness is bought with bandwidth.

**Not reproduced / different:**

1. **Absolute data-quality values are higher.** The paper does not define data quality; with the
   tolerance-normalised definition used here, every method scores higher than in the paper. Rankings among
   fixed, random, PPO and DQN agree.
2. **The threshold policy.** In the paper the threshold baseline has *lower* event detection than DQN. A
   send-on-delta rule with threshold equal to the tolerance can, by construction, never let the twin
   leave the tolerance under ideal delivery, so it detects 100 % of events with the fewest transmissions,
   but its average quality is low because the twin error sits just below ε most of the time. The paper's
   threshold rule is unspecified (it may threshold the raw reading rather than the twin error).
3. **Redundancy is higher for every method.** Redundancy is defined here against the twin state (a
   transmission while the twin is already within 0.5 ε). Under this definition periodic sampling of slowly
   varying indoor signals is mostly redundant (92 %). The relative ordering (fixed > random > DQN, PPO)
   matches the paper except for the threshold rule, which is never redundant by construction.
4. **Internal consistency of the original table.** In the paper the fixed-rate policy has the highest energy
   use but the lowest quality among non-random policies; with the same data, sampling every fourth slot
   gives high quality. This could not be reconciled without the paper's metric definitions.

## 7. Issues encountered

* No official code, hyper-parameters, metric formulas or data subset — every such choice is listed in Section 4.
* The paper's reference list is largely unrelated to its topic, so it could not be used to recover details.
* Intel Lab data contains corrupted readings (e.g. temperatures > 100 °C, low-battery drift after
  ~epoch 30000), handled by range filtering, a Hampel filter and by using the first 30 000 epochs.
* Training multiple agents in parallel on CPU; trained models are cached so runs can resume.

## 8. Deviations and their justification

* **Parameter sharing:** one Q-network is shared by all 120 stream agents (each stream is an independent
  agent at execution). This is standard for homogeneous fleets and keeps training to minutes on a CPU.
* **Energy reported as a ratio** to fixed-rate sampling, because the paper's energy horizon is unknown.
* **Ideal delivery:** the reproduction keeps the paper's implicit assumption that every transmitted reading
  reaches the receiver instantly. Relaxing this assumption is the research gap studied next
  (`scripts/exp3_ablation.py`, `results/exp2_gap.csv`).

## 9. Research gap identified

The same reproduced policies were deployed behind a **shared cloud ingestion gateway** (capacity 12
messages per 31 s slot for 120 streams, bounded buffer; `results/exp2_gap.csv`). Nothing else changed.

| Method | Event detection, ideal (%) | Event detection, shared gateway (%) | SLA violations, shared (%) | p95 delivery latency (s) |
|---|---|---|---|---|
| Fixed-rate (T=4) | 85.4 | 43.7 | 11.9 | 248 |
| Threshold (1.0 ε) | 100.0 | 48.6 | 6.8 | 243 |
| PPO | 100.0 | 46.3 | 8.5 | 245 |
| DQN (reference) | 99.7 | 46.3 | 7.9 | 244 |

Bursts in the lab are spatially correlated (lights switching, HVAC), so many sensors decide to transmit in
the same slot; the gateway queue grows, updates arrive minutes late or are dropped, and the twin falls out of
tolerance exactly during the events it should capture. The reference DQN is trained without any knowledge
of delivery delay or of the other streams, and its fixed reward weights give no handle on a target
violation rate. This motivates the proposed method (predictive mirrored twin, congestion awareness and an
SLA-constrained, run-time adjustable trade-off), evaluated below.

## 10. Proposed improvement

| | Component | Implementation |
|---|---|---|
| **A** | Predictive mirrored twin | damped-trend twin (φ = 0.9, slope window 6) mirrored at the edge; predicted-deviation features (`src/dtsync/env.py`, `features.py`) |
| **B** | Congestion awareness | gateway backlog, trend, in-flight count/age, acknowledged twin error, AoI, last latency as features; congestion externality price κ = 0.3 · send · backlog/μ |
| **C** | SLA-constrained control | two-head (cost / violation) DQN conditioned on log λ, λ randomised per stream in training; the cloud picks the cheapest λ that meets the 2 % SLA on the validation split (`src/dtsync/agents/dqn.py`, `scripts/common.py: select_lambda`) |

The baseline is unchanged and still available (`obs_kind="huang"`, `RewardCfg("huang")`); every component can be
switched off with `variant(A, B, C)` in `scripts/common.py`.

## 11. Ablation (Intel Lab, test split, shared gateway μ = 12, 3 seeds; `results/exp3_ablation.csv`)

| Variant | SLA violations (%) | Events (%) | Transmissions (%) | p95 latency (s) | Drops (%) |
|---|---|---|---|---|---|
| Original (DQN-ref, published setting) | 7.93 ± 0.22 | 46.3 | 6.57 | 244 | 18.2 |
| DQN-ref retrained behind gateway | 5.52 ± 0.91 | 57.9 | 4.95 | 239 | 7.0 |
| + A | 4.68 ± 0.44 | 54.4 | 5.13 | 237 | 6.8 |
| + B | 3.38 ± 0.12 | 60.9 | 5.62 | 89 | 0.0 |
| + C | 4.51 ± 0.95 | 61.4 | 2.89 | 230 | 4.5 |
| + A + B | 2.47 ± 0.11 | 67.3 | 5.04 | 61 | 0.0 |
| + A + C | 3.58 ± 0.86 | 65.5 | 2.77 | 207 | 4.4 |
| + B + C | 2.55 ± 0.28 | 76.0 | 2.94 | 79 | 0.0 |
| **+ A + B + C (proposed)** | **2.39 ± 0.28** | **70.9** | **2.78** | **66** | **0.0** |
| proposed with online SLA controller | 3.25 ± 0.24 | 69.4 | 2.61 | 65 | 0.0 |

## 12. Final comparison

| Dataset / setting | Method | Data quality | Events (%) | Energy (× fixed) | Transmissions (%) | SLA violations (%) |
|---|---|---|---|---|---|---|
| Intel, ideal delivery | DQN — original paper | 0.83 | 89.3 | 0.64 | — | — |
| | DQN — reproduced | 0.894 | 99.7 | 0.52 | 6.57 | 0.38 |
| | Proposed | 0.657 | 95.1 | 0.40 | 3.14 | 0.85 |
| Intel, shared gateway | DQN — reproduced | 0.784 | 46.3 | 0.52 | 6.57 | 7.93 |
| | Proposed | 0.640 | 70.9 | 0.38 | 2.78 | 2.39 |
| NAB, gateway μ = 18.5 | Send-on-delta | 0.837 | 100.0 | — | 30.6 | 0.00 |
| | DQN — reproduced, retrained | 0.627 | 82.0 | — | 6.48 | 22.39 |
| | Proposed, retrained | 0.841 | 100.0 | — | 32.0 | 0.38 |

Runtime and cloud resources (`results/training_time.csv`, `results/exp4_capacity.csv`):

| | DQN-ref | Proposed |
|---|---|---|
| Training time (2 CPU threads) | 55 s | 62 s |
| Parameters | 19 074 | 20 996 |
| Decision time per stream | 1.8 µs | 2.6 µs |
| Smallest gateway meeting the 2 % SLA | 32 msg/slot | 16 msg/slot |
| Ingestion cost, 1 000 streams, \$1 per million messages | \$5.49 / month | \$2.33 / month |

**Discussion.** The proposed method's gains come from congestion: behind a gateway that correlated bursts
overload (Intel Lab) it reduces violations by 70 %, catches 25 points more critical events and needs half the
ingestion capacity of the reproduced baseline. B (congestion awareness) is the largest single contribution and
C halves the messages at equal fidelity. Where the gateway is not congested (NAB, weakly correlated synthetic
bursts) threshold and dual-prediction rules are already sufficient and the proposed method only matches them.
Its average data quality is lower because the constrained objective keeps the twin inside the tolerance rather
than as close as possible within it. The online SLA controller was worse than the validated price on bursty
traces; this is reported as a limitation.
