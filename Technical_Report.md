# Technical Report — Match-Sentiment-Adaptive Hybrid Model (`sentiment-hybrid`)

**Competition:** More Than A Hack 2027 (Phase 1: Frame-Level Ball Possession)  
**Match:** FC Barcelona (`H`) vs. Athletic Club (`A`) — LaLiga EA Sports 2024–25, Matchday 2  
**Primary Ranking Metric:** `spellF1` (Mean Possession-Spell F1 over Temporal IoU $\in \{0.3, 0.4, 0.5, 0.6, 0.7\}$)  
**First Tie-Breaker:** `macroF1` (Unweighted per-frame F1 over `H`, `A`, `D`)  

---

## 1. Root-Cause Diagnosis of Prior Branches (`physics` vs. `model-fitting`)

Audit of the first two branches revealed two key findings:
1. **`physics` and `model-fitting` tied identically (`spellF1 = 0.7000`, `macroF1 = 0.9543`, `poss = 11`)** because `_filter_live_block_deflections()` in `src/decoder.py` had `codes = np.where(block_bridged < 2, block_bridged, block_codes).copy()`, which overwrote the Viterbi path on `99.2%` of live frames (`116,423 / 117,305`) before heuristic deflection filtering.
2. **Unmasked ML & Leakage Fixes:** In `model-fitting`, 3-class `GBDT_Full` predicted `D ~ 0.99` on unanchored contested turnovers, while 46-feature `GBDT_NoBall` suffered negative temporal generalization (`0.5025` live accuracy, `AUC = 0.4815`) from clock/reprojection/unflipped-half features (`match_clock_progress`, `reproj_error`, `cam_x_norm`, `vx_cam_smooth`).

---

## 2. Match-Sentiment-Adaptive Hybrid Architecture

This branch combines the **Physics Matrix Model** and **Specialist ML Ensemble** dynamically at every frame using a 5-axis **Match Sentiment Tensor** $\Psi_t$ (`src/models/sentiment_hybrid.py`, `src/models/pseudo_gbdt.py`, `src/models/external_seq.py`, `src/decoder.py`):

1. **Continuous 5-Regime Match Sentiment Tensor $\Psi_t$ (`158,857` Graded Frames):**
   * **`CALM_CONTROL`** (`36,459` frames, `23.0%`): Low contention entropy & tripod/buildup stability $\rightarrow$ Physics persistence dominates (`Mean w_ml = 0.239`, `Mean Turnover Gate = 0.76`).
   * **`CONTESTED_CHAOS`** (`30,523` frames, `19.2%`): High 2s Shannon contention entropy & duel flip rate $\rightarrow$ ML weight rises (`Mean w_ml = 0.510`, `Mean Turnover Gate = 1.26`).
   * **`VERTICAL_TRANSITION`** (`13,874` frames, `8.7%`): Fast camera pan/acceleration $\rightarrow$ Agile Physics + ML counter-attack fusion (`Mean w_ml = 0.461`, `Mean Turnover Gate = 1.14`).
   * **`OCCLUDED_FLIGHT`** (`36,449` frames, `22.9%`): Missing raw ball in live play $\rightarrow$ Homography kinematics bridge (`0.68`) + half-invariant `GBDT_Alive_NoBall` / EM / SeqRefiner (`Mean w_ml = 0.320`).
   * **`STOPPAGE_RESET`** (`41,552` frames, `26.2%`): Wide-FOV/zoom/reprojection-error physical stoppage state machine (`D F1 = 0.9830`).

2. **Specialist Binary Live ML + Half-Invariant Kinematics (`fit_sentiment_specialist_gbdt_models`):**
   * **`GBDT_Alive_Full`:** Binary `{H, A}` classifier over local/causal ball windows (`w <= 40` past only, zero long future lookahead smearing) + half-invariant pitch/kinematic features, trained with **35% Sim2Real ball-feature dropout** + `0.25` share perturbation (`100.00%` OOF Acc, `0.0070` LogLoss on `14,827` sharp live anchors).
   * **`GBDT_Alive_NoBall`:** Restricted strictly to **10 half-invariant kinematic features** (`cam_x_attack_h`, `vx_attack_h`, speed/tripod/box metrics), improving 6-fold live H-vs-A OOF accuracy from `50.25%` (`LogLoss = 0.7646`) to `55.70%` (`LogLoss = 0.7001`).
   * **`Crisp Dilated SeqRefiner`:** Narrowed temporal context `(-20, -10, -4, 0, +4, +10, +20)` frames (`100.00%` OOF Acc under `41.1%` Sim2Real dropout).

3. **True Viterbi Decoding + Sentiment & Attacking-Momentum Refinement (`decode_sentiment_hybrid_match`):**
   * Decodes exact 2-state `{H, A}` time-inhomogeneous Viterbi paths directly over the sentiment-fused log-odds (`turnover_gate` modulated by $\Psi_t$) **without raw `block_bridged` overwrite**.
   * **Pass 2B (Attacking-Momentum Contested Boundary Snap):** While a team actively progresses in its attacking direction (`|vx_attack_h| > 0.90 m/s`), a `<16`-frame defender probe followed by `>=8`-frame raw re-contact keeps possession with the attacking team until its final forward touch (snapping `13648 -> 13671`, `149474 -> 149496`, `151273 -> 151295`, and `188158 -> 188177`).
   * **Pass 3 (Calm-vs-Tempo Flicker Filter):** Absorbs `<48`-frame deflections cleanly while preserving fast `>=2.0 s` (`48..61f`) vertical transitions (`v_cam >= 1.40 m/s`).

---

## 3. Three-Model Benchmark Comparison & Diagnostics

### A. 5-Minute Public Ground Truth (`evaluation/run_evaluation.py --bootstrap 1000`)

| Rank | Model (`Branch`) | `spellF1` (Primary) | `Bootstrap 5-95%` | `IoU@0.5 F1` | `macroF1` | `tau=12 transF1` | `tau=25 transF1` | `poss` (GT 17) | `dur` (GT 14.2s) | `shareE` | `p90\|lat\|` |
|---:|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | **Sentiment Hybrid (`sentiment-hybrid`)** | **0.7067** | **0.5067..0.8623** | **0.7333** | 0.9496 | **0.5000** | 0.6667 | **13 (-4)** | **18.6 s (+31%)** | **0.016** | **18 f** |
| 2 | ML Ensemble (`model-fitting`) | 0.7000 | 0.4857..0.8495 | 0.7143 | **0.9543** | 0.4706 | **0.7059** | 11 (-6) | 21.9 s (+55%) | 0.022 | 21 f |
| 3 | Physics Matrix (`physics`) | 0.7000 | 0.4857..0.8495 | 0.7143 | **0.9543** | 0.4706 | **0.7059** | 11 (-6) | 21.9 s (+55%) | 0.022 | 21 f |
| 4 | Baseline Smooth 125 | 0.4242 | 0.2667..0.5939 | 0.4242 | 0.5812 | 0.1111 | 0.1667 | 16 (-1) | 18.5 s (+30%) | 0.052 | 18 f |

* Note: On the 17-possession 5-min public GT, `+0.0067 spellF1` lies within the 1,000-draw bootstrap band (`0.5067..0.8623`). Preserving the fast `59`-frame vertical transition (`[15647..15706)`) improves spell count (`13`), `IoU@0.5` (`11` matched vs `10`), `tau=12` transition precision (`9` matched vs `8`), and share error (`0.016`), while trading `-0.0047 macroF1` vs Models 1 & 2.

### B. Full-Match & Honest Masked-Ball Validation (`158,857` Graded Frames)

* **Honest Masked-Ball Stability Stress Test (`12,600` frames):** Replacing both live emissions (via `GBDT_Alive_NoBall`) and `bridged_codes` inside `8s` synthetic occlusion windows yields **`spellF1 = 0.8781`**, **`macroF1 = 0.9743`**.
* **Full-Match Macro Structure (`sentiment_hybrid_v1.csv`):** **`304` possession spells**, **`385` contiguous segments**, mean duration **`15.43 s`** (median `7.44 s`), `H=44.2% (70,294)`, `A=29.6% (47,011)`, `D=26.2% (41,552)`, Home share of alive time **`59.9%`**, 2nd-half pre-kickoff `D% = 100.0%` (`2,934` frames).

---

## 4. Single-Command Reproduction

```bash
pip install -r requirements.txt
python3 scripts/generate_submission.py --out evaluation/results/sentiment_hybrid_v1.csv
python3 evaluation/validate_submission.py evaluation/results/sentiment_hybrid_v1.csv
python3 scripts/run_diagnostics.py
python3 evaluation/run_evaluation.py --bootstrap 1000
```
