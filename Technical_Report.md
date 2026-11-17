# Technical Report — Final Multi-Modal Possession Hybrid (`branch-to-submit`)

**Competition:** More Than A Hack 2027 (Phase 1: Frame-Level Ball Possession)  
**Match:** FC Barcelona (`H`) vs. Athletic Club (`A`) — LaLiga EA Sports 2024–25, Matchday 2  
**Primary Ranking Metric:** `spellF1` (Mean Possession-Spell F1 over Temporal IoU $\in \{0.3, 0.4, 0.5, 0.6, 0.7\}$)  
**First Tie-Breaker:** `macroF1` (Unweighted per-frame F1 over `H`, `A`, `D`)  
**Submission File (Deterministically Reproducible):** `evaluation/results/branch_to_submit.csv`

---

## 1. System Architecture (`src/`)

Built strictly under [RULES.md](RULES.md) (zero hand-labels, zero external match events for this game, zero hyperparameter tuning on the 5-minute public alignment window; symmetric half-invariant geometry across all `199,319` frames with clean fallback when ATD tracks are absent):

1. **ATD Goalkeeper Re-ID & Trajectory Sanity Cleaning (`src/physics_matrices.py`, `src/preprocessing.py`):**
   * When broadcast/tactical cameras zoom tightly onto one penalty area, Metrica ATD periodically assigns the wrong goalkeeper tracklet slot (`gk_home = 23`, `gk_away = 45`) to the visible box keeper or a shadowed defender (`1,586` frames of `slot 23` and `13,458` frames of `slot 45` appearing `> 62 m` from their own goal).
   * `clean_atd_trajectories` masks wrong-end GK detections outside genuine late-half emergency corners (`d_own > 62 m`) and rejects single/two-frame tracklet teleport spikes (`> 3.6 m` over `2` frames) prior to Voronoi contact extraction, Matrix 5, Matrix 9, Matrix 10, and unsupervised retention scoring.
2. **Matrix 5 & Matrix 9 — Team Shape Tensors, Pressing Commitment & Defender Pinning (`src/physics_matrices.py`, `src/preprocessing.py`):**
   * `compute_team_spatial_inertia_tensors` extracts team centroid, surface area, length/width eigenvalues (`eig1`, `eig2`), and deepest/highest outfielder line coordinates (`low_line`, `high_line`).
   * `compute_press_and_pinning_matrices` quantifies goal-side attacker pinning (`pin_mean_h/a` discounted only by non-self deeper teammate cover `m != j`), unpinned closing-speed pressing pressure (`press_cap_on_h/a`), press-bait safety (`bait_safety_h/a`), open passing outlets (`>= 4.0 m` defender separation), and `net_press_hazard_h` (validated over `34,420` full-match frames: `-0.371 -> +0.071` shift before Home turnovers).
3. **Matrix 10 — Open-Space Ball Carrying (`TTI`) & Shooting-Range Decision Physics (`src/physics_matrices.py`, `src/features.py`):**
   * **Open-Space Heavy-Touch / Knock-On Carries (`knock_on_h/a`, `tti_margin_ha`, `open_carry_h/a`, `space_ahead_h/a`):** Projects ball velocity `+0.28 s` ahead to evaluate team Time-to-Intercept (`tti_margin_ha`), carrier chase/ball alignment, forward cone space, and an **origin-touch requirement** (`slot_touched_38`: carrier must have been `<= 2.25 m` from the ball in the prior `38` frames so opposing pressers chasing long passes never trigger fake carries). True sprint knock-ons (`knock_on_h = 2,117`, `knock_on_a = 1,308` frames) are exempted from `in_flight` damping and held steady across Viterbi & `Pass 2C/3B`.
   * **Shooting Range & Goal-Cone Blockers (`shot_takeon_open_h/a`, `shot_crowded_risk_h/a`, `shot_flight`):** Inside shooting distance (`d_goal <= 28 m`, `|y - 34| <= 20 m`), counts opponents inside the tapered cone to goalposts + `4 m` radius. Open lanes (`n_def_cone <= 1`, no defender `< 4 m`) stabilize carry/take-on likelihood, whereas crowded shooting cones & high-velocity goalward shots (`shot_flight = 1,457` frames) ease outward transition gates to capture blocks, goalkeeper parries, and rebounds.
4. **64-Feature Half-Invariant `NoBall` Territory Specialist & Dual `XGBoost` + `HistGBDT` (`src/features.py`, `src/models/pseudo_gbdt.py`):**
   * Expands `X_noball_inv` (`64` strictly ball-free, `nearest_slot`-free, H1/H2-symmetric features adding outfield line heights, penalty-box/attacking-third occupancy, anisotropic camera pitch control `compute_anisotropic_pitch_control`, and `25`-frame rolling levels/deltas) and `X_ctrl_atd` (`25` ball/carry/shooting features).
   * Trains 6-fold temporal block out-of-fold **Dual `XGBoost (3.0.2)` + `HistGradientBoostingClassifier` (`0.65 / 0.35`)** ensembles under `30%` Sim2Real ball-dropout and `15%` duel-flicker augmentation (`GBDT_Alive_Full = 99.40%` / `DuelStress = 95.38%`, **`GBDT_Alive_NoBall(64-inv) = 85.83%, AUC = 0.9380`**).
5. **Smart Tagging Restarts, `Crisp Dilated SeqRefiner` & Self-Supervised `Path-CRF` Boundary Scorer (`src/smart_tagging_priors.py`, `src/models/external_seq.py`, `src/models/sentiment_hybrid.py`, `src/decoder.py`):**
   * Forward-scans Smart Tagging `SET PIECES` until confirmed inbounds ball launch (`v_ball >= 3.6` or `max >= 5.5`), isolates post-shot `GOALS` tails, adds sustained strict-OOB (`oob_sust`) stoppages (**Dead-Ball F1 = `0.9895`**, **Alive/Dead F1 = `0.9974`**).
   * Trains a 6-fold self-supervised **`Path-CRF` edge classifier** (`fit_oof_path_crf_boundary_model`) using physical velocity reversals, TTI shifts, carry/shot dynamics, and pressing/chaos features cleanly modulating `turnover_gate_ha/ah` (`np.clip(0.94 + 0.46 * p_boundary, 0.94, 1.40)` + carry holds + shot-flight easing) via OOF `L-BFGS-B` Sentiment Meta-Stacking (`OOF Meta LogLoss = 0.2123`).
   * Stage 4 runs directed Viterbi + `Pass 2C` (shot/reception snapper protecting active knock-on carries) + `Pass 3B/3C`.

---

## 2. Benchmark & Validation Summary

### A. Full-Match Self-Supervised & Stress Benchmarks (`158,857` Graded Frames, Zero GT Leakage)
* **6-Fold OOF Specialist Accuracy:** `GBDT_Alive_Full = 99.40%` (`LogLoss = 0.0224`), `Harsh Duel Stress = 95.38%` (`LogLoss = 0.1110`), **`GBDT_Alive_NoBall(64-inv) = 85.83%` (`AUC = 0.9380`, `LogLoss = 0.3143`)**, `Crisp Dilated SeqRefiner = 99.98%` (`Stress = 99.11%`), `OOF Meta LogLoss = 0.2123`.
* **Honest Masked-Ball 8s Stability Stress Test (`12,600` artificially occluded visible frames):** Unmasked vs. Masked consistency **`spellF1 = 0.9347`**, **`macroF1 = 0.9872`**.
* **Full-Match Macro Structure (`branch_to_submit.csv`):** **`292` possession spells** (`417` segments), mean duration **`12.33 s`** (median `6.54 s`), `H = 32.3% (51,275)`, `A = 24.4% (38,701)`, `D = 43.4% (68,881)`, Home share of alive time **`57.0%`**, 2nd-half pre-kickoff `D% = 100.0% (2,934 frames)`.
* **Determinism & Unit Suites:** `tests/test_pressing_pinning.py` (`ALL_TESTS_PASS` — GK cleaning, Matrix 9/10, Half-2 mirror symmetry, presser-origin guard, and `has_full_atd=False` fallback) + byte-for-byte identical CSV reproduction (`RULES.md` Rule 3 verified via `cmp -s`).

### B. 5-Minute Public Alignment & Integrity Check (`evaluation/run_evaluation.py --gt public --bootstrap 1000`)

| Rank | Model (`Branch`) | `spellF1` | `IoU@0.5 F1` | `IoU@0.7 F1` | `macroF1` | `kappa` | `tau=12 F1` | `tau=25 F1` | `latency` | `poss` (GT 17) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **2 (Macro #1)** | **Final (`branch-to-submit`)** | **0.7143** | **0.7143** | **0.6429** | **0.9631** | **0.9343** | **0.6471** | **0.7647** | **-2.0 f** | **11 (-6)** |
| 1 | Pressing-Pinning (`pressing-pinning-hybrid`) | 0.7200 | 0.7333 | 0.6000 | 0.9563 | 0.9218 | 0.6111 | 0.7222 | -2.0 f | 13 (-4) |
| 3 | Sentiment v2 (`sentiment-hybrid-v2`) | 0.7067 | 0.7333 | — | 0.9496 | 0.9113 | 0.5000 | 0.6667 | -5.8 f | 13 (-4) |
| 5 | ML / Physics (`model-fitting` / `physics`) | 0.7000 | 0.7143 | — | 0.9543 | — | 0.4706 | 0.7059 | -3.8 f | 11 (-6) |
| 8 | Baseline Smooth 50 | 0.3256 | 0.3256 | 0.0930 | 0.6207 | 0.5202 | 0.0816 | 0.0816 | +4.0 f | 26 (+9) |

*(Note: Eliminating the phantom `slot 23` opponent-box GK split at `15647..15705` merged `14843..15879` cleanly, removing 2 spurious public transitions so `transF1@25` hit `0.7647` (`P=92.86%`, only 1 spurious transition) and `macroF1` hit `0.9631` (`bootstrap 0.8939..0.9841`), while `spellF1` `@0.7` rose `0.6000 -> 0.6429` and `@0.3` reached `precision = 1.0000`.)*

---

## 3. Reproduction

```bash
pip install numpy scipy pandas scikit-learn xgboost
python3 tests/test_pressing_pinning.py
python3 scripts/generate_submission.py --out evaluation/results/branch_to_submit.csv
python3 evaluation/validate_submission.py evaluation/results/branch_to_submit.csv
python3 evaluation/run_evaluation.py --gt public --bootstrap 1000
```
