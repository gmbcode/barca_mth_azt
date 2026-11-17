# Technical Report — Full-ATD Pressing-Pinning Hybrid (`pressing-pinning-hybrid`)

**Competition:** More Than A Hack 2027 (Phase 1: Frame-Level Ball Possession)  
**Match:** FC Barcelona (`H`) vs. Athletic Club (`A`) — LaLiga EA Sports 2024–25, Matchday 2  
**Primary Ranking Metric:** `spellF1` (Mean Possession-Spell F1 over Temporal IoU $\in \{0.3, 0.4, 0.5, 0.6, 0.7\}$)  
**First Tie-Breaker:** `macroF1` (Unweighted per-frame F1 over `H`, `A`, `D`)  

---

## 1. What `pressing-pinning-hybrid` Adds (Real 22-Player ATD + Smart Tagging + Matrix 9)

With full tracking (`BAR-ATH_tactical_atd.txt`, `BAR-ATH_tactical_atd.xml`) and `638` Smart Tagging annotations (`BAR-ATH_tactical_smart_tagging.xml`) unlocked (`has_full_atd = True`, `st_available = True`) alongside full no-ATD fallback safety (`tests/test_pressing_pinning.py`), this branch introduces five general, rule-compliant tactical & physical upgrades (`RULES.md` compliant — zero human labels or public-window hyperparameter tuning):

1. **Matrix 9: Pressing Commitment, Defender Pinning & Blind-Side Risk (`src/physics_matrices.py`, `src/preprocessing.py`):**
   * Formalizes the "attacker behind defender prevents jumping to press" law (`compute_press_and_pinning_matrices`):
     * Measures goal-side pinning `pin_mean_h/a` (weighted by absence of deeper teammate cover `m != j`), unpinned closing-speed press capacity `press_cap_on_h/a`, press-bait safety `bait_safety_h/a`, open passing outlets `n_free_outlets_h/a` (`>= 4.0 m` clear of defenders), and net turnover hazard `net_press_hazard_h`.
     * Full-match validation (`34,420` clean vs turnover frames): `net_press_hazard_h` shifts from `-0.371` (`H` keeps) to `+0.071` (`H` loses), and `press_cap_on_h` nearly doubles from `0.295 -> 0.531` right before Home turnovers.
   * Integrates Matrix 5 (`log_width_ratio`, `log_depth_ratio`) alongside NaN-safe finite-difference player velocities (no zero-fill dropout spikes).
2. **44-Feature Half-Invariant `NoBall` & Control Kinematics (`src/features.py`, `src/models/pseudo_gbdt.py`, `src/models/external_seq.py`):**
   * Expands `X_noball_inv` from `22` camera-only columns to **`44` strictly ball-free & nearest-slot-free half-invariant features** (`Matrix 5 + Matrix 9 + attacking team centroids + team speeds + Smart Tagging`), jumping 6-fold live `GBDT_NoBall` accuracy from **`56.76%` -> `83.57%` (`AUC = 0.9193`)** without any `nearest_slot` or ball-coordinate leakage.
   * Adds `X_ctrl_atd` (`d_home_1/2`, `d_away_1/2`, continuous Gaussian control proximity `ctrl_h/a`, ball kinematics `v_ball`, `a_ball`, `in_flight`, and unsupervised `track_retention_prior`) for `GBDT_Alive_Full` (`98.95%` clean / `93.85%` duel-stress OOF Acc), temp-calibrated HMM EM (`ll` correlation damping), and enriched `Crisp Dilated SeqRefiner` context.
3. **Ball-Launch Set-Piece Restart, Goal Celebration Tail & Strict-OOB Physics (`src/smart_tagging_priors.py`, `src/models/heuristic_fsm.py`):**
   * Fixes the Smart Tagging `SET PIECES` backward-scan bug (`65.5%` false `D` down to `9.4%`) by scanning **forward** from `s_f` until confirmed inbounds ball-launch speed (`v_ball >= 3.6 m/s` or `max >= 5.5`), isolating post-shot `GOALS` celebration tails, adding sustained strict out-of-bounds (`oob_sust`) stoppages, and extending resting stoppage exits (`< 3.6 m/s`) cleanly to the kick/throw launch frame (**Dead-Ball F1 = `0.9895`, `alive/dead F1 = 0.9974`**).
4. **Control-Weighted Tier 1 & Matrix 9 Directed Transition Gates (`src/models/heuristic_fsm.py`, `src/models/sentiment_hybrid.py`):**
   * Dampens aerial pass-in-flight frames (`in_flight`: `min_d > 2.7 m` or `v_ball > 9.5 m/s` while `min_d > 1.7 m`) in Tier 1 emissions and tightens mid-flight transition gates (`* 0.55`) while modulating directed `H -> A` (`0.765`) and `A -> H` (`0.816`) gates by Matrix 9 unpinned press capacity and bait safety (`OOF Meta LogLoss = 0.2178`).
5. **Physical Reception Boundary Snapper (`Pass 2C`) & Control-Verified Deflection Filtering (`src/decoder.py`):**
   * Shifts Voronoi mid-flight boundary flips (`-10.. -33f` early bias across `123` full-match turnovers) forward to receiver deceleration / ball-trap / clearance-kick frames (`Pass 2C`), followed by ATD control-proximity deflection (`Pass 3B`) and pre-stoppage last-touch clearance (`Pass 3C`) filtering — cutting mean transition latency by **`65%`** (`-5.8f` -> **`-2.0f`**) and lifting `transF1@tau=12` from `0.5000` -> **`0.6111`**.

---

## 2. Benchmark & Validation Summary

### A. 5-Minute Public Ground Truth (`evaluation/run_evaluation.py --bootstrap 1000`)

| Rank | Model (`Branch`) | `spellF1` | Bootstrap 5–95% | `IoU@0.5 F1` | `macroF1` | `tau=12` F1 | `tau=25` F1 | `latency` | `poss` (GT 17) | `shareE` |
|---:|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | **Pressing-Pinning (`pressing-pinning-hybrid`)** | **0.7200** | **0.5200..0.8753** | **0.7333** | **0.9563** | **0.6111** | **0.7222** | **-2.0 f** | **13 (-4)** | **0.004** |
| 2 | Sentiment v2 (`sentiment-hybrid-v2`) | 0.7067 | 0.5067..0.8623 | 0.7333 | 0.9496 | 0.5000 | 0.6667 | -5.8 f | 13 (-4) | 0.016 |
| 3 | Sentiment v1 (`sentiment-hybrid`) | 0.7067 | 0.5067..0.8623 | 0.7333 | 0.9496 | 0.5000 | 0.6667 | -5.8 f | 13 (-4) | 0.016 |
| 4 | ML Ensemble (`model-fitting`) | 0.7000 | 0.4857..0.8495 | 0.7143 | 0.9543 | 0.4706 | 0.7059 | -3.8 f | 11 (-6) | 0.022 |
| 5 | Physics Matrix (`physics`) | 0.7000 | 0.4857..0.8495 | 0.7143 | 0.9543 | 0.4706 | 0.7059 | -3.8 f | 11 (-6) | 0.022 |
| 6 | Baseline Smooth 50 | 0.3256 | 0.2140..0.4465 | 0.3256 | 0.6207 | 0.0816 | 0.0816 | +4.0 f | 26 (+9) | 0.048 |

### B. Full-Match & Stress Validation (`158,857` Graded Frames)

* **6-Fold Full-Match Specialists (Zero GT Leakage):** `GBDT_Full = 98.95%` (`DuelStress = 93.85%`), **`GBDT_NoBall(44-inv) = 83.57% (AUC = 0.9193)`**, `OOF Meta LogLoss = 0.2178`.
* **Full-Match Macro Structure (`pressing_pinning_hybrid.csv`):** **`11,911` frames refined vs. `v2`**; **`294` possession spells** (`419` contiguous `H/A/D` segments), mean duration **`12.24 s`** (median `6.56 s`), `H=32.4% (51,451)`, `A=24.3% (38,525)`, `D=43.4% (68,881)`, Home share of alive time **`57.18%`**, 2nd-half pre-kickoff `D% = 100.0%` (`2,934` frames).
* **Dual-Mode Fallback Verification:** Tested on no-ATD input (`tests/test_pressing_pinning.py` + `has_full_atd=False` pipeline run -> `len=199319`, valid cleanly).

---

## 3. Reproduction

```bash
pip install -r requirements.txt
python3 tests/test_pressing_pinning.py
python3 scripts/generate_submission.py --out evaluation/results/pressing_pinning_hybrid.csv
python3 evaluation/validate_submission.py evaluation/results/pressing_pinning_hybrid.csv
python3 evaluation/run_evaluation.py --gt public --bootstrap 1000
```
