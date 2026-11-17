# Technical Report — Match-Sentiment Hybrid v2 (`sentiment-hybrid-v2`)

**Competition:** More Than A Hack 2027 (Phase 1: Frame-Level Ball Possession)  
**Match:** FC Barcelona (`H`) vs. Athletic Club (`A`) — LaLiga EA Sports 2024–25, Matchday 2  
**Primary Ranking Metric:** `spellF1` (Mean Possession-Spell F1 over Temporal IoU $\in \{0.3, 0.4, 0.5, 0.6, 0.7\}$)  
**First Tie-Breaker:** `macroF1` (Unweighted per-frame F1 over `H`, `A`, `D`)  

---

## 1. What `sentiment-hybrid-v2` Adds Over `sentiment-hybrid` (`v1`)

This branch implements six structural upgrades plus rule-compliant **Hansi Flick tactical asymmetry** and **unsupervised player/zone retention priors** (`RULES.md` compliant — zero human labels or frame-specific tuning):

1. **Contested `OCCLUDED_FLIGHT` Handovers & Parked Stoppage Detection (`src/models/heuristic_fsm.py`):**
   * Previously, every live missing-ball gap locked the full interval to `pre_team` (`bridged_codes = pre_team`) and long multi-minute camera-parked gaps with slight drift (`disp >= 7.0 m`) were misclassified as live play.
   * **Parked Dead-Ball Stoppages (`+19,368` newly recovered `D` frames):** Long missing-ball gaps with high tripod lock (`>=400f`, `tripod >= 0.65`, `mean_v_cam < 0.65`; or `>=1000f`, `tripod >= 0.60`, `mean_v_cam < 0.80`) are classified as `STOPPAGE_RESET` (`D`), fixing multi-minute celebrations/injuries (e.g., `58099..61558`, `157564..163094`) and bringing full-match Dead-Ball ratio to realistic **`38.3%`** (`60,920` of `158,857` graded frames, `61.7%` effective ball-in-play time).
   * **Kinematic Contested Handover (`7,631` frames):** For live missing-ball gaps where `pre_team != post_team`, computes net directional forward velocity `vx_attack_h * s_post` to split handover timing rather than blindly locking the whole gap to `pre_team`, and raises ML `NoBall` routing weight (`mean = 0.499`).
2. **Non-Redundant Duel-Robust `GBDT_Alive_Full` & Spike-Resilient `SeqRefiner` (`src/models/pseudo_gbdt.py`, `src/models/external_seq.py`):**
   * Drops centered instant `w=8/w=20` shares (`sh_8, sh_20`, already encoded by Tier 1) and trains on clean + borderline live anchors (`16,594`) under `30%` ball-feature dropout + `15%` `w=8` flicker flips (`99.98%` clean OOF Acc; **`88.91%` Harsh Duel-Stress OOF Acc**).
   * Trains `Crisp Dilated SeqRefiner` with `18%` contiguous (`5–16f`) spike flips alongside `41.1%` Sim2Real dropout (**`100.00%` clean / `93.06%` stress-recovered OOF Acc**).
3. **22-Feature Half-Invariant `NoBall` Kinematics (`src/features.py`):**
   * Strips leaky clock/reprojection/unflipped-half variables and adds net attacking territorial displacement ($\Delta \tilde{x}$ at `1s, 3s, 6s`) and velocity reversal features (`vx_rev_1s, vx_rev_3s`), raising 6-fold CV live `H/A` `GBDT_Alive_NoBall` accuracy to **`58.26%`** (`LogLoss = 0.6847`, vs. `55.70%` in `v1` and `50.25%` in `model-fitting`).
4. **Evidence-Integrated Filtering (`src/decoder.py`):**
   * Protects low-chaos clean breakaways (`dur >= 34`, `chaos < 0.42`, `vx > 1.10 m/s`, `ev > 1.65`) while absorbing high-chaos stalled flickers without fragmenting full-match spells (`306` spells, mean `12.80 s`).
5. **Half-Invariant Forward-Backward HMM EM & OOF Meta-Router (`src/models/pseudo_gbdt.py`, `src/models/sentiment_hybrid.py`):**
   * Upgrades `fit_baum_welch_em(use_invariant_hmm=True)` to run a 3-state Forward-Backward HMM smoother over `X_noball_inv`.
   * Optimizes sentiment meta-parameters strictly via 6-fold OOF `L-BFGS-B` on stress-corrupted anchors (`w_chaos_boost = 0.4820`, `w_noball_in_ml = 0.1000`, `beta_flick_prior = 0.2872`, `oof_meta_logloss = 0.2402`).
6. **Hansi Flick Tactical System & Unsupervised Player/Zone Retention (`src/preprocessing.py`, `src/models/sentiment_hybrid.py`, `src/physics_matrices.py`):**
   * Uses out-of-fold empirical control balance (`oof_pi_h`), Barcelona high-line counter-press (`psi_high_press`), vertical Athletic breakaway detection (`psi_counter_break_a`), and zone + unsupervised track-level ball-retention propensity (`track_retention_prior`, activating automatically whenever full ATD is present) to drive directed Viterbi transition gates (**Mean `H -> A = 0.958`**, **Mean `A -> H = 1.014`**).

---

## 2. Benchmark & Validation Summary

### A. 5-Minute Public Ground Truth (`evaluation/run_evaluation.py --bootstrap 1000`)

| Rank | Model (`Branch`) | `spellF1` | Bootstrap 5–95% | `IoU@0.5 F1` | `macroF1` | `tau=12` F1 | `tau=25` F1 | `poss` (GT 17) | `dur` (GT 14.2s) | `shareE` |
|---:|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | **Sentiment v2 (`sentiment-hybrid-v2`)** | **0.7067** | **0.5067..0.8623** | **0.7333** | 0.9496 | **0.5000** | 0.6667 | **13 (-4)** | **18.6 s** | **0.016** |
| **1** | Sentiment v1 (`sentiment-hybrid`) | **0.7067** | **0.5067..0.8623** | **0.7333** | 0.9496 | **0.5000** | 0.6667 | **13 (-4)** | **18.6 s** | **0.016** |
| 3 | ML Ensemble (`model-fitting`) | 0.7000 | 0.4857..0.8495 | 0.7143 | **0.9543** | 0.4706 | **0.7059** | 11 (-6) | 21.9 s | 0.022 |
| 4 | Physics Matrix (`physics`) | 0.7000 | 0.4857..0.8495 | 0.7143 | **0.9543** | 0.4706 | **0.7059** | 11 (-6) | 21.9 s | 0.022 |

### B. Full-Match & Stress Validation (`158,857` Graded Frames)

* **Honest Masked-Ball Stability Stress Test (`12,600` masked frames):** **`spellF1 = 0.8864`** (`+0.0083` over `v1`'s `0.8781`), **`macroF1 = 0.9699`**.
* **Full-Match Macro Structure (`sentiment_hybrid_v2.csv`):** **`23,926` frames updated vs. `v1`** (`19,368` newly recovered parked `D` frames + live handover/Flick refinements); **`306` possession spells** (`397` contiguous `H/A/D` segments), mean duration **`12.80 s`** (median `6.78 s`), `H=35.9% (57,074)`, `A=25.7% (40,863)`, `D=38.3% (60,920)`, Home share of alive time **`58.3%`**, 2nd-half pre-kickoff `D% = 100.0%` (`2,934` frames).

---

## 3. Reproduction

```bash
pip install -r requirements.txt
python3 scripts/generate_submission.py --out evaluation/results/sentiment_hybrid_v2.csv
python3 evaluation/validate_submission.py evaluation/results/sentiment_hybrid_v2.csv
python3 scripts/run_diagnostics.py
python3 evaluation/run_evaluation.py --bootstrap 1000
```
