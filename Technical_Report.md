# Technical Report — Multi-Tier ML Ensemble (Ball-Dropout GBDT + Baum-Welch EM + Dilated Sequence Refiner)

**Competition:** More Than A Hack 2027 (Phase 1: Frame-Level Ball Possession)  
**Match:** FC Barcelona (`H`) vs. Athletic Club (`A`) — LaLiga EA Sports 2024–25, Matchday 2  
**Primary Ranking Metric:** `spellF1` (Mean Possession-Spell F1 over Temporal IoU $\in \{0.3, 0.4, 0.5, 0.6, 0.7\}$)  
**First Tie-Breaker:** `macroF1` (Unweighted per-frame F1 over `H`, `A`, `D`)  

---

## 1. Multi-Tier ML Architecture (`Proposed_Plan.md` Sections 4–7)

This branch implements the full **Multi-Tier Machine Learning & Sequence Modeling Plan** from `Proposed_Plan.md`:

1. **Stage 2 — 88 Engineered Multi-Modal Features (`src/features.py`):**
   * **`BALL_FEATURE_INDICES` (42 features):** Multi-scale centered, causal (past-only $[-w, 0]$), and anti-causal (future-only $[0, +w]$) contact shares over 5 horizons ($w \in \{8, 20, 40, 75, 150\}$ frames), forward-backward turnover gradients $\Delta p_H^{(w)}(t) = p_{H,\text{fut}}^{(w)} - p_{H,\text{past}}^{(w)}$, multi-scale log-odds, $1\text{s}/2\text{s}/4\text{s}$ duel flip rates, Shannon contention entropy $H_{\text{duel}}(t)$, and contiguous run lengths.
   * **`NOBALL_FEATURE_INDICES` (46 Ball-Independent Features):** Homography camera pose $(x_{\text{cam}}, y_{\text{cam}})$, half-normalized attacking coordinates & velocities $(\tilde{x}_{\text{cam}}, \tilde{v}_{x,\text{cam}})$, Kalman-RTS camera pan speed/acceleration/turn-angle, rolling `1s, 2s, 5s, 10s` speed statistics, horizontal & vertical FOV zoom $(\text{FOV}_X, \text{FOV}_Y)$, zoom dilatation rate $\dot{\text{FOV}}_X$, tripod-lock fractions, wide-FOV reset indicators, Inter-frame Optical-Flow Frobenius norm $\|M_t\|_F$, Spatial Jacobian $(\det J_t, \text{Tr} J_t, \kappa(J_t))$, reprojection error statistics, collective player speeds, Team Spatial Inertia ratio $\log\frac{\det(\Sigma_H)}{\det(\Sigma_A)}$, and Smart Tagging conjugate priors.
2. **Stage 3 (Model 2A) — Unsupervised Factored Baum-Welch EM HSMM (`src/models/pseudo_gbdt.py`):**
   * Factorizes emission likelihoods into conditionally independent Ball and Ball-Free Kinematic heads:
     $$p(\mathbf{f}_t \mid z_t) = p_{\text{kin}}(\mathbf{f}_{\text{kin}, t} \mid z_t) \cdot p_{\text{ST}}(\mathbf{f}_{\text{ST}, t} \mid z_t) \cdot \left[ p_{\text{ball}}(\mathbf{f}_{\text{ball}, t} \mid z_t) \right]^{\mathbb{I}[\text{ball\_seen}_t]}$$
3. **Stage 3 (Model 2B) — Out-of-Fold Temporal Block CV `GBDT_Full` & `GBDT_NoBall` (`src/models/pseudo_gbdt.py`):**
   * Selects `21,513` high-confidence algorithmic anchor samples across the match (zero human labels used) and runs **6-Fold Contiguous Temporal Block Cross-Validation** (`XGBClassifier` / `HistGradientBoostingClassifier`):
     * **`GBDT_Full` (88 features):** `93.91%` out-of-fold accuracy (`LogLoss = 0.1826`).
     * **`GBDT_NoBall` (46 ball-independent features — Masked-Ball Student):** Trained with all 42 ball features dropped out to predict possession state during the `40.7%` missing-ball gaps.
4. **Stage 3 (Model 3) — Multi-Stage Dilated Temporal Neural Refiner with Sim2Real 41% Ball-Dropout (`src/models/external_seq.py`):**
   * Gathers dilated 1D temporal context across $\delta \in \{-150, -75, -35, -15, -5, 0, +5, +15, +35, +75, +150\}$ frames ($\pm 6.0\text{s}$), trains an out-of-fold `MLPClassifier` under **Sim2Real 41% contiguous (`3–12s`) synthetic ball-dropout augmentation** (`87.70%` OOF accuracy), and applies **Truncated Temporal MSE Smoothing** before Stage 4 Time-Inhomogeneous Viterbi/HSMM decoding (`src/decoder.py`).

---

## 2. Validation & Scoreboard Summary

* **Validation Pillar #3 (Masked-Ball Stability Stress Test across `12,600` artificially masked visible frames):**
  * Unmasked vs. Masked-Ball Self-Consistency: **`spellF1 = 1.0000`**, **`macroF1 = 1.0000`** (Target `> 0.80`).
* **Public 5-Minute Benchmark (`evaluation/run_evaluation.py`):**
  * **`spellF1 = 0.7000`** (`@0.3: 0.7857, @0.4: 0.7857, @0.5: 0.7143, @0.6: 0.6429, @0.7: 0.5714`)
  * **`macroF1 = 0.9543`** (`H: 0.9494, A: 0.9306, D: 0.9830`)
  * **Full-Match Possession Count:** `321` spells, mean duration `14.62 s`, Home alive share `60.0%`.

---

## 3. Single-Command Reproduction

```bash
pip install -r requirements.txt
python scripts/generate_submission.py --out evaluation/results/ml_ensemble_v1.csv
python evaluation/validate_submission.py evaluation/results/ml_ensemble_v1.csv
python scripts/run_diagnostics.py
python evaluation/run_evaluation.py
```
