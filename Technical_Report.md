# Technical Report — Multi-Matrix Physics & Time-Inhomogeneous HSMM Pipeline

**Competition:** More Than A Hack 2027 (Phase 1: Frame-Level Ball Possession)  
**Match:** FC Barcelona (`H`) vs. Athletic Club (`A`) — LaLiga EA Sports 2024–25, Matchday 2  
**Primary Ranking Metric:** `spellF1` (Mean Possession-Spell F1 over Temporal IoU $\in \{0.3, 0.4, 0.5, 0.6, 0.7\}$)  
**First Tie-Breaker:** `macroF1` (Unweighted per-frame F1 over `H`, `A`, `D`)  

---

## 1. Executive Summary

This repository implements the 4-Stage Physics & Probabilistic Sequence Plan from `Proposed_Plan.md`, expanded with **8 explicit Physics & Kinematic Matrices** (`src/physics_matrices.py`):

1. **Projective Homography & Inverse Camera-Pose Matrix ($H_t, H_t^{-1} \in \mathbb{R}^{3 \times 3}$)**
2. **Inter-Frame Homography Optical-Flow Matrix ($M_t = H_t H_{t-1}^{-1} - I_3 \in \mathbb{R}^{3 \times 3}$) & Spatial Resolution Jacobian Matrix ($J_t(x,y) \in \mathbb{R}^{2 \times 2}$)**
3. **6D Constant-Acceleration Kalman Filter & Rauch-Tung-Striebel (RTS) Backward Smoother Matrices ($F, Q, R_t, K_{\infty}, G_{\infty}$)**
4. **$46 \times 46$ Pairwise Player Interaction & Pass-Corridor Shadowing Matrix ($D_t, V_t, C_t \in \mathbb{R}^{46 \times 46}$)**
5. **$2 \times 2$ Team Spatial Dispersion & Tactical Inertia Tensor ($\Sigma_H(t), \Sigma_A(t) \in \mathbb{R}^{2 \times 2}$)**
6. **Anisotropic Velocity-Weighted Pitch Control Tensor ($S_i(t) = R(\theta_i)\Lambda(s_i)R(\theta_i)^T \in \mathbb{R}^{2 \times 2}$)**
7. **Multi-Scale Temporal Evidence & Contention Entropy Matrix ($E_t \in \mathbb{R}^{N \times K \times 3}$)**
8. **Time-Inhomogeneous Non-Stationary HSMM Transition Matrix ($A_t \in \mathbb{R}^{N \times 3 \times 3}$)**

### Performance Summary (Evaluated via `evaluation/run_evaluation.py`)

| Submission | `spellF1` (Rank Key) | `macroF1` (Tie-Break) | `H` F1 | `A` F1 | `D` F1 | `transF1` ($\pm 25$f) | Spells (5m / Full) | Mean Dur (5m / Full) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`physics_matrix_v1` (Ours)** | **`0.7000`** | **`0.9543`** | **`0.9494`** | **`0.9306`** | **`0.9830`** | **`0.7059`** | `11` / **`321`** | `21.9s` / **`14.6s`** |
| `baseline_smooth125` | `0.4242` | `0.5812` | `0.7864` | `0.8108` | `0.1464` | `0.1667` | `16` / `334` | `18.5s` / `14.1s` |
| `baseline_smooth50` (Floor) | `0.3256` | `0.6207` | `0.7781` | `0.7615` | `0.3225` | `0.0816` | `26` / `594` | `10.8s` / `7.2s` |
| `baseline_smooth25` | `0.2868` | `0.6200` | `0.7620` | `0.7123` | `0.3857` | `0.2034` | `36` / `851` | `7.6s` / `4.8s` |
| `baseline_raw` | `0.0378` | `0.5972` | `0.7290` | `0.6430` | `0.4197` | `0.1410` | `131` / `1,949` | `2.0s` / `1.9s` |

---

## 2. Single-Command Reproduction

```bash
pip install -r requirements.txt
python scripts/generate_submission.py --out evaluation/results/physics_matrix_v1.csv
python evaluation/validate_submission.py evaluation/results/physics_matrix_v1.csv
python evaluation/run_evaluation.py
```
