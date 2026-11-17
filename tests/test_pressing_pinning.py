"""Unit & regression checks for Matrix 9 pressing/pinning geometry & dual-mode pipeline safety."""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.physics_matrices import (  # noqa: E402
    clean_atd_trajectories,
    compute_carry_and_shooting_matrices,
    compute_press_and_pinning_matrices,
)
from src.features import extract_physics_feature_dict  # noqa: E402
from src.models.heuristic_fsm import compute_tier1_emission_log_probs, detect_physical_regimes  # noqa: E402
from src.models.sentiment_hybrid import compute_match_sentiment_tensor, fuse_models_by_match_sentiment  # noqa: E402
from src.decoder import decode_sentiment_hybrid_match  # noqa: E402


def test_matrix9_pinning_cover_outlets_and_half_symmetry():
    """Verify defender-pinning behind pressers, cover rescue, free outlets, and Half-2 mirroring."""
    # Frame 0 (H1): Home carrier near (50, 34), Away defender (24) pressing at (55, 34),
    #               Home striker (0) pinned behind defender at (60, 34) with NO deeper Away cover.
    # Frame 1 (H1): Same setup PLUS deeper Away CB (25) covering at (65, 34) behind Home striker.
    # Frame 2 (H2, frame >= half_split_frame=2): Exact mirror x -> 105.3 - x of Frame 0.
    n_frames, n_slots = 3, 46
    px = np.full((n_frames, n_slots), np.nan, dtype=np.float64)
    py = np.full((n_frames, n_slots), np.nan, dtype=np.float64)
    vx = np.zeros((n_frames, n_slots), dtype=np.float64)
    vy = np.zeros((n_frames, n_slots), dtype=np.float64)
    team = np.array([0] * 24 + [1] * 22, dtype=np.int8)
    bx = np.array([50.0, 50.0, 105.3 - 50.0], dtype=np.float64)
    by = np.array([34.0, 34.0, 34.0], dtype=np.float64)

    for f in (0, 1):
        px[f, 1], py[f, 1] = 50.0, 34.0     # Home ball carrier
        px[f, 2], py[f, 2] = 44.0, 20.0     # Open Home passing outlet (>4m from Away)
        px[f, 0], py[f, 0] = 60.0, 34.0     # Home attacker behind presser (x_att=60 > 55)
        px[f, 24], py[f, 24] = 55.0, 34.0   # Away presser closing toward ball
        vx[f, 24] = -3.0
    # Frame 1 adds deeper Away cover CB behind striker at x=65
    px[1, 25], py[1, 25] = 65.0, 34.0

    # Frame 2 mirrors Frame 0 in Half 2 (attacking direction -X)
    for slot in (0, 1, 2, 24):
        px[2, slot] = 105.3 - px[0, slot]
        py[2, slot] = py[0, slot]
        vx[2, slot] = -vx[0, slot]

    m9 = compute_press_and_pinning_matrices(
        px, py, vx, vy, team, bx, by, pitch_length=105.3, half_split_frame=2, gk_slots=(23, 45)
    )

    # Pinned presser (Frame 0) vs Covered presser (Frame 1)
    assert m9["pin_mean_a"][0] > 0.55, f"Expected high pinning on uncovered presser, got {m9['pin_mean_a'][0]}"
    assert m9["pin_mean_a"][1] < m9["pin_mean_a"][0] * 0.65, "Deeper cover should relieve presser pinning"
    assert m9["press_cap_on_h"][1] > m9["press_cap_on_h"][0], "Deeper cover should unlock higher unpinned press"
    assert m9["bait_safety_h"][0] > 0.55, "Home carrier has high press-bait safety when presser is pinned"
    assert m9["n_free_outlets_h"][0] >= 1.0, "Open Home outlet at (44, 20) should be counted"

    # Half 1 <-> Half 2 spatial mirroring must be exact within float tolerance
    assert np.isclose(m9["pin_mean_a"][0], m9["pin_mean_a"][2], atol=1e-6)
    assert np.isclose(m9["press_cap_on_h"][0], m9["press_cap_on_h"][2], atol=1e-6)
    assert np.isclose(m9["bait_safety_h"][0], m9["bait_safety_h"][2], atol=1e-6)


def test_dual_mode_fallback_without_atd():
    """Verify fallback (has_full_atd=False) executes cleanly across all feature/model/decoder stages."""
    n = 400
    raw_codes = np.zeros(n, dtype=np.int8)
    raw_codes[160:260] = 1
    raw_codes[300:] = 2

    z = np.zeros(n, dtype=np.float64)
    half_pos = np.full(n, 0.5, dtype=np.float64)
    stage1 = {
        "n_frames": n,
        "pitch": (105.3, 68.0),
        "fps": 25.0,
        "half_blocks": [(1, 0, n - 1)],
        "reproj_error": np.full(n, 0.25),
        "valid_hom": np.ones(n, dtype=bool),
        "pose": {
            "cam_x_norm": half_pos,
            "cam_y_norm": half_pos,
            "cam_x_m": np.full(n, 52.65),
            "cam_y_m": np.full(n, 34.0),
            "fov_x_m": np.full(n, 60.0),
            "fov_y_m": np.full(n, 38.0),
            "dfov_x_ms": z,
        },
        "cam_kalman": {
            "x_smooth": np.full(n, 52.65),
            "y_smooth": np.full(n, 34.0),
            "vx_smooth": z,
            "vy_smooth": z,
            "speed_smooth": np.ones(n),
            "acc_mag_smooth": z,
            "turn_angle_rad": z,
            "mahalanobis_sq": z,
        },
        "flow_jac": {"frob_M": z, "det_J": np.ones(n), "cond_J": np.ones(n)},
        "atd": {
            "has_full_atd": False,
            "raw_codes": raw_codes,
            "ball_x_m": np.full(n, 52.65),
            "ball_y_m": np.full(n, 34.0),
            "ball_kalman": {
                "x_smooth": np.full(n, 52.65),
                "y_smooth": np.full(n, 34.0),
                "speed_smooth": np.ones(n),
                "acc_mag_smooth": z,
            },
            "mean_speed_all": np.full(n, 2.0),
            "p90_speed_all": np.full(n, 4.0),
            "log_area_ratio": z,
            "log_width_ratio": z,
            "log_depth_ratio": z,
            "pin_mean_a": z,
            "pin_mean_h": z,
            "press_cap_on_h": z,
            "press_cap_on_a": z,
            "bait_safety_h": half_pos,
            "bait_safety_a": half_pos,
            "n_free_outlets_h": z,
            "n_free_outlets_a": z,
            "net_press_hazard_h": z,
            "nearest_slot": np.full(n, -1, dtype=np.int16),
            "track_retention_prior": half_pos,
        },
        "ms": {
            "E_matrix": np.full((n, 5, 3), 1.0 / 3.0),
            "alive_home_share": np.full((n, 5), 0.5),
            "duel_flip_rate_2s": z,
            "contention_entropy_2s": z,
            "run_length_same": np.full(n, 25, dtype=np.int32),
        },
    }
    st_priors = {
        "available": False,
        "st_net_log_odds": z,
        "set_piece_pre_kick_d": np.zeros(n, dtype=bool),
        "set_piece_restart_team": np.full(n, -1, dtype=np.int8),
        "goal_dead_mask": np.zeros(n, dtype=bool),
    }

    feats = extract_physics_feature_dict(stage1, st_priors)
    assert feats["X_noball_inv"].shape == (n, 64)
    assert feats["X_ctrl_atd"].shape == (n, 25)
    reg = detect_physical_regimes(feats, stage1["half_blocks"], split_contested_occl=True)
    t1 = compute_tier1_emission_log_probs(feats, reg)
    sent = compute_match_sentiment_tensor(feats, reg)
    p_ha = np.full((n, 2), 0.5)
    seq_lp = np.log(np.full((n, 3), 1.0 / 3.0))
    hlp, tgha, tgah, lfus, _ = fuse_models_by_match_sentiment(feats, reg, sent, t1, t1, p_ha, p_ha, seq_lp)
    labels = decode_sentiment_hybrid_match(
        hlp, feats, reg, sent, 0.5 * (tgha + tgah), lfus, stage1["half_blocks"], tgha, tgah
    )
    assert len(labels) == n
    assert set(np.unique(labels)).issubset({"H", "A", "D", "X"})


def test_gk_cleaning_carry_and_shooting_matrices():
    """Verify GK wrong-end masking, open-space knock-on carry TTI, shooting-cone risk, and H2 symmetry."""
    n_frames, n_slots = 4, 46
    px = np.full((n_frames, n_slots), np.nan, dtype=np.float64)
    py = np.full((n_frames, n_slots), np.nan, dtype=np.float64)
    vx = np.zeros((n_frames, n_slots), dtype=np.float64)
    vy = np.zeros((n_frames, n_slots), dtype=np.float64)
    team = np.array([0] * 24 + [1] * 22, dtype=np.int8)

    # Check GK swap masking at frame 0 (H1): slot 23 at x=100.9 (wrong box) should be NaN-masked,
    # while real Home outfielder stays valid!
    px[:, 1], py[:, 1] = 10.0, 34.0
    px[0, 23], py[0, 23] = 100.9, 38.0
    px[1, 23], py[1, 23] = 9.5, 34.0
    px_cl, py_cl, _, _ = clean_atd_trajectories(
        px, py, team, gk_slots=(23, 45), half_split_frame=2, pitch_length=105.3
    )
    assert np.isnan(px_cl[0, 23]), "Wrong-end Home GK in H1 must be masked"
    assert np.isclose(px_cl[1, 23], 9.5), "Valid Home GK in own box must be preserved"

    # Matrix 10 checks:
    # Frame 0 (H1): Home winger knocking ball 3.5m ahead in open space, sprinting toward ball!
    # Frame 1 (H1): Shooting zone (x=88, y=34) with crowded Away cone (4 blockers + close defender) vs
    # Frame 2 (H1): Shooting zone (x=88, y=34) with open lane (1 GK only, no defender <4m) + shot speed
    # Frame 3 (H2, frame >= half_split_frame=3): Exact H2 mirror of Frame 0!
    px = np.full((n_frames, n_slots), np.nan, dtype=np.float64)
    py = np.full((n_frames, n_slots), np.nan, dtype=np.float64)
    vx = np.zeros((n_frames, n_slots), dtype=np.float64)
    vy = np.zeros((n_frames, n_slots), dtype=np.float64)

    bx = np.array([53.5, 88.0, 88.0, 105.3 - 53.5], dtype=np.float64)
    by = np.array([34.0, 34.0, 34.0, 34.0], dtype=np.float64)
    vbx = np.array([8.0, 2.0, 18.0, -8.0], dtype=np.float64)
    vby = np.zeros(n_frames, dtype=np.float64)
    vb = np.hypot(vbx, vby)

    # Frame 0: knock-on carry by Home slot 1
    px[0, 1], py[0, 1], vx[0, 1] = 50.0, 34.0, 6.5
    px[0, 2], py[0, 2] = 35.0, 20.0
    px[0, 24], py[0, 24] = 62.0, 34.0

    # Frame 1: crowded shooting zone
    px[1, 1], py[1, 1] = 87.5, 34.0
    for j, ox, oy in [(24, 89.2, 34.0), (25, 92.0, 33.5), (26, 95.0, 34.8), (45, 102.0, 34.0)]:
        px[1, j], py[1, j] = ox, oy

    # Frame 2: open shooting lane with fast shot
    px[2, 1], py[2, 1] = 87.5, 34.0
    px[2, 45], py[2, 45] = 102.0, 34.0
    px[2, 24], py[2, 24] = 82.0, 18.0

    # Frame 3: exact mirror of Frame 0 in H2
    for s in (1, 2, 24):
        px[3, s] = 105.3 - px[0, s]
        py[3, s] = py[0, s]
        vx[3, s] = -vx[0, s]

    m10 = compute_carry_and_shooting_matrices(
        px, py, vx, vy, team, bx, by, bx, by, vbx, vby, vb, bx, by,
        pitch_length=105.3, pitch_width=68.0, half_split_frame=3, gk_slots=(23, 45)
    )
    assert m10["knock_on_h"][0], "Open-space sprint knock-on should set knock_on_h"
    assert m10["tti_margin_ha"][0] > 0.3, "Sprinter chasing knock-on must have positive TTI margin"
    assert m10["shot_crowded_risk_h"][1] > m10["shot_crowded_risk_h"][2], "Crowded cone must elevate risk"
    assert m10["shot_takeon_open_h"][2] > m10["shot_takeon_open_h"][1], "Open shooting lane must raise takeon_open"
    assert m10["shot_flight"][2], "Fast shot inside 28m toward goal should trigger shot_flight"
    assert np.isclose(m10["tti_margin_ha"][0], m10["tti_margin_ha"][3], atol=1e-6)
    assert np.isclose(m10["open_carry_h"][0], m10["open_carry_h"][3], atol=1e-6)

    # Verify opponent presser chasing a long Home pass never triggers false knock_on_a when
    # Home had recent ball contact (t=0..4) and Away never came within <=2.25m!
    T = 12
    px_p = np.full((T, n_slots), np.nan, dtype=np.float64)
    py_p = np.full((T, n_slots), np.nan, dtype=np.float64)
    vx_p = np.zeros((T, n_slots), dtype=np.float64)
    vy_p = np.zeros((T, n_slots), dtype=np.float64)
    bx_p = np.linspace(10.0, 26.0, T)
    by_p = np.full(T, 34.0)
    vbx_p = np.full(T, 8.0)
    vby_p = np.zeros(T)
    vb_p = vbx_p.copy()
    px_p[:5, 1], py_p[:5, 1] = bx_p[:5] - 0.5, 34.0  # Home launcher in [0..4]
    px_p[5:, 1], py_p[5:, 1] = 15.0, 34.0
    px_p[:, 24], py_p[:, 24], vx_p[:, 24] = bx_p - 4.0, 34.0, 6.5  # Away presser sprinting behind pass
    m10_p = compute_carry_and_shooting_matrices(
        px_p, py_p, vx_p, vy_p, team, bx_p, by_p, bx_p, by_p, vbx_p, vby_p, vb_p, bx_p, by_p,
        pitch_length=105.3, pitch_width=68.0, half_split_frame=100, gk_slots=(23, 45)
    )
    assert not np.any(m10_p["knock_on_a"][5:]), "Presser without prior team touch must not fire knock_on_a"


if __name__ == "__main__":
    test_matrix9_pinning_cover_outlets_and_half_symmetry()
    test_dual_mode_fallback_without_atd()
    test_gk_cleaning_carry_and_shooting_matrices()
    print("ALL_TESTS_PASS")
