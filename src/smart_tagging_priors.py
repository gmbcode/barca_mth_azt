"""Stage 2D: Smart Tagging Conjugate Priors, 35s Set-Piece Decomposition & Goal Stoppage Windows."""

import os
import sys
import xml.etree.ElementTree as ET
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, GAMES, asset_paths, source_frames  # noqa: E402

ATTACKING_CODES = {
    "BUILD UP",
    "PROGRESSION",
    "CREATING CHANCES",
    "ATTACKING TRANSITION",
    "BALL IN FINAL THIRD",
    "BALL IN THE BOX",
    "PLAYERS IN THE BOX",
    "LONG BALL",
}

DEFENDING_CODES = {
    "DEFENDING IN ATTACKING THIRD",
    "DEFENDING IN MIDDLE THIRD",
    "DEFENDING IN DEFENSIVE THIRD",
    "DEFENSIVE TRANSITION",
}


def _team_to_code(team_str):
    """Map Smart Tagging Team string ('Barcelona', 'Atletic Club', 'N/A') -> 0 (H), 1 (A), or -1."""
    if not team_str:
        return -1
    t = team_str.strip().lower()
    if "barc" in t:
        return 0
    if "atlet" in t or "athlet" in t:
        return 1
    return -1


def compute_smart_tagging_priors(stage1_data, game=GAME, data_dir=None):
    """Parse BAR-ATH_tactical_smart_tagging.xml and compute conjugate priors & set-piece masks."""
    n_frames = stage1_data["n_frames"]
    fps = stage1_data["fps"]
    paths = asset_paths(game, data_dir, source="tactical")
    st_path = paths.get("smart_tagging", "")

    st_votes_h = np.zeros(n_frames, dtype=np.float64)
    st_votes_a = np.zeros(n_frames, dtype=np.float64)
    set_piece_pre_kick_d = np.zeros(n_frames, dtype=bool)
    set_piece_restart_team = np.full(n_frames, -1, dtype=np.int8)
    goal_dead_mask = np.zeros(n_frames, dtype=bool)
    kickoff_frames = []

    if not st_path or not os.path.isfile(st_path):
        return {
            "available": False,
            "st_votes_h": st_votes_h,
            "st_votes_a": st_votes_a,
            "st_net_log_odds": np.zeros(n_frames, dtype=np.float64),
            "set_piece_pre_kick_d": set_piece_pre_kick_d,
            "set_piece_restart_team": set_piece_restart_team,
            "goal_dead_mask": goal_dead_mask,
            "kickoff_frames": kickoff_frames,
        }

    v_cam = stage1_data["cam_kalman"]["speed_smooth"]
    atd = stage1_data["atd"]
    pitch = stage1_data["pitch"]
    has_full_atd = bool(atd.get("has_full_atd", False))
    raw_codes = atd["raw_codes"]
    bx = atd["ball_x_m"]
    by = atd["ball_y_m"]
    v_ball = atd["ball_kalman"]["speed_smooth"]
    pr_tot = (
        atd.get("press_cap_on_h", np.zeros(n_frames, dtype=np.float64))
        + atd.get("press_cap_on_a", np.zeros(n_frames, dtype=np.float64))
    )
    ms_all = atd["mean_speed_all"]
    inbounds = (
        (raw_codes < 2)
        & np.isfinite(bx)
        & np.isfinite(by)
        & (bx >= -0.3)
        & (bx <= pitch[0] + 0.3)
        & (by >= -0.3)
        & (by <= pitch[1] + 0.3)
    )
    strict_oob = (
        (raw_codes < 2)
        & np.isfinite(bx)
        & np.isfinite(by)
        & ((bx < -0.5) | (bx > pitch[0] + 0.5) | (by < -0.5) | (by > pitch[1] + 0.5))
    )

    events = []
    for instance in ET.parse(st_path).iter("instance"):
        start_s = float(instance.findtext("start"))
        end_s = float(instance.findtext("end"))
        code = (instance.findtext("code") or "").strip().upper()
        tags = {
            (lbl.findtext("group") or "").strip(): (lbl.findtext("text") or "").strip()
            for lbl in instance.findall("label")
        }
        s_f = max(0, min(n_frames, int(round(start_s * fps))))
        e_f = max(0, min(n_frames, int(round(end_s * fps))))
        events.append((code, s_f, e_f, tags))

    for code, s_f, e_f, tags in events:
        if e_f <= s_f:
            continue
        team_code = _team_to_code(tags.get("Team", ""))

        if code in ATTACKING_CODES and team_code in (0, 1):
            if team_code == 0:
                st_votes_h[s_f:e_f] += 1.0
            else:
                st_votes_a[s_f:e_f] += 1.0

        elif code in DEFENDING_CODES and team_code in (0, 1):
            # Conjugate inversion: Defending by Team Y means Opposing Team (1 - Y) has the ball!
            opp_code = 1 - team_code
            if opp_code == 0:
                st_votes_h[s_f:e_f] += 1.0
            else:
                st_votes_a[s_f:e_f] += 1.0

        elif code == "SET PIECES":
            sp_type = tags.get("Type", "").strip().lower()
            if "kick off (start)" in sp_type:
                # Locate exact kickoff strike inside the ~5s window
                mid = (s_f + e_f) // 2
                kickoff_frames.append(mid)
            else:
                # Forward-scan for true ball launch / restart kick inside 35s window [s_f, e_f]
                d_start = s_f
                limit = min(e_f, s_f + 650)
                t_kick = limit
                if has_full_atd:
                    while d_start > 0 and (s_f - d_start) < 350:
                        prev = d_start - 1
                        if (
                            strict_oob[prev]
                            or raw_codes[prev] == 2
                            or (pr_tot[prev] < 0.24 and v_ball[prev] < 2.6 and ms_all[prev] < 1.45)
                        ):
                            d_start -= 1
                        else:
                            break
                    for t in range(s_f, max(s_f, limit - 12)):
                        if np.mean(inbounds[t:t + 12]) >= 0.75 and (
                            np.mean(v_ball[t:t + 12] >= 3.6) >= 0.50
                            or (np.max(v_ball[t:t + 15]) >= 5.5 and np.mean(v_ball[t:t + 12]) >= 3.0)
                        ):
                            t_kick = t
                            break
                else:
                    search_end = min(e_f, s_f + 450)
                    t_kick = s_f
                    for t in range(s_f, max(s_f, search_end - 12)):
                        if np.mean(v_cam[t:t + 12] >= 0.45) >= 0.65:
                            t_kick = t
                            break
                if t_kick > d_start:
                    set_piece_pre_kick_d[d_start:t_kick] = True
                if team_code in (0, 1):
                    post_end = min(n_frames, t_kick + 75)
                    set_piece_restart_team[t_kick:post_end] = team_code
                else:
                    # For Throw-in (Team == N/A), infer from majority raw contact in [t_kick, t_kick+50]
                    post_slice = raw_codes[t_kick:min(n_frames, t_kick + 50)]
                    h_c = np.sum(post_slice == 0)
                    a_c = np.sum(post_slice == 1)
                    if h_c + a_c > 0:
                        set_piece_restart_team[t_kick:min(n_frames, t_kick + 50)] = (
                            0 if h_c >= a_c else 1
                        )

        elif code == "GOALS":
            # Mark ONLY the sustained post-shot dead celebration tail (never live build-up play!)
            if has_full_atd:
                live_act = inbounds & ((v_ball > 2.5) | (pr_tot > 0.35))
                last_live = np.flatnonzero(live_act[s_f:min(e_f, s_f + 900)])
                start_scan = s_f + (int(last_live[-1]) + 10 if len(last_live) else 0)
                if e_f - start_scan >= 300:
                    goal_dead_mask[start_scan:e_f] = True
            else:
                dead_tail = (raw_codes[s_f:e_f] == 2) & (v_cam[s_f:e_f] < 0.8)
                live_pt = np.flatnonzero(~dead_tail[:min(len(dead_tail), 900)])
                start_scan = s_f + (int(live_pt[-1]) + 10 if len(live_pt) else 0)
                if e_f - start_scan >= 300:
                    goal_dead_mask[start_scan:e_f] = True

    st_net_log_odds = np.clip(st_votes_h - st_votes_a, -3.0, 3.0) * 0.65

    return {
        "available": True,
        "st_votes_h": st_votes_h,
        "st_votes_a": st_votes_a,
        "st_net_log_odds": st_net_log_odds,
        "set_piece_pre_kick_d": set_piece_pre_kick_d,
        "set_piece_restart_team": set_piece_restart_team,
        "goal_dead_mask": goal_dead_mask,
        "kickoff_frames": kickoff_frames,
    }
