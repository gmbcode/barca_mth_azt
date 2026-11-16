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
    raw_codes = stage1_data["atd"]["raw_codes"]

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
                # 35s window [s_f, e_f]: find exact restart strike frame t_kick where camera/ball accelerates
                search_end = min(e_f, s_f + 450)
                if search_end > s_f + 25:
                    # Look for transition from low camera/ball speed to active play
                    low_v = v_cam[s_f:search_end] < 0.35
                    if np.any(low_v):
                        last_low = int(np.flatnonzero(low_v)[-1])
                        t_kick = min(search_end, s_f + last_low + 5)
                    else:
                        t_kick = s_f + int(np.argmin(v_cam[s_f:search_end])) + 5
                    set_piece_pre_kick_d[s_f:t_kick] = True
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
            # Mark celebration & walk-back inside GOALS window as D
            goal_dead_mask[s_f:e_f] = True

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
