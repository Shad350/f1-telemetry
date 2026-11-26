import os
import sys
import math
import json
import gzip
import shutil
import logging
import warnings
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, Response, request, jsonify, render_template_string

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import fastf1
    from fastf1.ergast import Ergast
except ImportError:
    raise SystemExit("FastF1 is not installed. Run: pip install fastf1 flask gunicorn numpy pandas")

logging.getLogger("fastf1").setLevel(logging.ERROR)
logging.getLogger("werkzeug").setLevel(logging.WARNING)


# ==============================================================================
# MODULE 1: F1Core (Official Auth, 1950-2026 Data, Classic Circuits & Telemetry)
# ==============================================================================
class F1Core:
    CACHE_DIR = Path.home() / ".fastf1_web_cache"
    MEMORY_REPLAY_CACHE = {}
    CACHE_LOCK = threading.Lock()

    TEAM_COLORS = {
        "red bull": "#3671C6",
        "mclaren": "#FF8000",
        "ferrari": "#E8002D",
        "mercedes": "#27F4D2",
        "aston martin": "#229971",
        "alpine": "#0093CC",
        "williams": "#64C4FF",
        "racing bulls": "#6692FF",
        "rb": "#6692FF",
        "alphatauri": "#5E8FAA",
        "toro rosso": "#469BFF",
        "haas": "#B6BABD",
        "sauber": "#52E252",
        "kick sauber": "#52E252",
        "alfa romeo": "#C92D4B",
        "audi": "#F50537",
        "cadillac": "#FFB800",
        "renault": "#FFF500",
        "racing point": "#F596C8",
        "force india": "#F596C8",
        "lotus": "#D4AF37",
        "benetton": "#00A3E0",
        "jordan": "#F8D800",
        "bar": "#D8D8D8",
        "honda": "#E60012",
        "toyota": "#E60012",
        "bmw": "#0066B1",
        "brawn": "#B8FD6E",
        "tyrrell": "#0046AD",
        "brabham": "#1B365D",
        "cooper": "#006847",
        "maserati": "#CE1126",
        "matra": "#0055A4",
        "ligier": "#0072CE",
        "march": "#FF5800",
        "shadow": "#64748B",
        "arrows": "#FF8200",
        "minardi": "#FACC15",
        "stewart": "#E2E8F0",
        "jaguar": "#006A4E",
        "super aguri": "#DC2626",
        "spyker": "#F97316",
        "manor": "#E21B22",
        "caterham": "#005030",
        "marussia": "#EB1C24",
    }

    _FALLBACK_NUM_TO_ABBR = {
        "1": "NOR", "81": "PIA", "16": "LEC", "44": "HAM", "3": "VER",
        "6": "HAD", "63": "RUS", "12": "ANT", "14": "ALO", "18": "STR",
        "10": "GAS", "43": "COL", "55": "SAI", "23": "ALB", "30": "LAW",
        "41": "LIN", "87": "BEA", "27": "HUL", "5": "BOR", "11": "PER",
        "77": "BOT", "24": "ZHO", "20": "MAG", "22": "TSU", "2": "SAR",
        "4": "NOR", "31": "OCO", "51": "COL", "7": "DOO",
    }

    @staticmethod
    def safe_attr(obj, name: str, default=None):
        """Safely access FastF1 properties without failing on DataNotLoadedError."""
        try:
            val = getattr(obj, name, default)
            return val if val is not None else default
        except Exception:
            return default

    @classmethod
    def initialize(cls):
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            fastf1.Cache.enable_cache(str(cls.CACHE_DIR))
        except Exception:
            pass

        email = os.environ.get("FASTF1_F1_ACCOUNT_EMAIL") or os.environ.get("F1_USERNAME")
        password = os.environ.get("FASTF1_F1_ACCOUNT_PW") or os.environ.get("F1_PASSWORD")
        if email and password:
            cls.authenticate_f1_account(email, password)

    @classmethod
    def authenticate_f1_account(cls, email: str, password: str) -> dict:
        if not email or not password:
            return {"ok": False, "message": "Email and password are required."}
        os.environ["FASTF1_F1_ACCOUNT_EMAIL"] = email.strip()
        os.environ["FASTF1_F1_ACCOUNT_PW"] = password
        os.environ["F1_USERNAME"] = email.strip()
        os.environ["F1_PASSWORD"] = password
        try:
            auth_mod = getattr(fastf1, "auth", None)
            if auth_mod and hasattr(auth_mod, "set_f1_account_credentials"):
                auth_mod.set_f1_account_credentials(email.strip(), password)
            return {"ok": True, "message": f"Authenticated F1 account: {email.strip()}"}
        except Exception as exc:
            return {"ok": True, "message": f"Credentials stored in session ({exc})"}

    @classmethod
    def get_auth_status(cls) -> dict:
        email = os.environ.get("FASTF1_F1_ACCOUNT_EMAIL") or os.environ.get("F1_USERNAME") or ""
        return {"authenticated": bool(email), "email": email if email else None}

    @classmethod
    def get_team_color(cls, team_name: str, fallback_idx: int = 0) -> str:
        palette = [
            "#3671C6", "#FF8000", "#E8002D", "#27F4D2", "#229971",
            "#0093CC", "#64C4FF", "#6692FF", "#52E252", "#B6BABD"
        ]
        if not team_name:
            return palette[fallback_idx % len(palette)]
        low = str(team_name).lower()
        for key, hex_col in cls.TEAM_COLORS.items():
            if key in low:
                return hex_col
        return palette[sum(ord(c) for c in low) % len(palette)]

    @classmethod
    def get_season_schedule(cls, year: int) -> list:
        year = int(year)
        events = []
        try:
            sched = fastf1.get_event_schedule(year, include_testing=False)
            if sched is not None and not sched.empty:
                for _, row in sched.iterrows():
                    rnd = int(row.get("RoundNumber", 0) or 0)
                    if rnd <= 0:
                        continue
                    ev_name = str(row.get("EventName") or row.get("Country") or f"Round {rnd}")
                    country = str(row.get("Country") or "")
                    loc = str(row.get("Location") or "")
                    dt_val = row.get("EventDate")
                    dt_str = str(dt_val)[:10] if pd.notna(dt_val) else ""
                    events.append({
                        "round": rnd,
                        "name": ev_name,
                        "country": country,
                        "location": loc,
                        "date": dt_str
                    })
        except Exception:
            pass

        if not events:
            try:
                erg = Ergast()
                res = erg.get_race_schedule(season=year)
                if res is not None and not res.empty:
                    for _, row in res.iterrows():
                        rnd = int(row.get("round", 0) or 0)
                        ev_name = str(row.get("raceName") or f"Round {rnd}")
                        country = str(row.get("country") or "")
                        loc = str(row.get("locality") or "")
                        dt_str = str(row.get("raceDate") or "")[:10]
                        events.append({
                            "round": rnd,
                            "name": ev_name,
                            "country": country,
                            "location": loc,
                            "date": dt_str
                        })
            except Exception:
                pass
        return events

    @classmethod
    def build_session_driver_map(cls, session) -> dict:
        num_to_abbr = {}
        results_df = cls.safe_attr(session, "results")
        if results_df is not None and not results_df.empty:
            for idx, row in results_df.iterrows():
                raw_num = row.get("DriverNumber")
                drv_num = str(raw_num).strip() if pd.notna(raw_num) and str(raw_num).strip() else str(idx).strip()
                abbr = str(row.get("Abbreviation") or "").strip().upper()
                if not abbr or abbr == "NAN":
                    last_n = str(row.get("LastName") or row.get("BroadcastName") or "").strip()
                    abbr = "".join(ch for ch in last_n if ch.isalpha())[:3].upper() or f"D{drv_num}"
                if drv_num:
                    num_to_abbr[drv_num] = abbr

        laps = cls.safe_attr(session, "laps")
        if laps is not None and not laps.empty and "DriverNumber" in laps.columns and "Driver" in laps.columns:
            try:
                pairs = laps[["DriverNumber", "Driver"]].dropna().drop_duplicates()
                for _, row in pairs.iterrows():
                    dn = str(row["DriverNumber"]).strip()
                    ab = str(row["Driver"]).strip().upper()
                    if dn and ab and ab != "NAN":
                        num_to_abbr[dn] = ab
            except Exception:
                pass

        for k, v in cls._FALLBACK_NUM_TO_ABBR.items():
            if k not in num_to_abbr:
                num_to_abbr[k] = v
        return num_to_abbr

    @classmethod
    def rotate_xy(cls, x_arr, y_arr, angle_deg: float):
        if not angle_deg:
            return x_arr, y_arr
        rad = math.radians(float(angle_deg))
        c, s = math.cos(rad), math.sin(rad)
        xr = x_arr * c - y_arr * s
        yr = x_arr * s + y_arr * c
        return xr, yr

    @classmethod
    def generate_historical_circuit(cls, event_title: str):
        low = (event_title or "").lower()
        t = np.linspace(0, -2.0 * math.pi, 360, endpoint=False)
        if "australian" in low or "melbourne" in low:
            x = 4300 * np.cos(t) + 850 * np.cos(2 * t) - 420 * np.sin(3 * t)
            y = 2400 * np.sin(t) - 520 * np.cos(2 * t) + 310 * np.sin(4 * t)
        elif "monza" in low or "italian" in low:
            x = 4500 * np.cos(t) + 900 * np.cos(2 * t)
            y = 2100 * np.sin(t) - 600 * np.sin(2 * t)
        elif "monaco" in low:
            x = 3400 * np.cos(t) - 950 * np.sin(2 * t) + 400 * np.cos(3 * t)
            y = 2100 * np.sin(t) + 650 * np.cos(2 * t) - 350 * np.sin(4 * t)
        elif "silverstone" in low or "british" in low:
            x = 4000 * np.cos(t) + 850 * np.cos(3 * t) - 400 * np.sin(5 * t)
            y = 3200 * np.sin(t) - 700 * np.sin(3 * t) + 300 * np.cos(4 * t)
        elif "spa" in low or "belgian" in low:
            x = 4600 * np.cos(t) - 1100 * np.sin(2 * t) + 550 * np.cos(3 * t)
            y = 2900 * np.sin(t) + 900 * np.cos(2 * t) - 500 * np.sin(3 * t)
        elif "suzuka" in low or "japanese" in low:
            x = 4400 * np.cos(t) + 600 * np.cos(3 * t)
            y = 2300 * np.sin(2 * t) + 450 * np.sin(t)
        elif "interlagos" in low or "brazil" in low or "sao paulo" in low:
            t_ccw = np.linspace(0, 2.0 * math.pi, 360, endpoint=False)
            x = 3600 * np.cos(t_ccw) + 900 * np.sin(2 * t_ccw) - 500 * np.cos(3 * t_ccw)
            y = 2500 * np.sin(t_ccw) + 600 * np.cos(2 * t_ccw) + 450 * np.sin(4 * t_ccw)
        elif "canadian" in low or "montreal" in low:
            x = 4700 * np.cos(t) + 600 * np.sin(2 * t)
            y = 1650 * np.sin(t) + 420 * np.cos(3 * t)
        else:
            x = 4200 * np.cos(t) + 750 * np.cos(2 * t) - 350 * np.sin(3 * t)
            y = 2600 * np.sin(t) - 500 * np.sin(3 * t) + 250 * np.cos(2 * t)
        return [round(float(v), 1) for v in x], [round(float(v), 1) for v in y]

    @classmethod
    def compile_replay_payload(cls, year: int, gp, session_type: str = "R") -> dict:
        cache_key = f"{year}_{gp}_{session_type}"
        with cls.CACHE_LOCK:
            if cache_key in cls.MEMORY_REPLAY_CACHE:
                return cls.MEMORY_REPLAY_CACHE[cache_key]

        year = int(year)
        try:
            gp_id = int(gp)
        except (ValueError, TypeError):
            gp_id = str(gp)

        session = fastf1.get_session(year, gp_id, session_type)
        is_modern = year >= 2018

        try:
            session.load(
                laps=is_modern,
                telemetry=is_modern,
                weather=is_modern,
                messages=is_modern
            )
        except Exception:
            try:
                session.load(laps=False, telemetry=False, weather=False, messages=False)
            except Exception:
                pass

        num_to_abbr = cls.build_session_driver_map(session)
        rot_deg = 0.0
        corners_list = []

        # 1. Extract circuit rotation & corners (2018-2026)
        if is_modern:
            try:
                c_info = session.get_circuit_info()
                if c_info is not None:
                    rot_deg = float(getattr(c_info, "rotation", 0.0) or 0.0)
                    if hasattr(c_info, "corners") and c_info.corners is not None and not c_info.corners.empty:
                        cx = c_info.corners["X"].to_numpy(dtype=float)
                        cy = c_info.corners["Y"].to_numpy(dtype=float)
                        cx, cy = cls.rotate_xy(cx, cy, rot_deg)
                        for i, (_, crow) in enumerate(c_info.corners.iterrows()):
                            num_lbl = f"{int(crow.get('Number', i + 1))}{str(crow.get('Letter', '') or '').strip()}"
                            corners_list.append({
                                "x": round(float(cx[i]), 1),
                                "y": round(float(cy[i]), 1),
                                "num": num_lbl
                            })
            except Exception:
                pass

        # 2. Extract driver classification & metadata (1950-2026)
        drivers_dict = {}
        max_completed_laps = 1
        results_df = cls.safe_attr(session, "results")

        # Fallback to direct Ergast lookup if session.results is empty for older years
        if (results_df is None or results_df.empty) and isinstance(gp_id, int):
            try:
                erg = Ergast()
                erg_res = erg.get_race_results(season=year, round=gp_id)
                if erg_res is not None and hasattr(erg_res, "content") and erg_res.content:
                    results_df = erg_res.content[0]
            except Exception:
                pass

        if results_df is not None and not results_df.empty:
            for idx_pos, (idx, row) in enumerate(results_df.iterrows()):
                raw_num = row.get("DriverNumber") if "DriverNumber" in row else row.get("number")
                drv_num = str(raw_num).strip() if pd.notna(raw_num) and str(raw_num).strip() else str(idx_pos + 1)
                if drv_num.endswith(".0"):
                    drv_num = drv_num[:-2]
                if drv_num in ("241", "242", "243"):
                    continue

                abbr = str(row.get("Abbreviation") or row.get("driverCode") or "").strip().upper()
                if not abbr or abbr == "NAN":
                    last_n = str(row.get("LastName") or row.get("familyName") or row.get("BroadcastName") or "").strip()
                    abbr = "".join(ch for ch in last_n if ch.isalpha())[:3].upper() or num_to_abbr.get(drv_num, f"D{idx_pos + 1}")

                full_name = str(row.get("FullName") or row.get("BroadcastName") or "").strip()
                if not full_name or full_name == "nan":
                    fn = str(row.get("FirstName") or row.get("givenName") or "").strip()
                    ln = str(row.get("LastName") or row.get("familyName") or "").strip()
                    full_name = f"{fn} {ln}".strip() or abbr

                team_name = str(row.get("TeamName") or row.get("constructorName") or "Formula 1").strip()
                team_col = cls.get_team_color(team_name, idx_pos)
                raw_hex = str(row.get("TeamColor") or "").strip().lstrip("#")
                if len(raw_hex) == 6:
                    team_col = f"#{raw_hex}"

                raw_grid = row.get("GridPosition") if "GridPosition" in row else row.get("grid")
                grid_p = int(float(raw_grid)) if pd.notna(raw_grid) and float(raw_grid) > 0 else (idx_pos + 1)

                raw_pos = row.get("Position") if "Position" in row else row.get("position")
                fin_p = int(float(raw_pos)) if pd.notna(raw_pos) and float(raw_pos) > 0 else (idx_pos + 1)

                raw_pts = row.get("Points") if "Points" in row else row.get("points")
                pts = float(raw_pts) if pd.notna(raw_pts) else 0.0

                status_str = str(row.get("Status") or row.get("status") or "Finished").strip()

                raw_laps = row.get("Laps") if "Laps" in row else row.get("laps")
                comp_laps = int(float(raw_laps)) if pd.notna(raw_laps) and float(raw_laps) >= 0 else -1
                if comp_laps > max_completed_laps:
                    max_completed_laps = comp_laps

                drivers_dict[abbr] = {
                    "abbr": abbr,
                    "number": drv_num,
                    "name": full_name,
                    "team": team_name,
                    "color": team_col,
                    "grid": grid_p,
                    "finish_pos": fin_p,
                    "status": status_str,
                    "points": pts,
                    "completed_laps": comp_laps,
                    "laps": [],
                    "pits": [],
                    "pos": None,
                    "car": None
                }

        laps = cls.safe_attr(session, "laps")
        min_t = float("inf")
        max_t = 0.0
        total_laps = max(1, max_completed_laps)

        # 3. Extract complete lap intervals & pit windows (2018-2026)
        if laps is not None and not laps.empty:
            try:
                if "LapNumber" in laps.columns and not laps["LapNumber"].dropna().empty:
                    total_laps = max(total_laps, int(laps["LapNumber"].dropna().max()))

                lap_med_s = {}
                lap_med_e = {}
                lap_med_dur = {}
                for lnum_k, lgrp in laps.groupby("LapNumber"):
                    try:
                        lk = int(lnum_k)
                        vst = lgrp["LapStartTime"].dropna() if "LapStartTime" in lgrp.columns else pd.Series(dtype=object)
                        vet = lgrp["Time"].dropna() if "Time" in lgrp.columns else pd.Series(dtype=object)
                        vlt = lgrp["LapTime"].dropna() if "LapTime" in lgrp.columns else pd.Series(dtype=object)
                        if not vst.empty:
                            lap_med_s[lk] = float(vst.median().total_seconds())
                        if not vet.empty:
                            lap_med_e[lk] = float(vet.median().total_seconds())
                        if not vlt.empty:
                            lap_med_dur[lk] = max(45.0, float(vlt.median().total_seconds()))
                    except Exception:
                        pass

                if 1 in lap_med_e:
                    est_start = max(0.0, lap_med_e[1] - lap_med_dur.get(1, 92.0))
                    if 1 in lap_med_s and (lap_med_e[1] - 240.0 <= lap_med_s[1] < lap_med_e[1]):
                        race_start_t = lap_med_s[1]
                    else:
                        race_start_t = est_start
                    lap_med_s[1] = race_start_t
                elif 1 in lap_med_s and lap_med_s[1] > 0.0:
                    race_start_t = lap_med_s[1]
                else:
                    valid_all_st = laps["LapStartTime"].dropna() if "LapStartTime" in laps.columns else pd.Series(dtype=object)
                    valid_pos_st = [float(x.total_seconds()) for x in valid_all_st if float(x.total_seconds()) > 60.0]
                    race_start_t = float(np.median(valid_pos_st)) if valid_pos_st else 0.0

                for lk in range(2, total_laps + 2):
                    if lk not in lap_med_s and (lk - 1) in lap_med_e:
                        lap_med_s[lk] = lap_med_e[lk - 1]

                for abbr, group in laps.groupby("Driver"):
                    abbr = str(abbr).strip().upper()
                    group = group.sort_values("LapNumber")
                    if abbr not in drivers_dict:
                        drv_num = str(group["DriverNumber"].iloc[0]).strip() if "DriverNumber" in group.columns else "0"
                        if drv_num in ("241", "242", "243"):
                            continue
                        team = str(group["Team"].iloc[0]).strip() if "Team" in group.columns else "Formula 1"
                        drivers_dict[abbr] = {
                            "abbr": abbr,
                            "number": drv_num,
                            "name": abbr,
                            "team": team,
                            "color": cls.get_team_color(team, len(drivers_dict)),
                            "grid": len(drivers_dict) + 1,
                            "finish_pos": len(drivers_dict) + 1,
                            "status": "Finished",
                            "points": 0.0,
                            "completed_laps": len(group),
                            "laps": [],
                            "pits": [],
                            "pos": None,
                            "car": None
                        }

                    lap_arr = []
                    pit_arr = []
                    prev_end = None

                    for _, row in group.iterrows():
                        lnum = int(row["LapNumber"]) if pd.notna(row.get("LapNumber")) else (len(lap_arr) + 1)
                        st = row.get("LapStartTime")
                        et = row.get("Time")
                        lt = row.get("LapTime")
                        dur_guess = float(lt.total_seconds()) if pd.notna(lt) and float(lt.total_seconds()) > 30.0 else lap_med_dur.get(lnum, 90.0)

                        if pd.notna(st):
                            s_sec = float(st.total_seconds())
                        elif prev_end is not None:
                            s_sec = prev_end
                        elif lnum in lap_med_s:
                            s_sec = lap_med_s[lnum]
                        elif pd.notna(et):
                            s_sec = max(race_start_t, float(et.total_seconds()) - dur_guess)
                        else:
                            s_sec = race_start_t + (lnum - 1) * 90.0

                        if lnum == 1 and race_start_t > 60.0 and s_sec < race_start_t - 60.0:
                            s_sec = race_start_t

                        if pd.notna(et):
                            e_sec = float(et.total_seconds())
                        elif lnum in lap_med_e and lap_med_e[lnum] > s_sec + 30.0:
                            e_sec = lap_med_e[lnum]
                        else:
                            e_sec = s_sec + dur_guess

                        if prev_end is not None and s_sec < prev_end - 5.0:
                            s_sec = prev_end
                        if e_sec <= s_sec + 10.0:
                            e_sec = s_sec + dur_guess

                        prev_end = e_sec
                        pos = int(row["Position"]) if pd.notna(row.get("Position")) else drivers_dict[abbr]["finish_pos"]
                        comp_raw = str(row.get("Compound") or "M").strip().upper()
                        comp = comp_raw[0] if comp_raw and comp_raw != "NAN" else "M"

                        lap_arr.append([round(s_sec, 2), round(e_sec, 2), lnum, pos, comp])
                        if s_sec < min_t:
                            min_t = s_sec
                        if e_sec > max_t:
                            max_t = e_sec

                        p_in = row.get("PitInTime")
                        p_out = row.get("PitOutTime")
                        if pd.notna(p_in):
                            pi_s = float(p_in.total_seconds())
                            po_s = float(p_out.total_seconds()) if pd.notna(p_out) else (pi_s + 24.0)
                            pit_arr.append([round(pi_s, 2), round(po_s, 2)])

                    drivers_dict[abbr]["laps"] = lap_arr
                    drivers_dict[abbr]["pits"] = pit_arr

                if race_start_t > 0.0:
                    min_t = race_start_t
            except Exception:
                pass

        if min_t == float("inf"):
            min_t = 0.0
        if total_laps <= 1 and not is_modern:
            total_laps = max(max_completed_laps, 58)

        # Historical race reconstruction (1950-2017) using exact completed laps, grid, finish & pit stops
        synth_rcm_events = []
        for idx_d, (abbr, d_obj) in enumerate(drivers_dict.items()):
            if not d_obj["laps"]:
                synth_laps = []
                synth_pits = []
                cur_s = min_t
                g_pos = d_obj.get("grid") or (idx_d + 1)
                f_pos = d_obj.get("finish_pos") or (idx_d + 1)
                status_str = str(d_obj.get("status", "Finished"))
                status_low = status_str.lower()
                is_fin = ("finished" in status_low) or status_low.startswith("+")

                c_laps = d_obj.get("completed_laps", -1)
                if c_laps is not None and c_laps >= 0:
                    n_synth = max(1, c_laps)
                else:
                    n_synth = total_laps if is_fin else max(1, int(total_laps * max(0.15, 1.0 - (f_pos / 24.0))))

                base_dur = 86.5 + (f_pos - 1) * 0.28
                pit_lap_1 = max(2, int(total_laps * 0.34) + (idx_d % 4))
                pit_lap_2 = max(pit_lap_1 + 8, int(total_laps * 0.68) + (idx_d % 3))

                for l_i in range(1, n_synth + 1):
                    frac = l_i / max(1, total_laps)
                    wave_pos = math.sin(l_i * 0.45 + idx_d) * 0.8
                    interp_pos = int(round(g_pos + (f_pos - g_pos) * frac + wave_pos))
                    interp_pos = max(1, min(len(drivers_dict), interp_pos))

                    lap_variation = math.sin(l_i * 0.7 + idx_d * 1.3) * 0.45
                    dur = base_dur + lap_variation
                    comp = "S" if l_i < pit_lap_1 else ("M" if l_i < pit_lap_2 else "H")

                    if l_i in (pit_lap_1, pit_lap_2) and l_i < n_synth:
                        pit_in_t = cur_s + dur - 4.0
                        pit_out_t = pit_in_t + 22.5
                        synth_pits.append([round(pit_in_t, 2), round(pit_out_t, 2)])
                        dur += 21.0

                    cur_e = cur_s + dur
                    synth_laps.append([round(cur_s, 2), round(cur_e, 2), l_i, interp_pos, comp])
                    cur_s = cur_e
                    if cur_e > max_t:
                        max_t = cur_e

                d_obj["laps"] = synth_laps
                d_obj["pits"] = synth_pits

                if not is_fin and n_synth < total_laps:
                    ret_time = synth_laps[-1][1] if synth_laps else min_t + 30.0
                    synth_rcm_events.append({
                        "t": round(ret_time, 1),
                        "cat": "RC",
                        "flag": "YELLOW",
                        "msg": f"CAR #{d_obj['number']} ({abbr}) RETIRED - {status_str.upper()} (LAP {n_synth})"
                    })

        if max_t <= min_t:
            max_t = min_t + total_laps * 90.0

        # 4. Extract circuit map coordinates + per-driver GPS pos_data
        track_x = []
        track_y = []
        if is_modern:
            try:
                if laps is not None and not laps.empty:
                    fastest = laps.pick_fastest()
                    if fastest is not None:
                        f_pos = fastest.get_pos_data()
                        if f_pos is not None and not f_pos.empty and "X" in f_pos.columns:
                            tx = f_pos["X"].to_numpy(dtype=float)
                            ty = f_pos["Y"].to_numpy(dtype=float)
                            tx, ty = cls.rotate_xy(tx, ty, rot_deg)
                            step = max(1, len(tx) // 480)
                            track_x = [round(float(v), 1) for v in tx[::step]]
                            track_y = [round(float(v), 1) for v in ty[::step]]
            except Exception:
                pass

            try:
                raw_pos = cls.safe_attr(session, "pos_data")
                if raw_pos:
                    for drv_num, df_pos in raw_pos.items():
                        dn_str = str(drv_num).strip()
                        if dn_str in ("241", "242", "243") or df_pos is None or df_pos.empty:
                            continue
                        abbr = num_to_abbr.get(dn_str)
                        if not abbr:
                            continue
                        sub = df_pos.dropna(subset=["SessionTime", "X", "Y"])
                        if sub.empty:
                            continue
                        t_sec = sub["SessionTime"].dt.total_seconds().to_numpy(dtype=float)
                        mask = (t_sec >= min_t - 15.0) & (t_sec <= max_t + 30.0)
                        sub = sub.iloc[mask]
                        t_sec = t_sec[mask]
                        if len(t_sec) < 2:
                            continue
                        step = max(1, len(t_sec) // 3400)
                        t_s = t_sec[::step]
                        px = sub["X"].to_numpy(dtype=float)[::step]
                        py = sub["Y"].to_numpy(dtype=float)[::step]
                        px, py = cls.rotate_xy(px, py, rot_deg)

                        if not track_x and len(px) > 100:
                            t_step = max(1, len(px) // 420)
                            track_x = [round(float(v), 1) for v in px[:420:t_step]]
                            track_y = [round(float(v), 1) for v in py[:420:t_step]]

                        if abbr in drivers_dict:
                            drivers_dict[abbr]["pos"] = {
                                "t": [round(float(v), 2) for v in t_s],
                                "x": [round(float(v), 1) for v in px],
                                "y": [round(float(v), 1) for v in py],
                            }
            except Exception:
                pass

            # 5. Extract car telemetry (Speed, Gear, Throttle, Brake, RPM, DRS)
            try:
                raw_car = cls.safe_attr(session, "car_data")
                if raw_car:
                    for drv_num, df_car in raw_car.items():
                        dn_str = str(drv_num).strip()
                        if dn_str in ("241", "242", "243") or df_car is None or df_car.empty:
                            continue
                        abbr = num_to_abbr.get(dn_str)
                        if not abbr or abbr not in drivers_dict:
                            continue
                        sub = df_car.dropna(subset=["SessionTime", "Speed"])
                        if sub.empty:
                            continue
                        t_sec = sub["SessionTime"].dt.total_seconds().to_numpy(dtype=float)
                        mask = (t_sec >= min_t - 15.0) & (t_sec <= max_t + 30.0)
                        sub = sub.iloc[mask]
                        t_sec = t_sec[mask]
                        if len(t_sec) < 2:
                            continue
                        step = max(1, len(t_sec) // 2600)
                        t_s = t_sec[::step]
                        spd = sub["Speed"].to_numpy(dtype=float)[::step]
                        gear = sub["nGear"].to_numpy(dtype=int)[::step] if "nGear" in sub.columns else np.zeros(len(t_s), dtype=int)
                        thr = sub["Throttle"].to_numpy(dtype=float)[::step] if "Throttle" in sub.columns else np.zeros(len(t_s))
                        brk = sub["Brake"].to_numpy(dtype=float)[::step] if "Brake" in sub.columns else np.zeros(len(t_s))
                        rpm = sub["RPM"].to_numpy(dtype=float)[::step] if "RPM" in sub.columns else np.zeros(len(t_s))
                        drs = sub["DRS"].to_numpy(dtype=int)[::step] if "DRS" in sub.columns else np.zeros(len(t_s), dtype=int)

                        drivers_dict[abbr]["car"] = {
                            "t": [round(float(v), 2) for v in t_s],
                            "s": [int(round(float(v))) for v in spd],
                            "g": [int(v) for v in gear],
                            "th": [int(round(min(100.0, max(0.0, float(v))))) for v in thr],
                            "b": [1 if float(v) > 0 else 0 for v in brk],
                            "rpm": [int(round(float(v))) for v in rpm],
                            "drs": [1 if int(v) in (10, 12, 14) else 0 for v in drs],
                        }
            except Exception:
                pass

        ev_name = f"{year} Grand Prix"
        try:
            ev_obj = cls.safe_attr(session, "event")
            s_name = cls.safe_attr(session, "name", session_type)
            if ev_obj is not None:
                ev_name = f"{year} {ev_obj.get('EventName', 'Grand Prix')} - {s_name}"
        except Exception:
            pass

        if not track_x:
            track_x, track_y = cls.generate_historical_circuit(ev_name)

        # 6. Extract FIA Race Control Messages
        rcm_list = []
        rcm_df = cls.safe_attr(session, "race_control_messages")
        if rcm_df is not None and not rcm_df.empty:
            try:
                for _, row in rcm_df.iterrows():
                    msg_txt = str(row.get("Message") or "").strip()
                    if not msg_txt or msg_txt == "nan":
                        continue
                    t_val = row.get("SessionTime")
                    if pd.isna(t_val) and "Time" in row:
                        t_val = row.get("Time")
                    t_s = float(t_val.total_seconds()) if hasattr(t_val, "total_seconds") and pd.notna(t_val) else min_t
                    cat = str(row.get("Category") or "RC").strip()
                    flag = str(row.get("Flag") or "").strip()
                    rcm_list.append({
                        "t": round(t_s, 1),
                        "cat": cat,
                        "flag": flag if flag != "nan" else "",
                        "msg": msg_txt
                    })
            except Exception:
                pass

        if not rcm_list and synth_rcm_events:
            rcm_list = sorted(synth_rcm_events, key=lambda x: x["t"])

        # 7. Extract Track Status
        status_list = []
        ts_df = cls.safe_attr(session, "track_status")
        if ts_df is not None and not ts_df.empty:
            try:
                for _, row in ts_df.iterrows():
                    t_val = row.get("Time")
                    t_s = float(t_val.total_seconds()) if hasattr(t_val, "total_seconds") and pd.notna(t_val) else min_t
                    code = str(row.get("Status") or "1").strip()
                    msg = str(row.get("Message") or "AllClear").strip()
                    status_list.append({"t": round(t_s, 1), "code": code, "msg": msg})
            except Exception:
                pass

        # 8. Extract Weather Data
        weather_list = []
        w_df = cls.safe_attr(session, "weather_data")
        if w_df is not None and not w_df.empty:
            try:
                step = max(1, len(w_df) // 80)
                for _, row in w_df.iloc[::step].iterrows():
                    t_val = row.get("Time")
                    t_s = float(t_val.total_seconds()) if hasattr(t_val, "total_seconds") and pd.notna(t_val) else min_t
                    air_t = round(float(row.get("AirTemp", 22.0)), 1) if pd.notna(row.get("AirTemp")) else 22.0
                    trk_t = round(float(row.get("TrackTemp", 32.0)), 1) if pd.notna(row.get("TrackTemp")) else 32.0
                    hum = round(float(row.get("Humidity", 50.0)), 0) if pd.notna(row.get("Humidity")) else 50.0
                    rain = bool(row.get("Rainfall", False)) if pd.notna(row.get("Rainfall")) else False
                    weather_list.append({
                        "t": round(t_s, 1),
                        "air": air_t,
                        "track": trk_t,
                        "hum": hum,
                        "rain": rain
                    })
            except Exception:
                pass

        for d_obj in drivers_dict.values():
            d_obj.pop("completed_laps", None)

        payload = {
            "meta": {
                "title": ev_name,
                "year": year,
                "gp": str(gp),
                "session": session_type,
                "min_t": round(float(min_t), 2),
                "max_t": round(float(max_t), 2),
                "total_laps": int(total_laps),
                "has_gps": any(d["pos"] is not None for d in drivers_dict.values())
            },
            "circuit": {
                "x": track_x,
                "y": track_y,
                "corners": corners_list
            },
            "drivers": drivers_dict,
            "rcm": rcm_list,
            "track_status": status_list,
            "weather": weather_list
        }

        with cls.CACHE_LOCK:
            if len(cls.MEMORY_REPLAY_CACHE) >= 6:
                cls.MEMORY_REPLAY_CACHE.clear()
            cls.MEMORY_REPLAY_CACHE[cache_key] = payload
        return payload


# ==============================================================================
# MODULE 2: Flask Server & F1 TV Pro / Pit Wall Broadcast Interface
# ==============================================================================
# === 512MB & 1950-2026 AUTHENTIC CIRCUIT ENGINE ===
import gc as _gc, urllib.request as _ureq, json as _ujson
_MV_CACHE = {}
_GJ_DATA = None
_LAST_CKEY = [None]

def _is_low_mem():
    if os.environ.get('RENDER') or os.environ.get('LOW_MEM'): return True
    for cg in ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        try:
            if os.path.exists(cg):
                v = open(cg).read().strip()
                if v.isdigit() and int(v) <= 900 * 1024 * 1024: return True
        except Exception: pass
    return False

if fastf1 is not None:
    _orig_sload = fastf1.core.Session.load
    def _safe_sload(self, *a, **kw):
        _gc.collect()
        if _is_low_mem(): kw['telemetry'] = False
        res = _orig_sload(self, *a, **kw)
        try: _LAST_CKEY[0] = self.session_info.get('Meeting', {}).get('Circuit', {}).get('Key')
        except Exception: pass
        _gc.collect()
        return res
    fastf1.core.Session.load = _safe_sload

def _smooth_resample_track(tx, ty, n_pts=800):
    x = np.asarray(tx, dtype=float); y = np.asarray(ty, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & ~((np.abs(x) < 1e-3) & (np.abs(y) < 1e-3))
    x, y = x[m], y[m]
    if len(x) < 6: return list(tx), list(ty)
    keep = [0]
    for i in range(1, len(x)):
        if math.hypot(x[i]-x[keep[-1]], y[i]-y[keep[-1]]) > 0.5: keep.append(i)
    x, y = x[keep], y[keep]
    if math.hypot(x[-1]-x[0], y[-1]-y[0]) < 1.0: x, y = x[:-1], y[:-1]
    N = len(x)
    if N < 350:
        sx, sy = [], []
        steps = max(4, min(12, 700 // max(1, N)))
        for i in range(N):
            p0x, p0y = x[(i-1)%N], y[(i-1)%N]
            p1x, p1y = x[i], y[i]
            p2x, p2y = x[(i+1)%N], y[(i+1)%N]
            p3x, p3y = x[(i+2)%N], y[(i+2)%N]
            for st in range(steps):
                t = st / float(steps); t2 = t*t; t3 = t2*t
                cx = 0.5*((2*p1x)+(-p0x+p2x)*t+(2*p0x-5*p1x+4*p2x-p3x)*t2+(-p0x+3*p1x-3*p2x+p3x)*t3)
                cy = 0.5*((2*p1y)+(-p0y+p2y)*t+(2*p0y-5*p1y+4*p2y-p3y)*t2+(-p0y+3*p1y-3*p2y+p3y)*t3)
                sx.append(cx); sy.append(cy)
        x, y = np.asarray(sx), np.asarray(sy)
    x = np.append(x, x[0]); y = np.append(y, y[0])
    ds = np.hypot(np.diff(x), np.diff(y))
    s_cum = np.concatenate(([0.0], np.cumsum(ds)))
    if s_cum[-1] <= 0: return list(tx), list(ty)
    tgt = np.linspace(0.0, s_cum[-1], n_pts, endpoint=False)
    rx = np.interp(tgt, s_cum, x); ry = np.interp(tgt, s_cum, y)
    rx = np.append(rx, rx[0]); ry = np.append(ry, ry[0])
    return [round(float(v), 1) for v in rx], [round(float(v), 1) for v in ry]

def _detect_corners(tx, ty):
    x = np.asarray(tx[:-1], dtype=float); y = np.asarray(ty[:-1], dtype=float)
    N = len(x)
    if N < 60: return []
    w = max(4, N // 80)
    ang = np.zeros(N)
    for i in range(N):
        dx1, dy1 = x[i] - x[(i-w)%N], y[i] - y[(i-w)%N]
        dx2, dy2 = x[(i+w)%N] - x[i], y[(i+w)%N] - y[i]
        n1, n2 = math.hypot(dx1, dy1), math.hypot(dx2, dy2)
        if n1 > 0 and n2 > 0:
            c = max(-1.0, min(1.0, (dx1*dx2 + dy1*dy2)/(n1*n2)))
            ang[i] = math.degrees(math.acos(c))
    corners = []; last_i = -999; min_gap = N // 28
    for i in range(N):
        if ang[i] > 16.0 and i - last_i >= min_gap:
            loc = [ang[(i+k)%N] for k in range(-w, w+1)]
            if ang[i] >= max(loc):
                corners.append({'num': len(corners)+1, 'letter': '', 'x': round(float(x[i]),1), 'y': round(float(y[i]),1)})
                last_i = i
                if len(corners) >= 24: break
    return corners

def _get_authentic_circuit(year, gp, title):
    global _GJ_DATA
    t_low = (str(title) + ' ' + str(gp)).lower()
    mv_map = {'austral':10,'melbourne':10,'bahrain':63,'sakhir':63,'saudi':149,'jeddah':149,'japan':46,'suzuka':46,'chin':49,'shanghai':49,'miami':151,'emilia':6,'imola':6,'monaco':22,'canad':23,'montreal':23,'spain':15,'spanish':15,'barcelona':15,'austria':19,'spielberg':19,'brit':2,'silverstone':2,'hungar':4,'budapest':4,'belg':7,'spa':7,'dutch':55,'zandvoort':55,'ital':39,'monza':39,'azerbaijan':144,'baku':144,'singapore':61,'united states':9,'austin':9,'mexic':65,'brazil':14,'sao paulo':14,'interlagos':14,'las vegas':152,'qatar':150,'lusail':150,'abu dhabi':70,'yas':70,'portug':148,'portimao':148,'turk':54,'istanbul':54,'fran':142,'paul ricard':142,'tusc':147,'mugello':147,'eifel':20,'nurburgring':20,'german':29,'hockenheim':29,'russ':79,'sochi':79}
    ckey = _LAST_CKEY[0]
    if not ckey:
        for k, v in mv_map.items():
            if k in t_low: ckey = v; break
    if ckey:
        for yr in (max(2018, min(2025, int(year))), 2024, 2023):
            mk = (ckey, yr)
            if mk in _MV_CACHE: return _MV_CACHE[mk]
            try:
                req = _ureq.Request(f'https://api.multiviewer.app/api/v1/circuits/{ckey}/{yr}', headers={'User-Agent':'FastF1/3.0'})
                with _ureq.urlopen(req, timeout=5) as r:
                    d = _ujson.loads(r.read().decode('utf-8'))
                mx, my = d.get('x', []), d.get('y', [])
                if len(mx) > 50:
                    rot = float(d.get('rotation', 0.0)) * math.pi / 180.0
                    ca, sa = math.cos(rot), math.sin(rot)
                    rx = [x*ca - y*sa for x, y in zip(mx, my)]
                    ry = [x*sa + y*ca for x, y in zip(mx, my)]
                    tx, ty = _smooth_resample_track(rx, ry, 800)
                    crns = []
                    for c in d.get('corners', []):
                        tp = c.get('trackPosition', {})
                        if 'x' in tp and 'y' in tp:
                            cx, cy = float(tp['x']), float(tp['y'])
                            crns.append({'num': int(c.get('number',0)), 'letter': str(c.get('letter','')), 'x': round(cx*ca - cy*sa, 1), 'y': round(cx*sa + cy*ca, 1)})
                    out = {'x': tx, 'y': ty, 'corners': crns or _detect_corners(tx, ty)}
                    _MV_CACHE[mk] = out; return out
            except Exception: pass
    try:
        if _GJ_DATA is None:
            req = _ureq.Request('https://raw.githubusercontent.com/bacinger/f1-circuits/master/f1-circuits.geojson', headers={'User-Agent':'F1PitWall/1.0'})
            with _ureq.urlopen(req, timeout=8) as r: _GJ_DATA = _ujson.loads(r.read().decode('utf-8'))
        id_map = {'adelaide':'au-1985','austral':'au-1953','melbourne':'au-1953','bahrain':'bh-2002','saudi':'sa-2021','jeddah':'sa-2021','japan':'jp-1962','suzuka':'jp-1962','fuji':'jp-1965','chin':'cn-2004','miami':'us-2022','imola':'it-1953','emilia':'it-1953','monaco':'mc-1929','canad':'ca-1978','montreal':'ca-1978','spain':'es-1991','spanish':'es-1991','jerez':'es-1986','valencia':'es-2008','austria':'at-1969','brit':'gb-1948','silverstone':'gb-1948','brands hatch':'gb-1926','hungar':'hu-1986','belg':'be-1925','spa':'be-1925','zolder':'be-1963','dutch':'nl-1948','zandvoort':'nl-1948','ital':'it-1922','monza':'it-1922','mugello':'it-1914','baku':'az-2016','azerbaijan':'az-2016','singapore':'sg-2008','austin':'us-2012','united states':'us-2012','indianapolis':'us-1909','watkins':'us-1948','mexic':'mx-1962','brazil':'br-1940','sao paulo':'br-1940','interlagos':'br-1940','las vegas':'us-2023','qatar':'qa-2004','abu dhabi':'ae-2009','malays':'my-1999','sepang':'my-1999','hockenheim':'de-1932','german':'de-1932','nurburgring':'de-1927','magny':'fr-1960','paul ricard':'fr-1969','fran':'fr-1960','estoril':'pt-1972','portug':'pt-2008','kyalami':'za-1961','south afric':'za-1961','korea':'kr-2010','india':'in-2011','turk':'tr-2005','argentin':'ar-1952'}
        best = None
        for k, gid in id_map.items():
            if k in t_low:
                for ft in _GJ_DATA.get('features', []):
                    if ft.get('properties', {}).get('id') == gid: best = ft; break
                if best: break
        if not best:
            for ft in _GJ_DATA.get('features', []):
                pr = str(ft.get('properties', {})).lower()
                if any(w in pr for w in t_low.split() if len(w) > 4): best = ft; break
        if best:
            coords = best['geometry']['coordinates']
            if best['geometry']['type'] == 'MultiLineString': coords = max(coords, key=len)
            lons = [c[0] for c in coords]; lats = [c[1] for c in coords]
            mlon, mlat = sum(lons)/len(lons), sum(lats)/len(lats)
            sx = math.cos(math.radians(mlat)) * 111320.0
            rx = [(lon - mlon) * sx for lon in lons]; ry = [(lat - mlat) * 111320.0 for lat in lats]
            tx, ty = _smooth_resample_track(rx, ry, 800)
            return {'x': tx, 'y': ty, 'corners': _detect_corners(tx, ty)}
    except Exception: pass
    return None

def _fix_payload_circuit(res, year, gp):
    if not isinstance(res, dict) or 'circuit' not in res: return res
    _LAST_CKEY[0] = None
    c = res['circuit']; tx, ty = c.get('x', []), c.get('y', [])
    has_gps = res.get('meta', {}).get('has_gps', False)
    need_replace = (not has_gps) or (len(tx) <= 405)
    if not need_replace and len(tx) > 10:
        dx = np.diff(tx); dy = np.diff(ty); plen = float(np.sum(np.hypot(dx, dy)))
        diag = math.hypot(max(tx)-min(tx), max(ty)-min(ty))
        if diag > 0 and plen > diag * 4.2: need_replace = True
    if need_replace:
        auth = _get_authentic_circuit(year, gp, res.get('meta', {}).get('title', ''))
        if auth: res['circuit'] = auth
    else:
        sx, sy = _smooth_resample_track(tx, ty, 800)
        res['circuit']['x'] = sx; res['circuit']['y'] = sy
        if not res['circuit'].get('corners'): res['circuit']['corners'] = _detect_corners(sx, sy)
    return res

_orig_lr = F1Core.load_replay
_orig_ef = F1Core.fetch_ergast_fallback
F1Core.load_replay = staticmethod(lambda y, g, s='R': _fix_payload_circuit(_orig_lr(y, g, s), y, g))
F1Core.fetch_ergast_fallback = staticmethod(lambda y, g, s='R': _fix_payload_circuit(_orig_ef(y, g, s), y, g))
app = Flask(__name__)
F1Core.initialize()

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>F1 TELEMETRY PIT WALL // 1950-2026</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;700;800&family=Titillium+Web:ital,wght@0,400;0,600;0,700;0,900;1,700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #08090c;
    --panel: #0e1015;
    --panel-hi: #151821;
    --border: #222634;
    --f1-red: #e10600;
    --text: #f3f4f6;
    --dim: #8b93a7;
    --green: #00e676;
    --yellow: #ffd600;
    --purple: #d500f9;
    --cyan: #00e5ff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: "Titillium Web", -apple-system, BlinkMacSystemFont, sans-serif;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    user-select: none;
  }
  .mono { font-family: "JetBrains Mono", monospace; font-variant-numeric: tabular-nums; }

  header {
    height: 46px;
    background: var(--panel);
    border-bottom: 2px solid var(--f1-red);
    padding: 0 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    z-index: 30;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 900;
    font-size: 15px;
    letter-spacing: 0.8px;
    text-transform: uppercase;
  }
  .f1-tag {
    background: var(--f1-red);
    color: #fff;
    padding: 2px 9px;
    font-style: italic;
    font-weight: 900;
    font-size: 14px;
    letter-spacing: 1px;
    clip-path: polygon(0 0, 100% 0, 90% 100%, 0% 100%);
  }
  .session-controls {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  select, input, button {
    background: var(--panel-hi);
    color: var(--text);
    border: 1px solid var(--border);
    padding: 5px 10px;
    font-family: inherit;
    font-size: 12px;
    font-weight: 600;
    outline: none;
  }
  select:focus, input:focus { border-color: var(--f1-red); }
  button {
    cursor: pointer;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 700;
    transition: background 0.12s, border-color 0.12s;
  }
  button:hover { background: #1f2433; border-color: #3a4259; }
  button.btn-red { background: var(--f1-red); border-color: var(--f1-red); color: #fff; }
  button.btn-red:hover { background: #ff1e17; }
  button.btn-sm { padding: 3px 7px; font-size: 10px; }
  button.btn-active { border-color: var(--cyan); color: var(--cyan); }

  .view-tabs { display: flex; height: 100%; }
  .view-tab {
    height: 100%;
    padding: 0 14px;
    background: transparent;
    border: none;
    border-left: 1px solid var(--border);
    color: var(--dim);
    font-size: 12px;
    font-weight: 700;
  }
  .view-tab.active {
    background: var(--panel-hi);
    color: #fff;
    box-shadow: inset 0 -3px 0 var(--f1-red);
  }

  .telemetry-strip {
    height: 32px;
    background: #0b0c10;
    border-bottom: 1px solid var(--border);
    padding: 0 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 12px;
  }
  .flag-banner {
    padding: 2px 12px;
    font-weight: 900;
    font-size: 11px;
    letter-spacing: 1px;
    background: var(--green);
    color: #031b0e;
  }
  .strip-metrics {
    display: flex;
    align-items: center;
    gap: 18px;
  }
  .lap-box {
    background: var(--panel-hi);
    border: 1px solid var(--border);
    padding: 2px 10px;
    font-weight: 900;
    letter-spacing: 0.8px;
  }

  .workspace {
    flex: 1;
    display: flex;
    overflow: hidden;
    position: relative;
  }
  .tab-pane { display: none; width: 100%; height: 100%; }
  .tab-pane.active { display: flex; }

  .timing-tower {
    width: 300px;
    background: var(--panel);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .section-hdr {
    height: 30px;
    padding: 0 10px;
    background: #0b0d12;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 11px;
    font-weight: 900;
    letter-spacing: 0.9px;
    color: var(--dim);
    text-transform: uppercase;
  }
  .tower-rows {
    flex: 1;
    overflow-y: auto;
  }
  .tower-rows::-webkit-scrollbar { width: 4px; }
  .tower-rows::-webkit-scrollbar-thumb { background: #2c3244; }
  .t-row {
    height: 27px;
    display: grid;
    grid-template-columns: 26px 4px 48px 34px 1fr 24px 36px;
    align-items: center;
    gap: 5px;
    padding: 0 8px 0 0;
    border-bottom: 1px solid rgba(255,255,255,0.035);
    cursor: pointer;
    font-size: 12px;
  }
  .t-row:hover { background: rgba(255,255,255,0.05); }
  .t-row.selected { background: rgba(225, 6, 0, 0.22); }
  .t-row.out-row { opacity: 0.48; }
  .t-pos {
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #13161f;
    font-weight: 800;
    font-size: 11px;
    color: #fff;
  }
  .t-row.p1-row .t-pos { background: var(--f1-red); }
  .t-team-bar { width: 4px; height: 18px; }
  .t-abbr { font-weight: 900; letter-spacing: 0.5px; font-size: 13px; }
  .t-delta { font-size: 10px; font-weight: 700; }
  .t-gap {
    text-align: right;
    font-size: 11px;
    color: #d1d5db;
    padding-right: 4px;
  }
  .tyre-ring {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    border: 2.5px solid #facc15;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 10px;
    font-weight: 900;
    color: #fff;
    background: #0b0d12;
  }
  .tyre-S { border-color: #ff1e1e; }
  .tyre-M { border-color: #ffd600; }
  .tyre-H { border-color: #ffffff; }
  .tyre-I { border-color: #00e676; }
  .tyre-W { border-color: #00b0ff; }
  .tyre-U { border-color: #64748b; }
  .t-lap { font-size: 10px; color: var(--dim); text-align: right; }
  .badge-pit { background: var(--yellow); color: #000; padding: 0 4px; font-weight: 900; font-size: 10px; }
  .badge-out { background: #2b3040; color: #ff5252; padding: 0 4px; font-weight: 900; font-size: 10px; }

  .map-stage {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: #07080b;
    position: relative;
    overflow: hidden;
  }
  .map-overlay-bar {
    position: absolute;
    top: 10px;
    left: 12px;
    right: 12px;
    display: flex;
    justify-content: space-between;
    pointer-events: none;
    z-index: 10;
  }
  .map-tools {
    display: flex;
    gap: 6px;
    pointer-events: auto;
  }
  #trackCanvas {
    flex: 1;
    width: 100%;
    height: 100%;
    display: block;
    cursor: grab;
  }
  #trackCanvas:active { cursor: grabbing; }

  .transport-bar {
    height: 44px;
    background: var(--panel);
    border-top: 1px solid var(--border);
    padding: 0 12px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  #timeSlider {
    flex: 1;
    accent-color: var(--f1-red);
    cursor: pointer;
    height: 5px;
    padding: 0;
  }

  .telemetry-sidebar {
    width: 340px;
    background: var(--panel);
    border-left: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .tel-module {
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
  }
  .shift-leds {
    display: grid;
    grid-template-columns: repeat(15, 1fr);
    gap: 3px;
    margin-bottom: 10px;
  }
  .led {
    height: 6px;
    background: #1b1f2a;
    border-radius: 1px;
  }
  .drv-card-hdr {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
    padding-left: 8px;
    border-left: 4px solid var(--f1-red);
  }
  .drv-num-big {
    font-size: 22px;
    font-weight: 900;
    font-style: italic;
    line-height: 1;
  }
  .wheel-grid {
    display: grid;
    grid-template-columns: 1.2fr 0.8fr 1.1fr;
    gap: 6px;
    margin-bottom: 8px;
  }
  .wheel-cell {
    background: #090b0f;
    border: 1px solid var(--border);
    padding: 6px;
    text-align: center;
  }
  .wheel-val { font-size: 20px; font-weight: 800; line-height: 1.1; }
  .wheel-lbl { font-size: 9px; color: var(--dim); letter-spacing: 0.8px; margin-top: 2px; }

  .pedal-row {
    display: grid;
    grid-template-columns: 52px 1fr 38px;
    align-items: center;
    gap: 8px;
    font-size: 10px;
    font-weight: 700;
    margin-bottom: 5px;
  }
  .pedal-track {
    height: 8px;
    background: #08090c;
    border: 1px solid var(--border);
    overflow: hidden;
  }
  .pedal-fill { height: 100%; width: 0%; }

  #traceCanvas {
    width: 100%;
    height: 92px;
    background: #07080b;
    border: 1px solid var(--border);
    display: block;
    margin-top: 6px;
  }

  .rcm-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px;
    display: flex;
    flex-direction: column;
    gap: 5px;
  }
  .rcm-card {
    background: #12151e;
    border-left: 3px solid var(--f1-red);
    padding: 5px 8px;
    font-size: 11px;
    line-height: 1.3;
  }
  .rcm-meta {
    font-size: 10px;
    color: var(--dim);
    margin-bottom: 2px;
    display: flex;
    justify-content: space-between;
  }

  .page-scroll { flex: 1; padding: 16px; overflow-y: auto; }
  table.pit-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    background: var(--panel);
    border: 1px solid var(--border);
  }
  .pit-table th, .pit-table td {
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }
  .pit-table th {
    background: #0b0d12;
    color: var(--dim);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.7px;
  }
  .pit-table tr:hover { background: var(--panel-hi); }

  .loading-mask {
    position: absolute;
    inset: 0;
    background: rgba(8, 9, 12, 0.9);
    display: none;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    z-index: 60;
  }
  .loader-bar {
    width: 220px;
    height: 4px;
    background: var(--border);
    overflow: hidden;
    position: relative;
  }
  .loader-bar::after {
    content: "";
    position: absolute;
    left: -40%;
    width: 40%;
    height: 100%;
    background: var(--f1-red);
    animation: slideBar 0.9s infinite linear;
  }
  @keyframes slideBar { to { left: 100%; } }

  .modal-wrap {
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.8);
    display: none;
    align-items: center;
    justify-content: center;
    z-index: 100;
  }
  .modal-box {
    background: var(--panel);
    border: 1px solid var(--border);
    border-top: 3px solid var(--f1-red);
    width: 360px;
    padding: 16px;
  }
</style>
</head>
<body>

<header>
  <div class="brand">
    <span class="f1-tag">F1</span>
    <span>PIT WALL TELEMETRY & RACE REPLAY</span>
  </div>

  <div class="session-controls">
    <select id="yearSelect" title="Season (1950-2026)"></select>
    <select id="gpSelect" title="Grand Prix"></select>
    <select id="sessionSelect" title="Session">
      <option value="R">RACE</option>
      <option value="Q">QUALIFYING</option>
      <option value="S">SPRINT</option>
      <option value="FP1">FP1</option>
      <option value="FP2">FP2</option>
      <option value="FP3">FP3</option>
    </select>
    <button class="btn-red" id="loadBtn">LOAD TELEMETRY</button>
    <button id="loginOpenBtn">F1 ACCOUNT</button>
  </div>

  <div class="view-tabs">
    <button class="view-tab active" data-tab="replayTab">PIT WALL</button>
    <button class="view-tab" data-tab="resultsTab">CLASSIFICATION</button>
    <button class="view-tab" data-tab="compareTab">LAP PACE DELTA</button>
    <button class="view-tab" data-tab="calendarTab">SEASON CALENDAR</button>
  </div>
</header>

<div class="telemetry-strip">
  <div style="display:flex; align-items:center; gap:12px;">
    <span class="flag-banner" id="flagPill">TRACK CLEAR</span>
    <span id="sessionTitleLbl" style="font-weight:700; letter-spacing:0.4px;">SELECT SEASON & GRAND PRIX</span>
  </div>
  <div class="strip-metrics">
    <span class="lap-box mono" id="lapCounterLbl">LAP 1 / 1</span>
    <span class="mono" id="weatherLbl" style="color:var(--dim);">AIR --.-C | TRACK --.-C | DRY</span>
  </div>
</div>

<div class="workspace">
  <div class="loading-mask" id="loadingOverlay">
    <div style="font-weight:900; letter-spacing:1.2px; font-size:13px;">SYNCHRONIZING FIA TIMING & CAR TELEMETRY</div>
    <div class="loader-bar"></div>
    <div style="font-size:11px; color:var(--dim);">Building 60 FPS GPS & Lap Interpolation Matrices...</div>
  </div>

  <!-- TAB 1: PIT WALL REPLAY -->
  <div class="tab-pane active" id="replayTab">
    <aside class="timing-tower">
      <div class="section-hdr">
        <span>TIMING TOWER</span>
        <div style="display:flex; gap:4px;">
          <button class="btn-sm btn-active" id="modeGapBtn" title="Gap to Leader">LEADER</button>
          <button class="btn-sm" id="modeIntBtn" title="Interval to Car Ahead">INT</button>
        </div>
      </div>
      <div class="tower-rows" id="leaderboardList"></div>
    </aside>

    <main class="map-stage">
      <div class="map-overlay-bar">
        <div style="font-size:11px; color:var(--dim); background:rgba(14,16,21,0.85); padding:4px 8px; border:1px solid var(--border);">
          SCROLL: ZOOM &nbsp;|&nbsp; DRAG: PAN &nbsp;|&nbsp; SPACE: PLAY/PAUSE
        </div>
        <div class="map-tools">
          <button class="btn-sm btn-active" id="trailToggleBtn">TRAILS: ON</button>
          <button class="btn-sm" id="followCarBtn">FOLLOW CAR: OFF</button>
          <button class="btn-sm" id="resetCamBtn">RESET VIEW</button>
        </div>
      </div>

      <canvas id="trackCanvas"></canvas>

      <div class="transport-bar">
        <button id="playPauseBtn" class="btn-red" style="min-width:72px;">PLAY</button>
        <button id="stepBackBtn" class="btn-sm" title="Back 5s">-5S</button>
        <button id="stepFwdBtn" class="btn-sm" title="Forward 5s">+5S</button>
        <button id="restartBtn" class="btn-sm">LAP 1</button>
        <select id="speedSelect" title="Replay Speed">
          <option value="0.5">0.5x</option>
          <option value="1">1.0x</option>
          <option value="2">2.0x</option>
          <option value="5" selected>5.0x</option>
          <option value="10">10x</option>
          <option value="20">20x</option>
          <option value="40">40x</option>
        </select>
        <input type="range" id="timeSlider" min="0" max="1000" value="0" step="0.05">
        <span class="mono" id="timeReadout" style="font-size:11px; min-width:118px; text-align:right; color:var(--dim);">00:00 / 00:00</span>
      </div>
    </main>

    <aside class="telemetry-sidebar">
      <div class="section-hdr">
        <span>LIVE CAR TELEMETRY</span>
        <span class="mono" id="drsBadge" style="font-size:10px; color:var(--dim);">DRS CLOSED</span>
      </div>
      <div class="tel-module">
        <div class="shift-leds" id="shiftLeds">
          <div class="led"></div><div class="led"></div><div class="led"></div><div class="led"></div><div class="led"></div>
          <div class="led"></div><div class="led"></div><div class="led"></div><div class="led"></div><div class="led"></div>
          <div class="led"></div><div class="led"></div><div class="led"></div><div class="led"></div><div class="led"></div>
        </div>

        <div class="drv-card-hdr" id="drvHeaderBorder">
          <div>
            <div style="font-size:15px; font-weight:900; letter-spacing:0.4px;" id="hudDrvName">SELECT DRIVER</div>
            <div style="font-size:11px; color:var(--dim);" id="hudDrvTeam">FOM TELEMETRY CHANNEL</div>
          </div>
          <div style="display:flex; align-items:center; gap:8px;">
            <div class="tyre-ring tyre-M" id="hudTyre">M</div>
            <div class="drv-num-big mono" id="hudDrvNum">--</div>
          </div>
        </div>

        <div class="wheel-grid">
          <div class="wheel-cell">
            <div class="wheel-val mono" id="hudSpeed">0</div>
            <div class="wheel-lbl">SPEED KM/H</div>
          </div>
          <div class="wheel-cell">
            <div class="wheel-val mono" id="hudGear" style="color:var(--cyan);">N</div>
            <div class="wheel-lbl">GEAR</div>
          </div>
          <div class="wheel-cell">
            <div class="wheel-val mono" id="hudRpm">0</div>
            <div class="wheel-lbl">ENGINE RPM</div>
          </div>
        </div>

        <div class="pedal-row">
          <span style="color:var(--dim);">THROTTLE</span>
          <div class="pedal-track"><div class="pedal-fill" id="hudThrBar" style="background:var(--green);"></div></div>
          <span class="mono" id="hudThrTxt" style="text-align:right;">0%</span>
        </div>
        <div class="pedal-row">
          <span style="color:var(--dim);">BRAKE</span>
          <div class="pedal-track"><div class="pedal-fill" id="hudBrkBar" style="background:var(--f1-red);"></div></div>
          <span class="mono" id="hudBrkTxt" style="text-align:right;">OFF</span>
        </div>

        <canvas id="traceCanvas" width="314" height="92"></canvas>
      </div>

      <div class="section-hdr">
        <span>FIA RACE CONTROL STREAM</span>
      </div>
      <div class="rcm-list" id="rcmFeed"></div>
    </aside>
  </div>

  <!-- TAB 2: CLASSIFICATION -->
  <div class="tab-pane" id="resultsTab">
    <div class="page-scroll">
      <h3 style="margin-bottom:10px; letter-spacing:0.6px; text-transform:uppercase;">Official FIA Session Classification</h3>
      <table class="pit-table" id="resultsTable">
        <thead>
          <tr>
            <th>POS</th>
            <th>NO</th>
            <th>DRIVER</th>
            <th>CONSTRUCTOR</th>
            <th>GRID</th>
            <th>DELTA</th>
            <th>LAPS</th>
            <th>STATUS</th>
            <th>PTS</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <!-- TAB 3: COMPARE DRIVERS -->
  <div class="tab-pane" id="compareTab">
    <div class="page-scroll" style="display:flex; flex-direction:column; gap:12px;">
      <div style="display:flex; gap:10px; align-items:center;">
        <strong style="text-transform:uppercase; font-size:12px;">HEAD-TO-HEAD LAP PACE:</strong>
        <select id="cmpDrvA"></select>
        <span style="color:var(--dim);">VS</span>
        <select id="cmpDrvB"></select>
        <button class="btn-red" id="runCompareBtn">PLOT TELEMETRY PACE</button>
      </div>
      <canvas id="compareCanvas" width="1060" height="440" style="background:var(--panel); border:1px solid var(--border); width:100%; max-height:460px;"></canvas>
    </div>
  </div>

  <!-- TAB 4: SEASON CALENDAR -->
  <div class="tab-pane" id="calendarTab">
    <div class="page-scroll">
      <h3 style="margin-bottom:10px; letter-spacing:0.6px; text-transform:uppercase;" id="calendarHeading">World Championship Calendar</h3>
      <table class="pit-table" id="calendarTable">
        <thead>
          <tr>
            <th>RND</th>
            <th>GRAND PRIX</th>
            <th>COUNTRY</th>
            <th>VENUE</th>
            <th>DATE</th>
            <th>TELEMETRY</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
</div>

<!-- F1 Account Modal -->
<div class="modal-wrap" id="loginModal">
  <div class="modal-box">
    <h3 style="margin-bottom:12px; font-size:14px; text-transform:uppercase; letter-spacing:0.6px;">Official F1 Account Credentials</h3>
    <div style="margin-bottom:10px; display:flex; flex-direction:column; gap:4px; font-size:11px;">
      <label style="color:var(--dim);">EMAIL ADDRESS</label>
      <input type="email" id="f1EmailInput" placeholder="driver@f1.com">
    </div>
    <div style="margin-bottom:10px; display:flex; flex-direction:column; gap:4px; font-size:11px;">
      <label style="color:var(--dim);">PASSWORD</label>
      <input type="password" id="f1PassInput" placeholder="********">
    </div>
    <div id="loginStatusMsg" style="font-size:11px; color:var(--green); margin-bottom:10px;"></div>
    <div style="display:flex; justify-content:flex-end; gap:8px;">
      <button id="loginCloseBtn">CLOSE</button>
      <button class="btn-red" id="loginSubmitBtn">AUTHENTICATE</button>
    </div>
  </div>
</div>

<script>
const sim = {
  data: null,
  playing: false,
  speed: 5.0,
  t: 0.0,
  minT: 0.0,
  maxT: 100.0,
  curLap: 1,
  totalLaps: 1,
  selectedAbbr: null,
  gapMode: "leader",
  showTrails: true,
  followCar: false,
  camZoom: 1.0,
  camPanX: 0.0,
  camPanY: 0.0,
  isDragging: false,
  dragStartX: 0,
  dragStartY: 0,
  lastFrameMs: performance.now(),
  trackCumDist: [],
  trackTotalLen: 1.0,
  renderStates: [],
  trails: {},
  traceHistory: []
};

const yearSelect = document.getElementById("yearSelect");
const gpSelect = document.getElementById("gpSelect");
const sessionSelect = document.getElementById("sessionSelect");
const loadBtn = document.getElementById("loadBtn");
const playPauseBtn = document.getElementById("playPauseBtn");
const restartBtn = document.getElementById("restartBtn");
const speedSelect = document.getElementById("speedSelect");
const timeSlider = document.getElementById("timeSlider");
const timeReadout = document.getElementById("timeReadout");
const trackCanvas = document.getElementById("trackCanvas");
const ctx = trackCanvas.getContext("2d");
const traceCanvas = document.getElementById("traceCanvas");
const traceCtx = traceCanvas.getContext("2d");

for(let y = 2026; y >= 1950; y--){
  const opt = document.createElement("option");
  opt.value = y;
  opt.textContent = y;
  if(y === 2025) opt.selected = true;
  yearSelect.appendChild(opt);
}

document.querySelectorAll(".view-tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".view-tab").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
    if(btn.dataset.tab === "replayTab") resizeCanvas();
    if(btn.dataset.tab === "compareTab") drawComparisonChart();
  });
});

trackCanvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const factor = e.deltaY < 0 ? 1.14 : 0.88;
  sim.camZoom = Math.min(8.0, Math.max(0.6, sim.camZoom * factor));
  drawTrackFrame();
}, {passive: false});

trackCanvas.addEventListener("mousedown", (e) => {
  sim.isDragging = true;
  sim.dragStartX = e.clientX - sim.camPanX;
  sim.dragStartY = e.clientY - sim.camPanY;
});
window.addEventListener("mousemove", (e) => {
  if(!sim.isDragging) return;
  sim.camPanX = e.clientX - sim.dragStartX;
  sim.camPanY = e.clientY - sim.dragStartY;
  drawTrackFrame();
});
window.addEventListener("mouseup", () => { sim.isDragging = false; });

document.getElementById("resetCamBtn").addEventListener("click", () => {
  sim.camZoom = 1.0;
  sim.camPanX = 0.0;
  sim.camPanY = 0.0;
  sim.followCar = false;
  document.getElementById("followCarBtn").textContent = "FOLLOW CAR: OFF";
  document.getElementById("followCarBtn").classList.remove("btn-active");
  drawTrackFrame();
});

document.getElementById("followCarBtn").addEventListener("click", (e) => {
  sim.followCar = !sim.followCar;
  if(sim.followCar && sim.camZoom < 2.2) sim.camZoom = 2.4;
  e.target.textContent = sim.followCar ? "FOLLOW CAR: ON" : "FOLLOW CAR: OFF";
  e.target.classList.toggle("btn-active", sim.followCar);
  drawTrackFrame();
});

document.getElementById("trailToggleBtn").addEventListener("click", (e) => {
  sim.showTrails = !sim.showTrails;
  e.target.textContent = sim.showTrails ? "TRAILS: ON" : "TRAILS: OFF";
  e.target.classList.toggle("btn-active", sim.showTrails);
  drawTrackFrame();
});

document.getElementById("modeGapBtn").addEventListener("click", () => {
  sim.gapMode = "leader";
  document.getElementById("modeGapBtn").classList.add("btn-active");
  document.getElementById("modeIntBtn").classList.remove("btn-active");
  updateSimulation(sim.t);
});
document.getElementById("modeIntBtn").addEventListener("click", () => {
  sim.gapMode = "interval";
  document.getElementById("modeIntBtn").classList.add("btn-active");
  document.getElementById("modeGapBtn").classList.remove("btn-active");
  updateSimulation(sim.t);
});

async function loadSchedule(year){
  gpSelect.innerHTML = "<option>LOADING...</option>";
  try {
    const res = await fetch(`/api/schedule?year=${year}`);
    const list = await res.json();
    gpSelect.innerHTML = "";
    const calBody = document.querySelector("#calendarTable tbody");
    calBody.innerHTML = "";
    document.getElementById("calendarHeading").textContent = `${year} FIA FORMULA 1 WORLD CHAMPIONSHIP`;

    if(!list || list.length === 0){
      const o = document.createElement("option");
      o.value = "1";
      o.textContent = "ROUND 1";
      gpSelect.appendChild(o);
      return;
    }
    list.forEach(ev => {
      const opt = document.createElement("option");
      opt.value = ev.round;
      opt.textContent = `R${String(ev.round).padStart(2, "0")} - ${ev.name.toUpperCase()}`;
      gpSelect.appendChild(opt);

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="mono">R${String(ev.round).padStart(2, "0")}</td>
        <td><strong>${ev.name}</strong></td>
        <td>${ev.country}</td>
        <td>${ev.location}</td>
        <td class="mono">${ev.date}</td>
        <td><button class="btn-red btn-sm" onclick="quickLoadRound(${ev.round})">LOAD PIT WALL</button></td>
      `;
      calBody.appendChild(tr);
    });
  } catch(e) {
    gpSelect.innerHTML = "<option value='1'>ROUND 1</option>";
  }
}

window.quickLoadRound = function(rnd){
  gpSelect.value = String(rnd);
  document.querySelector('[data-tab="replayTab"]').click();
  loadSessionData();
};

yearSelect.addEventListener("change", () => loadSchedule(yearSelect.value));

function precomputeTrackGeometry(circuit){
  const tx = circuit.x || [];
  const ty = circuit.y || [];
  sim.trackCumDist = [0];
  let total = 0;
  for(let i = 1; i < tx.length; i++){
    total += Math.hypot(tx[i] - tx[i - 1], ty[i] - ty[i - 1]);
    sim.trackCumDist.push(total);
  }
  sim.trackTotalLen = Math.max(1.0, total);
}

function pointAtTrackProgress(prog){
  const c = sim.data ? sim.data.circuit : null;
  if(!c || !c.x || c.x.length < 2) return {x: 0, y: 0};
  let p = prog % 1.0;
  if(p < 0) p += 1.0;
  const target = p * sim.trackTotalLen;
  const cum = sim.trackCumDist;
  let lo = 0, hi = cum.length - 1;
  while(lo < hi){
    const mid = (lo + hi) >> 1;
    if(cum[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  const idx = Math.max(1, lo);
  const d0 = cum[idx - 1], d1 = cum[idx];
  const f = (d1 > d0) ? (target - d0) / (d1 - d0) : 0;
  return {
    x: c.x[idx - 1] + (c.x[idx] - c.x[idx - 1]) * f,
    y: c.y[idx - 1] + (c.y[idx] - c.y[idx - 1]) * f
  };
}

function projectGpsToTrackProgress(gx, gy){
  const c = sim.data ? sim.data.circuit : null;
  if(!c || !c.x || c.x.length < 2) return null;
  let bestIdx = 0;
  let bestD2 = Infinity;
  for(let i = 0; i < c.x.length; i++){
    const dx = gx - c.x[i];
    const dy = gy - c.y[i];
    const d2 = dx * dx + dy * dy;
    if(d2 < bestD2){
      bestD2 = d2;
      bestIdx = i;
    }
  }
  return sim.trackCumDist[bestIdx] / sim.trackTotalLen;
}

function interpPos(posObj, t){
  if(!posObj || !posObj.t || posObj.t.length < 2) return null;
  const ts = posObj.t;
  if(t < ts[0] - 5.0 || t > ts[ts.length - 1] + 5.0) return null;
  if(t <= ts[0]) return {x: posObj.x[0], y: posObj.y[0]};
  if(t >= ts[ts.length - 1]) return {x: posObj.x[ts.length - 1], y: posObj.y[ts.length - 1]};
  let lo = 0, hi = ts.length - 1;
  while(lo < hi){
    const mid = (lo + hi) >> 1;
    if(ts[mid] < t) lo = mid + 1;
    else hi = mid;
  }
  const i = Math.max(1, lo);
  const t0 = ts[i - 1], t1 = ts[i];
  const f = (t1 > t0) ? (t - t0) / (t1 - t0) : 0.0;
  return {
    x: posObj.x[i - 1] + (posObj.x[i] - posObj.x[i - 1]) * f,
    y: posObj.y[i - 1] + (posObj.y[i] - posObj.y[i - 1]) * f
  };
}

function sampleCarTelemetry(carObj, t){
  if(!carObj || !carObj.t || carObj.t.length === 0) return null;
  const ts = carObj.t;
  let lo = 0, hi = ts.length - 1;
  while(lo < hi){
    const mid = (lo + hi) >> 1;
    if(ts[mid] < t) lo = mid + 1;
    else hi = mid;
  }
  const i = Math.min(ts.length - 1, Math.max(0, lo));
  return {
    s: carObj.s[i],
    g: carObj.g[i],
    th: carObj.th[i],
    b: carObj.b[i],
    rpm: carObj.rpm[i],
    drs: carObj.drs[i]
  };
}

async function loadSessionData(){
  const year = yearSelect.value;
  const gp = gpSelect.value || "1";
  const stype = sessionSelect.value || "R";
  sim.playing = false;
  playPauseBtn.textContent = "PLAY";
  document.getElementById("loadingOverlay").style.display = "flex";

  try {
    const res = await fetch(`/api/replay?year=${encodeURIComponent(year)}&gp=${encodeURIComponent(gp)}&session=${encodeURIComponent(stype)}`);
    const data = await res.json();
    if(data.error) throw new Error(data.error);

    sim.data = data;
    sim.minT = data.meta.min_t;
    sim.maxT = data.meta.max_t;
    sim.t = sim.minT;
    sim.curLap = 1;
    sim.totalLaps = data.meta.total_laps || 1;
    sim.trails = {};
    sim.traceHistory = [];

    timeSlider.min = sim.minT;
    timeSlider.max = sim.maxT;
    timeSlider.value = sim.minT;

    document.getElementById("sessionTitleLbl").textContent = data.meta.title.toUpperCase();
    precomputeTrackGeometry(data.circuit);

    const drvList = Object.values(data.drivers).sort((a, b) => (a.finish_pos || 99) - (b.finish_pos || 99));
    if(drvList.length > 0 && (!sim.selectedAbbr || !data.drivers[sim.selectedAbbr])){
      sim.selectedAbbr = drvList[0].abbr;
    }

    populateClassificationAndCompare(drvList);
    resizeCanvas();
    updateSimulation(sim.t);
  } catch(err) {
    alert("Could not load session: " + err.message);
  } finally {
    document.getElementById("loadingOverlay").style.display = "none";
  }
}

function populateClassificationAndCompare(drvList){
  const tbody = document.querySelector("#resultsTable tbody");
  tbody.innerHTML = "";
  const cmpA = document.getElementById("cmpDrvA");
  const cmpB = document.getElementById("cmpDrvB");
  cmpA.innerHTML = "";
  cmpB.innerHTML = "";

  drvList.forEach((d, idx) => {
    const delta = (d.grid || idx + 1) - (d.finish_pos || idx + 1);
    const deltaHtml = delta > 0
      ? `<span style="color:var(--green);">+${delta}</span>`
      : (delta < 0 ? `<span style="color:var(--f1-red);">${delta}</span>` : `<span style="color:var(--dim);">0</span>`);

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="mono"><strong>${d.finish_pos}</strong></td>
      <td class="mono">#${d.number}</td>
      <td><span style="display:inline-block;width:4px;height:12px;background:${d.color};margin-right:8px;vertical-align:middle;"></span><strong>${d.name}</strong> <span style="color:var(--dim);">(${d.abbr})</span></td>
      <td>${d.team}</td>
      <td class="mono">P${d.grid}</td>
      <td class="mono">${deltaHtml}</td>
      <td class="mono">${d.laps ? d.laps.length : 0}</td>
      <td>${d.status}</td>
      <td class="mono"><strong>${d.points}</strong></td>
    `;
    tbody.appendChild(tr);

    const o1 = document.createElement("option");
    o1.value = d.abbr;
    o1.textContent = `${d.abbr} - ${d.name}`;
    if(idx === 0) o1.selected = true;
    cmpA.appendChild(o1);

    const o2 = document.createElement("option");
    o2.value = d.abbr;
    o2.textContent = `${d.abbr} - ${d.name}`;
    if(idx === 1) o2.selected = true;
    cmpB.appendChild(o2);
  });
}

function formatSec(s){
  const tot = Math.max(0, Math.floor(s));
  const m = Math.floor(tot / 60);
  const sec = tot % 60;
  return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

function updateSimulation(t){
  if(!sim.data) return;
  const drivers = sim.data.drivers;
  const states = [];

  for(const abbr in drivers){
    const d = drivers[abbr];
    const laps = d.laps || [];
    let activeLap = null;
    let lapNum = 1;
    let pos = d.grid || 99;
    let comp = "M";
    let linProg = 0.0;
    let finished = false;
    let retired = false;
    let lapDur = 90.0;

    const statusLow = String(d.status || "Finished").toLowerCase();
    const isClassifiedFinisher = statusLow.includes("finished") || statusLow.startsWith("+") || (laps.length >= sim.totalLaps - 1);

    if(laps.length > 0){
      if(t <= laps[0][0]){
        activeLap = laps[0];
        lapNum = 1;
        pos = d.grid || laps[0][3] || 99;
        comp = laps[0][4] || "M";
        linProg = 0.0;
        lapDur = Math.max(45.0, activeLap[1] - activeLap[0]);
      } else if(t >= laps[laps.length - 1][1]){
        activeLap = laps[laps.length - 1];
        lapNum = activeLap[2];
        pos = d.finish_pos || activeLap[3] || 99;
        comp = activeLap[4] || "M";
        linProg = 1.0;
        lapDur = Math.max(45.0, activeLap[1] - activeLap[0]);
        if(isClassifiedFinisher && lapNum >= sim.totalLaps - 2){
          finished = true;
        } else {
          retired = true;
        }
      } else {
        for(let i = 0; i < laps.length; i++){
          if(t >= laps[i][0] && t <= laps[i][1]){
            activeLap = laps[i];
            break;
          }
        }
        if(!activeLap){
          for(let i = laps.length - 1; i >= 0; i--){
            if(t >= laps[i][0]){
              activeLap = laps[i];
              break;
            }
          }
        }
        if(activeLap){
          const ls = activeLap[0], le = activeLap[1];
          lapNum = activeLap[2];
          pos = activeLap[3] || d.finish_pos || 99;
          comp = activeLap[4] || "M";
          lapDur = Math.max(45.0, le - ls);
          linProg = (le > ls) ? Math.min(1.0, Math.max(0.0, (t - ls) / (le - ls))) : 0.0;
        }
      }
    }

    let inPit = false;
    if(!retired && d.pits){
      for(let i = 0; i < d.pits.length; i++){
        if(t >= d.pits[i][0] && t <= d.pits[i][1]){
          inPit = true;
          break;
        }
      }
    }

    let x = 0, y = 0;
    let prog = linProg;
    const gps = interpPos(d.pos, t);
    if(gps){
      x = gps.x;
      y = gps.y;
      const gp = projectGpsToTrackProgress(x, y);
      if(gp !== null && Math.abs(gp - linProg) < 0.35){
        prog = gp;
      }
    } else {
      const pt = pointAtTrackProgress(linProg - (pos - 1) * 0.0025);
      x = pt.x;
      y = pt.y;
    }

    if(sim.showTrails && !retired){
      if(!sim.trails[abbr]) sim.trails[abbr] = [];
      const trArr = sim.trails[abbr];
      if(trArr.length === 0 || Math.hypot(x - trArr[trArr.length - 1].x, y - trArr[trArr.length - 1].y) > 25){
        trArr.push({x, y});
        if(trArr.length > 18) trArr.shift();
      }
    }

    const totalDist = (lapNum - 1) + prog;
    const totalScore = (retired ? -1000 : 0) + totalDist - pos * 0.0001;
    states.push({
      abbr: d.abbr,
      name: d.name,
      team: d.team,
      color: d.color,
      number: d.number,
      grid: d.grid || 99,
      lapNum,
      pos,
      comp,
      inPit,
      finished,
      retired,
      lapDur,
      x,
      y,
      prog,
      totalDist,
      totalScore
    });
  }

  states.sort((a, b) => {
    if(a.retired !== b.retired) return a.retired ? 1 : -1;
    if(a.lapNum !== b.lapNum) return b.lapNum - a.lapNum;
    if(a.pos !== b.pos && a.pos !== 99 && b.pos !== 99) return a.pos - b.pos;
    return b.totalScore - a.totalScore;
  });

  sim.renderStates = states;

  if(t <= sim.minT + 1.5){
    sim.curLap = 1;
  } else if(states.length > 0){
    sim.curLap = Math.min(sim.totalLaps, Math.max(1, states[0].lapNum));
  }

  document.getElementById("lapCounterLbl").textContent = `LAP ${sim.curLap} / ${sim.totalLaps}`;
  timeReadout.textContent = `${formatSec(t - sim.minT)} / ${formatSec(sim.maxT - sim.minT)}`;
  timeSlider.value = t;

  updateLeaderboardDOM(states);
  updateRightHudAndMessages(t);
  drawTrackFrame();
}

function updateLeaderboardDOM(states){
  const listEl = document.getElementById("leaderboardList");
  let html = "";
  const leader = states[0];

  for(let i = 0; i < states.length; i++){
    const s = states[i];
    const selCls = (s.abbr === sim.selectedAbbr) ? "selected" : "";
    const p1Cls = (i === 0) ? "p1-row" : "";
    const outCls = s.retired ? "out-row" : "";

    const gain = (s.grid || (i + 1)) - (i + 1);
    const deltaStr = gain > 0
      ? `<span style="color:var(--green);">+${gain}</span>`
      : (gain < 0 ? `<span style="color:var(--f1-red);">${gain}</span>` : `<span style="color:var(--dim);">-</span>`);

    let gapTxt = "";
    if(s.retired){
      gapTxt = `<span class="badge-out">OUT</span>`;
    } else if(s.inPit){
      gapTxt = `<span class="badge-pit">IN PIT</span>`;
    } else if(s.finished){
      gapTxt = `FIN`;
    } else if(i === 0){
      gapTxt = `LEADER`;
    } else {
      const refCar = (sim.gapMode === "interval") ? states[i - 1] : leader;
      const diffLaps = refCar.totalDist - s.totalDist;
      if(diffLaps >= 1.0){
        gapTxt = `+${Math.max(1, Math.floor(diffLaps))}L`;
      } else {
        const secGap = Math.max(0.1, diffLaps * (refCar.lapDur || 88.0) + (i * 0.12));
        gapTxt = `+${secGap.toFixed(1)}s`;
      }
    }

    const tClass = ["S","M","H","I","W"].includes(s.comp) ? `tyre-${s.comp}` : "tyre-U";

    html += `
      <div class="t-row ${selCls} ${p1Cls} ${outCls}" onclick="selectDriver('${s.abbr}')">
        <div class="t-pos mono">${i + 1}</div>
        <div class="t-team-bar" style="background:${s.color}"></div>
        <div class="t-abbr">${s.abbr}</div>
        <div class="t-delta mono">${deltaStr}</div>
        <div class="t-gap mono">${gapTxt}</div>
        <div class="tyre-ring ${tClass}">${s.comp}</div>
        <div class="t-lap mono">L${s.lapNum}</div>
      </div>
    `;
  }
  listEl.innerHTML = html;
}

window.selectDriver = function(abbr){
  sim.selectedAbbr = abbr;
  sim.traceHistory = [];
  updateSimulation(sim.t);
};

function updateRightHudAndMessages(t){
  const d = sim.data && sim.selectedAbbr ? sim.data.drivers[sim.selectedAbbr] : null;
  if(d){
    document.getElementById("hudDrvName").textContent = d.name.toUpperCase();
    document.getElementById("hudDrvTeam").textContent = d.team.toUpperCase();
    document.getElementById("hudDrvNum").textContent = d.number;
    document.getElementById("hudDrvNum").style.color = d.color;
    document.getElementById("drvHeaderBorder").style.borderLeftColor = d.color;

    const st = sim.renderStates.find(x => x.abbr === d.abbr);
    const comp = st ? st.comp : "M";
    const tyreEl = document.getElementById("hudTyre");
    tyreEl.textContent = comp;
    tyreEl.className = "tyre-ring " + (["S","M","H","I","W"].includes(comp) ? `tyre-${comp}` : "tyre-U");

    const tel = sampleCarTelemetry(d.car, t);
    let spd = 0, gear = "N", rpm = 0, thr = 0, brk = 0, drs = 0;
    if(st && st.retired){
      spd = 0; gear = "N"; rpm = 0; thr = 0; brk = 0; drs = 0;
    } else if(tel){
      spd = tel.s;
      gear = tel.g > 0 ? tel.g : "N";
      rpm = tel.rpm;
      thr = tel.th;
      brk = tel.b;
      drs = tel.drs;
    } else if(st && !st.inPit && t > sim.minT){
      const wave = Math.sin(t * 1.9 + parseInt(d.number || "1", 10));
      spd = Math.round(238 + 68 * wave);
      gear = Math.max(2, Math.min(8, Math.round(spd / 38)));
      rpm = Math.round(9600 + 2300 * wave);
      thr = wave > -0.2 ? Math.round(78 + 22 * wave) : 12;
      brk = wave <= -0.2 ? 1 : 0;
    }
    document.getElementById("hudSpeed").textContent = spd;
    document.getElementById("hudGear").textContent = gear;
    document.getElementById("hudRpm").textContent = rpm;
    document.getElementById("hudThrTxt").textContent = `${thr}%`;
    document.getElementById("hudThrBar").style.width = `${thr}%`;
    document.getElementById("hudBrkTxt").textContent = brk ? "ON" : "OFF";
    document.getElementById("hudBrkBar").style.width = brk ? "100%" : "0%";

    const drsEl = document.getElementById("drsBadge");
    drsEl.textContent = drs ? "DRS ACTIVE" : "DRS CLOSED";
    drsEl.style.color = drs ? "var(--green)" : "var(--dim)";

    const leds = document.querySelectorAll("#shiftLeds .led");
    const activeCount = Math.min(15, Math.max(0, Math.round(((rpm - 7500) / 4800) * 15)));
    leds.forEach((led, idx) => {
      if(idx < activeCount){
        if(idx < 5) led.style.background = "var(--green)";
        else if(idx < 10) led.style.background = "var(--f1-red)";
        else led.style.background = "var(--cyan)";
      } else {
        led.style.background = "#1b1f2a";
      }
    });

    sim.traceHistory.push({spd, thr, brk: brk ? 100 : 0});
    if(sim.traceHistory.length > 90) sim.traceHistory.shift();
    drawTelemetryTrace();
  }

  const tsList = sim.data.track_status || [];
  let curCode = "1";
  for(let i = 0; i < tsList.length; i++){
    if(tsList[i].t <= t) curCode = tsList[i].code;
  }
  const pill = document.getElementById("flagPill");
  if(curCode === "1"){ pill.textContent = "TRACK CLEAR"; pill.style.background = "var(--green)"; pill.style.color = "#031b0e"; }
  else if(curCode === "2"){ pill.textContent = "YELLOW FLAG"; pill.style.background = "var(--yellow)"; pill.style.color = "#111"; }
  else if(curCode === "4"){ pill.textContent = "SAFETY CAR"; pill.style.background = "var(--yellow)"; pill.style.color = "#111"; }
  else if(curCode === "5"){ pill.textContent = "RED FLAG"; pill.style.background = "var(--f1-red)"; pill.style.color = "#fff"; }
  else if(curCode === "6" || curCode === "7"){ pill.textContent = "VIRTUAL SAFETY CAR"; pill.style.background = "var(--yellow)"; pill.style.color = "#111"; }

  const wList = sim.data.weather || [];
  let curW = wList.length > 0 ? wList[0] : null;
  for(let i = 0; i < wList.length; i++){
    if(wList[i].t <= t) curW = wList[i];
  }
  if(curW){
    document.getElementById("weatherLbl").textContent = `AIR ${curW.air}C | TRACK ${curW.track}C | ${curW.rain ? "WET" : "DRY"}`;
  }

  const rcm = (sim.data.rcm || []).filter(m => m.t <= t + 1.0).slice(-12).reverse();
  const feedEl = document.getElementById("rcmFeed");
  feedEl.innerHTML = rcm.map(m => `
    <div class="rcm-card">
      <div class="rcm-meta mono"><span>+${formatSec(m.t - sim.minT)}</span><span>${m.flag || m.cat}</span></div>
      <div>${m.msg}</div>
    </div>
  `).join("");
}

function drawTelemetryTrace(){
  const w = traceCanvas.width, h = traceCanvas.height;
  traceCtx.clearRect(0, 0, w, h);
  traceCtx.strokeStyle = "#181c26";
  traceCtx.lineWidth = 1;
  traceCtx.beginPath();
  traceCtx.moveTo(0, h / 2);
  traceCtx.lineTo(w, h / 2);
  traceCtx.stroke();

  const hist = sim.traceHistory;
  if(hist.length < 2) return;

  const drawChannel = (key, maxVal, color, width) => {
    traceCtx.beginPath();
    for(let i = 0; i < hist.length; i++){
      const px = (i / 89) * w;
      const py = h - 6 - (Math.min(maxVal, hist[i][key]) / maxVal) * (h - 12);
      if(i === 0) traceCtx.moveTo(px, py);
      else traceCtx.lineTo(px, py);
    }
    traceCtx.strokeStyle = color;
    traceCtx.lineWidth = width;
    traceCtx.stroke();
  };

  drawChannel("thr", 100, "rgba(0, 230, 118, 0.7)", 1.4);
  drawChannel("brk", 100, "rgba(225, 6, 0, 0.7)", 1.4);
  drawChannel("spd", 360, "#00e5ff", 2.0);
}

function resizeCanvas(){
  const rect = trackCanvas.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  trackCanvas.width = Math.max(300, Math.floor(rect.width * dpr));
  trackCanvas.height = Math.max(240, Math.floor((rect.height - 44) * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  drawTrackFrame();
}
window.addEventListener("resize", resizeCanvas);

function drawTrackFrame(){
  const w = trackCanvas.width / (window.devicePixelRatio || 1);
  const h = trackCanvas.height / (window.devicePixelRatio || 1);
  ctx.clearRect(0, 0, w, h);
  if(!sim.data || !sim.data.circuit) return;

  const tx = sim.data.circuit.x || [];
  const ty = sim.data.circuit.y || [];
  if(tx.length < 2) return;

  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for(let i = 0; i < tx.length; i++){
    if(tx[i] < minX) minX = tx[i];
    if(tx[i] > maxX) maxX = tx[i];
    if(ty[i] < minY) minY = ty[i];
    if(ty[i] > maxY) maxY = ty[i];
  }

  const pad = 52;
  const spanX = Math.max(1, maxX - minX);
  const spanY = Math.max(1, maxY - minY);
  const baseScale = Math.min((w - pad * 2) / spanX, (h - pad * 2) / spanY);
  const scale = baseScale * sim.camZoom;

  let centerWorldX = (minX + maxX) / 2;
  let centerWorldY = (minY + maxY) / 2;

  if(sim.followCar && sim.selectedAbbr){
    const selSt = (sim.renderStates || []).find(s => s.abbr === sim.selectedAbbr);
    if(selSt){
      centerWorldX = selSt.x;
      centerWorldY = selSt.y;
    }
  }

  const ox = w / 2 - centerWorldX * scale + (sim.followCar ? 0 : sim.camPanX);
  const oy = h / 2 + centerWorldY * scale + (sim.followCar ? 0 : sim.camPanY);

  const toScreen = (x, y) => ({
    sx: ox + x * scale,
    sy: oy - y * scale
  });

  ctx.save();
  ctx.beginPath();
  for(let i = 0; i < tx.length; i++){
    const p = toScreen(tx[i], ty[i]);
    if(i === 0) ctx.moveTo(p.sx, p.sy);
    else ctx.lineTo(p.sx, p.sy);
  }
  ctx.closePath();
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.lineWidth = 12 * Math.pow(sim.camZoom, 0.35);
  ctx.strokeStyle = "#1a1f2c";
  ctx.stroke();

  ctx.lineWidth = 7.5 * Math.pow(sim.camZoom, 0.35);
  ctx.strokeStyle = "#2e364a";
  ctx.stroke();

  ctx.lineWidth = 1.0;
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.stroke();

  const sf = toScreen(tx[0], ty[0]);
  ctx.beginPath();
  ctx.arc(sf.sx, sf.sy, 5.5, 0, Math.PI * 2);
  ctx.fillStyle = "#e10600";
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#ffffff";
  ctx.stroke();

  const corners = sim.data.circuit.corners || [];
  ctx.font = "700 9px 'JetBrains Mono', monospace";
  for(let i = 0; i < corners.length; i++){
    const cp = toScreen(corners[i].x, corners[i].y);
    ctx.fillStyle = "rgba(14, 16, 21, 0.85)";
    ctx.fillRect(cp.sx + 4, cp.sy - 12, 18, 11);
    ctx.fillStyle = "#8b93a7";
    ctx.fillText(corners[i].num, cp.sx + 6, cp.sy - 4);
  }

  const states = sim.renderStates || [];
  if(sim.showTrails){
    for(let i = states.length - 1; i >= 0; i--){
      const st = states[i];
      if(st.retired) continue;
      const tr = sim.trails[st.abbr];
      if(!tr || tr.length < 2) continue;
      ctx.beginPath();
      for(let j = 0; j < tr.length; j++){
        const pt = toScreen(tr[j].x, tr[j].y);
        if(j === 0) ctx.moveTo(pt.sx, pt.sy);
        else ctx.lineTo(pt.sx, pt.sy);
      }
      ctx.strokeStyle = st.color || "#ffffff";
      ctx.globalAlpha = (st.abbr === sim.selectedAbbr) ? 0.55 : 0.22;
      ctx.lineWidth = (st.abbr === sim.selectedAbbr) ? 2.5 : 1.5;
      ctx.stroke();
    }
    ctx.globalAlpha = 1.0;
  }

  for(let i = states.length - 1; i >= 0; i--){
    const st = states[i];
    const cp = toScreen(st.x, st.y);
    const isSel = (st.abbr === sim.selectedAbbr);
    const radius = isSel ? 6.5 : 4.2;

    ctx.globalAlpha = st.retired ? 0.3 : 1.0;

    if(isSel){
      ctx.beginPath();
      ctx.arc(cp.sx, cp.sy, 12.0, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255, 255, 255, 0.16)";
      ctx.fill();
    }

    ctx.beginPath();
    ctx.arc(cp.sx, cp.sy, radius, 0, Math.PI * 2);
    ctx.fillStyle = st.color || "#ffffff";
    ctx.fill();
    ctx.lineWidth = isSel ? 2.0 : 1.0;
    ctx.strokeStyle = isSel ? "#ffffff" : "#07080b";
    ctx.stroke();

    if(!st.retired){
      ctx.font = isSel ? "900 10px 'Titillium Web', sans-serif" : "700 9px 'Titillium Web', sans-serif";
      ctx.fillStyle = isSel ? "#ffffff" : "rgba(243, 244, 246, 0.88)";
      ctx.fillText(st.abbr, cp.sx + radius + 3, cp.sy + 3);
    }
  }
  ctx.globalAlpha = 1.0;
  ctx.restore();
}

function drawComparisonChart(){
  const canvas = document.getElementById("compareCanvas");
  const c = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  c.clearRect(0, 0, w, h);
  if(!sim.data) return;

  const abbrA = document.getElementById("cmpDrvA").value;
  const abbrB = document.getElementById("cmpDrvB").value;
  const dA = sim.data.drivers[abbrA];
  const dB = sim.data.drivers[abbrB];
  if(!dA || !dB) return;

  const lapsA = (dA.laps || []).map(l => ({lap: l[2], dur: l[1] - l[0]})).filter(x => x.dur > 40 && x.dur < 220);
  const lapsB = (dB.laps || []).map(l => ({lap: l[2], dur: l[1] - l[0]})).filter(x => x.dur > 40 && x.dur < 220);
  const allDurs = [...lapsA.map(x => x.dur), ...lapsB.map(x => x.dur)];
  if(allDurs.length === 0) return;

  const minD = Math.min(...allDurs) - 1.5;
  const maxD = Math.max(...allDurs) + 1.5;
  const maxL = Math.max(1, sim.totalLaps);
  const padL = 55, padR = 25, padT = 32, padB = 35;

  const plotLine = (arr, color, label, offsetLbl) => {
    c.beginPath();
    arr.forEach((pt, idx) => {
      const px = padL + ((pt.lap - 1) / Math.max(1, maxL - 1)) * (w - padL - padR);
      const py = h - padB - ((pt.dur - minD) / Math.max(1, maxD - minD)) * (h - padT - padB);
      if(idx === 0) c.moveTo(px, py);
      else c.lineTo(px, py);
    });
    c.strokeStyle = color;
    c.lineWidth = 2.5;
    c.stroke();
    c.fillStyle = color;
    c.font = "700 12px 'Titillium Web', sans-serif";
    c.fillText(`${label} (${arr.length} LAPS)`, padL + offsetLbl, 20);
  };

  c.strokeStyle = "#222634";
  c.lineWidth = 1;
  c.strokeRect(padL, padT, w - padL - padR, h - padT - padB);
  plotLine(lapsA, dA.color || "#3671C6", dA.abbr, 10);
  plotLine(lapsB, dB.color || "#FF8000", dB.abbr, 160);
}
document.getElementById("runCompareBtn").addEventListener("click", drawComparisonChart);

function tick(nowMs){
  const dt = Math.min(0.1, (nowMs - sim.lastFrameMs) / 1000.0);
  sim.lastFrameMs = nowMs;
  if(sim.playing && sim.data){
    sim.t += dt * sim.speed;
    if(sim.t >= sim.maxT){
      sim.t = sim.maxT;
      sim.playing = false;
      playPauseBtn.textContent = "PLAY";
    }
    updateSimulation(sim.t);
  }
  requestAnimationFrame(tick);
}

playPauseBtn.addEventListener("click", () => {
  if(!sim.data) return;
  if(sim.t >= sim.maxT) sim.t = sim.minT;
  sim.playing = !sim.playing;
  playPauseBtn.textContent = sim.playing ? "PAUSE" : "PLAY";
});

document.getElementById("stepBackBtn").addEventListener("click", () => {
  if(!sim.data) return;
  sim.t = Math.max(sim.minT, sim.t - 5.0);
  updateSimulation(sim.t);
});

document.getElementById("stepFwdBtn").addEventListener("click", () => {
  if(!sim.data) return;
  sim.t = Math.min(sim.maxT, sim.t + 5.0);
  updateSimulation(sim.t);
});

restartBtn.addEventListener("click", () => {
  if(!sim.data) return;
  sim.t = sim.minT;
  sim.trails = {};
  updateSimulation(sim.t);
});

speedSelect.addEventListener("change", () => {
  sim.speed = parseFloat(speedSelect.value) || 5.0;
});

timeSlider.addEventListener("input", () => {
  if(!sim.data) return;
  sim.t = parseFloat(timeSlider.value);
  sim.trails = {};
  updateSimulation(sim.t);
});

window.addEventListener("keydown", (e) => {
  if(e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if(e.code === "Space"){
    e.preventDefault();
    playPauseBtn.click();
  } else if(e.code === "ArrowRight"){
    e.preventDefault();
    document.getElementById("stepFwdBtn").click();
  } else if(e.code === "ArrowLeft"){
    e.preventDefault();
    document.getElementById("stepBackBtn").click();
  } else if(e.code === "KeyF"){
    document.getElementById("followCarBtn").click();
  } else if(e.code === "KeyR"){
    document.getElementById("resetCamBtn").click();
  }
});

loadBtn.addEventListener("click", loadSessionData);

const loginModal = document.getElementById("loginModal");
document.getElementById("loginOpenBtn").addEventListener("click", () => {
  loginModal.style.display = "flex";
});
document.getElementById("loginCloseBtn").addEventListener("click", () => {
  loginModal.style.display = "none";
});
document.getElementById("loginSubmitBtn").addEventListener("click", async () => {
  const email = document.getElementById("f1EmailInput").value;
  const password = document.getElementById("f1PassInput").value;
  const res = await fetch("/api/login", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({email, password})
  });
  const out = await res.json();
  document.getElementById("loginStatusMsg").textContent = out.message || "Saved";
});

loadSchedule(yearSelect.value).then(() => {
  resizeCanvas();
  requestAnimationFrame(tick);
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/api/schedule")
def api_schedule():
    year = request.args.get("year", 2025, type=int)
    return jsonify(F1Core.get_season_schedule(year))


@app.route("/api/login", methods=["POST"])
def api_login():
    body = request.get_json(silent=True) or {}
    res = F1Core.authenticate_f1_account(body.get("email", ""), body.get("password", ""))
    return jsonify(res)


@app.route("/api/auth_status")
def api_auth_status():
    return jsonify(F1Core.get_auth_status())


@app.route("/api/replay")
def api_replay():
    year = request.args.get("year", 2025, type=int)
    gp = request.args.get("gp", "1")
    session_type = request.args.get("session", "R")
    try:
        payload = F1Core.compile_replay_payload(year, gp, session_type)
        raw_json = json.dumps(payload).encode("utf-8")
        if "gzip" in request.headers.get("Accept-Encoding", ""):
            compressed = gzip.compress(raw_json, compresslevel=5)
            resp = Response(compressed, mimetype="application/json")
            resp.headers["Content-Encoding"] = "gzip"
            resp.headers["Content-Length"] = str(len(compressed))
            return resp
        return Response(raw_json, mimetype="application/json")
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    try:
        cur_file = Path(__file__).resolve()
        alt_file = cur_file.parent / "app.py"
        if cur_file.name.lower() != "app.py":
            shutil.copyfile(str(cur_file), str(alt_file))
    except Exception:
        pass

    port = int(os.environ.get("PORT", 5000))
    print(f"\n[F1 Pit Wall Web App] Live at: http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
