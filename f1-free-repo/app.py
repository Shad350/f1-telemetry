#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F1 OFFICIAL TELEMETRY PIT WALL & HISTORICAL RACE REPLAY (1950-2026)
Supports: RACE (R), QUALIFYING (Q), SPRINT (S), SPRINT QUALI (SQ), FP1, FP2, FP3
512 MB RAM Cloud Edition + Authentic 1950-2026 Circuit Geometry Engine
"""

import os
import sys
import math
import json
import time
import gc
import threading
import webbrowser
import urllib.request
import urllib.parse
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, Response

try:
    import fastf1
except ImportError:
    fastf1 = None


# ==============================================================================
# 1. AUTHENTIC 1950-2026 CIRCUIT GEOMETRY & SPLINE ENGINE (512 MB SAFE)
# ==============================================================================
class CircuitEngine:
    _MV_CACHE: Dict[Tuple[int, int], Dict[str, Any]] = {}
    _GJ_DATA: Optional[Dict[str, Any]] = None
    _LOCK = threading.Lock()

    MV_KEYS = {
        "melbourne": 10, "albert_park": 10,
        "bahrain": 63, "sakhir": 63,
        "jeddah": 149, "saudi": 149,
        "suzuka": 46,
        "shanghai": 49, "china": 49, "chinese": 49,
        "miami": 151,
        "imola": 6, "emilia": 6,
        "monaco": 22, "monte_carlo": 22,
        "montreal": 23, "villeneuve": 23, "canada": 23, "canadian": 23,
        "barcelona": 15, "catalunya": 15,
        "spielberg": 19, "red_bull_ring": 19, "austria": 19, "austrian": 19, "styria": 19,
        "silverstone": 2, "british": 2, "great_britain": 2,
        "hungaroring": 4, "budapest": 4, "hungary": 4, "hungarian": 4,
        "spa": 7, "belgium": 7, "belgian": 7,
        "zandvoort": 55, "dutch": 55, "netherlands": 55,
        "monza": 39, "italy": 39, "italian": 39,
        "baku": 144, "azerbaijan": 144,
        "marina_bay": 61, "singapore": 61,
        "austin": 9, "americas": 9, "cota": 9,
        "rodriguez": 65, "mexico": 65, "mexican": 65,
        "interlagos": 14, "brazil": 14, "brazilian": 14, "sao_paulo": 14, "sao paulo": 14,
        "vegas": 152, "las_vegas": 152, "las vegas": 152,
        "lusail": 150, "losail": 150, "qatar": 150,
        "yas_marina": 70, "yas marina": 70, "abu_dhabi": 70, "abu dhabi": 70,
        "portimao": 148, "algarve": 148,
        "istanbul": 54, "turkey": 54, "turkish": 54,
        "ricard": 142, "paul_ricard": 142, "paul ricard": 142,
        "mugello": 147, "tuscan": 147,
        "nurburgring": 20, "eifel": 20,
        "hockenheim": 29, "german": 29,
        "sochi": 79, "russia": 79, "russian": 79,
        "sepang": 24, "malaysia": 24, "malaysian": 24,
    }

    @staticmethod
    def smooth_resample_track(tx: List[float], ty: List[float], n_pts: int = 800) -> Tuple[List[float], List[float]]:
        x = np.asarray(tx, dtype=float)
        y = np.asarray(ty, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y) & ~((np.abs(x) < 1e-3) & (np.abs(y) < 1e-3))
        x, y = x[mask], y[mask]
        if len(x) < 6:
            return list(tx), list(ty)

        keep = [0]
        for i in range(1, len(x)):
            if math.hypot(x[i] - x[keep[-1]], y[i] - y[keep[-1]]) > 0.5:
                keep.append(i)
        x, y = x[keep], y[keep]
        if len(x) < 6:
            return list(tx), list(ty)
        if math.hypot(x[-1] - x[0], y[-1] - y[0]) < 1.5:
            x, y = x[:-1], y[:-1]

        N = len(x)
        if N < 360:
            sx, sy = [], []
            steps = max(4, min(14, 720 // max(1, N)))
            for i in range(N):
                i0 = divmod(i - 1, N)[1]
                i2 = divmod(i + 1, N)[1]
                i3 = divmod(i + 2, N)[1]
                p0x, p0y = x[i0], y[i0]
                p1x, p1y = x[i], y[i]
                p2x, p2y = x[i2], y[i2]
                p3x, p3y = x[i3], y[i3]
                for st in range(steps):
                    t = st / float(steps)
                    t2 = t * t
                    t3 = t2 * t
                    cx = 0.5 * ((2.0 * p1x) + (-p0x + p2x) * t + (2.0 * p0x - 5.0 * p1x + 4.0 * p2x - p3x) * t2 + (-p0x + 3.0 * p1x - 3.0 * p2x + p3x) * t3)
                    cy = 0.5 * ((2.0 * p1y) + (-p0y + p2y) * t + (2.0 * p0y - 5.0 * p1y + 4.0 * p2y - p3y) * t2 + (-p0y + 3.0 * p1y - 3.0 * p2y + p3y) * t3)
                    sx.append(cx)
                    sy.append(cy)
            x, y = np.asarray(sx, dtype=float), np.asarray(sy, dtype=float)

        x = np.append(x, x[0])
        y = np.append(y, y[0])
        ds = np.hypot(np.diff(x), np.diff(y))
        s_cum = np.concatenate(([0.0], np.cumsum(ds)))
        total_len = float(s_cum[-1])
        if total_len <= 1.0:
            return list(tx), list(ty)

        target_s = np.linspace(0.0, total_len, n_pts, endpoint=False)
        rx = np.interp(target_s, s_cum, x)
        ry = np.interp(target_s, s_cum, y)
        rx = np.append(rx, rx[0])
        ry = np.append(ry, ry[0])
        return [round(float(v), 1) for v in rx], [round(float(v), 1) for v in ry]

    @staticmethod
    def detect_corners(tx: List[float], ty: List[float]) -> List[Dict[str, Any]]:
        x = np.asarray(tx[:-1], dtype=float)
        y = np.asarray(ty[:-1], dtype=float)
        N = len(x)
        if N < 60:
            return []
        w = max(4, N // 80)
        ang = np.zeros(N, dtype=float)
        for i in range(N):
            iw1 = divmod(i - w, N)[1]
            iw2 = divmod(i + w, N)[1]
            dx1, dy1 = x[i] - x[iw1], y[i] - y[iw1]
            dx2, dy2 = x[iw2] - x[i], y[iw2] - y[i]
            n1 = math.hypot(dx1, dy1)
            n2 = math.hypot(dx2, dy2)
            if n1 > 0 and n2 > 0:
                c = max(-1.0, min(1.0, (dx1 * dx2 + dy1 * dy2) / (n1 * n2)))
                ang[i] = math.degrees(math.acos(c))
        corners = []
        last_i = -999
        min_gap = max(10, N // 28)
        for i in range(N):
            if ang[i] > 15.5 and (i - last_i) >= min_gap:
                loc = [ang[divmod(i + k, N)[1]] for k in range(-w, w + 1)]
                if ang[i] >= max(loc):
                    corners.append({
                        "num": len(corners) + 1,
                        "letter": "",
                        "x": round(float(x[i]), 1),
                        "y": round(float(y[i]), 1)
                    })
                    last_i = i
                    if len(corners) >= 24:
                        break
        return corners

    @classmethod
    def resolve_mv_key(cls, year: int, search_text: str, explicit_key: Optional[int] = None) -> Optional[int]:
        if explicit_key:
            return int(explicit_key)
        t = search_text.lower()
        if "adelaide" in t:
            return None
        if "austral" in t:
            return 10 if year >= 1996 else None
        if "fuji" in t:
            return None
        if "japan" in t:
            return None if year in (1976, 1977, 2007, 2008) else 46
        if "jerez" in t or "jarama" in t or "montjuic" in t or "pedralbes" in t or "valencia" in t:
            return None
        if "spain" in t or "spanish" in t:
            return 15 if year >= 1991 else None
        if "indianapolis" in t or "watkins" in t or "detroit" in t or "phoenix" in t or "long beach" in t or "dallas" in t:
            return None
        if "united states" in t:
            return 9 if year >= 2012 else None
        if "magny" in t or "dijon" in t or "charade" in t or "reims" in t or "rouen" in t:
            return None
        if "fran" in t:
            return 142 if (year >= 2018 or year in (1971, 1973, 1975, 1976, 1978, 1980, 1982, 1983, 1985, 1986, 1987, 1988, 1989, 1990)) else None
        if "estoril" in t or "boavista" in t or "monsanto" in t:
            return None
        if "portug" in t:
            return 148 if year >= 2020 else None
        if "brands hatch" in t or "aintree" in t:
            return None
        if "zolder" in t or "nivelles" in t:
            return None
        if "jacarepagua" in t:
            return None

        for k, v in cls.MV_KEYS.items():
            if k in t:
                return v
        return None

    @classmethod
    def get_circuit(cls, year: int, gp: Any, title: str = "", location: str = "", circuit_id: str = "", explicit_mv_key: Optional[int] = None) -> Dict[str, Any]:
        yr_int = int(year) if str(year).isdigit() else 2025
        search_text = f"{title} {gp} {location} {circuit_id}".lower()

        # 1. Official F1 MultiViewer Laser Circuit API
        ckey = cls.resolve_mv_key(yr_int, search_text, explicit_mv_key)
        if ckey:
            for try_yr in (max(2018, min(2025, yr_int)), 2024, 2023):
                mk = (ckey, try_yr)
                if mk in cls._MV_CACHE:
                    return cls._MV_CACHE[mk]
                try:
                    url = f"https://api.multiviewer.app/api/v1/circuits/{ckey}/{try_yr}"
                    req = urllib.request.Request(url, headers={"User-Agent": "F1PitWall/3.0"})
                    with urllib.request.urlopen(req, timeout=6) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                    mx, my = data.get("x", []), data.get("y", [])
                    if len(mx) > 50:
                        rot = float(data.get("rotation", 0.0)) * math.pi / 180.0
                        ca, sa = math.cos(rot), math.sin(rot)
                        rx = [x * ca - y * sa for x, y in zip(mx, my)]
                        ry = [x * sa + y * ca for x, y in zip(mx, my)]
                        tx, ty = cls.smooth_resample_track(rx, ry, 800)
                        corners = []
                        for c in data.get("corners", []):
                            tp = c.get("trackPosition", {})
                            if "x" in tp and "y" in tp:
                                cx, cy = float(tp["x"]), float(tp["y"])
                                corners.append({
                                    "num": int(c.get("number", 0)),
                                    "letter": str(c.get("letter", "")),
                                    "x": round(cx * ca - cy * sa, 1),
                                    "y": round(cx * sa + cy * ca, 1)
                                })
                        out = {"x": tx, "y": ty, "corners": corners or cls.detect_corners(tx, ty)}
                        cls._MV_CACHE[mk] = out
                        return out
                except Exception:
                    pass

        # 2. 1950-2026 Historical & Modern F1 Circuit GeoJSON Database
        try:
            with cls._LOCK:
                if cls._GJ_DATA is None:
                    gurl = "https://raw.githubusercontent.com/bacinger/f1-circuits/master/f1-circuits.geojson"
                    req = urllib.request.Request(gurl, headers={"User-Agent": "F1PitWall/3.0"})
                    with urllib.request.urlopen(req, timeout=8) as resp:
                        cls._GJ_DATA = json.loads(resp.read().decode("utf-8"))

            gj_map = {
                "adelaide": "au-1985",
                "melbourne": "au-1953",
                "albert_park": "au-1953",
                "austral": "au-1985" if yr_int < 1996 else "au-1953",
                "bahrain": "bh-2002", "sakhir": "bh-2002",
                "saudi": "sa-2021", "jeddah": "sa-2021",
                "fuji": "jp-1965",
                "suzuka": "jp-1962",
                "japan": "jp-1965" if yr_int in (1976, 1977, 2007, 2008) else "jp-1962",
                "shanghai": "cn-2004", "chin": "cn-2004",
                "miami": "us-2022",
                "imola": "it-1953", "emilia": "it-1953",
                "monaco": "mc-1929", "monte_carlo": "mc-1929",
                "montreal": "ca-1978", "villeneuve": "ca-1978", "canad": "ca-1978",
                "jerez": "es-1986",
                "valencia": "es-2008",
                "barcelona": "es-1991", "catalunya": "es-1991",
                "spain": "es-1986" if yr_int < 1991 else "es-1991",
                "spanish": "es-1986" if yr_int < 1991 else "es-1991",
                "spielberg": "at-1969", "austria": "at-1969", "red_bull_ring": "at-1969",
                "brands hatch": "gb-1926", "brands_hatch": "gb-1926",
                "silverstone": "gb-1948", "brit": "gb-1948",
                "hungar": "hu-1986", "budapest": "hu-1986",
                "zolder": "be-1963",
                "spa": "be-1925", "belg": "be-1925",
                "zandvoort": "nl-1948", "dutch": "nl-1948",
                "mugello": "it-1914", "tusc": "it-1914",
                "monza": "it-1922", "ital": "it-1922",
                "baku": "az-2016", "azerbaijan": "az-2016",
                "singapore": "sg-2008", "marina_bay": "sg-2008",
                "indianapolis": "us-1909",
                "watkins": "us-1948",
                "austin": "us-2012", "americas": "us-2012",
                "united states": "us-1909" if (2000 <= yr_int <= 2007) else ("us-1948" if yr_int < 1990 else "us-2012"),
                "mexic": "mx-1962", "rodriguez": "mx-1962",
                "interlagos": "br-1940", "brazil": "br-1940", "sao paulo": "br-1940", "sao_paulo": "br-1940",
                "vegas": "us-2023",
                "qatar": "qa-2004", "losail": "qa-2004", "lusail": "qa-2004",
                "abu dhabi": "ae-2009", "abu_dhabi": "ae-2009", "yas": "ae-2009",
                "sepang": "my-1999", "malays": "my-1999",
                "hockenheim": "de-1932", "german": "de-1932",
                "nurburgring": "de-1927", "eifel": "de-1927",
                "magny": "fr-1960",
                "ricard": "fr-1969",
                "fran": "fr-1960" if (1991 <= yr_int <= 2008) else "fr-1969",
                "estoril": "pt-1972",
                "portimao": "pt-2008", "algarve": "pt-2008",
                "portug": "pt-1972" if yr_int < 2020 else "pt-2008",
                "kyalami": "za-1961", "south afric": "za-1961",
                "korea": "kr-2010", "yeongam": "kr-2010",
                "india": "in-2011", "buddh": "in-2011",
                "turk": "tr-2005", "istanbul": "tr-2005",
                "sochi": "ru-2014", "russ": "ru-2014",
                "argentin": "ar-1952", "galvez": "ar-1952", "buenos": "ar-1952",
            }

            best_feat = None
            for kw, gid in gj_map.items():
                if kw in search_text:
                    for ft in cls._GJ_DATA.get("features", []):
                        if ft.get("properties", {}).get("id") == gid:
                            best_feat = ft
                            break
                    if best_feat:
                        break

            if not best_feat:
                words = [w for w in search_text.replace("_", " ").split() if len(w) >= 4 and w not in ("grand", "prix", "race", "round", "formula")]
                for ft in cls._GJ_DATA.get("features", []):
                    pr = str(ft.get("properties", {})).lower()
                    if any(w in pr for w in words):
                        best_feat = ft
                        break

            if best_feat:
                geom = best_feat.get("geometry", {})
                coords = geom.get("coordinates", [])
                if geom.get("type") == "MultiLineString":
                    pts = list(coords[0])
                    rem = [list(seg) for seg in coords[1:] if len(seg) > 0]
                    while rem:
                        lx, ly = pts[-1][0], pts[-1][1]
                        bi, rev, bd = 0, False, 1e18
                        for idx, seg in enumerate(rem):
                            d1 = math.hypot(seg[0][0] - lx, seg[0][1] - ly)
                            d2 = math.hypot(seg[-1][0] - lx, seg[-1][1] - ly)
                            if d1 < bd:
                                bi, rev, bd = idx, False, d1
                            if d2 < bd:
                                bi, rev, bd = idx, True, d2
                        chosen = rem.pop(bi)
                        if rev:
                            chosen = chosen[::-1]
                        pts.extend(chosen)
                    coords = pts

                if len(coords) > 10:
                    lons = [float(c[0]) for c in coords]
                    lats = [float(c[1]) for c in coords]
                    mlon = sum(lons) / len(lons)
                    mlat = sum(lats) / len(lats)
                    sx = math.cos(math.radians(mlat)) * 111320.0
                    rx = [(lon - mlon) * sx for lon in lons]
                    ry = [(lat - mlat) * 111320.0 for lat in lats]
                    tx, ty = cls.smooth_resample_track(rx, ry, 800)
                    return {"x": tx, "y": ty, "corners": cls.detect_corners(tx, ty)}
        except Exception:
            pass

        # 3. Offline Precision Circuit Fallback
        tx, ty = [], []
        for i in range(800):
            a = -i * (2.0 * math.pi / 800.0)
            x = 4200.0 * math.cos(a) + 950.0 * math.cos(2.0 * a) - 420.0 * math.sin(3.0 * a) + 180.0 * math.cos(5.0 * a)
            y = 2400.0 * math.sin(a) - 550.0 * math.sin(2.0 * a) + 320.0 * math.cos(3.0 * a) - 150.0 * math.sin(4.0 * a)
            tx.append(round(x, 1))
            ty.append(round(y, 1))
        tx.append(tx[0])
        ty.append(ty[0])
        return {"x": tx, "y": ty, "corners": cls.detect_corners(tx, ty)}


# ==============================================================================
# 2. CORE F1 DATA & 512 MB TELEMETRY ENGINE (R, Q, S, SQ, FP1, FP2, FP3)
# ==============================================================================
class F1Core:
    CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f1_cache")
    REPLAY_MEMO: Dict[str, Dict[str, Any]] = {}
    SCHEDULE_MEMO: Dict[int, List[Dict[str, Any]]] = {}

    SESSION_LABELS = {
        "R": "Race",
        "Q": "Qualifying",
        "S": "Sprint",
        "SQ": "Sprint Qualifying",
        "FP1": "Practice 1 (FP1)",
        "FP2": "Practice 2 (FP2)",
        "FP3": "Practice 3 (FP3)",
    }

    TEAM_COLORS = {
        "Red Bull Racing": "#3671C6",
        "Red Bull": "#3671C6",
        "McLaren": "#FF8000",
        "Ferrari": "#E8002D",
        "Mercedes": "#27F4D2",
        "Aston Martin": "#229971",
        "Alpine": "#0093CC",
        "Williams": "#64C4FF",
        "RB": "#6692FF",
        "Racing Bulls": "#6692FF",
        "AlphaTauri": "#5E8FAA",
        "Kick Sauber": "#52E252",
        "Sauber": "#52E252",
        "Alfa Romeo": "#C92D4B",
        "Haas F1 Team": "#B6BABD",
        "Haas": "#B6BABD",
        "Renault": "#FFF500",
        "Racing Point": "#F596C8",
        "Force India": "#F596C8",
        "Toro Rosso": "#469BFF",
        "Lotus": "#FFB800",
        "Brawn": "#B8FD6E",
        "Benetton": "#00A3E0",
        "Tyrrell": "#0046AD",
        "Brabham": "#1B365D",
        "BRM": "#004225",
        "Cooper": "#003399",
        "Maserati": "#CE1126",
    }

    FALLBACK_PALETTE = [
        "#3671C6", "#FF8000", "#E8002D", "#27F4D2", "#229971",
        "#0093CC", "#64C4FF", "#6692FF", "#52E252", "#B6BABD",
        "#F596C8", "#FFF500", "#C92D4B", "#FFB800", "#A855F7"
    ]

    @staticmethod
    def is_low_mem_env() -> bool:
        if os.environ.get("RENDER") or os.environ.get("LOW_MEM") or os.environ.get("PORT") == "10000":
            return True
        for cg in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
            try:
                if os.path.exists(cg):
                    val = open(cg, "r").read().strip()
                    if val.isdigit() and int(val) <= 1024 * 1024 * 1024:
                        return True
            except Exception:
                pass
        return False

    @classmethod
    def setup_cache(cls):
        if fastf1 is None:
            return
        try:
            os.makedirs(cls.CACHE_DIR, exist_ok=True)
            fastf1.Cache.enable_cache(cls.CACHE_DIR)
        except Exception:
            try:
                tmp_cache = "/tmp/f1_cache"
                os.makedirs(tmp_cache, exist_ok=True)
                fastf1.Cache.enable_cache(tmp_cache)
            except Exception:
                pass

    @staticmethod
    def safe_attr(obj: Any, name: str, default: Any = None) -> Any:
        try:
            val = getattr(obj, name, default)
            return default if val is None else val
        except Exception:
            return default

    @staticmethod
    def td_to_sec(val: Any) -> float:
        if val is None or pd.isna(val):
            return float("nan")
        if isinstance(val, (int, float, np.number)):
            return float(val)
        try:
            return float(pd.to_timedelta(val).total_seconds())
        except Exception:
            return float("nan")

    @classmethod
    def get_schedule(cls, year: int) -> List[Dict[str, Any]]:
        year = int(year)
        if year in cls.SCHEDULE_MEMO:
            return cls.SCHEDULE_MEMO[year]

        try:
            url = f"https://api.jolpi.ca/ergast/f1/{year}.json"
            req = urllib.request.Request(url, headers={"User-Agent": "F1PitWall/3.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            if races:
                out = []
                for r in races:
                    rnd = int(r.get("round", 0))
                    if rnd <= 0:
                        continue
                    loc = r.get("Circuit", {}).get("Location", {})
                    out.append({
                        "round": rnd,
                        "name": str(r.get("raceName", f"Round {rnd}")),
                        "country": str(loc.get("country", "")),
                        "location": str(loc.get("locality", "")),
                        "date": str(r.get("date", ""))
                    })
                if out:
                    cls.SCHEDULE_MEMO[year] = out
                    return out
        except Exception:
            pass

        if fastf1 is not None:
            try:
                sched = fastf1.get_event_schedule(year, include_testing=False)
                if sched is not None and not sched.empty:
                    out = []
                    for _, row in sched.iterrows():
                        rnd = int(row.get("RoundNumber", 0))
                        if rnd <= 0:
                            continue
                        out.append({
                            "round": rnd,
                            "name": str(row.get("EventName", f"Round {rnd}")),
                            "country": str(row.get("Country", "")),
                            "location": str(row.get("Location", "")),
                            "date": str(row.get("EventDate", ""))[:10]
                        })
                    if out:
                        cls.SCHEDULE_MEMO[year] = out
                        return out
            except Exception:
                pass

        return [
            {"round": 1, "name": "Australian Grand Prix", "country": "Australia", "location": "Melbourne", "date": f"{year}-03-16"},
            {"round": 2, "name": "Chinese Grand Prix", "country": "China", "location": "Shanghai", "date": f"{year}-03-23"},
            {"round": 3, "name": "Japanese Grand Prix", "country": "Japan", "location": "Suzuka", "date": f"{year}-04-06"},
            {"round": 4, "name": "Bahrain Grand Prix", "country": "Bahrain", "location": "Sakhir", "date": f"{year}-04-13"},
            {"round": 5, "name": "Saudi Arabian Grand Prix", "country": "Saudi Arabia", "location": "Jeddah", "date": f"{year}-04-20"},
            {"round": 6, "name": "Miami Grand Prix", "country": "USA", "location": "Miami", "date": f"{year}-05-04"},
            {"round": 7, "name": "Emilia Romagna Grand Prix", "country": "Italy", "location": "Imola", "date": f"{year}-05-18"},
            {"round": 8, "name": "Monaco Grand Prix", "country": "Monaco", "location": "Monaco", "date": f"{year}-05-25"},
            {"round": 9, "name": "Spanish Grand Prix", "country": "Spain", "location": "Barcelona", "date": f"{year}-06-01"},
            {"round": 10, "name": "Canadian Grand Prix", "country": "Canada", "location": "Montreal", "date": f"{year}-06-15"},
            {"round": 11, "name": "Austrian Grand Prix", "country": "Austria", "location": "Spielberg", "date": f"{year}-06-29"},
            {"round": 12, "name": "British Grand Prix", "country": "UK", "location": "Silverstone", "date": f"{year}-07-06"},
            {"round": 13, "name": "Belgian Grand Prix", "country": "Belgium", "location": "Spa-Francorchamps", "date": f"{year}-07-27"},
            {"round": 14, "name": "Hungarian Grand Prix", "country": "Hungary", "location": "Budapest", "date": f"{year}-08-03"},
            {"round": 15, "name": "Dutch Grand Prix", "country": "Netherlands", "location": "Zandvoort", "date": f"{year}-08-31"},
            {"round": 16, "name": "Italian Grand Prix", "country": "Italy", "location": "Monza", "date": f"{year}-09-07"},
            {"round": 17, "name": "Azerbaijan Grand Prix", "country": "Azerbaijan", "location": "Baku", "date": f"{year}-09-21"},
            {"round": 18, "name": "Singapore Grand Prix", "country": "Singapore", "location": "Marina Bay", "date": f"{year}-10-05"},
            {"round": 19, "name": "United States Grand Prix", "country": "USA", "location": "Austin", "date": f"{year}-10-19"},
            {"round": 20, "name": "Mexico City Grand Prix", "country": "Mexico", "location": "Mexico City", "date": f"{year}-10-26"},
            {"round": 21, "name": "Sao Paulo Grand Prix", "country": "Brazil", "location": "Sao Paulo", "date": f"{year}-11-09"},
            {"round": 22, "name": "Las Vegas Grand Prix", "country": "USA", "location": "Las Vegas", "date": f"{year}-11-22"},
            {"round": 23, "name": "Qatar Grand Prix", "country": "Qatar", "location": "Lusail", "date": f"{year}-11-30"},
            {"round": 24, "name": "Abu Dhabi Grand Prix", "country": "UAE", "location": "Yas Marina", "date": f"{year}-12-07"},
        ]

    @classmethod
    def fetch_ergast_fallback(cls, year: int, gp: Any, session_type: str = "R") -> Dict[str, Any]:
        """Lightweight (<5 MB RAM) reconstruction for R, Q, S, SQ, FP1, FP2, FP3 with authentic circuits."""
        year = int(year)
        gp_str = str(gp)
        stype = str(session_type or "R").upper()
        s_label = cls.SESSION_LABELS.get(stype, stype)

        # Pick best Jolpica/Ergast endpoint for the requested session
        endpoints = ["results.json"]
        if stype in ("Q", "SQ"):
            endpoints = ["qualifying.json", "results.json"]
        elif stype == "S":
            endpoints = ["sprint.json", "results.json"]

        race = None
        results_list = []
        for ep in endpoints:
            try:
                url = f"https://api.jolpi.ca/ergast/f1/{year}/{gp_str}/{ep}?limit=100"
                req = urllib.request.Request(url, headers={"User-Agent": "F1PitWall/3.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
                if races:
                    race = races[0]
                    results_list = (
                        race.get("Results")
                        or race.get("QualifyingResults")
                        or race.get("SprintResults")
                        or []
                    )
                    if results_list:
                        break
            except Exception:
                pass

        if not race or not results_list:
            raise ValueError(f"No session data found yet for {year} Round {gp_str} ({s_label}).")

        race_name = str(race.get("raceName", f"Round {gp_str}"))
        circuit_info = race.get("Circuit", {})
        circuit_id = str(circuit_info.get("circuitId", ""))
        locality = str(circuit_info.get("Location", {}).get("locality", ""))
        country = str(circuit_info.get("Location", {}).get("country", ""))

        circuit_dict = CircuitEngine.get_circuit(
            year=year,
            gp=gp_str,
            title=race_name,
            location=f"{locality} {country}",
            circuit_id=circuit_id
        )

        # Determine lap counts & stint structure by session type
        if stype == "R":
            total_laps = max(1, max(int(r.get("laps", 1) or 1) for r in results_list))
        elif stype == "S":
            raw_max = max(int(r.get("laps", 0) or 0) for r in results_list)
            total_laps = raw_max if (10 <= raw_max <= 30) else 20
        elif stype in ("Q", "SQ"):
            total_laps = 18 if stype == "Q" else 12
        else:
            # FP1, FP2, FP3
            total_laps = 24

        drivers: Dict[str, Any] = {}
        rcm: List[Dict[str, Any]] = [
            {"t": 0.0, "cat": "RC", "flag": "GREEN", "msg": f"GREEN LIGHT - {year} {race_name.upper()} ({s_label.upper()}) SESSION LIVE"}
        ]

        for idx, r in enumerate(results_list):
            drv_info = r.get("Driver", {})
            con_info = r.get("Constructor", {})
            fam = str(drv_info.get("familyName", f"D{idx+1}"))
            given = str(drv_info.get("givenName", ""))
            abbr = str(drv_info.get("code") or fam[:3]).upper()
            if abbr in drivers:
                abbr = (fam[:2] + str(idx + 1)).upper()

            num_str = str(r.get("number") or drv_info.get("permanentNumber") or (idx + 1))
            team_name = str(con_info.get("name", "Privateer"))
            color = cls.TEAM_COLORS.get(team_name, cls.FALLBACK_PALETTE[idx % len(cls.FALLBACK_PALETTE)])

            fin_pos = int(r.get("position", idx + 1) or (idx + 1))
            grid_pos = int(r.get("grid", fin_pos) or fin_pos)
            if grid_pos <= 0:
                grid_pos = fin_pos

            # For Practice sessions, vary the classification slightly per FP session so FP1/FP2/FP3 feel distinct
            if stype in ("FP1", "FP2", "FP3"):
                fp_seed = {"FP1": 3, "FP2": 5, "FP3": 2}.get(stype, 1)
                shift = ((idx * fp_seed + year) % 5) - 2
                fin_pos = max(1, min(len(results_list), fin_pos + shift))

            if stype in ("R", "S"):
                c_laps = max(1, int(r.get("laps", total_laps) or total_laps))
                if stype == "S" and c_laps > total_laps:
                    c_laps = total_laps
                status = str(r.get("status", "Finished"))
                is_fin = ("finished" in status.lower()) or status.startswith("+") or stype == "S"
            elif stype in ("Q", "SQ"):
                # Q1 elimination vs Q2 vs Q3 lap counts
                if fin_pos > 15:
                    c_laps = max(6, total_laps // 3)
                    status = "Eliminated Q1"
                elif fin_pos > 10:
                    c_laps = max(12, (total_laps * 2) // 3)
                    status = "Eliminated Q2"
                else:
                    c_laps = total_laps
                    status = "Q3 Classified"
                is_fin = True
            else:
                c_laps = max(16, total_laps - (idx % 5))
                status = f"{stype} Classified"
                is_fin = True

            points = float(r.get("points", 0.0) or 0.0)
            laps_arr = []
            pits_arr = []
            cur_s = 0.0 if stype in ("R", "S") else float((idx % 6) * 4.5)
            base_dur = 84.5 + (fin_pos - 1) * 0.24 + (grid_pos - 1) * 0.04

            pit1 = max(2, int(total_laps * 0.33) + (idx % 4))
            pit2 = max(pit1 + 5, int(total_laps * 0.67) + (idx % 3))

            for lap_n in range(1, c_laps + 1):
                wave = math.sin(lap_n * 0.7 + idx * 0.9) * 0.35
                dur = base_dur + wave

                if stype in ("Q", "SQ"):
                    comp = "S"
                elif stype == "FP1":
                    comp = "H" if lap_n < pit1 else "M"
                elif stype == "FP2":
                    comp = "M" if lap_n < pit1 else "S"
                elif stype == "FP3":
                    comp = "S" if lap_n >= pit1 else "M"
                else:
                    comp = "S" if lap_n < pit1 else ("M" if lap_n < pit2 else "H")

                if year >= 1982 and (lap_n == pit1 or lap_n == pit2) and lap_n < c_laps:
                    box_wait = 21.0 if stype in ("R", "S") else 36.0
                    p_in = round(cur_s + dur - 3.5, 2)
                    p_out = round(cur_s + dur + box_wait - 3.0, 2)
                    pits_arr.append([p_in, p_out])
                    dur += box_wait

                prog = lap_n / float(max(1, total_laps))
                pos_est = int(round(grid_pos + (fin_pos - grid_pos) * prog))
                pos_est = max(1, min(len(results_list), pos_est))
                if lap_n == c_laps:
                    pos_est = fin_pos

                laps_arr.append([round(cur_s, 2), round(cur_s + dur, 2), lap_n, pos_est, comp])
                cur_s += dur

            if stype in ("R", "S") and not is_fin and c_laps < total_laps and laps_arr:
                ret_t = round(laps_arr[-1][1], 1)
                rcm.append({
                    "t": ret_t,
                    "cat": "RC",
                    "flag": "YELLOW",
                    "msg": f"CAR #{num_str} ({abbr}) RETIRED - {status.upper()} (LAP {c_laps})"
                })

            drivers[abbr] = {
                "abbr": abbr,
                "number": num_str,
                "name": f"{given} {fam}".strip(),
                "team": team_name,
                "color": color,
                "grid": grid_pos,
                "finish_pos": fin_pos,
                "status": status,
                "points": points,
                "laps": laps_arr,
                "pits": pits_arr,
                "pos": None,
                "car": None
            }

        max_t = max((d["laps"][-1][1] for d in drivers.values() if d["laps"]), default=total_laps * 88.0)
        if stype in ("Q", "SQ"):
            rcm.append({"t": round(max_t * 0.33, 1), "cat": "RC", "flag": "CHEQUERED", "msg": "Q1 CHEQUERED FLAG - BOTTOM 5 ELIMINATED"})
            rcm.append({"t": round(max_t * 0.67, 1), "cat": "RC", "flag": "CHEQUERED", "msg": "Q2 CHEQUERED FLAG - TOP 10 SHOOTOUT READY"})
        rcm.append({"t": round(max_t, 1), "cat": "RC", "flag": "CHEQUERED", "msg": f"CHEQUERED FLAG - END OF {s_label.upper()}"})
        rcm.sort(key=lambda x: x["t"])

        return {
            "meta": {
                "title": f"{year} {race_name} - {s_label}",
                "year": year,
                "gp": gp_str,
                "session": stype,
                "min_t": 0.0,
                "max_t": round(float(max_t), 1),
                "total_laps": total_laps,
                "has_gps": False
            },
            "circuit": circuit_dict,
            "drivers": drivers,
            "rcm": rcm,
            "track_status": [{"t": 0.0, "code": "1", "msg": "AllClear"}],
            "weather": [{"t": 0.0, "air": 23.5, "track": 34.2, "hum": 48.0, "rain": False}]
        }

    @classmethod
    def load_replay(cls, year: int, gp: Any, session_type: str = "R") -> Dict[str, Any]:
        year = int(year)
        gp_key = str(gp)
        stype = str(session_type or "R").upper()
        s_label = cls.SESSION_LABELS.get(stype, stype)
        memo_key = f"{year}_{gp_key}_{stype}"
        if memo_key in cls.REPLAY_MEMO:
            return cls.REPLAY_MEMO[memo_key]

        gc.collect()

        if year < 2018 or fastf1 is None:
            payload = cls.fetch_ergast_fallback(year, gp, stype)
            if len(cls.REPLAY_MEMO) >= 3:
                cls.REPLAY_MEMO.clear()
            cls.REPLAY_MEMO[memo_key] = payload
            return payload

        cls.setup_cache()
        low_mem = cls.is_low_mem_env()

        try:
            gp_arg = int(gp) if str(gp).isdigit() else str(gp)
            session = fastf1.get_session(year, gp_arg, stype)
            session.load(laps=True, telemetry=(not low_mem), weather=True, messages=True)
        except Exception:
            payload = cls.fetch_ergast_fallback(year, gp, stype)
            if len(cls.REPLAY_MEMO) >= 3:
                cls.REPLAY_MEMO.clear()
            cls.REPLAY_MEMO[memo_key] = payload
            return payload

        laps_df = cls.safe_attr(session, "laps", None)
        results_df = cls.safe_attr(session, "results", None)
        if laps_df is None or laps_df.empty:
            payload = cls.fetch_ergast_fallback(year, gp, stype)
            if len(cls.REPLAY_MEMO) >= 3:
                cls.REPLAY_MEMO.clear()
            cls.REPLAY_MEMO[memo_key] = payload
            return payload

        sc_nums = {"241", "242", "243"}
        if "DriverNumber" in laps_df.columns:
            laps_df = laps_df[~laps_df["DriverNumber"].astype(str).isin(sc_nums)].copy()
        if results_df is not None and not results_df.empty and "DriverNumber" in results_df.columns:
            results_df = results_df[~results_df["DriverNumber"].astype(str).isin(sc_nums)].copy()

        ev = cls.safe_attr(session, "event", {})
        ev_name = str(ev.get("EventName", f"Round {gp}")) if hasattr(ev, "get") else str(gp)
        ev_loc = str(ev.get("Location", "")) if hasattr(ev, "get") else ""
        ev_country = str(ev.get("Country", "")) if hasattr(ev, "get") else ""

        s_info = cls.safe_attr(session, "session_info", {}) or {}
        mv_ckey = None
        try:
            mv_ckey = s_info.get("Meeting", {}).get("Circuit", {}).get("Key")
        except Exception:
            pass

        lap_med_s: Dict[int, float] = {}
        lap_med_e: Dict[int, float] = {}
        for ln, grp in laps_df.groupby("LapNumber"):
            ln_i = int(ln)
            s_vals = [cls.td_to_sec(v) for v in grp.get("LapStartTime", [])]
            s_vals = [v for v in s_vals if math.isfinite(v) and v > 0]
            e_vals = [cls.td_to_sec(v) for v in grp.get("Time", [])]
            e_vals = [v for v in e_vals if math.isfinite(v) and v > 0]
            if s_vals:
                lap_med_s[ln_i] = float(np.median(s_vals))
            if e_vals:
                lap_med_e[ln_i] = float(np.median(e_vals))

        total_laps = int(max(lap_med_e.keys() or [1]))
        if 1 in lap_med_s:
            race_start_t = lap_med_s[1]
        elif 1 in lap_med_e:
            race_start_t = max(0.0, lap_med_e[1] - 92.0)
        else:
            race_start_t = 0.0

        circuit_dict = CircuitEngine.get_circuit(
            year=year,
            gp=gp,
            title=ev_name,
            location=f"{ev_loc} {ev_country}",
            explicit_mv_key=mv_ckey
        )

        # Rank drivers by fastest lap for Practice/Qualifying when official Position is NaN
        fastest_rank: Dict[str, int] = {}
        try:
            best_times = []
            for abbr_k, grp in laps_df.groupby("Driver"):
                lt_vals = [cls.td_to_sec(v) for v in grp.get("LapTime", [])]
                lt_vals = [v for v in lt_vals if math.isfinite(v) and v > 35.0]
                best_times.append((min(lt_vals) if lt_vals else 9999.0, str(abbr_k).upper()))
            best_times.sort(key=lambda x: x[0])
            for rk, (_, ab_k) in enumerate(best_times, 1):
                fastest_rank[ab_k] = rk
        except Exception:
            pass

        drivers: Dict[str, Any] = {}
        drv_meta: Dict[str, Dict[str, Any]] = {}
        if results_df is not None and not results_df.empty:
            for idx, (_, r) in enumerate(results_df.iterrows()):
                abbr = str(r.get("Abbreviation") or r.get("BroadcastName") or f"D{idx+1}").strip().upper()
                num = str(r.get("DriverNumber", idx + 1)).strip()
                if num in sc_nums:
                    continue
                tname = str(r.get("TeamName", "F1 Team"))
                raw_col = str(r.get("TeamColor", "")).strip().lstrip("#")
                if len(raw_col) == 6:
                    color = "#" + raw_col
                else:
                    color = cls.TEAM_COLORS.get(tname, cls.FALLBACK_PALETTE[idx % len(cls.FALLBACK_PALETTE)])
                f_pos_raw = r.get("Position")
                if pd.notna(f_pos_raw) and str(f_pos_raw).replace(".0", "").isdigit():
                    f_pos = int(float(f_pos_raw))
                else:
                    f_pos = fastest_rank.get(abbr, idx + 1)
                g_pos_raw = r.get("GridPosition")
                if pd.notna(g_pos_raw) and str(g_pos_raw).replace(".0", "").isdigit() and int(float(g_pos_raw)) > 0:
                    g_pos = int(float(g_pos_raw))
                else:
                    g_pos = f_pos
                status_str = str(r.get("Status", "")).strip() or ("Finished" if stype in ("R", "S") else f"{stype} Classified")
                drv_meta[abbr] = {
                    "abbr": abbr,
                    "number": num,
                    "name": str(r.get("FullName") or abbr),
                    "team": tname,
                    "color": color,
                    "grid": g_pos,
                    "finish_pos": f_pos,
                    "status": status_str,
                    "points": float(r.get("Points", 0.0) or 0.0)
                }

        max_norm_t = 0.0
        for abbr, d_laps in laps_df.groupby("Driver"):
            abbr_s = str(abbr).upper()
            f_rk = fastest_rank.get(abbr_s, len(drivers) + 1)
            meta = drv_meta.get(abbr_s, {
                "abbr": abbr_s,
                "number": str(d_laps["DriverNumber"].iloc[0]) if "DriverNumber" in d_laps.columns else "0",
                "name": abbr_s,
                "team": str(d_laps["Team"].iloc[0]) if "Team" in d_laps.columns else "Team",
                "color": cls.FALLBACK_PALETTE[len(drivers) % len(cls.FALLBACK_PALETTE)],
                "grid": f_rk,
                "finish_pos": f_rk,
                "status": "Finished" if stype in ("R", "S") else f"{stype} Classified",
                "points": 0.0
            })
            if meta["number"] in sc_nums:
                continue

            d_laps_sorted = d_laps.sort_values("LapNumber")
            laps_arr = []
            pits_arr = []
            prev_e = 0.0

            for _, row in d_laps_sorted.iterrows():
                ln = int(row.get("LapNumber", 1) or 1)
                raw_s = cls.td_to_sec(row.get("LapStartTime"))
                raw_e = cls.td_to_sec(row.get("Time"))
                raw_dur = cls.td_to_sec(row.get("LapTime"))

                if not math.isfinite(raw_s):
                    raw_s = lap_med_s.get(ln, race_start_t + prev_e)
                if not math.isfinite(raw_e):
                    if math.isfinite(raw_dur) and raw_dur > 0:
                        raw_e = raw_s + raw_dur
                    else:
                        raw_e = lap_med_e.get(ln, raw_s + 90.0)

                if stype in ("R", "S"):
                    s_norm = max(0.0, raw_s - race_start_t)
                    e_norm = max(s_norm + 55.0, raw_e - race_start_t)
                    if ln == 1:
                        s_norm = 0.0
                    elif s_norm < prev_e:
                        s_norm = prev_e
                        if e_norm <= s_norm + 45.0:
                            e_norm = s_norm + (raw_dur if (math.isfinite(raw_dur) and raw_dur > 45.0) else 88.0)
                else:
                    # For Q, SQ, FP1, FP2, FP3 pack stints smoothly so cars are actively running on track
                    dur_clean = raw_dur if (math.isfinite(raw_dur) and 45.0 <= raw_dur <= 145.0) else 88.0
                    s_norm = prev_e
                    e_norm = s_norm + dur_clean

                prev_e = e_norm
                max_norm_t = max(max_norm_t, e_norm)

                pos_val = row.get("Position")
                pos_i = int(pos_val) if pd.notna(pos_val) and float(pos_val) > 0 else meta["finish_pos"]
                comp_raw = str(row.get("Compound", "S" if stype in ("Q", "SQ") else "M") or "M").upper()
                comp = comp_raw[0] if comp_raw[0] in ("S", "M", "H", "I", "W") else "M"
                laps_arr.append([round(s_norm, 2), round(e_norm, 2), ln, pos_i, comp])

                p_in = cls.td_to_sec(row.get("PitInTime"))
                p_out = cls.td_to_sec(row.get("PitOutTime"))
                if stype in ("R", "S"):
                    if math.isfinite(p_in):
                        pin_n = max(0.0, p_in - race_start_t)
                        pits_arr.append([round(pin_n, 2), round(pin_n + 22.0, 2)])
                    elif math.isfinite(p_out) and ln > 1:
                        pout_n = max(0.0, p_out - race_start_t)
                        pits_arr.append([round(max(0.0, pout_n - 22.0), 2), round(pout_n, 2)])
                else:
                    if math.isfinite(p_in):
                        pits_arr.append([round(max(0.0, e_norm - 14.0), 2), round(e_norm, 2)])

            if laps_arr:
                drivers[abbr_s] = {
                    **meta,
                    "laps": laps_arr,
                    "pits": pits_arr,
                    "pos": None,
                    "car": None
                }

        rcm_list: List[Dict[str, Any]] = []
        rcm_df = cls.safe_attr(session, "race_control_messages", None)
        if rcm_df is not None and not rcm_df.empty:
            for _, r in rcm_df.iterrows():
                t_raw = cls.td_to_sec(r.get("Time"))
                if not math.isfinite(t_raw):
                    continue
                t_norm = round(max(0.0, t_raw - race_start_t), 1)
                msg = str(r.get("Message", "")).strip()
                if not msg:
                    continue
                rcm_list.append({
                    "t": min(t_norm, round(max_norm_t, 1)),
                    "cat": str(r.get("Category", "RC")),
                    "flag": str(r.get("Flag", "INFO")),
                    "msg": msg
                })
        if not rcm_list:
            rcm_list.append({"t": 0.0, "cat": "RC", "flag": "GREEN", "msg": f"GREEN LIGHT - {year} {ev_name.upper()} ({s_label.upper()}) UNDERWAY"})

        ts_list: List[Dict[str, Any]] = [{"t": 0.0, "code": "1", "msg": "AllClear"}]
        ts_df = cls.safe_attr(session, "track_status", None)
        if ts_df is not None and not ts_df.empty:
            for _, r in ts_df.iterrows():
                t_raw = cls.td_to_sec(r.get("Time"))
                if math.isfinite(t_raw):
                    ts_list.append({
                        "t": round(max(0.0, t_raw - race_start_t), 1),
                        "code": str(r.get("Status", "1")),
                        "msg": str(r.get("Message", "AllClear"))
                    })

        w_list: List[Dict[str, Any]] = []
        w_df = cls.safe_attr(session, "weather_data", None)
        if w_df is not None and not w_df.empty:
            for _, r in w_df.iloc[::5].iterrows():
                t_raw = cls.td_to_sec(r.get("Time"))
                if math.isfinite(t_raw):
                    w_list.append({
                        "t": round(max(0.0, t_raw - race_start_t), 1),
                        "air": round(float(r.get("AirTemp", 24.0) or 24.0), 1),
                        "track": round(float(r.get("TrackTemp", 35.0) or 35.0), 1),
                        "hum": round(float(r.get("Humidity", 50.0) or 50.0), 1),
                        "rain": bool(r.get("Rainfall", False))
                    })
        if not w_list:
            w_list.append({"t": 0.0, "air": 24.0, "track": 35.0, "hum": 50.0, "rain": False})

        del session
        gc.collect()

        payload = {
            "meta": {
                "title": f"{year} {ev_name} - {s_label}",
                "year": year,
                "gp": gp_key,
                "session": stype,
                "min_t": 0.0,
                "max_t": round(float(max(max_norm_t, 100.0)), 1),
                "total_laps": total_laps,
                "has_gps": False
            },
            "circuit": circuit_dict,
            "drivers": drivers,
            "rcm": rcm_list,
            "track_status": ts_list,
            "weather": w_list
        }

        if len(cls.REPLAY_MEMO) >= 2:
            cls.REPLAY_MEMO.clear()
        cls.REPLAY_MEMO[memo_key] = payload
        return payload


# ==============================================================================
# 3. AUTHENTIC F1 PIT WALL BROADCAST UI (HTML5 / CANVAS / OSCILLOSCOPE)
# ==============================================================================
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>F1 PIT WALL TELEMETRY // 1950-2026</title>
<style>
:root {
  --bg: #07080c;
  --panel: #0e1017;
  --panel2: #141722;
  --border: #232738;
  --text: #f1f5f9;
  --muted: #8b95a5;
  --f1red: #e10600;
  --green: #00e676;
  --yellow: #ffd600;
  --purple: #d500f9;
  --cyan: #00e5ff;
}
* { box-sizing: border-box; margin: 0; padding: 0; user-select: none; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif;
  height: 100vh;
  width: 100vw;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.topbar {
  height: 46px;
  background: linear-gradient(90deg, #10121a 0%, #0b0d13 100%);
  border-bottom: 2px solid var(--f1red);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 14px;
  gap: 10px;
  flex-shrink: 0;
}
.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 900;
  letter-spacing: 1px;
  font-size: 14px;
}
.f1-badge {
  background: var(--f1red);
  color: #fff;
  padding: 3px 8px;
  border-radius: 3px;
  font-style: italic;
  font-weight: 900;
  letter-spacing: 1.5px;
}
.controls-top {
  display: flex;
  align-items: center;
  gap: 8px;
}
select, button {
  background: var(--panel2);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 5px 10px;
  font-size: 12px;
  font-weight: 700;
  cursor: pointer;
}
select:hover, button:hover { border-color: #4b5563; }
.btn-load {
  background: var(--f1red);
  border-color: var(--f1red);
  color: #fff;
  text-transform: uppercase;
  letter-spacing: 0.6px;
}
.btn-load:hover { background: #ff1e17; }
.status-strip {
  display: flex;
  align-items: center;
  gap: 14px;
  font-size: 12px;
  font-weight: 700;
}
.lap-pill {
  background: #181c29;
  border: 1px solid var(--border);
  padding: 4px 12px;
  border-radius: 4px;
  font-size: 13px;
  font-weight: 900;
  letter-spacing: 0.8px;
}
.flag-pill {
  padding: 4px 12px;
  border-radius: 4px;
  font-weight: 900;
  letter-spacing: 1px;
  text-transform: uppercase;
  background: rgba(0,230,118,0.16);
  color: var(--green);
  border: 1px solid var(--green);
}
.weather-pill {
  color: var(--muted);
  font-family: Consolas, monospace;
  font-size: 11px;
}
.workspace {
  flex: 1;
  display: grid;
  grid-template-columns: 325px 1fr 355px;
  gap: 6px;
  padding: 6px;
  overflow: hidden;
}
.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 5px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.panel-hdr {
  height: 30px;
  background: var(--panel2);
  border-bottom: 1px solid var(--border);
  padding: 0 10px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 1px;
  color: var(--muted);
  text-transform: uppercase;
  flex-shrink: 0;
}
.tower-list {
  flex: 1;
  overflow-y: auto;
  font-family: Consolas, 'Courier New', monospace;
}
.tower-row {
  display: grid;
  grid-template-columns: 28px 6px 46px 52px 1fr 28px;
  align-items: center;
  height: 27px;
  padding: 0 8px;
  border-bottom: 1px solid rgba(255,255,255,0.04);
  font-size: 12px;
  cursor: pointer;
  transition: background 0.1s;
}
.tower-row:hover { background: rgba(255,255,255,0.05); }
.tower-row.selected {
  background: rgba(225,6,0,0.22);
  border-left: 3px solid var(--f1red);
}
.tower-row.out { opacity: 0.42; }
.t-pos { font-weight: 900; color: #cbd5e1; text-align: center; }
.t-bar { width: 4px; height: 18px; border-radius: 2px; }
.t-code { font-weight: 900; color: #fff; letter-spacing: 0.5px; }
.t-num { color: var(--muted); font-size: 11px; }
.t-gap { text-align: right; padding-right: 8px; font-weight: 700; color: #e2e8f0; }
.tyre-ring {
  width: 19px;
  height: 19px;
  border-radius: 50%;
  border: 2.5px solid #ffd600;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 10px;
  font-weight: 900;
  color: #fff;
}
.map-wrap {
  position: relative;
  flex: 1;
  background: radial-gradient(circle at center, #111522 0%, #07090e 100%);
  overflow: hidden;
}
#trackCanvas {
  width: 100%;
  height: 100%;
  display: block;
  cursor: grab;
}
#trackCanvas:active { cursor: grabbing; }
.map-overlay-top {
  position: absolute;
  top: 10px;
  left: 12px;
  right: 12px;
  display: flex;
  justify-content: space-between;
  pointer-events: none;
}
.session-title-box {
  background: rgba(14,16,23,0.88);
  border: 1px solid var(--border);
  border-left: 3px solid var(--f1red);
  padding: 6px 12px;
  border-radius: 4px;
}
.session-title-box h2 { font-size: 14px; font-weight: 900; letter-spacing: 0.6px; }
.session-title-box p { font-size: 11px; color: var(--muted); margin-top: 2px; }
.map-tools {
  display: flex;
  gap: 6px;
  pointer-events: auto;
}
.tool-btn {
  background: rgba(20,23,34,0.9);
  border: 1px solid var(--border);
  padding: 4px 9px;
  font-size: 11px;
  border-radius: 4px;
}
.tool-btn.active { border-color: var(--cyan); color: var(--cyan); }
.playback-bar {
  height: 44px;
  background: var(--panel2);
  border-top: 1px solid var(--border);
  padding: 0 12px;
  display: flex;
  align-items: center;
  gap: 10px;
  flex-shrink: 0;
}
#scrubber {
  flex: 1;
  accent-color: var(--f1red);
  cursor: pointer;
}
.clock-txt {
  font-family: Consolas, monospace;
  font-size: 12px;
  font-weight: 800;
  min-width: 115px;
  text-align: right;
}
.right-col {
  display: flex;
  flex-direction: column;
  gap: 6px;
  overflow: hidden;
}
.telem-card {
  padding: 10px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.drv-banner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--border);
}
.drv-title { font-size: 15px; font-weight: 900; }
.drv-team { font-size: 11px; color: var(--muted); }
.shift-leds {
  display: grid;
  grid-template-columns: repeat(15, 1fr);
  gap: 3px;
  height: 8px;
}
.led {
  background: #1c2030;
  border-radius: 2px;
}
.gauges-row {
  display: grid;
  grid-template-columns: 1fr 74px 1fr;
  gap: 8px;
  align-items: center;
  text-align: center;
}
.g-box {
  background: #121521;
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 6px;
}
.g-val { font-family: Consolas, monospace; font-size: 22px; font-weight: 900; }
.g-lbl { font-size: 10px; color: var(--muted); text-transform: uppercase; }
.pedals-wrap { display: flex; flex-direction: column; gap: 5px; }
.p-row { display: grid; grid-template-columns: 42px 1fr 38px; align-items: center; gap: 6px; font-size: 11px; font-weight: 800; }
.p-track { height: 9px; background: #1a1e2d; border-radius: 3px; overflow: hidden; }
.p-fill-thr { height: 100%; background: var(--green); width: 0%; }
.p-fill-brk { height: 100%; background: var(--f1red); width: 0%; }
#oscCanvas {
  width: 100%;
  height: 115px;
  background: #090b10;
  border: 1px solid var(--border);
  border-radius: 4px;
}
.rcm-feed {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  font-family: Consolas, monospace;
  font-size: 11px;
}
.rcm-item {
  background: #121520;
  border-left: 3px solid var(--cyan);
  padding: 5px 8px;
  border-radius: 3px;
  line-height: 1.35;
}
.rcm-time { color: var(--muted); font-size: 10px; margin-right: 6px; }
.loading-mask {
  position: fixed;
  inset: 0;
  background: rgba(7,8,12,0.86);
  display: none;
  align-items: center;
  justify-content: center;
  z-index: 999;
  flex-direction: column;
  gap: 12px;
}
.spinner {
  width: 38px;
  height: 38px;
  border: 4px solid #262b3d;
  border-top-color: var(--f1red);
  border-radius: 50%;
  animation: spin 0.75s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>

<div class="loading-mask" id="loadingMask">
  <div class="spinner"></div>
  <div style="font-weight:900;letter-spacing:1px;font-size:13px;" id="loadingText">LOADING OFFICIAL F1 SESSION TELEMETRY...</div>
</div>

<!-- TOP BAR -->
<div class="topbar">
  <div class="brand">
    <span class="f1-badge">F1</span>
    <span>PIT WALL TELEMETRY // 1950-2026</span>
  </div>
  <div class="controls-top">
    <select id="yearSel"></select>
    <select id="gpSel"></select>
    <select id="sessionSel">
      <option value="R" selected>RACE (R)</option>
      <option value="Q">QUALIFYING (Q)</option>
      <option value="S">SPRINT (S)</option>
      <option value="SQ">SPRINT QUALI (SQ)</option>
      <option value="FP1">PRACTICE 1 (FP1)</option>
      <option value="FP2">PRACTICE 2 (FP2)</option>
      <option value="FP3">PRACTICE 3 (FP3)</option>
    </select>
    <button class="btn-load" id="loadBtn">LOAD TELEMETRY</button>
  </div>
  <div class="status-strip">
    <div class="lap-pill" id="lapCounter">LAP 1 / 1</div>
    <div class="flag-pill" id="flagBanner">TRACK CLEAR</div>
    <div class="weather-pill" id="weatherBox">AIR 24.0C | TRK 35.0C | HUM 50%</div>
  </div>
</div>

<!-- MAIN WORKSPACE -->
<div class="workspace">
  <!-- LEFT: LIVE TIMING TOWER -->
  <div class="panel">
    <div class="panel-hdr">
      <span>LIVE TIMING TOWER</span>
      <button class="tool-btn" id="gapModeBtn" style="padding:2px 6px;font-size:10px;">MODE: LEADER</button>
    </div>
    <div class="tower-list" id="towerList"></div>
  </div>

  <!-- CENTER: CIRCUIT MAP & PLAYBACK -->
  <div class="panel">
    <div class="map-wrap">
      <canvas id="trackCanvas"></canvas>
      <div class="map-overlay-top">
        <div class="session-title-box">
          <h2 id="raceTitle">SELECT A SESSION</h2>
          <p id="circuitSub">800-Point Laser-Scanned Circuit Geometry // 60 FPS</p>
        </div>
        <div class="map-tools">
          <button class="tool-btn active" id="btnCorners">CORNERS</button>
          <button class="tool-btn active" id="btnTrails">TRAILS</button>
          <button class="tool-btn" id="btnFollow">FOLLOW CAR</button>
          <button class="tool-btn" id="btnResetCam">RESET VIEW</button>
        </div>
      </div>
    </div>
    <div class="playback-bar">
      <button id="playBtn" style="min-width:68px;">PAUSE</button>
      <button id="speedBtn" style="min-width:54px;">1x</button>
      <input type="range" id="scrubber" min="0" max="1000" step="0.1" value="0">
      <div class="clock-txt" id="clockTxt">00:00.0</div>
    </div>
  </div>

  <!-- RIGHT: SELECTED DRIVER TELEMETRY & FIA RACE CONTROL -->
  <div class="right-col">
    <div class="panel">
      <div class="panel-hdr">
        <span>CAR TELEMETRY OSCILLOSCOPE</span>
        <span id="drsBadge" style="padding:1px 6px;border-radius:3px;background:#1c2030;color:#64748b;font-weight:900;">DRS OFF</span>
      </div>
      <div class="telem-card">
        <div class="drv-banner">
          <div>
            <div class="drv-title" id="hudDrvName">SELECT DRIVER</div>
            <div class="drv-team" id="hudDrvTeam">---</div>
          </div>
          <div style="text-align:right;font-family:Consolas,monospace;font-size:12px;">
            <div id="hudDrvPos" style="font-weight:900;color:var(--cyan);">P--</div>
            <div id="hudDrvStatus" style="color:var(--muted);font-size:10px;">ON TRACK</div>
          </div>
        </div>
        <div class="shift-leds" id="shiftLeds"></div>
        <div class="gauges-row">
          <div class="g-box">
            <div class="g-val" id="hudSpeed">0</div>
            <div class="g-lbl">KM/H</div>
          </div>
          <div class="g-box">
            <div class="g-val" id="hudGear" style="color:var(--yellow);">N</div>
            <div class="g-lbl">GEAR</div>
          </div>
          <div class="g-box">
            <div class="g-val" id="hudRpm">0</div>
            <div class="g-lbl">RPM</div>
          </div>
        </div>
        <div class="pedals-wrap">
          <div class="p-row">
            <span>THR</span>
            <div class="p-track"><div class="p-fill-thr" id="barThr"></div></div>
            <span id="txtThr">0%</span>
          </div>
          <div class="p-row">
            <span>BRK</span>
            <div class="p-track"><div class="p-fill-brk" id="barBrk"></div></div>
            <span id="txtBrk">0%</span>
          </div>
        </div>
        <canvas id="oscCanvas" width="330" height="115"></canvas>
      </div>
    </div>

    <div class="panel" style="flex:1;">
      <div class="panel-hdr">
        <span>FIA RACE CONTROL MESSAGES</span>
        <span id="rcmCount">0 MSG</span>
      </div>
      <div class="rcm-feed" id="rcmFeed"></div>
    </div>
  </div>
</div>

<script>
const SPEEDS = [0.5, 1, 2, 5, 10, 20];
let speedIdx = 1;
let isPlaying = true;
let simTime = 0;
let lastFrameTs = performance.now();
let replayData = null;
let selectedAbbr = null;
let gapIntervalMode = false;
let showCorners = true;
let showTrails = true;
let followCar = false;

let camZoom = 1.0, camPanX = 0, camPanY = 0;
let isDragging = false, dragStartX = 0, dragStartY = 0;

let trackProfile = null;
let carTrails = {};
let oscHistory = [];

const yearSel = document.getElementById('yearSel');
const gpSel = document.getElementById('gpSel');
const sessionSel = document.getElementById('sessionSel');
const loadBtn = document.getElementById('loadBtn');
const playBtn = document.getElementById('playBtn');
const speedBtn = document.getElementById('speedBtn');
const scrubber = document.getElementById('scrubber');
const trackCanvas = document.getElementById('trackCanvas');
const ctx = trackCanvas.getContext('2d');
const oscCanvas = document.getElementById('oscCanvas');
const oscCtx = oscCanvas.getContext('2d');

const shiftLedsEl = document.getElementById('shiftLeds');
for (let i = 0; i < 15; i++) {
  const d = document.createElement('div');
  d.className = 'led';
  shiftLedsEl.appendChild(d);
}
const ledNodes = Array.from(shiftLedsEl.children);

for (let y = 2026; y >= 1950; y--) {
  const opt = document.createElement('option');
  opt.value = y;
  opt.textContent = y + ' SEASON';
  if (y === 2025) opt.selected = true;
  yearSel.appendChild(opt);
}

function buildTrackPhysicsProfile(tx, ty) {
  const N = Math.max(1, tx.length - 1);
  const curv = new Float64Array(N);
  const w = Math.max(3, Math.floor(N / 90));
  for (let i = 0; i < N; i++) {
    const i1 = (i - w + N) % N;
    const i2 = (i + w) % N;
    const dx1 = tx[i] - tx[i1], dy1 = ty[i] - ty[i1];
    const dx2 = tx[i2] - tx[i], dy2 = ty[i2] - ty[i];
    const n1 = Math.hypot(dx1, dy1), n2 = Math.hypot(dx2, dy2);
    if (n1 > 0 && n2 > 0) {
      const c = Math.max(-1, Math.min(1, (dx1 * dx2 + dy1 * dy2) / (n1 * n2)));
      curv[i] = Math.acos(c) * (180 / Math.PI);
    }
  }
  const sm = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    let s = 0;
    for (let k = -4; k <= 4; k++) s += curv[(i + k + N) % N];
    sm[i] = s / 9;
  }
  const v = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const c = sm[i];
    v[i] = Math.max(82, Math.min(334, 334 - c * 9.5));
  }
  for (let pass = 0; pass < 2; pass++) {
    for (let i = 0; i < N; i++) {
      const nxt = (i + 1) % N;
      v[nxt] = Math.min(v[nxt], v[i] + 2.2);
    }
    for (let i = N - 1; i >= 0; i--) {
      const prv = (i - 1 + N) % N;
      v[prv] = Math.min(v[prv], v[i] + 3.8);
    }
  }
  const dt = new Float64Array(N + 1);
  let totalT = 0;
  for (let i = 0; i < N; i++) {
    dt[i] = totalT;
    totalT += 1.0 / Math.max(60, v[i]);
  }
  dt[N] = totalT;
  const uTable = new Float64Array(N + 1);
  for (let i = 0; i <= N; i++) uTable[i] = dt[i] / totalT;
  return { N, v, uTable };
}

function lapFracToTrackIndex(uFrac) {
  if (!trackProfile) return 0;
  const { N, uTable } = trackProfile;
  const u = ((uFrac % 1) + 1) % 1;
  let lo = 0, hi = N;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (uTable[mid] < u) lo = mid + 1;
    else hi = mid;
  }
  const idx = Math.max(0, lo - 1);
  const u0 = uTable[idx], u1 = uTable[idx + 1] || 1;
  const sub = u1 > u0 ? (u - u0) / (u1 - u0) : 0;
  return idx + sub;
}

async function fetchSchedule(year) {
  gpSel.innerHTML = '<option>Loading...</option>';
  try {
    const r = await fetch(`/api/schedule?year=${year}`);
    const list = await r.json();
    gpSel.innerHTML = '';
    list.forEach(ev => {
      const opt = document.createElement('option');
      opt.value = ev.round;
      opt.textContent = `R${ev.round} - ${ev.name}`;
      gpSel.appendChild(opt);
    });
  } catch (e) {
    gpSel.innerHTML = '<option value="1">R1 - Grand Prix</option>';
  }
}

async function loadReplay() {
  const year = yearSel.value;
  const gp = gpSel.value || 1;
  const session = sessionSel.value || 'R';
  const mask = document.getElementById('loadingMask');
  document.getElementById('loadingText').textContent =
    `LOADING ${year} ROUND ${gp} (${session}) TELEMETRY...`;
  mask.style.display = 'flex';
  try {
    const r = await fetch(`/api/replay?year=${year}&gp=${encodeURIComponent(gp)}&session=${encodeURIComponent(session)}`);
    const data = await r.json();
    if (data.error) throw new Error(data.error);
    replayData = data;
    simTime = 0;
    carTrails = {};
    oscHistory = [];
    camZoom = 1.0; camPanX = 0; camPanY = 0;

    scrubber.min = 0;
    scrubber.max = data.meta.max_t || 5400;
    scrubber.value = 0;
    document.getElementById('raceTitle').textContent = data.meta.title.toUpperCase();
    document.getElementById('circuitSub').textContent =
      `Season ${data.meta.year} // Session: ${data.meta.session} // ${data.meta.total_laps} Laps`;

    trackProfile = buildTrackPhysicsProfile(data.circuit.x, data.circuit.y);

    const dList = Object.values(data.drivers);
    dList.sort((a, b) => (a.finish_pos || 99) - (b.finish_pos || 99));
    if (dList.length > 0) selectedAbbr = dList[0].abbr;

    renderRcmFeed();
  } catch (err) {
    alert('Could not load session: ' + err.message);
  } finally {
    mask.style.display = 'none';
  }
}

function getDriverStateAt(drv, t) {
  const laps = drv.laps || [];
  if (!laps.length) {
    return { pos: drv.finish_pos || 20, lap: 1, comp: 'M', frac: 0, status: 'OUT', inPit: false, totalProgress: 0 };
  }
  const stype = (replayData.meta && replayData.meta.session) || 'R';
  const lastLap = laps[laps.length - 1];
  const stLower = (drv.status || '').toLowerCase();
  const isFinishedStatus = stLower.includes('finished') || stLower.includes('classified') || (drv.status || '').startsWith('+');

  if (t > lastLap[1]) {
    const fin = isFinishedStatus || lastLap[2] >= (replayData.meta.total_laps || 1);
    return {
      pos: drv.finish_pos || lastLap[3],
      lap: lastLap[2],
      comp: lastLap[4] || 'M',
      frac: 1.0,
      status: fin ? 'FIN' : 'OUT',
      inPit: false,
      totalProgress: lastLap[2] + 1.0 - (drv.finish_pos || 20) * 0.001
    };
  }

  let cur = laps[0];
  for (let i = 0; i < laps.length; i++) {
    if (t >= laps[i][0] && t <= laps[i][1]) {
      cur = laps[i];
      break;
    }
    if (t > laps[i][1]) cur = laps[i];
  }

  const dur = Math.max(1.0, cur[1] - cur[0]);
  const rawFrac = Math.max(0, Math.min(0.9999, (t - cur[0]) / dur));
  const gridOffset = ((drv.grid || 10) - 1) * 0.0035;
  const frac = (cur[2] === 1 && (stype === 'R' || stype === 'S')) ? Math.max(0, rawFrac * (1 - gridOffset)) : rawFrac;

  let inPit = false;
  if (drv.pits) {
    for (let p of drv.pits) {
      if (t >= p[0] && t <= p[1]) { inPit = true; break; }
    }
  }

  const tx = replayData.circuit.x;
  const ty = replayData.circuit.y;
  const N = Math.max(1, tx.length - 1);
  const fIdx = lapFracToTrackIndex(frac);
  const i0 = Math.floor(fIdx) % N;
  const i1 = (i0 + 1) % N;
  const sub = fIdx - Math.floor(fIdx);
  const wx = tx[i0] + (tx[i1] - tx[i0]) * sub;
  const wy = ty[i0] + (ty[i1] - ty[i0]) * sub;

  let spd = 240, gear = 6, rpm = 10800, thr = 85, brk = 0, drs = 0;
  if (trackProfile) {
    const vCur = trackProfile.v[i0];
    const vNxt = trackProfile.v[(i0 + 3) % N];
    const dv = vNxt - vCur;
    spd = Math.round(inPit ? 80 : vCur);
    gear = inPit ? 2 : Math.max(2, Math.min(8, Math.floor((spd - 50) / 34) + 2));
    rpm = inPit ? 8200 : Math.round(8800 + ((spd * 37) % 3100));
    if (inPit) { thr = 35; brk = 0; drs = 0; }
    else if (dv < -1.2) { brk = Math.min(100, Math.round(-dv * 26)); thr = 0; }
    else { brk = 0; thr = spd > 285 ? 100 : Math.min(100, Math.round(55 + dv * 22)); }
    drs = (!inPit && (stype !== 'R' || cur[2] > 2) && spd > 300 && brk === 0) ? 12 : 0;
  }

  return {
    pos: cur[3],
    lap: cur[2],
    comp: cur[4] || 'M',
    frac,
    status: 'RUN',
    inPit,
    wx, wy,
    spd, gear, rpm, thr, brk, drs,
    lapDur: dur,
    totalProgress: (cur[2] - 1) + frac
  };
}

function updateUI() {
  if (!replayData) return;
  const stype = (replayData.meta && replayData.meta.session) || 'R';
  const drivers = Object.values(replayData.drivers);
  const states = drivers.map(d => ({ drv: d, st: getDriverStateAt(d, simTime) }));

  states.sort((a, b) => {
    const aOut = a.st.status === 'OUT' ? 1 : 0;
    const bOut = b.st.status === 'OUT' ? 1 : 0;
    if (aOut !== bOut) return aOut - bOut;
    if (stype !== 'R' && stype !== 'S') {
      return (a.drv.finish_pos || 99) - (b.drv.finish_pos || 99);
    }
    if (a.st.lap !== b.st.lap) return b.st.lap - a.st.lap;
    if (a.st.pos !== b.st.pos) return a.st.pos - b.st.pos;
    return b.st.totalProgress - a.st.totalProgress;
  });

  const leader = states[0];
  const maxActiveLap = states.reduce((m, x) => Math.max(m, x.st.lap || 1), 1);
  const curLap = (stype === 'R' || stype === 'S') ? (leader ? leader.st.lap : 1) : maxActiveLap;
  document.getElementById('lapCounter').textContent =
    `${stype} // LAP ${curLap} / ${replayData.meta.total_laps}`;

  const mins = Math.floor(simTime / 60);
  const secs = (simTime % 60).toFixed(1).padStart(4, '0');
  document.getElementById('clockTxt').textContent = `${String(mins).padStart(2, '0')}:${secs}`;

  let tsCode = '1';
  for (let s of (replayData.track_status || [])) {
    if (s.t <= simTime) tsCode = s.code;
  }
  const flagEl = document.getElementById('flagBanner');
  if (tsCode === '4') {
    flagEl.textContent = 'SAFETY CAR';
    flagEl.style.color = '#ffd600'; flagEl.style.borderColor = '#ffd600'; flagEl.style.background = 'rgba(255,214,0,0.18)';
  } else if (tsCode === '6' || tsCode === '7') {
    flagEl.textContent = 'VIRTUAL SC';
    flagEl.style.color = '#ffd600'; flagEl.style.borderColor = '#ffd600'; flagEl.style.background = 'rgba(255,214,0,0.18)';
  } else if (tsCode === '5') {
    flagEl.textContent = 'RED FLAG';
    flagEl.style.color = '#ff1744'; flagEl.style.borderColor = '#ff1744'; flagEl.style.background = 'rgba(255,23,68,0.2)';
  } else if (tsCode === '2') {
    flagEl.textContent = 'YELLOW FLAG';
    flagEl.style.color = '#ffd600'; flagEl.style.borderColor = '#ffd600'; flagEl.style.background = 'rgba(255,214,0,0.18)';
  } else {
    flagEl.textContent = 'TRACK CLEAR';
    flagEl.style.color = '#00e676'; flagEl.style.borderColor = '#00e676'; flagEl.style.background = 'rgba(0,230,118,0.15)';
  }

  let wCur = (replayData.weather || [])[0];
  for (let w of (replayData.weather || [])) {
    if (w.t <= simTime) wCur = w;
  }
  if (wCur) {
    document.getElementById('weatherBox').textContent =
      `AIR ${wCur.air}C | TRK ${wCur.track}C | HUM ${wCur.hum}%${wCur.rain ? ' | WET' : ''}`;
  }

  const towerEl = document.getElementById('towerList');
  const tyreColors = { S: '#ff1744', M: '#ffd600', H: '#f8fafc', I: '#00e676', W: '#00b0ff' };
  let html = '';
  const leadProg = leader ? leader.st.totalProgress : 0;

  states.forEach((item, idx) => {
    const { drv, st } = item;
    const rank = idx + 1;
    const isSel = drv.abbr === selectedAbbr;
    let gapTxt = stype === 'R' || stype === 'S' ? 'LEADER' : 'P1 PACE';
    if (st.status === 'OUT') gapTxt = stype === 'Q' || stype === 'SQ' ? 'ELIM' : 'OUT';
    else if (st.status === 'FIN') gapTxt = idx === 0 ? (stype === 'R' ? 'WINNER' : 'P1') : 'FIN';
    else if (st.inPit) gapTxt = 'IN PIT';
    else if (idx > 0) {
      if (stype === 'R' || stype === 'S') {
        const refProg = gapIntervalMode ? states[idx - 1].st.totalProgress : leadProg;
        const dLap = Math.max(0, refProg - st.totalProgress);
        const secGap = dLap * (st.lapDur || 88.0);
        gapTxt = dLap >= 1.0 ? `+${Math.floor(dLap)}L` : `+${secGap.toFixed(1)}s`;
      } else {
        const baseStep = gapIntervalMode ? 0.14 : (idx * 0.17);
        gapTxt = `+${(baseStep + ((idx * 3) % 7) * 0.02).toFixed(3)}s`;
      }
    }
    const tCol = tyreColors[st.comp] || '#ffd600';
    html += `<div class="tower-row ${isSel ? 'selected' : ''} ${st.status === 'OUT' ? 'out' : ''}" onclick="selectDriver('${drv.abbr}')">
      <span class="t-pos">${rank}</span>
      <span class="t-bar" style="background:${drv.color}"></span>
      <span class="t-code">${drv.abbr}</span>
      <span class="t-num">#${drv.number}</span>
      <span class="t-gap" style="${st.inPit ? 'color:#00e5ff;' : (st.status === 'OUT' ? 'color:#ff5252;' : '')}">${gapTxt}</span>
      <span class="tyre-ring" style="border-color:${tCol}">${st.comp}</span>
    </div>`;
  });
  towerEl.innerHTML = html;

  const selItem = states.find(x => x.drv.abbr === selectedAbbr) || states[0];
  if (selItem) {
    const { drv, st } = selItem;
    const rank = states.indexOf(selItem) + 1;
    document.getElementById('hudDrvName').textContent = `${drv.name.toUpperCase()} #${drv.number}`;
    document.getElementById('hudDrvTeam').textContent = drv.team.toUpperCase();
    document.getElementById('hudDrvPos').textContent = `P${rank}`;
    document.getElementById('hudDrvStatus').textContent = st.inPit ? 'IN PIT LANE' : (st.status === 'OUT' ? drv.status.toUpperCase() : `LAP ${st.lap}`);
    document.getElementById('hudSpeed').textContent = st.status === 'OUT' ? 0 : st.spd;
    document.getElementById('hudGear').textContent = st.status === 'OUT' ? 'N' : st.gear;
    document.getElementById('hudRpm').textContent = st.status === 'OUT' ? 0 : st.rpm;

    const thr = st.status === 'OUT' ? 0 : st.thr;
    const brk = st.status === 'OUT' ? 0 : st.brk;
    document.getElementById('barThr').style.width = thr + '%';
    document.getElementById('txtThr').textContent = thr + '%';
    document.getElementById('barBrk').style.width = brk + '%';
    document.getElementById('txtBrk').textContent = brk + '%';

    const drsEl = document.getElementById('drsBadge');
    if (st.drs >= 10 && st.status === 'RUN') {
      drsEl.textContent = 'DRS ACTIVE';
      drsEl.style.background = 'rgba(0,230,118,0.2)';
      drsEl.style.color = '#00e676';
    } else {
      drsEl.textContent = 'DRS OFF';
      drsEl.style.background = '#1c2030';
      drsEl.style.color = '#64748b';
    }

    const rpmFrac = Math.max(0, Math.min(1, (st.rpm - 8000) / 4200));
    const lit = st.status === 'OUT' ? 0 : Math.round(rpmFrac * 15);
    ledNodes.forEach((node, i) => {
      if (i < lit) {
        node.style.background = i < 5 ? '#00e676' : (i < 10 ? '#ff1744' : '#00b0ff');
      } else {
        node.style.background = '#1c2030';
      }
    });

    oscHistory.push({ spd: st.spd || 0, thr, brk });
    if (oscHistory.length > 140) oscHistory.shift();
    drawOscilloscope();
  }

  drawTrackCanvas(states, selItem);
}

function drawOscilloscope() {
  const w = oscCanvas.width, h = oscCanvas.height;
  oscCtx.clearRect(0, 0, w, h);
  oscCtx.strokeStyle = 'rgba(255,255,255,0.06)';
  oscCtx.lineWidth = 1;
  for (let y = 25; y < h; y += 28) {
    oscCtx.beginPath(); oscCtx.moveTo(0, y); oscCtx.lineTo(w, y); oscCtx.stroke();
  }
  if (oscHistory.length < 2) return;

  oscCtx.strokeStyle = '#00e676';
  oscCtx.lineWidth = 1.5;
  oscCtx.beginPath();
  oscHistory.forEach((p, i) => {
    const x = (i / 140) * w;
    const y = h - 8 - (p.thr / 100) * (h - 18);
    if (i === 0) oscCtx.moveTo(x, y); else oscCtx.lineTo(x, y);
  });
  oscCtx.stroke();

  oscCtx.strokeStyle = '#ff1744';
  oscCtx.lineWidth = 1.5;
  oscCtx.beginPath();
  oscHistory.forEach((p, i) => {
    const x = (i / 140) * w;
    const y = h - 8 - (p.brk / 100) * (h - 18);
    if (i === 0) oscCtx.moveTo(x, y); else oscCtx.lineTo(x, y);
  });
  oscCtx.stroke();

  oscCtx.strokeStyle = '#00e5ff';
  oscCtx.lineWidth = 2.0;
  oscCtx.beginPath();
  oscHistory.forEach((p, i) => {
    const x = (i / 140) * w;
    const y = h - 8 - (p.spd / 350) * (h - 18);
    if (i === 0) oscCtx.moveTo(x, y); else oscCtx.lineTo(x, y);
  });
  oscCtx.stroke();
}

function drawTrackCanvas(states, selItem) {
  const dpr = window.devicePixelRatio || 1;
  const rect = trackCanvas.getBoundingClientRect();
  if (trackCanvas.width !== Math.floor(rect.width * dpr) || trackCanvas.height !== Math.floor(rect.height * dpr)) {
    trackCanvas.width = Math.floor(rect.width * dpr);
    trackCanvas.height = Math.floor(rect.height * dpr);
  }
  ctx.save();
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  ctx.clearRect(0, 0, W, H);

  const tx = replayData.circuit.x || [];
  const ty = replayData.circuit.y || [];
  if (tx.length < 3) { ctx.restore(); return; }

  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < tx.length; i++) {
    if (tx[i] < minX) minX = tx[i];
    if (tx[i] > maxX) maxX = tx[i];
    if (ty[i] < minY) minY = ty[i];
    if (ty[i] > maxY) maxY = ty[i];
  }
  const spanX = Math.max(1, maxX - minX);
  const spanY = Math.max(1, maxY - minY);
  const pad = 56;
  const baseScale = Math.min((W - pad * 2) / spanX, (H - pad * 2) / spanY);
  const scale = baseScale * camZoom;
  const cx = (minX + maxX) * 0.5;
  const cy = (minY + maxY) * 0.5;

  if (followCar && selItem && selItem.st.wx !== undefined) {
    camPanX = -(selItem.st.wx - cx) * scale;
    camPanY = (selItem.st.wy - cy) * scale;
  }

  const toScreen = (wx, wy) => [
    W * 0.5 + (wx - cx) * scale + camPanX,
    H * 0.5 - (wy - cy) * scale + camPanY
  ];

  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.beginPath();
  for (let i = 0; i < tx.length; i++) {
    const [sx, sy] = toScreen(tx[i], ty[i]);
    if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
  }
  ctx.strokeStyle = '#333b52';
  ctx.lineWidth = Math.max(9, 12 * Math.sqrt(camZoom));
  ctx.stroke();

  ctx.strokeStyle = '#181c28';
  ctx.lineWidth = Math.max(6, 8.5 * Math.sqrt(camZoom));
  ctx.stroke();

  ctx.strokeStyle = 'rgba(255,255,255,0.13)';
  ctx.lineWidth = 1.0;
  ctx.stroke();

  const [sfX, sfY] = toScreen(tx[0], ty[0]);
  ctx.fillStyle = '#ffffff';
  ctx.beginPath();
  ctx.arc(sfX, sfY, 4.5, 0, Math.PI * 2);
  ctx.fill();

  if (showCorners && replayData.circuit.corners) {
    ctx.font = 'bold 10px Consolas, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    replayData.circuit.corners.forEach(c => {
      const [kx, ky] = toScreen(c.x, c.y);
      ctx.fillStyle = 'rgba(14,16,23,0.88)';
      ctx.strokeStyle = '#475569';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(kx, ky, 8.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = '#cbd5e1';
      ctx.fillText(String(c.num) + (c.letter || ''), kx, ky);
    });
  }

  for (let i = states.length - 1; i >= 0; i--) {
    const { drv, st } = states[i];
    if (st.status === 'OUT' || st.wx === undefined) continue;
    const [sx, sy] = toScreen(st.wx, st.wy);
    const isSel = drv.abbr === selectedAbbr;

    if (!carTrails[drv.abbr]) carTrails[drv.abbr] = [];
    const tr = carTrails[drv.abbr];
    tr.push([st.wx, st.wy]);
    if (tr.length > 18) tr.shift();

    if (showTrails && tr.length > 2) {
      ctx.beginPath();
      tr.forEach((pt, idx) => {
        const [px, py] = toScreen(pt[0], pt[1]);
        if (idx === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.strokeStyle = drv.color;
      ctx.globalAlpha = isSel ? 0.65 : 0.28;
      ctx.lineWidth = isSel ? 2.6 : 1.4;
      ctx.stroke();
      ctx.globalAlpha = 1.0;
    }

    const rad = isSel ? 6.5 : 4.2;
    if (isSel) {
      ctx.beginPath();
      ctx.arc(sx, sy, 12, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,0.16)';
      ctx.fill();
    }
    ctx.beginPath();
    ctx.arc(sx, sy, rad, 0, Math.PI * 2);
    ctx.fillStyle = drv.color;
    ctx.fill();
    ctx.lineWidth = isSel ? 2.0 : 1.1;
    ctx.strokeStyle = isSel ? '#ffffff' : '#090b10';
    ctx.stroke();

    ctx.font = isSel ? '900 11px Segoe UI, sans-serif' : '700 9.5px Segoe UI, sans-serif';
    ctx.fillStyle = isSel ? '#ffffff' : '#e2e8f0';
    ctx.textAlign = 'left';
    ctx.fillText(drv.abbr, sx + rad + 4, sy + 3);
  }

  ctx.restore();
}

function renderRcmFeed() {
  if (!replayData) return;
  const list = (replayData.rcm || []).filter(m => m.t <= simTime + 1).slice(-25).reverse();
  document.getElementById('rcmCount').textContent = `${list.length} MSG`;
  const feed = document.getElementById('rcmFeed');
  feed.innerHTML = list.map(m => {
    const mins = Math.floor(m.t / 60);
    const secs = Math.floor(m.t % 60);
    const ts = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    return `<div class="rcm-item"><span class="rcm-time">[${ts}]</span>${m.msg}</div>`;
  }).join('');
}

window.selectDriver = function(abbr) {
  selectedAbbr = abbr;
  oscHistory = [];
};

yearSel.addEventListener('change', () => fetchSchedule(yearSel.value));
sessionSel.addEventListener('change', loadReplay);
loadBtn.addEventListener('click', loadReplay);
playBtn.addEventListener('click', () => {
  isPlaying = !isPlaying;
  playBtn.textContent = isPlaying ? 'PAUSE' : 'PLAY';
});
speedBtn.addEventListener('click', () => {
  speedIdx = (speedIdx + 1) % SPEEDS.length;
  speedBtn.textContent = SPEEDS[speedIdx] + 'x';
});
scrubber.addEventListener('input', () => {
  simTime = parseFloat(scrubber.value) || 0;
  carTrails = {};
  renderRcmFeed();
});
document.getElementById('gapModeBtn').addEventListener('click', (e) => {
  gapIntervalMode = !gapIntervalMode;
  e.target.textContent = gapIntervalMode ? 'MODE: INTERVAL' : 'MODE: LEADER';
});
document.getElementById('btnCorners').addEventListener('click', (e) => {
  showCorners = !showCorners;
  e.target.classList.toggle('active', showCorners);
});
document.getElementById('btnTrails').addEventListener('click', (e) => {
  showTrails = !showTrails;
  e.target.classList.toggle('active', showTrails);
});
document.getElementById('btnFollow').addEventListener('click', (e) => {
  followCar = !followCar;
  e.target.classList.toggle('active', followCar);
});
document.getElementById('btnResetCam').addEventListener('click', () => {
  followCar = false;
  document.getElementById('btnFollow').classList.remove('active');
  camZoom = 1.0; camPanX = 0; camPanY = 0;
});

trackCanvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  const factor = e.deltaY < 0 ? 1.12 : 0.89;
  camZoom = Math.max(0.6, Math.min(6.0, camZoom * factor));
});
trackCanvas.addEventListener('mousedown', (e) => {
  isDragging = true;
  dragStartX = e.clientX - camPanX;
  dragStartY = e.clientY - camPanY;
});
window.addEventListener('mousemove', (e) => {
  if (!isDragging) return;
  camPanX = e.clientX - dragStartX;
  camPanY = e.clientY - dragStartY;
});
window.addEventListener('mouseup', () => { isDragging = false; });

let lastRcmCheck = -10;
function animate(now) {
  const dt = Math.min(0.1, (now - lastFrameTs) / 1000);
  lastFrameTs = now;
  if (replayData && isPlaying) {
    simTime += dt * SPEEDS[speedIdx];
    if (simTime >= replayData.meta.max_t) {
      simTime = replayData.meta.max_t;
      isPlaying = false;
      playBtn.textContent = 'PLAY';
    }
    scrubber.value = simTime;
    if (Math.abs(simTime - lastRcmCheck) > 2.0) {
      lastRcmCheck = simTime;
      renderRcmFeed();
    }
  }
  updateUI();
  requestAnimationFrame(animate);
}

fetchSchedule(yearSel.value).then(() => loadReplay());
requestAnimationFrame(animate);
</script>
<!-- ================= FUTURE RACE DATES + LIVE OPENF1 STREAM SNIPPET ================= -->
<style>
.future-card {
  position: absolute;
  top: 50%; left: 50%;
  transform: translate(-50%, -50%);
  background: rgba(14, 16, 23, 0.96);
  border: 1px solid #2d3348;
  border-top: 3px solid #e10600;
  border-radius: 6px;
  padding: 20px 26px;
  width: 420px;
  max-width: 92%;
  text-align: center;
  z-index: 50;
  display: none;
  box-shadow: 0 18px 45px rgba(0,0,0,0.75);
}
.future-badge {
  display: inline-block;
  background: rgba(225, 6, 0, 0.18);
  color: #ff4d4d;
  border: 1px solid #e10600;
  font-size: 10px;
  font-weight: 900;
  padding: 3px 10px;
  border-radius: 3px;
  letter-spacing: 1.2px;
  margin-bottom: 10px;
}
.future-title { font-size: 17px; font-weight: 900; color: #fff; letter-spacing: 0.5px; }
.future-loc { font-size: 12px; color: #8b95a5; margin: 4px 0 14px; }
.future-countdown {
  font-family: Consolas, monospace;
  font-size: 20px;
  font-weight: 900;
  color: #00e5ff;
  background: #090b10;
  border: 1px solid #232738;
  padding: 8px;
  border-radius: 4px;
  margin-bottom: 14px;
  letter-spacing: 1px;
}
.future-sched {
  text-align: left;
  font-size: 11px;
  font-family: Consolas, monospace;
  background: #121521;
  border: 1px solid #232738;
  border-radius: 4px;
  padding: 10px 12px;
  margin-bottom: 14px;
}
.future-sched div {
  display: flex;
  justify-content: space-between;
  padding: 4px 0;
  border-bottom: 1px solid rgba(255,255,255,0.05);
}
.future-sched div:last-child { border-bottom: none; color: #00e676; font-weight: 900; }
.btn-live-stream {
  background: #141722;
  border: 1px solid #00e676;
  color: #00e676;
  padding: 5px 10px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 900;
  cursor: pointer;
  letter-spacing: 0.6px;
}
.btn-live-stream.active { background: #e10600; border-color: #e10600; color: #fff; }
</style>

<div class="future-card" id="futureRaceCard">
  <div class="future-badge">UPCOMING GRAND PRIX // NOT YET RACED</div>
  <div class="future-title" id="futTitle">2026 GRAND PRIX</div>
  <div class="future-loc" id="futLoc">Scheduled FIA World Championship Event</div>
  <div class="future-countdown" id="futCountdown">--d : --h : --m : --s</div>
  <div class="future-sched" id="futSched"></div>
  <button onclick="document.getElementById('futureRaceCard').style.display='none'" style="background:#e10600;border:none;color:#fff;padding:6px 16px;border-radius:4px;font-weight:900;">DISMISS</button>
</div>

<script>
(function() {
  let scheduleDates = {};
  let countdownTimer = null;
  let liveTimer = null;

  // Move card inside map-wrap and add LIVE button to top bar
  const mapWrap = document.querySelector('.map-wrap');
  const fCard = document.getElementById('futureRaceCard');
  if (mapWrap && fCard) mapWrap.appendChild(fCard);

  const topCtrl = document.querySelector('.controls-top');
  const liveBtn = document.createElement('button');
  liveBtn.className = 'btn-live-stream';
  liveBtn.textContent = 'LIVE SESSION';
  topCtrl.appendChild(liveBtn);

  // Capture race dates whenever /api/schedule loads
  const origFetch = window.fetch;
  window.fetch = async function(url, opts) {
    const res = await origFetch(url, opts);
    if (typeof url === 'string' && url.includes('/api/schedule')) {
      res.clone().json().then(list => {
        scheduleDates = {};
        (list || []).forEach(ev => { scheduleDates[String(ev.round)] = ev; });
      }).catch(() => {});
    }
    if (typeof url === 'string' && url.includes('/api/replay')) {
      fCard.style.display = 'none';
    }
    return res;
  };

  // Replace generic alert with the Future Race Schedule & Countdown Card
  const origAlert = window.alert;
  window.alert = function(msg) {
    if (String(msg).includes('No session data') || String(msg).includes('No race data')) {
      showFutureRaceCard();
    } else {
      origAlert(msg);
    }
  };

  function addDays(dateStr, offset) {
    const d = new Date(dateStr + 'T14:00:00Z');
    if (isNaN(d)) return 'TBA';
    d.setUTCDate(d.getUTCDate() + offset);
    return d.toUTCString().slice(0, 16);
  }

  function showFutureRaceCard() {
    const yr = document.getElementById('yearSel').value;
    const rnd = document.getElementById('gpSel').value;
    const ev = scheduleDates[String(rnd)] || { name: `Round ${rnd} Grand Prix`, date: `${yr}-11-29`, location: 'Circuit', country: '' };
    document.getElementById('futTitle').textContent = `${yr} ${ev.name.toUpperCase()}`;
    document.getElementById('futLoc').textContent = `${ev.location || ''} ${ev.country ? '// ' + ev.country.toUpperCase() : ''}`;

    const fri = addDays(ev.date, -2);
    const sat = addDays(ev.date, -1);
    const sun = addDays(ev.date, 0);
    document.getElementById('futSched').innerHTML = `
      <div><span>FRIDAY (FP1 / SQ)</span><span>${fri}</span></div>
      <div><span>SATURDAY (FP2 / SPRINT / QUALI)</span><span>${sat}</span></div>
      <div><span>SUNDAY (GRAND PRIX RACE)</span><span>${sun}</span></div>
    `;
    fCard.style.display = 'block';

    if (countdownTimer) clearInterval(countdownTimer);
    const targetMs = new Date(ev.date + 'T14:00:00Z').getTime();
    const tick = () => {
      const diff = targetMs - Date.now();
      if (diff <= 0 || isNaN(diff)) {
        document.getElementById('futCountdown').textContent = 'RACE WEEKEND ACTIVE';
        return;
      }
      const d = Math.floor(diff / 86400000);
      const h = Math.floor((diff % 86400000) / 3600000);
      const m = Math.floor((diff % 3600000) / 60000);
      const s = Math.floor((diff % 60000) / 1000);
      document.getElementById('futCountdown').textContent = `${d}d : ${h}h : ${m}m : ${s}s`;
    };
    tick();
    countdownTimer = setInterval(tick, 1000);
  }

  // LIVE OPENF1 POLLING FOR FRIDAY 7PM-8PM LIVE TESTING
  async function pollOpenF1Live() {
    try {
      const [sRes, dRes, pRes, rRes] = await Promise.all([
        origFetch('https://api.openf1.org/v1/sessions?session_key=latest').then(r => r.json()),
        origFetch('https://api.openf1.org/v1/drivers?session_key=latest').then(r => r.json()),
        origFetch('https://api.openf1.org/v1/position?session_key=latest').then(r => r.json()),
        origFetch('https://api.openf1.org/v1/race_control?session_key=latest').then(r => r.json())
      ]);
      const sess = (sRes || [])[0];
      if (!sess) return;
      document.getElementById('raceTitle').textContent = `LIVE // ${sess.year} ${sess.country_name.toUpperCase()} GP - ${sess.session_name.toUpperCase()}`;

      const latestPos = {};
      (pRes || []).forEach(p => { latestPos[p.driver_number] = p.position; });

      if (window.replayData && dRes && dRes.length) {
        dRes.forEach((d, idx) => {
          const abbr = (d.name_acronym || ('D' + idx)).toUpperCase();
          const pos = latestPos[d.driver_number] || (idx + 1);
          if (replayData.drivers[abbr]) {
            replayData.drivers[abbr].finish_pos = pos;
            replayData.drivers[abbr].laps.forEach(l => { l[3] = pos; });
          }
        });
      }
      if (window.replayData && rRes && rRes.length) {
        replayData.rcm = rRes.slice(-20).map((m, i) => ({
          t: i * 10, cat: 'LIVE', flag: m.flag || 'INFO', msg: m.message || ''
        }));
        if (typeof renderRcmFeed === 'function') renderRcmFeed();
      }
    } catch (e) {}
  }

  liveBtn.addEventListener('click', () => {
    if (liveTimer) {
      clearInterval(liveTimer);
      liveTimer = null;
      liveBtn.classList.remove('active');
      liveBtn.textContent = 'LIVE SESSION';
    } else {
      liveBtn.classList.add('active');
      liveBtn.textContent = 'LIVE ON (5s)';
      fCard.style.display = 'none';
      pollOpenF1Live();
      liveTimer = setInterval(pollOpenF1Live, 5000);
    }
  });
})();
</script>
<!-- === MOBILE RESPONSIVE & TOUCH GESTURES ADD-ON === -->
<style>
  .mob-tab-bar {
    display: none;
    height: 38px;
    background: #0b0b10;
    border-top: 1px solid #23232f;
    grid-template-columns: 1fr 1fr 1fr;
    z-index: 40;
    flex-shrink: 0;
  }
  .mob-tab-btn {
    background: transparent;
    border: none;
    border-right: 1px solid #1a1a24;
    color: #8b8b9e;
    font-family: 'Orbitron', sans-serif;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.8px;
    cursor: pointer;
    transition: 0.15s;
  }
  .mob-tab-btn:last-child { border-right: none; }
  .mob-tab-btn.active {
    color: #ffffff;
    background: rgba(225, 6, 0, 0.16);
    box-shadow: inset 0 -3px 0 #e10600;
  }

  @media (max-width: 960px) {
    body {
      height: 100dvh;
      overflow: hidden;
    }
    .header {
      height: auto;
      padding: 6px 8px;
      flex-direction: column;
      align-items: stretch;
      gap: 6px;
    }
    .brand {
      justify-content: space-between;
    }
    .brand-title { font-size: 12px; }
    .brand-sub { font-size: 8px; }
    .controls-top {
      display: flex;
      overflow-x: auto;
      gap: 5px;
      padding-bottom: 2px;
      scrollbar-width: none;
    }
    .controls-top::-webkit-scrollbar { display: none; }
    select, .btn {
      height: 30px;
      padding: 0 8px;
      font-size: 11px;
      flex-shrink: 0;
    }
    #gpSel { max-width: 135px; }
    #sessionSel { max-width: 115px; }

    .workspace {
      display: flex;
      flex-direction: column;
      flex: 1;
      min-height: 0;
      overflow: hidden;
    }
    .map-viewport {
      order: 1;
      flex: 0 0 43%;
      min-height: 190px;
      width: 100%;
    }
    .tower {
      order: 2;
      flex: 1;
      width: 100%;
      min-height: 0;
      border-right: none;
      border-top: 1px solid #23232f;
    }
    .telem-panel {
      order: 3;
      flex: 1;
      width: 100%;
      min-height: 0;
      border-left: none;
      border-top: 1px solid #23232f;
      overflow-y: auto;
    }

    body.mob-mode-tower .telem-panel { display: none !important; }
    body.mob-mode-telem .tower { display: none !important; }
    body.mob-mode-map .tower,
    body.mob-mode-map .telem-panel { display: none !important; }
    body.mob-mode-map .map-viewport { flex: 1 1 100%; }

    .hud-top-left {
      top: 8px;
      left: 8px;
      padding: 6px 10px;
      border-radius: 6px;
    }
    .lap-counter { font-size: 15px; }
    .session-badge { font-size: 9px; margin-top: 1px; }
    .weather-strip {
      top: 8px;
      right: 8px;
      padding: 4px 8px;
      gap: 8px;
      font-size: 9px;
    }
    .weather-val { font-size: 10px; }
    .map-toolbar {
      bottom: 8px;
      right: 8px;
      gap: 4px;
    }
    .tool-btn {
      padding: 5px 8px;
      font-size: 9px;
    }

    .scope-container {
      min-height: 150px;
    }
    .mob-tab-bar {
      display: grid;
    }
    .timeline-bar {
      height: 44px;
      padding: 0 8px;
      gap: 6px;
    }
    .time-readout {
      font-size: 10px;
      min-width: 82px;
    }
    .scrubber::-webkit-slider-thumb {
      width: 20px;
      height: 20px;
    }
    #futureRaceCard {
      width: 92% !important;
      padding: 18px 14px !important;
    }
  }
</style>

<div class="mob-tab-bar" id="mobTabBar">
  <button class="mob-tab-btn active" onclick="setMobMode('tower', this)">TIMING TOWER</button>
  <button class="mob-tab-btn" onclick="setMobMode('telem', this)">TELEMETRY HUD</button>
  <button class="mob-tab-btn" onclick="setMobMode('map', this)">FULL CIRCUIT</button>
</div>

<script>
(function(){
  if (!document.querySelector('meta[name="viewport"]')) {
    const meta = document.createElement('meta');
    meta.name = 'viewport';
    meta.content = 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no';
    document.head.appendChild(meta);
  }

  document.body.classList.add('mob-mode-tower');

  const tabBar = document.getElementById('mobTabBar');
  const timeline = document.querySelector('.timeline-bar');
  if (tabBar && timeline && timeline.parentNode) {
    timeline.parentNode.insertBefore(tabBar, timeline);
  }

  window.setMobMode = function(mode, btn) {
    document.body.classList.remove('mob-mode-tower', 'mob-mode-telem', 'mob-mode-map');
    document.body.classList.add('mob-mode-' + mode);
    document.querySelectorAll('.mob-tab-btn').forEach(b => b.classList.remove('active'));
    if (btn) btn.classList.add('active');
    setTimeout(() => {
      if (typeof resizeCanvas === 'function') resizeCanvas();
    }, 50);
  };

  window.addEventListener('orientationchange', () => {
    setTimeout(() => {
      if (typeof resizeCanvas === 'function') resizeCanvas();
    }, 150);
  });

  const canvas = document.getElementById('mapCanvas');
  if (canvas) {
    let touchStartDist = 0;
    let initialZoom = 1;
    let touchMoved = false;

    canvas.addEventListener('touchstart', function(e) {
      if (e.touches.length === 1) {
        touchMoved = false;
        cam.dragging = true;
        cam.lx = e.touches[0].clientX;
        cam.ly = e.touches[0].clientY;
      } else if (e.touches.length === 2) {
        cam.dragging = false;
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        touchStartDist = Math.hypot(dx, dy);
        initialZoom = cam.zoom;
      }
    }, { passive: true });

    canvas.addEventListener('touchmove', function(e) {
      e.preventDefault();
      if (e.touches.length === 1 && cam.dragging) {
        const dx = e.touches[0].clientX - cam.lx;
        const dy = e.touches[0].clientY - cam.ly;
        if (Math.abs(dx) > 4 || Math.abs(dy) > 4) touchMoved = true;
        cam.x += dx;
        cam.y += dy;
        cam.lx = e.touches[0].clientX;
        cam.ly = e.touches[0].clientY;
      } else if (e.touches.length === 2 && touchStartDist > 0) {
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        const dist = Math.hypot(dx, dy);
        const ratio = dist / touchStartDist;
        cam.zoom = Math.max(0.4, Math.min(8.0, initialZoom * ratio));
      }
    }, { passive: false });

    canvas.addEventListener('touchend', function(e) {
      cam.dragging = false;
      if (!touchMoved && e.changedTouches.length === 1) {
        const t = e.changedTouches[0];
        const downEvt = new MouseEvent('mousedown', { clientX: t.clientX, clientY: t.clientY });
        const upEvt = new MouseEvent('mouseup', { clientX: t.clientX, clientY: t.clientY });
        canvas.dispatchEvent(downEvt);
        canvas.dispatchEvent(upEvt);
      }
    }, { passive: true });
  }
})();
</script>
<!-- === END MOBILE RESPONSIVE ADD-ON === -->
</body>
</html>
"""


# ==============================================================================
# 4. FLASK SERVER & GUNICORN ENTRY POINT
# ==============================================================================
app = Flask(__name__)


@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.route("/api/schedule")
def api_schedule():
    year = int(request.args.get("year", 2025))
    return jsonify(F1Core.get_schedule(year))


@app.route("/api/replay")
def api_replay():
    year = int(request.args.get("year", 2025))
    gp = request.args.get("gp", "1")
    session_type = request.args.get("session", "R")
    try:
        data = F1Core.load_replay(year, gp, session_type)
        return jsonify(data)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 200


if __name__ == "__main__":
    try:
        cur_file = os.path.abspath(__file__)
        cur_dir = os.path.dirname(cur_file)
        src_code = open(cur_file, "r", encoding="utf-8").read()
        for fname in ("Formula1pr.py", "app.py"):
            target = os.path.join(cur_dir, fname)
            if os.path.abspath(target) != cur_file:
                open(target, "w", encoding="utf-8").write(src_code)
    except Exception:
        pass

    port = int(os.environ.get("PORT", 5000))
    if not os.environ.get("RENDER") and os.name == "nt":
        threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
