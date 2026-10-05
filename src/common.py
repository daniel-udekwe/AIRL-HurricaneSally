"""Shared helpers: config loading, routes, folds, phases, seeding, logging."""
import importlib.util
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    path = Path(path or os.environ.get("EVAC_CONFIG", ROOT / "config.py"))
    spec = importlib.util.spec_from_file_location("evac_cfg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = {k: getattr(mod, k) for k in dir(mod) if k.isupper()}
    if os.environ.get("EVAC_QUICK") == "1":
        cfg.update(cfg.get("QUICK_OVERRIDES", {}))
    cfg["WORK_DIR"] = Path(cfg["WORK_DIR"])
    cfg["WORK_DIR"].mkdir(parents=True, exist_ok=True)
    return cfg


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def wpath(cfg, *parts):
    p = cfg["WORK_DIR"].joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
def _flip(s):
    return s[1:] if s.startswith("-") else "-" + s


def read_routes(cfg):
    """Rows (seg = way ID as it appears in the data, route label, order_in_file)."""
    mode = cfg.get("ROUTE_DIRECTIONS", "listed")
    rows = []
    for route, f in cfg["ROUTE_FILES"].items():
        segs = [s.strip() for s in Path(f).read_text().split() if s.strip()]
        for k, s in enumerate(segs):
            rows.append((s, route, k))
            if mode == "merged":
                rows.append((_flip(s), route, k))
            elif mode == "separate":
                rows.append((_flip(s), f"{route}_rev", k))
    df = pd.DataFrame(rows, columns=["seg", "route", "order_in_file"])
    dup = df[df.seg.duplicated(keep=False)]
    if len(dup):
        raise ValueError(f"Way IDs assigned to more than one route:\n{dup}")
    return df


def route_list(cfg):
    """Route series to model: all defined labels, minus any with no data (see s02)."""
    labels = list(dict.fromkeys(read_routes(cfg).route))
    p = cfg["WORK_DIR"] / "active_routes.txt"
    if p.exists():
        active = set(p.read_text().split())
        labels = [r for r in labels if r in active]
    return labels


# --------------------------------------------------------------------------
# Time, folds, phases
# --------------------------------------------------------------------------
def landfall(cfg):
    return pd.Timestamp(cfg["LANDFALL_UTC"], tz="UTC")


def local_midnight_utc(cfg, date_str):
    return pd.Timestamp(date_str).tz_localize(cfg["LOCAL_TZ"]).tz_convert("UTC")


def make_folds(cfg):
    r = cfg["ROLLING"]
    folds = []
    t = local_midnight_utc(cfg, r["start"])
    end = local_midnight_utc(cfg, r["end"])
    step = pd.Timedelta(hours=r["step_hours"])
    while t < end:
        t_end = min(t + step, end)
        loc = t.tz_convert(cfg["LOCAL_TZ"])
        folds.append({"fold": f"r{loc:%m%d}", "origin": t, "end": t_end, "kind": "rolling"})
        t = t_end
    for name, a, b in cfg["EXTRA_FOLDS"]:
        folds.append({"fold": name, "origin": local_midnight_utc(cfg, a),
                      "end": local_midnight_utc(cfg, b), "kind": "extra"})
    return folds


def assign_phase(cfg, ts_utc, folds=None):
    """Phase label for each UTC timestamp (vectorized)."""
    ts = pd.to_datetime(pd.Series(ts_utc), utc=True)
    L = landfall(cfg)
    lo, hi = cfg["PHASE_LANDFALL_WINDOW_H"]
    hrs = (ts - L).dt.total_seconds() / 3600.0
    out = np.where(hrs < lo, "pre_landfall", np.where(hrs < hi, "landfall", "recovery"))
    out = pd.Series(out, index=ts.index, dtype=object)
    for f in (folds or make_folds(cfg)):
        if f["kind"] == "extra":
            m = (ts >= f["origin"]) & (ts < f["end"])
            out[m] = f["fold"]
    return out.values


def expand_phase_groups(cfg, df, col="phase"):
    """Return df with extra copies of rows for each phase group (e.g. 'acute')."""
    parts = [df]
    for g, members in cfg["PHASE_GROUPS"].items():
        sub = df[df[col].isin(members)].copy()
        sub[col] = g
        parts.append(sub)
    sub = df.copy()
    sub[col] = "all"
    parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def local_date(cfg, ts_utc):
    return pd.to_datetime(pd.Series(ts_utc), utc=True).dt.tz_convert(cfg["LOCAL_TZ"]).dt.date.values


# --------------------------------------------------------------------------
# Header handling
# --------------------------------------------------------------------------
def read_header_tokens(path):
    txt = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    return [t.strip().strip('"') for t in re.split(r"[,\t\r\n]+", txt) if t.strip()]


def resolve_columns(available, candidates):
    """Map logical -> actual column name. Raises if a required column is missing."""
    low = {a.lower(): a for a in available}
    out, missing = {}, []
    for logical, (cands, required) in candidates.items():
        hit = next((low[c.lower()] for c in cands if c.lower() in low), None)
        if hit is None and required:
            missing.append(f"{logical} (tried {cands})")
        out[logical] = hit
    if missing:
        raise KeyError("Required columns not found: " + "; ".join(missing)
                       + f"\nAvailable columns: {available}")
    return out


def dataset_files(cfg, ds):
    base = Path(cfg["DATA_ROOT"]) / cfg["DATASETS"][ds]["dir"]
    trajs = sorted(p for p in base.rglob("*.gz") if p.parent.name == "trajs")
    trips = sorted(p for p in base.rglob("*.gz") if p.parent.name == "trips")
    th = next(iter(base.rglob("TripBulkReportTrajectoriesHeaders.csv")), None)
    ph = next(iter(base.rglob("TripBulkReportTripsHeaders.csv")), None)
    return {"trajs": trajs, "trips": trips, "traj_header": th, "trip_header": ph}


def write_manifest(cfg, steps, argv):
    """Record configuration, software versions and input files for this invocation."""
    import json, platform, importlib
    from datetime import datetime, timezone
    vers = {}
    for m in ["numpy", "pandas", "pyarrow", "duckdb", "scipy", "sklearn", "torch", "matplotlib"]:
        try:
            vers[m] = importlib.import_module(m).__version__
        except Exception:
            vers[m] = None
    inputs = {}
    for ds in cfg["DATASETS"]:
        f = dataset_files(cfg, ds)
        inputs[ds] = [dict(path=str(p), bytes=p.stat().st_size) for p in f["trajs"]]
    cfgs = {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.items()}
    man = dict(utc=datetime.now(timezone.utc).isoformat(), argv=argv, steps=list(steps),
               python=sys.version, platform=platform.platform(), packages=vers,
               config=cfgs, inputs=inputs,
               routes={r: list(g.seg) for r, g in read_routes(cfg).groupby("route")})
    p = wpath(cfg, "manifest", datetime.now().strftime("run_%Y%m%d_%H%M%S.json"))
    p.write_text(json.dumps(man, indent=1, default=str))
    return p
