"""
Create a small synthetic dataset with the same folder layout and headerless CSV
format as the INRIX Trip Bulk Report, using the real route segment IDs. Only for
testing that the pipeline runs end to end - the numbers mean nothing.

  python tools/make_synthetic.py <out_root>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAJ_COLS = ["TripId", "DeviceId", "ProviderId", "TripTimezone", "TrajIdx", "TrajRawDistanceM",
             "TrajRawDurationMillis", "SegmentId", "SegmentIdx", "LengthM", "CrossingStartOffsetM",
             "CrossingEndOffsetM", "CrossingStartDateUtc", "CrossingEndDateUtc", "CrossingSpeedKph",
             "OnRoadNetworkSnapCount", "ErrorCodes"]
TRIP_COLS = ["TripId", "DeviceId", "ProviderId", "StartDate", "EndDate", "StartLocLat", "StartLocLon",
             "EndLocLat", "EndLocLon", "TripMeanSpeedKph", "TripMaxSpeedKph", "TripDistanceM",
             "MovementType", "StartTimezone", "EndTimezone", "WaypointFreqSeconds"]
LANDFALL = pd.Timestamp("2020-09-16 09:45", tz="UTC")


def iso(ts):
    return ts.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def make(out, ds_dir, start, end, event, seed, trips_per_day=35):
    rng = np.random.default_rng(seed)
    routes = {r: (ROOT / "routes" / f"route{r}.txt").read_text().split() for r in "ABCD"}
    base_speed = {"A": 95, "B": 60, "C": 55, "D": 85}
    days = pd.date_range(start, end, freq="D", inclusive="left", tz="America/Chicago")
    traj, trips, tid = [], [], 0
    for day in days:
        for r, segs in routes.items():
            n = rng.poisson(trips_per_day * (1.8 if event and abs((day.tz_convert("UTC") - LANDFALL).days) <= 2 else 1))
            for _ in range(n):
                tid += 1
                t = (day + pd.Timedelta(hours=float(rng.uniform(5, 23)))).tz_convert("UTC")
                h = (t - LANDFALL).total_seconds() / 3600 if event else 999
                a = int(rng.integers(0, max(1, len(segs) - 2)))
                b = min(len(segs), a + int(rng.integers(2, 15)))
                path = segs[a:b]
                if rng.random() < 0.4:                    # reverse direction: negative way IDs
                    path = ["-" + x for x in path[::-1]]
                # exit choice: funnels toward exit 0 close to landfall
                p_exit0 = 0.85 if -48 < h < 0 else 0.4
                path = path + [f"9{path[-1].lstrip('-')[-6:]}{0 if rng.random() < p_exit0 else int(rng.integers(1, 4))}"]
                hr = t.tz_convert("America/Chicago").hour
                v = base_speed[r] * (0.75 if hr in (7, 8, 16, 17) else 1.0)
                if -24 < h < 12:
                    v *= rng.uniform(0.25, 0.6)
                speeds = np.clip(v * rng.normal(1, 0.12, len(path)), 3, 150)
                cur, k, rows = t, 0, []
                for s, sp in zip(path, speeds):          # each way = 1-3 sub-segments "<way>_<j>"
                    for j in range(int(rng.integers(1, 4))):
                        L = float(rng.uniform(60, 400))
                        dt = pd.Timedelta(seconds=L / (sp / 3.6))
                        rows.append([f"T{seed}_{tid}", f"D{tid % 997}", f"P{tid % 7}", "America/Chicago", 0,
                                     None, None, f"{s}_{j}", k, L, 0, L, iso(cur), iso(cur + dt),
                                     round(sp, 1), 1, ""])
                        cur += dt; k += 1
                dist = sum(r[9] for r in rows); dur = (cur - t).total_seconds() * 1000
                for r_ in rows:
                    r_[5], r_[6] = dist, dur
                traj += rows
                trips.append([f"T{seed}_{tid}", f"D{tid % 997}", f"P{tid % 7}", iso(t), iso(cur), 30.4, -87.2,
                              30.5, -87.3, round(float(speeds.mean()), 1), round(float(speeds.max()), 1), 5000,
                              1, "America/Chicago", "America/Chicago", 5])
    base = out / ds_dir / "date=2025-07-08" / "reportId=1" / "v1"
    (base / "data" / "trajs").mkdir(parents=True, exist_ok=True)
    (base / "data" / "trips").mkdir(parents=True, exist_ok=True)
    (base / "schema").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(traj, columns=TRAJ_COLS)
    half = len(df) // 2
    df.iloc[:half].to_csv(base / "data" / "trajs" / "trajs.csv.gz", header=False, index=False)
    df.iloc[half:].to_csv(base / "data" / "trajs" / "trajs.csv1.gz", header=False, index=False)
    pd.DataFrame(trips, columns=TRIP_COLS).to_csv(base / "data" / "trips" / "trips.csv.gz", header=False, index=False)
    (base / "schema" / "TripBulkReportTrajectoriesHeaders.csv").write_text(",".join(TRAJ_COLS) + "\n")
    (base / "schema" / "TripBulkReportTripsHeaders.csv").write_text(",".join(TRIP_COLS) + "\n")
    print(ds_dir, len(df), "crossings", len(trips), "trips")


if __name__ == "__main__":
    out = Path(sys.argv[1])
    make(out, "escambia-affected", "2020-08-16", "2020-11-16", True, 1)
    make(out, "escambia-controlpy", "2019-08-16", "2019-11-16", False, 2)
