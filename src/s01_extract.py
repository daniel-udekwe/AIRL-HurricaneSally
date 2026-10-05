"""
Step 1 - Extract route crossings from the raw INRIX Trip Bulk Report.

For each dataset (2020 affected, 2019 control):
  pass 1: trip IDs that touch any route segment
  pass 2: every crossing of those trips (needed to find the *next* segment,
          which is usually off-route)
  pass 3: crossings on route segments, with next segment / next start time
  trips : per-trip attributes (mean speed, provider) for touching trips

Outputs (WORK_DIR/<dataset>/): crossings_route.parquet, trips.parquet, extract_summary.csv
"""
import sys
import gzip

import duckdb
import pandas as pd

from common import (dataset_files, load_config, log, read_header_tokens,
                    read_routes, resolve_columns, wpath)


def _q(p):
    return "'" + str(p).replace("\\", "/").replace("'", "''") + "'"


def detect_schema(cfg, files, header_file, kind):
    """Return (column names, has_header_row, logical->actual mapping)."""
    with gzip.open(files[0], "rt", encoding="utf-8-sig", errors="replace") as fh:
        first = fh.readline().rstrip("\r\n")
    first_tokens = [t.strip().strip('"') for t in first.split(cfg["CSV_DELIM"])]
    hdr = read_header_tokens(header_file) if header_file else []
    cands = cfg["COLUMN_CANDIDATES"][kind]
    known = {c.lower() for v in cands.values() for c in v[0]}
    has_header = sum(t.lower() in known for t in first_tokens) >= 2
    if has_header:
        names = first_tokens
    elif hdr:
        names = hdr
        if len(names) != len(first_tokens):
            raise ValueError(
                f"{kind}: header file lists {len(names)} columns but data rows have "
                f"{len(first_tokens)}. Check {header_file} and CSV_DELIM.")
    else:
        raise ValueError(f"{kind}: no header row and no header file found.")
    return names, has_header, resolve_columns(names, cands)


def csv_rel(cfg, files, names, has_header):
    flist = "[" + ",".join(_q(f) for f in files) + "]"
    d = cfg["CSV_DELIM"]
    if has_header:
        return (f"read_csv({flist}, header=true, all_varchar=true, delim='{d}', "
                f"compression='gzip', ignore_errors=true)")
    cols = "{" + ",".join(f"'{n}':'VARCHAR'" for n in names) + "}"
    return (f"read_csv({flist}, header=false, columns={cols}, delim='{d}', "
            f"compression='gzip', ignore_errors=true)")


def ts_expr(col):
    c = f'"{col}"'
    num = f"TRY_CAST({c} AS DOUBLE)"
    return (f"COALESCE(TRY_CAST({c} AS TIMESTAMPTZ), "
            f"CASE WHEN {num} > 1e11 THEN to_timestamp({num}/1000.0) "
            f"WHEN {num} IS NOT NULL THEN to_timestamp({num}) END)")


def col_or_null(m, key, cast=None):
    if m.get(key) is None:
        return "NULL"
    c = f'"{m[key]}"'
    return f"TRY_CAST({c} AS {cast})" if cast else c


def way_expr(cfg, col):
    sep = cfg.get("SEGMENT_WAY_SEPARATOR")
    c = f'"{col}"'
    return f"split_part({c}, '{sep}', 1)" if sep else c


def extract_dataset(cfg, con, ds, routes):
    files = dataset_files(cfg, ds)
    if not files["trajs"]:
        raise FileNotFoundError(f"No trajs/*.gz found for {ds}")
    names, hh, m = detect_schema(cfg, files["trajs"], files["traj_header"], "traj")
    rel = csv_rel(cfg, files["trajs"], names, hh)
    out = wpath(cfg, ds, "x").parent
    touch, cx_all = out / "trips_touch.parquet", out / "crossings_trips.parquet"
    cx_route, trips_out = out / "crossings_route.parquet", out / "trips.parquet"
    way = way_expr(cfg, m["seg"])

    log(f"{ds}: pass 1/3 - trips touching route ways ({len(files['trajs'])} files)")
    con.execute(f"""COPY (SELECT DISTINCT "{m['trip']}" AS trip FROM {rel}
                    WHERE {way} IN (SELECT seg FROM route_segs))
                    TO {_q(touch)} (FORMAT PARQUET)""")
    n_trips = con.execute(f"SELECT count(*) FROM read_parquet({_q(touch)})").fetchone()[0]
    log(f"{ds}: {n_trips:,} trips touch the routes")
    if n_trips == 0:
        raise RuntimeError(f"{ds}: no trips match the route IDs. Check SEGMENT_WAY_SEPARATOR "
                           "and that the route files list IDs in the same form as the data.")

    log(f"{ds}: pass 2/3 - all crossings of those trips")
    con.execute(f"""COPY (
        SELECT "{m['trip']}" AS trip,
               {col_or_null(m, 'provider')} AS provider,
               {col_or_null(m, 'traj_idx', 'BIGINT')} AS traj_idx,
               {col_or_null(m, 'traj_dist', 'DOUBLE')} AS traj_dist,
               {col_or_null(m, 'traj_dur', 'DOUBLE')} AS traj_dur,
               "{m['seg']}" AS seg_raw,
               {way} AS way,
               {col_or_null(m, 'seg_idx', 'BIGINT')} AS seg_idx,
               {col_or_null(m, 'length', 'DOUBLE')} AS length_m,
               {ts_expr(m['t_start'])} AS t_start,
               TRY_CAST("{m['speed']}" AS DOUBLE) AS speed_kph
        FROM {rel}
        WHERE "{m['trip']}" IN (SELECT trip FROM read_parquet({_q(touch)}))
    ) TO {_q(cx_all)} (FORMAT PARQUET)""")

    log(f"{ds}: pass 3/3 - merge sub-segments into way traversals; find next way")
    order = "traj_idx NULLS LAST, seg_idx NULLS LAST, t_start"
    con.execute(f"""COPY (
        WITH o AS (
            SELECT *, CASE WHEN way IS NOT DISTINCT FROM LAG(way) OVER w THEN 0 ELSE 1 END AS chg
            FROM read_parquet({_q(cx_all)})
            WINDOW w AS (PARTITION BY trip ORDER BY {order})
        ), r AS (
            SELECT *, SUM(chg) OVER (PARTITION BY trip ORDER BY {order}
                                     ROWS UNBOUNDED PRECEDING) AS run
            FROM o
        ), runs AS (
            SELECT trip, run, any_value(way) AS way, any_value(provider) AS provider,
                   min(t_start) AS t_start, count(*) AS n_sub,
                   sum(CASE WHEN speed_kph > 0 AND length_m > 0 THEN length_m END) AS length_m,
                   sum(CASE WHEN speed_kph > 0 AND length_m > 0
                            THEN length_m / (speed_kph / 3.6) END) AS tt_s,
                   max(traj_dist) AS traj_dist, max(traj_dur) AS traj_dur
            FROM r GROUP BY trip, run
        ), nx AS (
            SELECT *, LEAD(way) OVER v AS next_seg, LEAD(t_start) OVER v AS next_t
            FROM runs WINDOW v AS (PARTITION BY trip ORDER BY run)
        )
        SELECT nx.trip, nx.way AS seg, nx.next_seg, nx.t_start, nx.next_t, nx.n_sub,
               nx.length_m, nx.length_m / nx.tt_s * 3.6 AS speed_kph,
               nx.provider,
               CASE WHEN nx.traj_dur > 0 THEN nx.traj_dist / (nx.traj_dur / 1000.0) * 3.6 END
                   AS trip_mean_speed,
               rs.route
        FROM nx JOIN route_segs rs ON nx.way = rs.seg
    ) TO {_q(cx_route)} (FORMAT PARQUET)""")

    # trip-level attributes come from the trajectory file (the trips file has
    # rows with inconsistent field counts and is not needed)
    con.execute(f"""COPY (SELECT trip, any_value(provider) AS provider,
                         avg(trip_mean_speed) AS trip_mean_speed
                         FROM read_parquet({_q(cx_route)}) GROUP BY trip)
                    TO {_q(trips_out)} (FORMAT PARQUET)""")

    s = con.execute(f"""SELECT route, count(*) AS traversals, count(DISTINCT trip) AS trips,
                         count(DISTINCT seg) AS ways_seen, avg(n_sub) AS avg_subsegments,
                         min(t_start) AS first, max(t_start) AS last,
                         avg(CASE WHEN t_start IS NULL THEN 1 ELSE 0 END) AS frac_bad_time,
                         avg(CASE WHEN next_seg IS NULL THEN 1 ELSE 0 END) AS frac_no_next
                         FROM read_parquet({_q(cx_route)}) GROUP BY route ORDER BY route""").df()
    s.insert(0, "dataset", ds)
    seg_tot = routes.groupby("route").size().rename("ways_listed")
    s = s.merge(seg_tot, left_on="route", right_index=True, how="right")
    s.to_csv(out / "extract_summary.csv", index=False)
    print(s.to_string(index=False))
    return s


def main():
    cfg = load_config()
    routes = read_routes(cfg)
    cfg["DUCKDB_TEMP"].mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    con.execute(f"SET memory_limit='{cfg['DUCKDB_MEMORY']}'")
    con.execute(f"SET temp_directory={_q(cfg['DUCKDB_TEMP'])}")
    con.register("route_segs_df", routes[["seg", "route"]])
    con.execute("CREATE TABLE route_segs AS SELECT * FROM route_segs_df")
    summaries = [extract_dataset(cfg, con, ds, routes) for ds in cfg["DATASETS"]]
    pd.concat(summaries).to_csv(wpath(cfg, "extract_summary.csv"), index=False)
    log("extract done")


if __name__ == "__main__":
    sys.exit(main())
