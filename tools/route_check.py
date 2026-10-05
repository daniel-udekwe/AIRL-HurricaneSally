"""
Check how well each route's listed way IDs are covered by the 2020-map data.

For each route, listed direction and reverse:
  * listed IDs never observed (IDs created after the July 2020 map cannot appear);
  * connectivity of observed ways via observed way-to-way transitions
    (1 component = contiguous corridor; >1 = gaps);
  * estimated corridor length (sum of each way's 90th-percentile traversal length);
  * off-route 2020 ways most often traversed BETWEEN two route ways within a trip
    (<= 3 ways apart) - candidates for ways that fill gaps.

Run after step 1:  python tools/route_check.py
Writes work/tables/route_check.csv and work/tables/route_gap_candidates.csv
"""
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import load_config, wpath  # noqa: E402


def flip(s):
    return s[1:] if s.startswith("-") else "-" + s


def n_components(nodes, edges):
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    return len({find(n) for n in nodes})


def main():
    cfg = load_config()
    con = duckdb.connect()
    cfg["DUCKDB_TEMP"].mkdir(parents=True, exist_ok=True)
    con.execute(f"SET memory_limit='{cfg['DUCKDB_MEMORY']}'")
    con.execute(f"SET temp_directory='{str(cfg['DUCKDB_TEMP']).replace(chr(92), '/')}'")
    files = [str(wpath(cfg, ds, "crossings_trips.parquet")).replace("\\", "/") for ds in cfg["DATASETS"]]
    flist = "[" + ",".join(f"'{f}'" for f in files) + "]"
    print("building way runs (one pass over all crossings of touching trips)...", flush=True)
    order = "traj_idx NULLS LAST, seg_idx NULLS LAST, t_start"
    con.execute(f"""CREATE TABLE runs AS
        WITH o AS (SELECT trip, way, length_m, {order.replace(' NULLS LAST', '')},
                   CASE WHEN way IS NOT DISTINCT FROM LAG(way) OVER w THEN 0 ELSE 1 END AS chg
                   FROM read_parquet({flist}) WINDOW w AS (PARTITION BY trip ORDER BY {order})),
        r AS (SELECT *, SUM(chg) OVER (PARTITION BY trip ORDER BY {order} ROWS UNBOUNDED PRECEDING) AS rn FROM o)
        SELECT trip, rn, any_value(way) AS way, sum(length_m) AS length FROM r GROUP BY trip, rn""")
    con.execute("CREATE TABLE runs2 AS SELECT *, LEAD(way) OVER (PARTITION BY trip ORDER BY rn) AS next_way FROM runs")

    rows, gaps = [], []
    for route, f in cfg["ROUTE_FILES"].items():
        ids = [s.strip() for s in Path(f).read_text().split() if s.strip()]
        for direction, seq in [("listed", ids), ("reverse", [flip(s) for s in ids])]:
            con.register("rset", pd.DataFrame({"way": seq}))
            obs = con.execute("""SELECT way, count(*) n, quantile_cont(length, 0.9) l90
                                 FROM runs2 WHERE way IN (SELECT way FROM rset) GROUP BY way""").df()
            edges = con.execute("""SELECT DISTINCT way, next_way FROM runs2
                                   WHERE way IN (SELECT way FROM rset) AND next_way IN (SELECT way FROM rset)""").df()
            seen = set(obs.way)
            unseen = [s for s in seq if s not in seen]
            rows.append(dict(route=route, direction=direction, listed=len(seq), seen=len(seen),
                             unseen=len(unseen),
                             connected_components=n_components(seen, zip(edges.way, edges.next_way)) if seen else 0,
                             est_length_km=round(obs.l90.sum() / 1000, 2), traversals=int(obs.n.sum()),
                             max_seen_id=max((int(s.lstrip("-")) for s in seen), default=None),
                             min_unseen_id=min((int(s.lstrip("-")) for s in unseen), default=None)))
            if direction == "listed" and seen:
                g = con.execute("""
                    WITH t AS (SELECT trip, rn, way, way IN (SELECT way FROM rset) AS on_r FROM runs),
                    u AS (SELECT *,
                        max(CASE WHEN on_r THEN rn END) OVER (PARTITION BY trip ORDER BY rn
                            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_on,
                        min(CASE WHEN on_r THEN rn END) OVER (PARTITION BY trip ORDER BY rn
                            ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING) AS next_on
                        FROM t)
                    SELECT way, count(DISTINCT trip) AS trips_between_route_ways FROM u
                    WHERE NOT on_r AND prev_on IS NOT NULL AND next_on IS NOT NULL AND next_on - prev_on <= 4
                    GROUP BY way ORDER BY 2 DESC LIMIT 15""").df()
                g.insert(0, "route", route)
                gaps.append(g)
    out = pd.DataFrame(rows)
    out.to_csv(wpath(cfg, "tables", "route_check.csv"), index=False)
    gp = pd.concat(gaps) if gaps else pd.DataFrame()
    gp.to_csv(wpath(cfg, "tables", "route_gap_candidates.csv"), index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    if len(gp):
        print("\nOff-route ways most often traversed between two route ways (top 5 per route):")
        print(gp.groupby("route").head(5).to_string(index=False))


if __name__ == "__main__":
    main()
