"""
Step 0 - Inspect raw files before the long extraction.

Prints, for each dataset: number of files, header files, first rows, the
resolved column mapping, and how many route segments appear in a sample of
the first trajectory file. Run this first; if a mapping fails, edit
COLUMN_CANDIDATES (or CSV_DELIM) in config.py.
"""
import sys
import gzip

import duckdb

from common import dataset_files, load_config, log, read_routes
from s01_extract import csv_rel, detect_schema, way_expr


def main():
    cfg = load_config()
    routes = read_routes(cfg)
    print(routes.groupby("route").size().rename("ways_listed").to_string(), "\n")
    con = duckdb.connect()
    con.register("rs", routes[["seg", "route"]])
    for ds in cfg["DATASETS"]:
        f = dataset_files(cfg, ds)
        print("=" * 78, f"\n{ds}: {len(f['trajs'])} traj files (trips files are not used)")
        print("traj header file:", f["traj_header"])
        if not f["trajs"]:
            print("  !! no traj files"); continue
        with gzip.open(f["trajs"][0], "rt", errors="replace") as fh:
            print(f"\n  first 3 lines of {f['trajs'][0].name}:")
            for _ in range(3):
                print("   ", fh.readline().rstrip()[:300])
        names, hh, m = detect_schema(cfg, f["trajs"], f["traj_header"], "traj")
        print(f"  header row in data: {hh}\n  columns: {names}\n  mapping: {m}")
        rel = csv_rel(cfg, f["trajs"][:1], names, hh)
        way = way_expr(cfg, m["seg"])
        df = con.execute(f"""SELECT r.route, count(*) AS subsegment_rows, count(DISTINCT t.w) AS ways
            FROM (SELECT {way} AS w FROM {rel} LIMIT 2000000) t JOIN rs r ON t.w = r.seg
            GROUP BY 1 ORDER BY 1""").df()
        print("  route matches in first 2M rows of first file:\n", df.to_string(index=False))
        if df.empty:
            print("  !! no route IDs matched - check SEGMENT_WAY_SEPARATOR in config.py")
    log("inspect done")


if __name__ == "__main__":
    sys.exit(main())
