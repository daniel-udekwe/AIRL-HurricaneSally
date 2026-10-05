"""
Run the full revision pipeline.

  python run_all.py                    # everything
  python run_all.py --from 3           # resume from step 3
  python run_all.py --only 0           # inspect raw files only
  python run_all.py --quick            # smoke test (few seeds/folds/epochs)
  python run_all.py --config my_cfg.py
  python run_all.py --only 4 --folds r0914 r0915   # one process per fold subset

Steps: 0 inspect | 1 extract | 2 panels | 3 behavior | 4 forecast | 5 classify | 6 stats | 7 report
Steps 3-5 skip fits whose output already exists, so they can be interrupted and resumed.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

STEPS = ["s00_inspect", "s01_extract", "s02_panels", "s03_behavior",
         "s04_forecast", "s05_classify", "s06_stats", "s07_report"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=0)
    ap.add_argument("--to", dest="stop", type=int, default=7)
    ap.add_argument("--only", type=int)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--config")
    ap.add_argument("--folds", nargs="*")
    a = ap.parse_args()
    env = dict(os.environ)
    if a.quick:
        env["EVAC_QUICK"] = "1"
    if a.config:
        env["EVAC_CONFIG"] = str(Path(a.config).resolve())
    src = Path(__file__).resolve().parent / "src"
    steps = [a.only] if a.only is not None else range(a.start, a.stop + 1)
    # record config, package versions and inputs for this invocation
    os.environ.update(env)
    sys.path.insert(0, str(src))
    from common import load_config, write_manifest
    print("manifest:", write_manifest(load_config(), steps, sys.argv), flush=True)
    for k in steps:
        cmd = [sys.executable, str(src / f"{STEPS[k]}.py")]
        if a.folds and k in (3, 4, 5):
            cmd += ["--folds", *a.folds]
        print(f"\n===== step {k}: {STEPS[k]} =====", flush=True)
        r = subprocess.run(cmd, env=env, cwd=src)
        if r.returncode:
            sys.exit(f"step {k} failed (exit {r.returncode})")


if __name__ == "__main__":
    main()
