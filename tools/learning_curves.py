"""
Regenerate learning curves for forecast fits that were run before curve logging
existed. Re-trains the selected fits with the same seeds and writes only the curve
files (work/curves/...); saved predictions are NOT touched.

  python tools/learning_curves.py                 # last rolling fold, figure horizon
  python tools/learning_curves.py --folds r0915 r0926 --conds none airl route_id
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import load_config, log, make_folds  # noqa: E402
import s04_forecast as F  # noqa: E402


def main():
    cfg = load_config()
    rolling = [f for f in make_folds(cfg) if f["kind"] == "rolling"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="*", default=[rolling[-1]["fold"]])
    ap.add_argument("--h", nargs="*", type=int, default=[cfg["FIG_HORIZON_STEPS"]])
    ap.add_argument("--conds", nargs="*", default=["none", "airl"])
    ap.add_argument("--models", nargs="*", default=["lstm", "rnn"])
    a = ap.parse_args()
    torch.set_num_threads(cfg["N_THREADS"])
    panel = pd.read_parquet(F.wpath(cfg, "panel15.parquet"))
    panel["bin"] = pd.to_datetime(panel.bin, utc=True)
    panel["obs"] = panel.obs.astype(float)
    folds = [f for f in make_folds(cfg) if f["fold"] in a.folds]
    for fold in folds:
        for h in a.h:
            for m in a.models:
                for c in (a.conds if m == "lstm" else ["none"]):
                    for s in cfg["SEEDS"]:
                        log(f"curve {fold['fold']} {m} {c} h{h} s{s}")
                        F.run_one(cfg, panel, fold, cfg["PRIMARY_REGIME"], m, c, h, s, force=True, save_pred=False)


if __name__ == "__main__":
    main()
