"""
Step 5 - Long-term (6-h) congestion classification with rolling origins (editor point 4).

Features from window (t-6h, t]; label = SPI class of (t, t+6h]. Training uses only
windows whose LABEL window ends at or before the fold origin; test windows have
t in [origin, end). Saves class probabilities so per-class metrics, binary F1 and
PR-AUC can be computed later, and records training class counts per fold (before
landfall the training set may contain almost no congested windows - report it).

Outputs: WORK_DIR/classify/<fold>/<model>_<cond>_s<seed>.parquet,
         WORK_DIR/classify/train_class_counts.csv
"""
import sys
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from common import load_config, log, make_folds, set_seed, wpath
from s04_forecast import attach_behavior

BASE6 = ["speed", "spi", "obs", "log_cross", "log_trips", "trip_speed",
         "slot_0", "slot_1", "slot_2", "slot_3", "dow_sin", "dow_cos", "htl", "pre_flag"]


def prep(cfg, panel, fold, cond, seed):
    p, extra = attach_behavior(cfg, panel, cfg["PRIMARY_REGIME"], fold, cond, seed,
                               tcol="wstart", suffix="f6")
    cols = BASE6 + extra
    tr = ((p.t + pd.Timedelta(hours=6)) <= fold["origin"]) & p.label.notna()
    te = (p.t >= fold["origin"]) & (p.t < fold["end"]) & p.label.notna()
    for c in extra:
        p[c] = p[c].fillna(p.loc[p.t <= fold["origin"], c].mean()).fillna(0.0)
    mu = p.loc[tr, cols].mean()
    sd = p.loc[tr, cols].std().replace(0, 1).fillna(1)
    X = ((p[cols] - mu) / sd).to_numpy(np.float32)
    y = p.label.fillna(-1).to_numpy(int)
    return X, y, tr.values, te.values, p


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3))

    def forward(self, x):
        return self.net(x)


def fit_mlp(cfg, X, y, t, seed):
    c = cfg["CLASSIFY"]
    gen = torch.Generator().manual_seed(seed)
    order = np.argsort(t, kind="stable")
    n_va = max(1, int(len(order) * c["val_frac"]))
    va = np.zeros(len(y), bool); va[order[-n_va:]] = True
    w = torch.tensor([c["class_weights"][k] for k in range(3)], dtype=torch.float32)
    model = MLP(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=c["mlp_lr"], weight_decay=1e-4)
    Xt, yt = torch.tensor(X[~va]), torch.tensor(y[~va])
    Xv, yv = torch.tensor(X[va]), torch.tensor(y[va])
    best, state, bad = np.inf, None, 0
    hist = []
    for ep in range(c["mlp_epochs"]):
        model.train()
        tl, nb = 0.0, 0
        for b in torch.randperm(len(Xt), generator=gen).split(c["batch"]):
            loss = nn.functional.cross_entropy(model(Xt[b]), yt[b], weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(b); nb += len(b)
        model.eval()
        with torch.no_grad():
            v = nn.functional.cross_entropy(model(Xv), yv, weight=w).item()
        hist.append(dict(epoch=ep + 1, train_loss=tl / max(nb, 1), val_loss=v))
        if v < best - 1e-6:
            best, state, bad = v, {k: x.clone() for k, x in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= c["patience"]:
                break
    model.load_state_dict(state)
    b = int(np.argmin([r["val_loss"] for r in hist]))
    for k, r in enumerate(hist):
        r["selected"] = (k == b)
    fit_mlp.last_history = hist
    return lambda Z: torch.softmax(model(torch.tensor(Z)), -1).detach().numpy()


def full_proba(est, Z):
    pr = np.zeros((len(Z), 3))
    pr[:, est.classes_.astype(int)] = est.predict_proba(Z)
    return pr


def run_one(cfg, panel, fold, model_name, cond, seed, counts):
    out = wpath(cfg, "classify", fold["fold"], f"{model_name}_{cond}_s{seed}.parquet")
    if out.exists():
        return
    set_seed(seed)
    X, y, tr, te, p = prep(cfg, panel, fold, cond, seed)
    if te.sum() == 0:
        return
    cc = np.bincount(y[tr], minlength=3)
    counts.append(dict(fold=fold["fold"], cond=cond, seed=seed, n0=cc[0], n1=cc[1], n2=cc[2]))
    cw = cfg["CLASSIFY"]["class_weights"]
    if len(np.unique(y[tr])) < 2:
        proba = np.tile(np.eye(3)[np.bincount(y[tr]).argmax()], (te.sum(), 1))
    elif model_name == "mlp":
        proba = fit_mlp(cfg, X[tr], y[tr], p.t.values[tr].astype("int64"), seed)(X[te])
        pd.DataFrame(fit_mlp.last_history).assign(fold=fold["fold"], model=model_name, cond=cond, seed=seed).to_csv(
            wpath(cfg, "curves", "classify", fold["fold"], f"{model_name}_{cond}_s{seed}.csv"), index=False)
    elif model_name == "svm":
        est = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=seed,
                  class_weight={k: v for k, v in cw.items() if k in set(y[tr])}).fit(X[tr], y[tr])
        proba = full_proba(est, X[te])
    else:
        est = KNeighborsClassifier(n_neighbors=min(15, tr.sum()), weights="distance", p=2).fit(X[tr], y[tr])
        proba = full_proba(est, X[te])
    res = p.loc[te, ["route", "t"]].copy()
    res["y"] = y[te]
    res["yhat"] = proba.argmax(1)
    res[["p0", "p1", "p2"]] = proba
    res["fold"], res["model"], res["cond"], res["seed"] = fold["fold"], model_name, cond, seed
    res.to_parquet(out, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="*")
    args = ap.parse_args()
    cfg = load_config()
    torch.set_num_threads(cfg["N_THREADS"])
    panel = pd.read_parquet(wpath(cfg, "panel6h.parquet"))
    panel["t"] = pd.to_datetime(panel.t, utc=True)
    panel["obs"] = panel.obs.astype(float)
    for k in range(4):
        panel[f"slot_{k}"] = (panel.wslot == k).astype(float)
    folds = [f for f in make_folds(cfg) if not args.folds or f["fold"] in args.folds]
    counts = []
    for fold in folds:
        log(f"classify fold {fold['fold']}")
        for m in cfg["CLASSIFY"]["models"]:
            for c in cfg["CONDITIONS"]:
                for s in cfg["SEEDS"]:
                    run_one(cfg, panel, fold, m, c, s, counts)
    if counts:
        cp = wpath(cfg, "classify", "train_class_counts.csv")
        new = pd.DataFrame(counts)
        if cp.exists():
            old = pd.read_csv(cp)
            new = pd.concat([old, new]).drop_duplicates(["fold", "cond", "seed"], keep="last")
        new.to_csv(cp, index=False)
    log("classify done")


if __name__ == "__main__":
    sys.exit(main())
