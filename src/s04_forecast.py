"""
Step 4 - Short-term speed forecasting with rolling origins (editor points 1-3).

For every fold x regime x horizon x seed x (baseline model | LSTM feature condition):
train on samples whose TARGET time is before the fold origin, test on samples whose
target falls in [origin, end). The model is frozen at the origin; inputs stream in.

Outputs: WORK_DIR/forecast/<regime>/<fold>/<model>_<cond>_h<h>_s<seed>.parquet
(one row per test sample: route, t_issue, t_target, y, yhat). Existing files are
skipped, so the step can be stopped and resumed, or split across processes with
--folds.
"""
import sys
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge

from common import load_config, log, make_folds, route_list, set_seed, wpath

BASE = ["speed", "obs", "log_since", "log_cross", "log_trips", "trip_speed",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "htl", "pre_flag"]


def _single_columns(cfg, cond):
    if cond == "none":
        return []
    if cond == "route_id":
        return [f"route_{r}" for r in route_list(cfg)]
    if cond == "topology":
        return ["logA"]
    if cond == "airl_H":
        return ["H"]
    if cond == "airl_P":
        return ["Pmax"]
    return ["H", "Pmax"]


def cond_columns(cfg, cond):
    """Columns for a condition; "a+b" combines conditions (e.g. route_id+airl)."""
    cols = []
    for part in cond.split("+"):
        cols += [c for c in _single_columns(cfg, part) if c not in cols]
    return cols


def behavior_method(cond):
    """Route-choice model whose features a condition uses (None if it uses none)."""
    for part in cond.split("+"):
        if part in ("none", "route_id"):
            continue
        return {"topology": "empirical", "airl_H": "airl", "airl_P": "airl"}.get(part, part)
    return None


BEHAVIOR_REGIME = {"finetune": "pooled", "pretrain_only": "pooled"}


def attach_behavior(cfg, panel, regime, fold, cond, seed, tcol="bin", suffix="f15"):
    """Merge route-level behavior features for this fold into a panel; fill gaps causally."""
    p = panel.copy()
    for r in route_list(cfg):
        p[f"route_{r}"] = (p.route == r).astype(float)
    cols = cond_columns(cfg, cond)
    need = [c for c in cols if c in ("H", "Pmax", "logA")]
    if not need:
        return p, cols
    breg = BEHAVIOR_REGIME.get(regime, regime)
    method = behavior_method(cond)
    s = seed if method != "empirical" else cfg["SEEDS"][0]
    f = pd.read_parquet(wpath(cfg, "behavior", breg, fold["fold"], f"{method}_s{s}_{suffix}.parquet"))
    if tcol == "bin":
        f["bin"] = pd.to_datetime(f["bin"], utc=True)
    p = p.merge(f[["dataset", "route", tcol] + need], on=["dataset", "route", tcol], how="left")
    for c in need:   # forward-fill within series (causal), then training mean
        p[c] = p.groupby(["dataset", "route"])[c].ffill()
    return p, cols


def windows(cfg, p, cols, h, L):
    """Sequence samples per (dataset, route): X (n,L,d), y, meta."""
    Xs, ys, meta = [], [], []
    for (ds, r), g in p.groupby(["dataset", "route"], sort=False):
        g = g.sort_values("bin")
        F_ = g[cols].to_numpy(np.float32)
        y = g.speed_obs.to_numpy(np.float32)
        last = g.speed_raw.to_numpy(np.float32)
        t = pd.DatetimeIndex(g.bin)
        n = len(g)
        idx = np.arange(L - 1, n - h)
        ok = ~np.isnan(y[idx + h])
        idx = idx[ok]
        if not len(idx):
            continue
        win = np.lib.stride_tricks.sliding_window_view(F_, (L, F_.shape[1]))[:, 0]   # (n-L+1, L, d)
        Xs.append(win[idx - (L - 1)])
        ys.append(y[idx + h])
        meta.append(pd.DataFrame({"dataset": ds, "route": r, "t_issue": t[idx],
                                  "t_target": t[idx + h], "persist": last[idx]}))
    return np.concatenate(Xs), np.concatenate(ys), pd.concat(meta, ignore_index=True)


class SeqNet(nn.Module):
    def __init__(self, d_in, hidden, cell):
        super().__init__()
        self.rnn = (nn.LSTM if cell == "lstm" else nn.RNN)(d_in, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


def train_seq(model, Xtr, ytr, Xva, yva, lr, max_epochs, patience, batch, clip, seed,
              history=None, stage="train"):
    """Train with early stopping; appends per-epoch losses to `history` (list) if given."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr, ytr = torch.tensor(Xtr), torch.tensor(ytr)
    Xva, yva = torch.tensor(Xva), torch.tensor(yva)
    best, best_state, bad = np.inf, None, 0
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        tl, nb = 0.0, 0
        for b in perm.split(batch):
            loss = nn.functional.mse_loss(model(Xtr[b]), ytr[b])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            tl += loss.item() * len(b); nb += len(b)
        model.eval()
        with torch.no_grad():
            v = nn.functional.mse_loss(model(Xva), yva).item() if len(Xva) else 0.0
        if history is not None:
            history.append(dict(stage=stage, epoch=ep + 1, train_mse=tl / max(nb, 1), val_mse=v,
                                n_train=len(Xtr), n_val=len(Xva)))
        if v < best - 1e-6:
            best, best_state, bad = v, {k: x.clone() for k, x in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    if history is not None:
        rows = [r for r in history if r["stage"] == stage]
        if rows:
            b = int(np.argmin([r["val_mse"] for r in rows]))
            for k, r in enumerate(rows):
                r["selected"] = (k == b)
    return model


def chrono_val(meta_tr, frac):
    order = np.argsort(meta_tr.t_target.values, kind="stable")
    n_va = max(1, int(len(order) * frac))
    va = np.zeros(len(order), bool)
    va[order[-n_va:]] = True
    return va


def run_one(cfg, panel, fold, regime, model_name, cond, h, seed, force=False, save_pred=True):
    stem = f"{model_name}_{cond}_h{h}_s{seed}"
    out = wpath(cfg, "forecast", regime, fold["fold"], stem + ".parquet")
    curve = wpath(cfg, "curves", "forecast", regime, fold["fold"], stem + ".csv")
    if out.exists() and not force:
        return
    hist = []
    set_seed(seed)
    p, extra = attach_behavior(cfg, panel, regime, fold, cond, seed)
    cols = BASE + extra
    if regime == "2020_only":
        p = p[p.dataset.map(lambda d: cfg["DATASETS"][d]["event"])]
    p = p[p.bin < fold["end"] + pd.Timedelta(minutes=15 * (h + 1))]
    # training rows for scaling: bins strictly before origin (2019 only for pretrain_only)
    trm = p.bin < fold["origin"]
    if regime == "pretrain_only":
        trm &= ~p.dataset.map(lambda d: cfg["DATASETS"][d]["event"])
    for c in extra:   # fill remaining gaps with training mean (no test information)
        p[c] = p[c].fillna(p.loc[trm, c].mean()).fillna(0.0)
    p["speed_raw"] = p["speed"]
    mu = p.loc[trm, cols].mean()
    sd = p.loc[trm, cols].std().replace(0, 1).fillna(1)
    p[cols] = (p[cols] - mu) / sd
    X, y, meta = windows(cfg, p, cols, h, cfg["SEQ_LEN"])
    tr = np.array(meta.t_target < fold["origin"], dtype=bool)
    if regime == "pretrain_only":
        tr &= ~np.array(meta.dataset.map(lambda d: cfg["DATASETS"][d]["event"]), dtype=bool)
    te = np.array((meta.t_target >= fold["origin"]) & (meta.t_target < fold["end"]), dtype=bool)
    if te.sum() == 0 or tr.sum() == 0:
        log(f"   skip {fold['fold']} {model_name} {cond} h{h}: no train/test samples")
        return
    ymu, ysd = y[tr].mean(), y[tr].std() or 1.0
    ys = (y - ymu) / ysd
    lc = cfg["LSTM"]

    if model_name == "persistence":
        yhat = meta.persist.values[te]
    elif model_name == "ridge":
        Xf = X.reshape(len(X), -1)          # base features + any condition columns
        yhat = Ridge(alpha=1.0).fit(Xf[tr], ys[tr]).predict(Xf[te]) * ysd + ymu
    else:
        cell = "rnn" if model_name == "rnn" else "lstm"
        net = SeqNet(X.shape[2], lc["hidden"], cell)
        if regime == "finetune":
            event = meta.dataset.map(lambda d: cfg["DATASETS"][d]["event"]).values
            pre = tr & ~event
            va = chrono_val(meta[pre], lc["val_frac"])
            Xp, yp = X[pre], ys[pre]
            net = train_seq(net, Xp[~va], yp[~va], Xp[va], yp[va], lc["lr"], lc["max_epochs"],
                            lc["patience"], lc["batch"], lc["clip"], seed, history=hist, stage="pretrain")
            ft = tr & event
            va = chrono_val(meta[ft], lc["val_frac"])
            Xf_, yf_ = X[ft], ys[ft]
            net = train_seq(net, Xf_[~va], yf_[~va], Xf_[va], yf_[va], cfg["FINETUNE"]["lr"],
                            cfg["FINETUNE"]["max_epochs"], lc["patience"], lc["batch"], lc["clip"], seed, history=hist, stage="finetune")
        else:
            va = chrono_val(meta[tr], lc["val_frac"])
            Xt, yt = X[tr], ys[tr]
            net = train_seq(net, Xt[~va], yt[~va], Xt[va], yt[va], lc["lr"], lc["max_epochs"],
                            lc["patience"], lc["batch"], lc["clip"], seed, history=hist, stage="train")
        net.eval()
        with torch.no_grad():
            yhat = net(torch.tensor(X[te])).numpy() * ysd + ymu
        if cfg.get("SAVE_MODELS"):
            torch.save(net.state_dict(), wpath(cfg, "models", regime, fold["fold"], stem + ".pt"))
    if hist:
        pd.DataFrame(hist).assign(fold=fold["fold"], regime=regime, model=model_name, cond=cond,
                                  h=h, seed=seed).to_csv(curve, index=False)
    if not save_pred:
        return
    res = meta[te][["route", "t_issue", "t_target"]].copy()
    res["y"], res["yhat"] = y[te], yhat
    res["fold"], res["regime"], res["model"], res["cond"], res["h"], res["seed"] = (
        fold["fold"], regime, model_name, cond, h, seed)
    res.to_parquet(out, index=False)


def _seeds_for(cfg, cond):
    m = behavior_method(cond)
    return cfg["SEEDS"] if m in ("airl", "bc", "logit") else cfg["SEEDS"][:1]


def jobs(cfg):
    J = []
    reg = cfg["PRIMARY_REGIME"]
    for h in cfg["HORIZONS_STEPS"]:
        J.append((reg, "persistence", "none", h, cfg["SEEDS"][0]))
        if "rnn" in cfg["BASELINE_MODELS"]:
            J += [(reg, "rnn", "none", h, s) for s in cfg["SEEDS"]]
        for c in cfg.get("RIDGE_CONDITIONS", ["none"]):
            J += [(reg, "ridge", c, h, s) for s in _seeds_for(cfg, c)]
        J += [(reg, "lstm", c, h, s) for c in cfg["CONDITIONS"] for s in cfg["SEEDS"]]
        for r2 in cfg["SECONDARY_REGIMES"]:
            J += [(r2, "lstm", c, h, s) for c in cfg["SECONDARY_CONDITIONS"] for s in cfg["SEEDS"]]
    return J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="*")
    args = ap.parse_args()
    cfg = load_config()
    torch.set_num_threads(cfg["N_THREADS"])
    panel = pd.read_parquet(wpath(cfg, "panel15.parquet"))
    panel["bin"] = pd.to_datetime(panel.bin, utc=True)
    panel["obs"] = panel.obs.astype(float)
    folds = [f for f in make_folds(cfg) if not args.folds or f["fold"] in args.folds]
    J = jobs(cfg)
    for fold in folds:
        log(f"forecast fold {fold['fold']} ({len(J)} fits)")
        for k, (reg, m, c, h, s) in enumerate(J):
            run_one(cfg, panel, fold, reg, m, c, h, s)
    log("forecast done")


if __name__ == "__main__":
    sys.exit(main())
