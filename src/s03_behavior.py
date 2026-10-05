"""
Step 3 - Route-choice models fit per fold, and the features they produce.

MDP (as in the manuscript, Sec. 3.3.1) at the segment level:
  state  s = (segment i, speed bin, landfall bin)
  action a = next segment, a in A(i) = next segments observed from i in TRAINING data

Four ways to get P(a|s), all over the SAME state space and action sets
(editor point 2 - attribution to the adversarial component):
  empirical : Dirichlet-smoothed counts with backoff (i,sb,lb) -> (i,lb) -> (i)
  logit     : conditional logit, linear in popularity/on-route x bin interactions
  bc        : behavior cloning - the policy network pi_phi trained by cross-entropy
  airl      : the manuscript's AIRL-inspired scheme (Eq. 11-13): f = R_theta + log pi_phi,
              GAN-style BCE of observed vs sampled alternatives, P = softmax(f) over A(i)
From each: H(s) (Eq. 14) and Pmax(s) (Eq. 15). Also log|A(i)| for the topology baseline.

Leakage: every model, action set, speed-bin threshold and seen-landfall-bin set
is built only from transitions with next_t < fold origin. Seeds vary per fit.

Outputs (WORK_DIR/behavior/<regime>/<fold>/):
  <method>_s<seed>_f15.parquet, <method>_s<seed>_f6.parquet  (route-level features)
  <method>_s<seed>_curve.parquet  (landfall-conditioned H, Pmax at fixed speed bin)
  airl_s<seed>_history.csv       (training dynamics, replaces old Fig. 7)
WORK_DIR/behavior/coverage.csv     (test transitions outside training action sets)
"""
import sys
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (landfall, load_config, log, make_folds, read_routes, set_seed, wpath)

SB_EDGES = [0.5, 0.75, 0.9, 1.1]                      # -> 5 speed bins
LB_EDGES = [-96, -48, -24, -12, 0, 12, 24, 48, 96]    # -> 10 landfall bins
N_SB, NO_EVENT, UNK, N_LB = 5, 10, 11, 12


# --------------------------------------------------------------------------
# States
# --------------------------------------------------------------------------
def add_states(cfg, cx, seg_med):
    ratio = (cx.speed_kph / cx.seg.map(seg_med)).fillna(1.0)
    sb = np.digitize(ratio.values, SB_EDGES)
    event = cx.dataset.map(lambda d: cfg["DATASETS"][d]["event"]).values
    h = (cx.t_start - landfall(cfg)).dt.total_seconds().values / 3600.0
    lb = np.where(event, np.digitize(h, LB_EDGES), NO_EVENT)
    return cx.assign(sb=sb.astype(int), lb=lb.astype(int))


def training_transitions(cfg, cx, origin, include_control):
    m = cx.next_seg.notna() & (cx.next_t < origin)
    if not include_control:
        m &= cx.dataset.map(lambda d: cfg["DATASETS"][d]["event"])
    return cx[m]


# --------------------------------------------------------------------------
# Shared tensors for the neural / logit models
# --------------------------------------------------------------------------
class Space:
    def __init__(self, tuples, routes):
        acts = tuples.groupby("seg").next_seg.unique().apply(sorted)
        self.sources = list(acts.index)
        vocab = sorted(set(self.sources) | set(tuples.next_seg))
        self.vid = {s: k + 1 for k, s in enumerate(vocab)}            # 0 = PAD
        self.sid = {s: k for k, s in enumerate(self.sources)}
        self.K = int(acts.apply(len).max())
        S = len(self.sources)
        cand = np.zeros((S, self.K), dtype=np.int64)
        mask = np.zeros((S, self.K), dtype=bool)
        for i, s in enumerate(self.sources):
            ids = [self.vid[a] for a in acts[s]]
            cand[i, :len(ids)], mask[i, :len(ids)] = ids, True
        self.cand, self.mask = torch.tensor(cand), torch.tensor(mask)
        self.acts = acts
        self.n_vocab = len(vocab) + 1
        # popularity + on-route features for the logit model
        n_ia = tuples.groupby(["seg", "next_seg"]).w.sum()
        seg_route = dict(zip(routes.seg, routes.route))
        pop = np.zeros((S, self.K), dtype=np.float32)
        onr = np.zeros((S, self.K), dtype=np.float32)
        for i, s in enumerate(self.sources):
            tot, k = n_ia.loc[s].sum(), len(acts[s])
            for j, a in enumerate(acts[s]):
                pop[i, j] = np.log((n_ia.loc[(s, a)] + 0.5) / (tot + 0.5 * k))
                onr[i, j] = float(seg_route.get(a) == seg_route.get(s))
        self.pop, self.onr = torch.tensor(pop), torch.tensor(onr)

    def encode(self, df):
        src = torch.tensor(df.seg.map(self.sid).values, dtype=torch.long)
        sb = torch.tensor(df.sb.values, dtype=torch.long)
        lb = torch.tensor(df.lb.values, dtype=torch.long)
        return src, sb, lb

    def target_pos(self, df):
        pos = np.empty(len(df), dtype=np.int64)
        for k, (s, a) in enumerate(zip(df.seg.values, df.next_seg.values)):
            pos[k] = self.acts[s].index(a)
        return torch.tensor(pos)


class PairScorer(nn.Module):
    """Scores every candidate action of a state: (B,) states -> (B,K) scores."""

    def __init__(self, n_src, n_vocab, emb, hidden):
        super().__init__()
        self.src = nn.Embedding(n_src, emb)
        self.act = nn.Embedding(n_vocab, emb, padding_idx=0)
        self.sb = nn.Embedding(N_SB, 4)
        self.lb = nn.Embedding(N_LB, 4)
        self.mlp = nn.Sequential(nn.Linear(2 * emb + 8, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, src, sb, lb, cand):
        st = torch.cat([self.src(src), self.sb(sb), self.lb(lb)], -1)
        B, K = cand.shape
        x = torch.cat([st[:, None, :].expand(B, K, st.shape[-1]), self.act(cand)], -1)
        return self.mlp(x).squeeze(-1)


class LogitScorer(nn.Module):
    """Conditional logit: linear in [pop, pop x sb, pop x lb, onroute, onroute x lb]."""

    def __init__(self, space):
        super().__init__()
        self.space = space
        self.w = nn.Linear(1 + N_SB + N_LB + 1 + N_LB, 1, bias=False)

    def forward(self, src, sb, lb, cand):
        pop, onr = self.space.pop[src], self.space.onr[src]          # (B,K)
        sb1 = F.one_hot(sb, N_SB).float()[:, None, :]
        lb1 = F.one_hot(lb, N_LB).float()[:, None, :]
        x = torch.cat([pop[..., None], pop[..., None] * sb1, pop[..., None] * lb1,
                       onr[..., None], onr[..., None] * lb1], -1)
        return self.w(x).squeeze(-1)


def masked(scores, mask):
    return scores.masked_fill(~mask, -1e9)


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------
def _batches(w, cfg, gen):
    bc = cfg["BEHAVIOR"]
    p = torch.tensor(w / w.sum(), dtype=torch.double)
    draws = int(min(bc["draws_per_epoch"], max(w.sum(), bc["batch"])))
    idx = torch.multinomial(p, draws, replacement=True, generator=gen)
    return idx.split(bc["batch"])


def _unk(lb, seen_lb, p, gen):
    lb = lb.clone()
    lb[~torch.isin(lb, seen_lb)] = UNK
    if p > 0:
        drop = torch.rand(lb.shape, generator=gen) < p
        lb[drop & (lb != NO_EVENT)] = UNK
    return lb


def fit_ce(model, space, tup, cfg, seed, seen_lb):
    """Cross-entropy over A(i): used for bc and logit."""
    bc = cfg["BEHAVIOR"]
    gen = torch.Generator().manual_seed(seed)
    src, sb, lb = space.encode(tup)
    tgt = space.target_pos(tup)
    opt = torch.optim.Adam(model.parameters(), lr=bc["lr"], weight_decay=bc["weight_decay"])
    hist = []
    for ep in range(bc["epochs"]):
        tot, n = 0.0, 0
        for b in _batches(tup.w.values, cfg, gen):
            s = src[b]
            lbb = _unk(lb[b], seen_lb, bc["unk_p"] if isinstance(model, PairScorer) else 0, gen)
            logits = masked(model(s, sb[b], lbb, space.cand[s]), space.mask[s])
            loss = F.cross_entropy(logits, tgt[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b); n += len(b)
        hist.append(dict(epoch=ep + 1, cross_entropy=tot / max(n, 1)))
    model.history = pd.DataFrame(hist)
    return model


class AIRLInspired(nn.Module):
    def __init__(self, space, cfg):
        super().__init__()
        bc = cfg["BEHAVIOR"]
        self.R = PairScorer(len(space.sources), space.n_vocab, bc["emb"], bc["hidden"])
        self.pi = PairScorer(len(space.sources), space.n_vocab, bc["emb"], bc["hidden"])
        self.space = space

    def f(self, src, sb, lb):
        cand, mask = self.space.cand[src], self.space.mask[src]
        logpi = F.log_softmax(masked(self.pi(src, sb, lb, cand), mask), -1)
        return masked(self.R(src, sb, lb, cand) + logpi, mask)


def fit_airl(space, tup, cfg, seed, seen_lb):
    """Eq. 12: maximize E[log s(f(s,a))] + E[log(1 - s(f(s,a~)))] over theta, phi."""
    bc = cfg["BEHAVIOR"]
    gen = torch.Generator().manual_seed(seed)
    model = AIRLInspired(space, cfg)
    tup = tup[tup.seg.map(lambda s: len(space.acts[s])) >= 2]      # need an alternative
    if len(tup) == 0:
        return model, pd.DataFrame()
    src, sb, lb = space.encode(tup)
    tgt = space.target_pos(tup)
    opt = torch.optim.Adam(model.parameters(), lr=bc["lr"], weight_decay=bc["weight_decay"])
    hist = []
    for ep in range(bc["epochs"]):
        stats = []
        for b in _batches(tup.w.values, cfg, gen):
            s = src[b]
            lbb = _unk(lb[b], seen_lb, bc["unk_p"], gen)
            f = model.f(s, sb[b], lbb)
            t = tgt[b]
            # negative: uniform over valid alternatives != observed action
            r = torch.rand(f.shape, generator=gen).masked_fill(~space.mask[s], -1)
            r[torch.arange(len(t)), t] = -1
            neg = r.argmax(-1)
            fp = f[torch.arange(len(t)), t]
            fn = f[torch.arange(len(t)), neg]
            loss = (F.softplus(-fp) + F.softplus(fn)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            stats.append([loss.item(), torch.sigmoid(fp).mean().item(), torch.sigmoid(fn).mean().item()])
        st = np.mean(stats, 0)
        with torch.no_grad():
            P = F.softmax(model.f(src[:4096], sb[:4096], _unk(lb[:4096], seen_lb, 0, gen)), -1)
            H = -(P * torch.log(P.clamp_min(1e-12))).sum(-1).mean().item()
            Pm = P.max(-1).values.mean().item()
        hist.append(dict(epoch=ep + 1, loss=st[0], D_expert=st[1], D_policy=st[2], entropy=H, pmax=Pm))
    return model, pd.DataFrame(hist)


# --------------------------------------------------------------------------
# Feature computation for unique states
# --------------------------------------------------------------------------
def entropy_pmax(P):
    H = -(P * np.log(np.clip(P, 1e-12, None))).sum(-1)
    return H, P.max(-1)


def nn_features(model_fn, space, states, seen_lb):
    known = states.seg.isin(space.sid)
    H = np.full(len(states), np.nan)
    Pm = np.full(len(states), np.nan)
    if known.any():
        st = states[known]
        src, sb, lb = space.encode(st)
        lb = _unk(lb, seen_lb, 0, None)
        out_h, out_p = [], []
        with torch.no_grad():
            for k in range(0, len(st), 8192):
                sl = slice(k, k + 8192)
                P = F.softmax(model_fn(src[sl], sb[sl], lb[sl]), -1).numpy()
                h, p = entropy_pmax(P)
                out_h.append(h); out_p.append(p)
        H[known.values], Pm[known.values] = np.concatenate(out_h), np.concatenate(out_p)
    return H, Pm


def empirical_features(tup, states, alpha):
    acts = tup.groupby("seg").next_seg.unique()
    known = states[states.seg.isin(acts.index)].reset_index()
    long = known.merge(acts.rename("a").explode().reset_index(), on="seg")
    long["K"] = long.seg.map(acts.apply(len))
    n_sa = tup.groupby(["seg", "sb", "lb", "next_seg"]).w.sum().rename("n_sa")
    n_s = tup.groupby(["seg", "sb", "lb"]).w.sum().rename("n_s")
    n_ila = tup.groupby(["seg", "lb", "next_seg"]).w.sum().rename("n_ila")
    n_il = tup.groupby(["seg", "lb"]).w.sum().rename("n_il")
    n_ia = tup.groupby(["seg", "next_seg"]).w.sum().rename("n_ia")
    n_i = tup.groupby("seg").w.sum().rename("n_i")
    long = (long.join(n_sa, on=["seg", "sb", "lb", "a"]).join(n_s, on=["seg", "sb", "lb"])
            .join(n_ila, on=["seg", "lb", "a"]).join(n_il, on=["seg", "lb"])
            .join(n_ia, on=["seg", "a"]).join(n_i, on="seg")).fillna(0)
    p_i = (long.n_ia + alpha / long.K) / (long.n_i + alpha)
    p_il = (long.n_ila + alpha * p_i) / (long.n_il + alpha)
    long["p"] = (long.n_sa + alpha * p_il) / (long.n_s + alpha)
    long["plogp"] = -long.p * np.log(long.p.clip(1e-12))
    g = long.groupby("index").agg(H=("plogp", "sum"), P=("p", "max"))
    H = np.full(len(states), np.nan); Pm = np.full(len(states), np.nan)
    H[g.index.values], Pm[g.index.values] = g.H.values, g.P.values
    return H, Pm


# --------------------------------------------------------------------------
# Aggregation to route panels
# --------------------------------------------------------------------------
def aggregate(cfg, cx_feat):
    """Mean of crossing-level features per route x 15-min bin and route x 6-h window."""
    cols = ["H", "Pmax", "logA"]
    f15 = (cx_feat.assign(bin=cx_feat.t_start.dt.floor("15min"))
           .groupby(["dataset", "route", "bin"])[cols].mean().reset_index())
    loc = cx_feat.t_start.dt.tz_convert(cfg["LOCAL_TZ"]).dt.tz_localize(None)
    f6 = (cx_feat.assign(wstart=loc.dt.floor("6h"))
          .groupby(["dataset", "route", "wstart"])[cols].mean().reset_index())
    return f15, f6


def landfall_curve(model_fn, space, routes, seen_lb, sb_fixed, method, tup, alpha):
    segs = routes[routes.seg.isin(space.sid)]
    rows = [(s, r, sb_fixed, lb) for s, r in zip(segs.seg, segs.route) for lb in range(NO_EVENT + 1)]
    st = pd.DataFrame(rows, columns=["seg", "route", "sb", "lb"])
    if method == "empirical":
        H, P = empirical_features(tup, st, alpha)
    else:
        H, P = nn_features(model_fn, space, st, seen_lb)
    st["H"], st["Pmax"] = H, P
    st["lb_seen"] = st.lb.isin(seen_lb.numpy())
    return st.groupby(["route", "lb", "lb_seen"])[["H", "Pmax"]].mean().reset_index()


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run_fold(cfg, cx, routes, fold, regime, seeds, methods, coverage_rows):
    include_control = regime in ("pooled", "finetune")
    tr = training_transitions(cfg, cx, fold["origin"], include_control)
    tr_all = cx[cx.t_start < fold["origin"]] if include_control else \
        cx[(cx.t_start < fold["origin"]) & cx.dataset.map(lambda d: cfg["DATASETS"][d]["event"])]
    seg_med = tr_all.groupby("seg").speed_kph.median()
    tr = add_states(cfg, tr, seg_med)
    cxs = add_states(cfg, cx, seg_med)
    tup = tr.groupby(["seg", "sb", "lb", "next_seg"]).size().rename("w").reset_index()
    seen_lb = torch.tensor(sorted(set(tup.lb)), dtype=torch.long)
    acts = tup.groupby("seg").next_seg.unique()
    logA = np.log(acts.apply(len))

    # coverage of TEST transitions by training action sets
    te = cxs[(cxs.t_start >= fold["origin"]) & (cxs.t_start < fold["end"]) & cxs.next_seg.notna()]
    if len(te):
        src_known = te.seg.isin(acts.index)
        in_A = [a in set(acts.get(s, [])) for s, a in zip(te.seg, te.next_seg)]
        coverage_rows.append(dict(regime=regime, fold=fold["fold"], test_transitions=len(te),
                                  frac_source_unseen=1 - src_known.mean(),
                                  frac_action_outside_A=1 - np.mean(in_A),
                                  test_lb_unseen=float((~te.lb.isin(seen_lb.numpy())).mean())))

    states = cxs[["seg", "sb", "lb"]].drop_duplicates().reset_index(drop=True)
    space = Space(tup, routes)
    out_dir = wpath(cfg, "behavior", regime, fold["fold"], "x").parent
    for method in methods:
        for seed in (seeds if method != "empirical" else [seeds[0]]):
            f15p = out_dir / f"{method}_s{seed}_f15.parquet"
            if f15p.exists():
                continue
            set_seed(seed)
            model_fn = None
            if method == "empirical":
                H, P = empirical_features(tup, states, cfg["BEHAVIOR"]["alpha"])
            elif method == "logit":
                m = fit_ce(LogitScorer(space), space, tup, cfg, seed, seen_lb)
                model_fn = lambda s, sb, lb, m=m: masked(m(s, sb, lb, space.cand[s]), space.mask[s])
            elif method == "bc":
                m = fit_ce(PairScorer(len(space.sources), space.n_vocab, cfg["BEHAVIOR"]["emb"],
                                      cfg["BEHAVIOR"]["hidden"]), space, tup, cfg, seed, seen_lb)
                model_fn = lambda s, sb, lb, m=m: masked(m(s, sb, lb, space.cand[s]), space.mask[s])
            elif method == "airl":
                m, hist = fit_airl(space, tup, cfg, seed, seen_lb)
                hist.to_csv(out_dir / f"airl_s{seed}_history.csv", index=False)
                model_fn = lambda s, sb, lb, m=m: m.f(s, sb, lb)
            if method in ("bc", "logit"):
                m.history.to_csv(out_dir / f"{method}_s{seed}_history.csv", index=False)
            if model_fn is not None:
                H, P = nn_features(model_fn, space, states, seen_lb)
            st = states.assign(H=H, Pmax=P, logA=states.seg.map(logA).values)
            feat = cxs[["dataset", "route", "t_start", "seg", "sb", "lb"]].merge(st, on=["seg", "sb", "lb"], how="left")
            # deterministic choice for single-action segments
            single = feat.logA == 0
            feat.loc[single, "H"], feat.loc[single, "Pmax"] = 0.0, 1.0
            f15, f6 = aggregate(cfg, feat)
            f15.to_parquet(f15p, index=False)
            f6.to_parquet(out_dir / f"{method}_s{seed}_f6.parquet", index=False)
            landfall_curve(model_fn, space, routes, seen_lb, cfg["SB_FIXED_FOR_CURVES"], method,
                           tup, cfg["BEHAVIOR"]["alpha"]).to_parquet(
                out_dir / f"{method}_s{seed}_curve.parquet", index=False)
    log(f"  behavior {regime}/{fold['fold']}: {len(tup):,} unique transitions, "
        f"{len(space.sources)} source segments, max |A|={space.K}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="*", help="subset of fold names to run")
    args = ap.parse_args()
    cfg = load_config()
    torch.set_num_threads(cfg["N_THREADS"])
    cx = pd.read_parquet(wpath(cfg, "crossings.parquet"))
    cx["t_start"] = pd.to_datetime(cx.t_start, utc=True)
    cx["next_t"] = pd.to_datetime(cx.next_t, utc=True)
    routes = read_routes(cfg)
    folds = [f for f in make_folds(cfg) if not args.folds or f["fold"] in args.folds]
    coverage = []
    # finetune and pretrain_only reuse the pooled behavior models (fitted on 2019 + 2020 pre-origin)
    regimes = [cfg["PRIMARY_REGIME"]] + [r for r in cfg["SECONDARY_REGIMES"]
                                         if r not in ("finetune", "pretrain_only")]
    for regime in regimes:
        methods = cfg["BEHAVIOR"]["methods"] if regime == cfg["PRIMARY_REGIME"] else ["airl"]
        for fold in folds:
            run_fold(cfg, cx, routes, fold, regime, cfg["SEEDS"], methods, coverage)
    if coverage:
        cov = pd.DataFrame(coverage)
        p = wpath(cfg, "behavior", "coverage.csv")
        if p.exists() and args.folds:
            old = pd.read_csv(p)
            cov = pd.concat([old[~old.set_index(["regime", "fold"]).index.isin(
                cov.set_index(["regime", "fold"]).index)], cov])
        cov.to_csv(p, index=False)
    log("behavior done")


if __name__ == "__main__":
    sys.exit(main())
