"""
Step 6 - Statistics (editor points 3 and 5).

* Run-to-run variance: every metric is computed per seed and reported as mean +- SD.
* Dependence-robust inference. With 4 routes a route-clustered bootstrap has only 4
  clusters, so we use instead:
    - Diebold-Mariano test on the loss differential (averaged over seeds and routes at
      each target time), Newey-West HAC variance, Harvey-Leybourne-Newbold correction;
    - day-block bootstrap (resample whole local days, all routes together) for CIs.
* Holm correction within each table family (table x regime x phase).

Outputs (WORK_DIR/stats/): forecast_metrics.csv, forecast_comparisons.csv,
classify_metrics.csv, classify_comparisons.csv, classify_confusion.csv, class_counts.csv
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats as st
from scipy.stats import binomtest

from common import (assign_phase, expand_phase_groups, load_config, local_date, log,
                    make_folds, wpath)


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.sum() == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    m = len(pv)
    adj = np.maximum.accumulate((m - np.arange(m)) * pv[order])
    res = np.empty(m); res[order] = np.minimum(adj, 1)
    out[ok] = res
    return out


def dm_test(d, h):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    T = len(d)
    if T < 10:
        return np.nan, np.nan
    dbar = d.mean()
    L = max(h - 1, int(np.floor(4 * (T / 100) ** (2 / 9))))
    dc = d - dbar
    lrv = dc @ dc / T
    for k in range(1, L + 1):
        lrv += 2 * (1 - k / (L + 1)) * (dc[k:] @ dc[:-k]) / T
    if lrv <= 0:
        return np.nan, np.nan
    stat = dbar / np.sqrt(lrv / T)
    hln = np.sqrt(max((T + 1 - 2 * h + h * (h - 1) / T) / T, 1e-12))
    stat *= hln
    return stat, 2 * (1 - st.t.cdf(abs(stat), T - 1))


def day_boot(values, days, B, rng, stat=np.mean):
    """Bootstrap of stat(values) resampling whole days. values: array or callable(idx)."""
    ud = np.unique(days)
    groups = {d: np.where(days == d)[0] for d in ud}
    out = np.empty(B)
    for b in range(B):
        pick = rng.choice(ud, len(ud), replace=True)
        idx = np.concatenate([groups[d] for d in pick])
        out[b] = values(idx) if callable(values) else stat(values[idx])
    return out


def boot_summary(est, boots):
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    p = 2 * min(np.mean(boots <= 0), np.mean(boots >= 0))
    return lo, hi, min(p, 1.0)


# --------------------------------------------------------------------------
# Forecast
# --------------------------------------------------------------------------
def load_forecast(cfg):
    files = list(wpath(cfg, "forecast", "x").parent.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError("no forecast outputs; run s04 first")
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    df["t_target"] = pd.to_datetime(df.t_target, utc=True)
    df["phase"] = assign_phase(cfg, df.t_target, make_folds(cfg))
    df["ae"] = (df.y - df.yhat).abs()
    df["se"] = (df.y - df.yhat) ** 2
    df["ape"] = df.ae / df.y.abs().clip(lower=1.0) * 100
    return df


def forecast_metrics(cfg, df):
    e = expand_phase_groups(cfg, df)
    m = (e.groupby(["regime", "model", "cond", "h", "seed", "phase"])
         .agg(n=("ae", "size"), MAE=("ae", "mean"), MSE=("se", "mean"), MAPE=("ape", "mean"))
         .reset_index())
    m["RMSE"] = np.sqrt(m.MSE)
    s = (m.groupby(["regime", "model", "cond", "h", "phase"])
         .agg(n=("n", "first"), seeds=("seed", "nunique"),
              MAE=("MAE", "mean"), MAE_sd=("MAE", "std"),
              RMSE=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
              MAPE=("MAPE", "mean"), MAPE_sd=("MAPE", "std")).reset_index())
    return m, s


def compare(cfg, df, ref, cand, family, rng):
    """ref/cand = (regime, model, cond). Positive delta => cand better (lower error)."""
    rows = []
    key = ["route", "t_target", "h"]
    A = df[(df.regime == ref[0]) & (df.model == ref[1]) & (df.cond == ref[2])]
    Bd = df[(df.regime == cand[0]) & (df.model == cand[1]) & (df.cond == cand[2])]
    if A.empty or Bd.empty:
        return rows
    # per-seed deltas (for run-to-run SD); seed-constant baselines broadcast
    a_s = A.groupby(key + ["seed"]).ae.mean().rename("a").reset_index()
    b_s = Bd.groupby(key + ["seed"]).ae.mean().rename("b").reset_index()
    if a_s.seed.nunique() == 1:
        a_s = a_s.drop(columns="seed")
    if b_s.seed.nunique() == 1:
        b_s = b_s.drop(columns="seed")
    on = key + (["seed"] if "seed" in a_s and "seed" in b_s else [])
    ps = a_s.merge(b_s, on=on)
    if "seed" not in ps:
        ps["seed"] = 0
    ps["phase"] = assign_phase(cfg, ps.t_target, make_folds(cfg))
    ps = expand_phase_groups(cfg, ps)
    for (h, ph), g in ps.groupby(["h", "phase"]):
        per_seed = g.groupby("seed").apply(lambda x: (x.a - x.b).mean(), include_groups=False)
        rowavg = g.groupby(["route", "t_target"]).agg(a=("a", "mean"), b=("b", "mean")).reset_index()
        d = (rowavg.a - rowavg.b).values
        series = rowavg.assign(d=d).groupby("t_target").d.mean().sort_index()
        stat, p_dm = dm_test(series.values, int(h))
        days = local_date(cfg, rowavg.t_target)
        boots = day_boot(d, days, cfg["BOOT_B"], rng)
        lo, hi, p_b = boot_summary(d.mean(), boots)
        rows.append(dict(family=family, phase=ph, h=int(h), ref="/".join(ref), cand="/".join(cand),
                         n=len(d), n_days=len(np.unique(days)), MAE_ref=rowavg.a.mean(),
                         MAE_cand=rowavg.b.mean(), dMAE=d.mean(), dMAE_seed_sd=per_seed.std(),
                         dMAE_pct=100 * d.mean() / rowavg.a.mean(), ci_lo=lo, ci_hi=hi,
                         dm_stat=stat, p_dm=p_dm, p_boot=p_b))
    return rows


def forecast_comparisons(cfg, df, rng):
    reg = cfg["PRIMARY_REGIME"]
    rows = []
    single = [c for c in cfg["CONDITIONS"] if "+" not in c and c != "none"]
    combos = [c for c in cfg["CONDITIONS"] if c.startswith("route_id+")]
    for m in cfg["BASELINE_MODELS"]:          # LSTM (no behavior features) vs baselines
        rows += compare(cfg, df, (reg, m, "none"), (reg, "lstm", "none"), "baselines", rng)
    for c in single:                          # feature ladder vs no-feature LSTM
        rows += compare(cfg, df, (reg, "lstm", "none"), (reg, "lstm", c), "features", rng)
    for c in ["empirical", "logit", "bc"]:    # AIRL vs non-adversarial alternatives
        if c in cfg["CONDITIONS"] and "airl" in cfg["CONDITIONS"]:
            rows += compare(cfg, df, (reg, "lstm", c), (reg, "lstm", "airl"), "airl_vs_alt", rng)
    for c in combos:                          # value beyond route identity (route-aware LSTM)
        rows += compare(cfg, df, (reg, "lstm", "route_id"), (reg, "lstm", c), "beyond_route", rng)
    if "route_id+airl" in combos:
        for c in [x for x in combos if x != "route_id+airl"]:
            rows += compare(cfg, df, (reg, "lstm", c), (reg, "lstm", "route_id+airl"),
                            "beyond_route_airl_vs_alt", rng)
    rc = cfg.get("RIDGE_CONDITIONS", ["none"])
    for c in [x for x in rc if x != "none"]:  # behavioral features on the linear AR model
        rows += compare(cfg, df, (reg, "ridge", "none"), (reg, "ridge", c), "ridge_features", rng)
    for ref_c, cand_c in [("empirical", "airl"), ("route_id+empirical", "route_id+airl"),
                          ("route_id", "route_id+airl"), ("route_id", "route_id+empirical")]:
        if ref_c in rc and cand_c in rc:
            rows += compare(cfg, df, (reg, "ridge", ref_c), (reg, "ridge", cand_c), "ridge_contrasts", rng)
    regs = [reg] + list(cfg["SECONDARY_REGIMES"])
    if "2020_only" in regs:                   # transfer learning: vs training from scratch
        for c in cfg["SECONDARY_CONDITIONS"]:
            for r2 in [r for r in regs if r != "2020_only"]:
                rows += compare(cfg, df, ("2020_only", "lstm", c), (r2, "lstm", c), "transfer", rng)
    if "none" in cfg["SECONDARY_CONDITIONS"] and "airl" in cfg["SECONDARY_CONDITIONS"]:
        for r in regs:                        # AIRL effect within each training regime
            rows += compare(cfg, df, (r, "lstm", "none"), (r, "lstm", "airl"), "airl_by_regime", rng)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for col in ["p_dm", "p_boot"]:
        out[col + "_holm"] = out.groupby(["family", "phase"])[col].transform(lambda x: holm(x.values))
    return out


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------
def macro_f1(y, yhat, k=3):
    cm = np.bincount(y * k + yhat, minlength=k * k).reshape(k, k)
    tp = np.diag(cm)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(k), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(k), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)
    present = cm.sum(1) > 0
    return f1[present].mean() if present.any() else np.nan, prec, rec, f1


def pr_auc(y_bin, score):
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return np.nan
    from sklearn.metrics import average_precision_score
    return average_precision_score(y_bin, score)


def load_classify(cfg):
    files = [f for f in wpath(cfg, "classify", "x").parent.rglob("*.parquet")]
    if not files:
        return None
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    df["t"] = pd.to_datetime(df.t, utc=True)
    df["phase"] = assign_phase(cfg, df.t, make_folds(cfg))
    return expand_phase_groups(cfg, df)


def classify_stats(cfg, df, rng):
    mrows = []
    for (m, c, s, ph), g in df.groupby(["model", "cond", "seed", "phase"]):
        y, yh = g.y.values.astype(int), g.yhat.values.astype(int)
        mf, prec, rec, f1 = macro_f1(y, yh)
        yb = (y > 0).astype(int); yhb = (yh > 0).astype(int)
        tpb = (yb & yhb).sum()
        f1b = 2 * tpb / (yb.sum() + yhb.sum()) if (yb.sum() + yhb.sum()) else np.nan
        mrows.append(dict(model=m, cond=c, seed=s, phase=ph, n=len(g), n_light=(y == 1).sum(),
                          n_heavy=(y == 2).sum(), accuracy=(y == yh).mean(), macro_f1=mf,
                          f1_congested_binary=f1b, pr_auc_congested=pr_auc(yb, g.p1.values + g.p2.values),
                          recall_none=rec[0], recall_light=rec[1], recall_heavy=rec[2]))
    met = pd.DataFrame(mrows)
    summ = (met.groupby(["model", "cond", "phase"])
            .agg(n=("n", "first"), n_light=("n_light", "first"), n_heavy=("n_heavy", "first"),
                 accuracy=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
                 macro_f1=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
                 f1_bin=("f1_congested_binary", "mean"), f1_bin_sd=("f1_congested_binary", "std"),
                 pr_auc=("pr_auc_congested", "mean"), pr_auc_sd=("pr_auc_congested", "std"),
                 recall_light=("recall_light", "mean"), recall_heavy=("recall_heavy", "mean"))
            .reset_index())

    crow, conf = [], []
    for (m, ph), g in df.groupby(["model", "phase"]):
        base = g[g.cond == "none"]
        for c in g.cond.unique():
            sub = g[g.cond == c]
            # majority-vote confusion matrix across seeds
            mv = sub.groupby(["route", "t"]).agg(y=("y", "first"),
                                                 yhat=("yhat", lambda x: np.bincount(x, minlength=3).argmax()))
            cm = np.bincount(mv.y.astype(int) * 3 + mv.yhat.astype(int), minlength=9).reshape(3, 3)
            conf.append(dict(model=m, cond=c, phase=ph, **{f"true{i}_pred{j}": cm[i, j]
                                                           for i in range(3) for j in range(3)}))
            if c == "none":
                continue
            j = base.merge(sub, on=["route", "t", "seed"], suffixes=("_b", "_c"))
            if j.empty:
                continue
            seeds = j.seed.unique()
            days = local_date(cfg, j.t)
            yb = j.y_b.values.astype(int)
            hb, hc = j.yhat_b.values.astype(int), j.yhat_c.values.astype(int)
            sd_idx = {s: (j.seed.values == s) for s in seeds}

            def dstat(idx):
                vals = []
                for s in seeds:
                    ii = idx[sd_idx[s][idx]]
                    if len(ii):
                        vals.append(macro_f1(yb[ii], hc[ii])[0] - macro_f1(yb[ii], hb[ii])[0])
                return np.nanmean(vals) if vals else np.nan

            allidx = np.arange(len(j))
            est = dstat(allidx)
            per_seed = [macro_f1(yb[sd_idx[s]], hc[sd_idx[s]])[0] - macro_f1(yb[sd_idx[s]], hb[sd_idx[s]])[0]
                        for s in seeds]
            boots = day_boot(dstat, days, cfg["BOOT_B"], rng)
            lo, hi, p_b = boot_summary(est, boots)
            # McNemar exact on majority-vote correctness (secondary)
            mvj = j.groupby(["route", "t"]).agg(y=("y_b", "first"),
                                                hb=("yhat_b", lambda x: np.bincount(x, minlength=3).argmax()),
                                                hc=("yhat_c", lambda x: np.bincount(x, minlength=3).argmax()))
            cb, cc = mvj.hb == mvj.y, mvj.hc == mvj.y
            n01, n10 = int((cb & ~cc).sum()), int((~cb & cc).sum())
            p_mc = binomtest(n10, n01 + n10, 0.5).pvalue if (n01 + n10) else 1.0
            crow.append(dict(model=m, cond=c, phase=ph, n=int(len(j) / len(seeds)),
                             n_days=len(np.unique(days)), dMacroF1=est, dMacroF1_seed_sd=np.std(per_seed, ddof=1)
                             if len(per_seed) > 1 else np.nan, ci_lo=lo, ci_hi=hi, p_boot=p_b,
                             mcnemar_b_correct_c_wrong=n01, mcnemar_b_wrong_c_correct=n10, p_mcnemar=p_mc))
    comp = pd.DataFrame(crow)
    if not comp.empty:
        for col in ["p_boot", "p_mcnemar"]:
            comp[col + "_holm"] = comp.groupby(["phase"])[col].transform(lambda x: holm(x.values))
    return met, summ, comp, pd.DataFrame(conf)


def main():
    cfg = load_config()
    rng = np.random.default_rng(12345)
    out = wpath(cfg, "stats", "x").parent
    df = load_forecast(cfg)
    per_seed, summ = forecast_metrics(cfg, df)
    per_seed.to_csv(out / "forecast_metrics_by_seed.csv", index=False)
    summ.to_csv(out / "forecast_metrics.csv", index=False)
    comp = forecast_comparisons(cfg, df, rng)
    comp.to_csv(out / "forecast_comparisons.csv", index=False)
    log(f"forecast: {len(summ)} metric rows, {len(comp)} comparisons")
    cdf = load_classify(cfg)
    if cdf is not None:
        met, csumm, ccomp, conf = classify_stats(cfg, cdf, rng)
        met.to_csv(out / "classify_metrics_by_seed.csv", index=False)
        csumm.to_csv(out / "classify_metrics.csv", index=False)
        ccomp.to_csv(out / "classify_comparisons.csv", index=False)
        conf.to_csv(out / "classify_confusion.csv", index=False)
        cnt = (cdf[(cdf.model == cdf.model.iloc[0]) & (cdf.cond == "none") & (cdf.seed == cdf.seed.min())]
               .groupby(["phase", "y"]).size().unstack(fill_value=0))
        cnt.to_csv(out / "class_counts_test.csv")
        log("test class counts by phase:\n" + cnt.to_string())
    log("stats done")


if __name__ == "__main__":
    sys.exit(main())
