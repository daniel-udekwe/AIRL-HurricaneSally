"""
Step 7 - Tables (CSV + LaTeX) and figures for the revised manuscript.

WORK_DIR/tables/ : T1_data_summary, T2_baselines, T3_feature_ladder, T4_airl_vs_alternatives,
                   T5_classification, T6_classification_deltas, T7_class_counts,
                   T8_action_coverage, T9_regimes, headline_numbers.json
WORK_DIR/figures/: fig_heldout_landfall_forecasts, fig_mae_by_phase,
                   fig_landfall_conditioned_features, fig_airl_training, fig_action_coverage
"""
import sys
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import landfall, load_config, log, make_folds, route_list, wpath

H_MIN = {1: 15, 2: 30, 3: 45, 4: 60, 6: 90, 8: 120}
PHASE_ORDER = ["pre_landfall", "landfall", "recovery", "acute", "rolling_all", "late", "all"]


def save(df, name, cfg, index=False, floatfmt="%.3f"):
    d = wpath(cfg, "tables", "x").parent
    df.to_csv(d / f"{name}.csv", index=index)
    try:
        (d / f"{name}.tex").write_text(df.to_latex(index=index, float_format=floatfmt.__mod__, na_rep="--"))
    except Exception as e:  # jinja2 missing etc.
        log(f"  (LaTeX export skipped for {name}: {e})")


def fmt_pm(m, s, nd=2):
    return f"{m:.{nd}f} ± {s:.{nd}f}" if pd.notna(s) else f"{m:.{nd}f}"


def stars(p):
    return "" if pd.isna(p) else "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def comp_table(comp, fam):
    c = comp[comp.family == fam].copy()
    if c.empty:
        return c
    c["horizon_min"] = c.h.map(H_MIN)
    c["dMAE [95% CI]"] = c.apply(lambda r: f"{r.dMAE:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]", axis=1)
    c["seed SD"] = c.dMAE_seed_sd.round(2)
    c["p_DM (Holm)"] = c.apply(lambda r: f"{r.p_dm_holm:.3f}{stars(r.p_dm_holm)}", axis=1)
    c["p_boot (Holm)"] = c.apply(lambda r: f"{r.p_boot_holm:.3f}{stars(r.p_boot_holm)}", axis=1)
    c["phase"] = pd.Categorical(c.phase, [p for p in PHASE_ORDER if p in set(c.phase)] +
                                sorted(set(c.phase) - set(PHASE_ORDER)))
    return c.sort_values(["phase", "h", "cand"])[
        ["phase", "horizon_min", "ref", "cand", "n", "n_days", "MAE_ref", "MAE_cand",
         "dMAE [95% CI]", "seed SD", "dMAE_pct", "p_DM (Holm)", "p_boot (Holm)"]]


def fig_forecasts(cfg, figs):
    h = cfg["FIG_HORIZON_STEPS"]
    base = wpath(cfg, "forecast", cfg["PRIMARY_REGIME"], "x").parent
    files = [f for f in base.rglob(f"*_h{h}_s*.parquet")
             if f.parent.name.startswith("r") and f.name.split("_")[1] in ("none", "airl", "empirical")]
    if not files:
        return
    df = pd.concat(pd.read_parquet(f) for f in files)
    df["t_target"] = pd.to_datetime(df.t_target, utc=True).dt.tz_convert(cfg["LOCAL_TZ"])
    routes = route_list(cfg)
    fig, axes = plt.subplots(len(routes), 1, figsize=(11, 2.6 * len(routes)), sharex=True)
    axes = np.atleast_1d(axes)
    L = landfall(cfg).tz_convert(cfg["LOCAL_TZ"])
    styles = {("persistence", "none"): ("Persistence", "0.6", ":"),
              ("lstm", "none"): ("LSTM, no behavior features", "C0", "-"),
              ("lstm", "empirical"): ("LSTM + empirical-transition features", "C2", "-"),
              ("lstm", "airl"): ("LSTM + AIRL-inspired features", "C3", "-")}
    for ax, r in zip(axes, routes):
        g = df[df.route == r]
        obs = g.groupby("t_target").y.first()
        ax.plot(obs.index, obs.values, "k-", lw=1.2, label="Observed")
        for (m, c), (lab, col, ls) in styles.items():
            s = g[(g.model == m) & (g.cond == c)].groupby("t_target").yhat.agg(["mean", "std"])
            if s.empty:
                continue
            ax.plot(s.index, s["mean"], color=col, ls=ls, lw=1, label=lab)
            if s["std"].notna().any():
                ax.fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"], color=col, alpha=.15, lw=0)
        ax.axvline(L, color="k", ls="--", lw=.8)
        for f in make_folds(cfg):
            if f["kind"] == "rolling":
                ax.axvline(f["origin"].tz_convert(cfg["LOCAL_TZ"]), color="0.85", lw=.5, zorder=0)
        ax.set_ylabel(f"Route {r}\nspeed (kph)")
    axes[0].legend(ncol=3, fontsize=8, loc="lower left")
    axes[-1].set_xlabel("Local time (dashed: landfall; grey: daily forecast origins)")
    fig.suptitle(f"Held-out {H_MIN.get(h, h * 15)}-min forecasts around landfall "
                 f"(each day predicted by a model trained only on earlier data; band = ±1 SD over seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(figs / "fig_heldout_landfall_forecasts.png", dpi=200)
    plt.close(fig)


def fig_mae_phase(cfg, figs, summ):
    h = cfg["FIG_HORIZON_STEPS"]
    s = summ[(summ.regime == cfg["PRIMARY_REGIME"]) & (summ.h == h) &
             (summ.phase.isin(["pre_landfall", "landfall", "recovery", "late"]))]
    s = s[(s.model == "lstm") | (s.cond == "none")].copy()
    s["label"] = np.where(s.model == "lstm", "LSTM:" + s.cond, s.model)
    phases = [p for p in ["pre_landfall", "landfall", "recovery", "late"] if p in set(s.phase)]
    labels = list(dict.fromkeys(s.sort_values(["model", "cond"]).label))
    fig, ax = plt.subplots(figsize=(11, 4))
    w = 0.8 / max(len(labels), 1)
    for k, lab in enumerate(labels):
        sub = s[s.label == lab].set_index("phase").reindex(phases)
        ax.bar(np.arange(len(phases)) + k * w, sub.MAE, w, yerr=sub.MAE_sd.fillna(0), label=lab, capsize=2)
    ax.set_xticks(np.arange(len(phases)) + 0.4 - w / 2)
    ax.set_xticklabels(phases)
    ax.set_ylabel(f"MAE (kph), {H_MIN.get(h, h * 15)}-min horizon")
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout(); fig.savefig(figs / "fig_mae_by_phase.png", dpi=200); plt.close(fig)


def fig_curves(cfg, figs):
    rolling = [f for f in make_folds(cfg) if f["kind"] == "rolling"]
    if not rolling:
        return
    last = rolling[-1]["fold"]
    d = wpath(cfg, "behavior", cfg["PRIMARY_REGIME"], last, "x").parent
    parts = []
    for f in d.glob("*_curve.parquet"):
        m, s = f.stem.split("_")[0], int(f.stem.split("_")[1][1:])
        parts.append(pd.read_parquet(f).assign(method=m, seed=s))
    if not parts:
        return
    c = pd.concat(parts)
    c.to_csv(wpath(cfg, "tables", "landfall_conditioned_features.csv"), index=False)
    lbl = ["<-96", "-96:-48", "-48:-24", "-24:-12", "-12:0", "0:12", "12:24", "24:48", "48:96", ">96", "no event\n(2019)"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    for col, method in enumerate(["airl", "empirical"]):
        sub = c[c.method == method]
        for row, feat in enumerate(["H", "Pmax"]):
            ax = axes[row, col]
            for r in route_list(cfg):
                g = sub[sub.route == r].groupby("lb")[feat].agg(["mean", "std"]).reindex(range(11))
                seen = sub[sub.route == r].groupby("lb").lb_seen.first().reindex(range(11)).fillna(False)
                ax.errorbar(range(11), g["mean"], yerr=g["std"].fillna(0), marker="o", ms=3, label=f"Route {r}")
                ax.scatter(np.where(~seen)[0], g["mean"][~seen.values], facecolor="none", edgecolor="k", s=40, zorder=3)
            ax.set_title(f"{method}: {feat} vs hours to landfall (speed bin fixed)", fontsize=9)
            ax.set_xticks(range(11)); ax.set_xticklabels(lbl, rotation=60, fontsize=7)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Frozen policy from fold {last}; open circles = landfall bins absent from training", fontsize=9)
    fig.tight_layout(); fig.savefig(figs / "fig_landfall_conditioned_features.png", dpi=200); plt.close(fig)


def fig_airl_training(cfg, figs):
    rolling = [f for f in make_folds(cfg) if f["kind"] == "rolling"]
    if not rolling:
        return
    d = wpath(cfg, "behavior", cfg["PRIMARY_REGIME"], rolling[-1]["fold"], "x").parent
    hs = [pd.read_csv(f).assign(seed=f.stem) for f in d.glob("airl_s*_history.csv")]
    hs = [h for h in hs if len(h)]
    if not hs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for h in hs:
        axes[0].plot(h.epoch, h.D_expert, "C0", alpha=.6); axes[0].plot(h.epoch, h.D_policy, "C1", alpha=.6)
        axes[1].plot(h.epoch, h.loss, "C2", alpha=.6)
        axes[2].plot(h.epoch, h.entropy, "C3", alpha=.6)
    axes[0].set_title("Discriminator output (blue: observed, orange: sampled)", fontsize=8)
    axes[1].set_title("Discriminator loss (Eq. 12)", fontsize=8)
    axes[2].set_title("Mean entropy of P(a|s)", fontsize=8)
    for a in axes:
        a.set_xlabel("epoch")
    fig.suptitle(f"AIRL-inspired training, fold {rolling[-1]['fold']}, all seeds", fontsize=9)
    fig.tight_layout(); fig.savefig(figs / "fig_airl_training.png", dpi=200); plt.close(fig)


def training_summary(cfg):
    """Stopping epochs for every network fit (from learning-curve files)."""
    rows = []
    base = wpath(cfg, "curves", "x").parent
    for f in base.rglob("*.csv"):
        d = pd.read_csv(f)
        if "selected" not in d:
            continue
        task = f.relative_to(base).parts[0]
        stages = d.stage.unique() if "stage" in d else ["train"]
        for st in stages:
            g = d[d.stage == st] if "stage" in d else d
            sel = g[g.selected.astype(bool)]
            rows.append(dict(task=task, regime=g.get("regime", pd.Series(["pooled"])).iloc[0],
                             model=g.model.iloc[0], cond=g.cond.iloc[0], stage=st,
                             epochs_run=int(g.epoch.max()),
                             selected_epoch=int(sel.epoch.iloc[0]) if len(sel) else None))
    if not rows:
        return None
    t = pd.DataFrame(rows)
    cap = {"forecast": cfg["LSTM"]["max_epochs"], "classify": cfg["CLASSIFY"]["mlp_epochs"]}
    t["hit_max"] = t.apply(lambda r: r.epochs_run >= (cfg["FINETUNE"]["max_epochs"] if r.stage == "finetune"
                                                    else cap.get(r.task, 1e9)), axis=1)
    s = (t.groupby(["task", "regime", "model", "cond", "stage"])
         .agg(fits=("epochs_run", "size"), median_selected_epoch=("selected_epoch", "median"),
              q25=("selected_epoch", lambda x: x.quantile(.25)), q75=("selected_epoch", lambda x: x.quantile(.75)),
              frac_reached_max_epochs=("hit_max", "mean")).reset_index())
    return s


def fig_learning_curves(cfg, figs):
    rolling = [f for f in make_folds(cfg) if f["kind"] == "rolling"]
    if not rolling:
        return
    fold, h = rolling[-1]["fold"], cfg["FIG_HORIZON_STEPS"]
    panels = [("pooled LSTM, no behavioral features", cfg["PRIMARY_REGIME"], "lstm_none"),
              ("pooled LSTM, AIRL-inspired features", cfg["PRIMARY_REGIME"], "lstm_airl"),
              ("pretrain (2019) + fine-tune, no behavioral features", "finetune", "lstm_none")]
    panels = [p for p in panels if list(wpath(cfg, "curves", "forecast", p[1], fold, "x").parent.glob(f"{p[2]}_h{h}_s*.csv"))]
    mlp = list(wpath(cfg, "curves", "classify", fold, "x").parent.glob("mlp_none_s*.csv"))
    n = len(panels) + (1 if mlp else 0)
    if n == 0:
        return
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax, (title, reg, stem) in zip(axes, panels):
        for f in sorted(wpath(cfg, "curves", "forecast", reg, fold, "x").parent.glob(f"{stem}_h{h}_s*.csv")):
            d = pd.read_csv(f)
            off = 0
            for st, g in d.groupby("stage", sort=False):
                ax.plot(g.epoch + off, g.train_mse, "C0-", alpha=.5, lw=1)
                ax.plot(g.epoch + off, g.val_mse, "C1-", alpha=.7, lw=1)
                s = g[g.selected.astype(bool)]
                ax.plot(s.epoch + off, s.val_mse, "ko", ms=3)
                if st == "pretrain":
                    off = g.epoch.max()
                    ax.axvline(off + .5, color="0.6", ls=":", lw=.8)
        ax.set_title(title, fontsize=8); ax.set_xlabel("epoch"); ax.set_ylabel("MSE (standardized speed)")
    if mlp:
        ax = axes[n - 1]
        for f in sorted(mlp):
            d = pd.read_csv(f)
            ax.plot(d.epoch, d.train_loss, "C0-", alpha=.5, lw=1); ax.plot(d.epoch, d.val_loss, "C1-", alpha=.7, lw=1)
            s = d[d.selected.astype(bool)]; ax.plot(s.epoch, s.val_loss, "ko", ms=3)
        ax.set_title("MLP classifier, no behavioral features", fontsize=8); ax.set_xlabel("epoch")
        ax.set_ylabel("weighted cross-entropy")
    axes[0].plot([], [], "C0-", label="training"); axes[0].plot([], [], "C1-", label="validation (chronological)")
    axes[0].plot([], [], "ko", label="selected epoch"); axes[0].legend(fontsize=7)
    fig.suptitle(f"Learning curves, fold {fold}, {H_MIN.get(h, h * 15)}-min horizon, {len(cfg['SEEDS'])} seeds", fontsize=9)
    fig.tight_layout(); fig.savefig(figs / "fig_learning_curves.png", dpi=200); plt.close(fig)


def main():
    cfg = load_config()
    st = wpath(cfg, "stats", "x").parent
    figs = wpath(cfg, "figures", "x").parent
    headline = {}

    ds = wpath(cfg, "tables", "data_summary.csv")
    if ds.exists():
        save(pd.read_csv(ds), "T1_data_summary", cfg)

    summ = pd.read_csv(st / "forecast_metrics.csv")
    comp = pd.read_csv(st / "forecast_comparisons.csv")
    save(comp_table(comp, "baselines"), "T2_baselines", cfg)
    save(comp_table(comp, "features"), "T3_feature_ladder", cfg)
    save(comp_table(comp, "airl_vs_alt"), "T4_airl_vs_alternatives", cfg)
    for fam, name in [("beyond_route", "T10_beyond_route_identity"),
                      ("beyond_route_airl_vs_alt", "T10b_route_aware_airl_vs_alternatives"),
                      ("ridge_features", "T11_ridge_features"),
                      ("ridge_contrasts", "T11b_ridge_contrasts"),
                      ("transfer", "T12_transfer_learning"),
                      ("airl_by_regime", "T13_airl_effect_by_regime")]:
        if (comp.family == fam).any():
            save(comp_table(comp, fam), name, cfg)
    tab = summ.copy()
    tab["MAE"] = [fmt_pm(m, s) for m, s in zip(summ.MAE, summ.MAE_sd)]
    tab["RMSE"] = [fmt_pm(m, s) for m, s in zip(summ.RMSE, summ.RMSE_sd)]
    tab["MAPE"] = [fmt_pm(m, s, 1) for m, s in zip(summ.MAPE, summ.MAPE_sd)]
    save(tab[["regime", "phase", "h", "model", "cond", "n", "seeds", "MAE", "RMSE", "MAPE"]],
         "T3b_forecast_metrics_mean_sd", cfg)

    for ph in ["acute", "landfall", "late"]:
        f = comp[(comp.family == "features") & (comp.phase == ph) & (comp.cand.str.endswith("/airl"))]
        if len(f):
            headline[f"airl_vs_none_{ph}"] = {int(r.h * 15): dict(dMAE=round(r.dMAE, 3), ci=[round(r.ci_lo, 3), round(r.ci_hi, 3)],
                                                               seed_sd=round(r.dMAE_seed_sd, 3), p_dm_holm=round(r.p_dm_holm, 4))
                                             for r in f.itertuples()}
        f = comp[(comp.family == "airl_vs_alt") & (comp.phase == ph)]
        if len(f):
            headline[f"airl_vs_alternatives_{ph}"] = {f"{r.ref.split('/')[-1]}_h{int(r.h * 15)}": round(r.dMAE, 3)
                                                      for r in f.itertuples()}

    if (st / "classify_metrics.csv").exists():
        cm = pd.read_csv(st / "classify_metrics.csv")
        cm["macro_F1"] = [fmt_pm(m, s, 3) for m, s in zip(cm.macro_f1, cm.macro_f1_sd)]
        cm["F1_congested"] = [fmt_pm(m, s, 3) for m, s in zip(cm.f1_bin, cm.f1_bin_sd)]
        cm["PR_AUC"] = [fmt_pm(m, s, 3) for m, s in zip(cm.pr_auc, cm.pr_auc_sd)]
        cm["accuracy"] = [fmt_pm(m, s, 3) for m, s in zip(cm.accuracy, cm.accuracy_sd)]
        save(cm[["phase", "model", "cond", "n", "n_light", "n_heavy", "accuracy", "macro_F1",
                 "F1_congested", "PR_AUC", "recall_light", "recall_heavy"]], "T5_classification", cfg)
        cc = pd.read_csv(st / "classify_comparisons.csv")
        save(cc, "T6_classification_deltas", cfg)
        save(pd.read_csv(st / "class_counts_test.csv"), "T7_class_counts", cfg)
        conf = pd.read_csv(st / "classify_confusion.csv")
        save(conf, "T7b_confusion_matrices", cfg)
        tcc = wpath(cfg, "classify", "train_class_counts.csv")
        if tcc.exists():
            t = pd.read_csv(tcc)
            save(t[t.cond == "none"].groupby("fold")[["n0", "n1", "n2"]].first().reset_index(),
                 "T7c_train_class_counts_by_fold", cfg)
        a = cc[(cc.cond == "airl") & cc.phase.isin(["acute", "all", "late"])]
        headline["classification_airl_vs_none"] = {f"{r.model}_{r.phase}": dict(
            dMacroF1=round(r.dMacroF1, 4), ci=[round(r.ci_lo, 4), round(r.ci_hi, 4)], n=int(r.n))
            for r in a.itertuples()}

    cov = wpath(cfg, "behavior", "coverage.csv")
    if cov.exists():
        c = pd.read_csv(cov)
        save(c, "T8_action_coverage", cfg)
        fig, ax = plt.subplots(figsize=(8, 3))
        cp = c[c.regime == cfg["PRIMARY_REGIME"]]
        ax.plot(cp.fold, cp.frac_action_outside_A, "o-", label="next segment outside training A(i)")
        ax.plot(cp.fold, cp.test_lb_unseen, "s-", label="landfall bin unseen in training")
        ax.set_ylabel("fraction of test transitions"); ax.legend(fontsize=8)
        plt.setp(ax.get_xticklabels(), rotation=45)
        fig.tight_layout(); fig.savefig(figs / "fig_action_coverage.png", dpi=200); plt.close(fig)

    ts = training_summary(cfg)
    if ts is not None:
        save(ts, "T14_training_summary", cfg)
    fig_learning_curves(cfg, figs)
    fig_forecasts(cfg, figs)
    fig_mae_phase(cfg, figs, summ)
    fig_curves(cfg, figs)
    fig_airl_training(cfg, figs)
    wpath(cfg, "tables", "headline_numbers.json").write_text(json.dumps(headline, indent=2, default=str))
    log("report done -> " + str(wpath(cfg, "tables", "x").parent) + " and " + str(figs))


if __name__ == "__main__":
    sys.exit(main())
