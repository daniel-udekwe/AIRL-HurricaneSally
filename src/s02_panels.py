"""
Step 2 - Clean crossings and build route-level panels.

panel15.parquet : (dataset, route, 15-min bin) traffic + time features
panel6h.parquet : (dataset, route, 6-h local window) features + next-window SPI label
crossings.parquet : cleaned route crossings (input to behavior models)
ref_speed.csv : SPI reference speed per route
"""
import sys
import numpy as np
import pandas as pd

from common import landfall, load_config, log, read_routes, route_list, wpath


def load_clean(cfg):
    parts = []
    for ds, spec in cfg["DATASETS"].items():
        cx = pd.read_parquet(wpath(cfg, ds, "crossings_route.parquet"))
        tr = pd.read_parquet(wpath(cfg, ds, "trips.parquet"))[["trip", "provider", "trip_mean_speed"]]
        tr = tr.drop_duplicates("trip")
        cx["t_start"] = pd.to_datetime(cx["t_start"], utc=True)
        cx["next_t"] = pd.to_datetime(cx["next_t"], utc=True)
        lo = pd.Timestamp(spec["start"]).tz_localize(cfg["LOCAL_TZ"]).tz_convert("UTC")
        hi = pd.Timestamp(spec["end"]).tz_localize(cfg["LOCAL_TZ"]).tz_convert("UTC")
        n0 = len(cx)
        cx = cx[cx.t_start.notna() & (cx.t_start >= lo) & (cx.t_start < hi)]
        cx = cx[cx.speed_kph.between(cfg["SPEED_MIN_KPH"], cfg["SPEED_MAX_KPH"])]
        med_len = cx[cx.length_m > 0].groupby("seg").length_m.median()
        cx["length_m"] = cx.length_m.where(cx.length_m > 0, cx.seg.map(med_len)).fillna(100.0)
        gap = (cx.next_t - cx.t_start).dt.total_seconds() / 60
        ok = cx.next_seg.notna() & (cx.next_seg != cx.seg) & gap.between(0, cfg["MAX_TRANSITION_GAP_MIN"])
        cx["next_seg"] = cx.next_seg.where(ok)
        if "trip_mean_speed" not in cx.columns:
            cx = cx.merge(tr, on="trip", how="left")
        cx["dataset"] = ds
        log(f"{ds}: kept {len(cx):,}/{n0:,} crossings; {ok.mean():.1%} with a valid next segment")
        parts.append(cx)
    cx = pd.concat(parts, ignore_index=True)
    cx["tt_s"] = cx.length_m / (cx.speed_kph / 3.6)
    return cx


def since_obs_minutes(obs, step_min):
    idx = np.where(obs, np.arange(len(obs)), np.nan)
    last = pd.Series(idx).ffill().values
    out = (np.arange(len(obs)) - last) * step_min
    return np.where(np.isnan(out), 7 * 24 * 60, out)


def time_features(cfg, df, tcol, event):
    loc = df[tcol].dt.tz_convert(cfg["LOCAL_TZ"])
    hr = loc.dt.hour + loc.dt.minute / 60.0
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    dw = loc.dt.dayofweek + hr / 24.0
    df["dow_sin"], df["dow_cos"] = np.sin(2 * np.pi * dw / 7), np.cos(2 * np.pi * dw / 7)
    clip = cfg["HTL_CLIP_H"]
    if event:
        h = (df[tcol] - landfall(cfg)).dt.total_seconds() / 3600.0
        df["htl"] = h.clip(-clip, clip) / clip
        df["pre_flag"] = ((h < 0) & (h > -clip)).astype(float)
    else:
        df["htl"], df["pre_flag"] = 1.0, 0.0
    return df


def agg_traffic(g):
    a = g.agg(sum_len=("length_m", "sum"), sum_tt=("tt_s", "sum"), n_cross=("seg", "size"),
              n_trips=("trip", "nunique"), trip_speed=("trip_mean_speed", "mean"))
    a["speed"] = a.sum_len / a.sum_tt * 3.6
    return a.drop(columns=["sum_len", "sum_tt"])


def build_panel15(cfg, cx):
    cx = cx.assign(bin=cx.t_start.dt.floor("15min"))
    agg = agg_traffic(cx.groupby(["dataset", "route", "bin"]))
    out = []
    for ds, spec in cfg["DATASETS"].items():
        lo = pd.Timestamp(spec["start"]).tz_localize(cfg["LOCAL_TZ"]).tz_convert("UTC")
        hi = pd.Timestamp(spec["end"]).tz_localize(cfg["LOCAL_TZ"]).tz_convert("UTC")
        bins = pd.date_range(lo, hi, freq="15min", inclusive="left")
        for r in route_list(cfg):
            p = pd.DataFrame({"bin": bins})
            if (ds, r) in agg.index.droplevel(2).unique():
                p = p.merge(agg.loc[(ds, r)].reset_index(), on="bin", how="left")
            else:
                p = p.assign(n_cross=np.nan, n_trips=np.nan, trip_speed=np.nan, speed=np.nan)
            p["dataset"], p["route"] = ds, r
            p["obs"] = p.n_cross.fillna(0) > 0
            p["speed_obs"] = p.speed.where(p.obs)            # target (NaN if unobserved)
            p["speed"] = p.speed_obs.ffill().bfill()          # input (last observed)
            p["trip_speed"] = p.trip_speed.ffill().bfill().fillna(p.speed)
            p["n_cross"], p["n_trips"] = p.n_cross.fillna(0), p.n_trips.fillna(0)
            p["mins_since_obs"] = since_obs_minutes(p.obs.values, 15)
            out.append(time_features(cfg, p, "bin", spec["event"]))
    panel = pd.concat(out, ignore_index=True)
    panel["log_cross"] = np.log1p(panel.n_cross)
    panel["log_trips"] = np.log1p(panel.n_trips)
    panel["log_since"] = np.log1p(panel.mins_since_obs)
    return panel


def ref_speeds(cfg, cx):
    if cfg["REF_SPEED_SOURCE"] == "control2019":
        sub = cx[~cx.dataset.map(lambda d: cfg["DATASETS"][d]["event"])]
    else:
        lo = pd.Timestamp("2020-08-16").tz_localize(cfg["LOCAL_TZ"]).tz_convert("UTC")
        sub = cx[(cx.dataset == "affected2020") & (cx.t_start >= lo) & (cx.t_start < lo + pd.Timedelta(days=7))]
    g = sub.groupby("route")
    ref = (g.length_m.sum() / g.tt_s.sum() * 3.6).rename("ref_speed")
    return ref.reindex(route_list(cfg))


def spi_class(spi):
    return np.select([spi > 75, spi > 50], [0, 1], 2).astype(float)


def build_panel6h(cfg, cx, ref):
    loc = cx.t_start.dt.tz_convert(cfg["LOCAL_TZ"]).dt.tz_localize(None)
    cx = cx.assign(wstart=loc.dt.floor("6h"))
    agg = agg_traffic(cx.groupby(["dataset", "route", "wstart"]))
    out = []
    for ds, spec in cfg["DATASETS"].items():
        ws = pd.date_range(spec["start"], spec["end"], freq="6h", inclusive="left")
        for r in route_list(cfg):
            p = pd.DataFrame({"wstart": ws})
            if (ds, r) in agg.index.droplevel(2).unique():
                p = p.merge(agg.loc[(ds, r)].reset_index(), on="wstart", how="left")
            else:
                p = p.assign(n_cross=np.nan, n_trips=np.nan, trip_speed=np.nan, speed=np.nan)
            p["dataset"], p["route"] = ds, r
            p["obs"] = p.n_cross.fillna(0) > 0
            spi = p.speed / ref[r] * 100
            p["spi_obs"] = spi.where(p.obs)
            p["spi"] = p.spi_obs.ffill().bfill()
            p["speed"] = p.speed.ffill().bfill()
            p["trip_speed"] = p.trip_speed.ffill().bfill().fillna(p.speed)
            p["n_cross"], p["n_trips"] = p.n_cross.fillna(0), p.n_trips.fillna(0)
            # label = congestion class of the NEXT, non-overlapping window
            nxt = p.spi_obs.shift(-1)
            p["label"] = np.where(nxt.notna(), spi_class(nxt.fillna(100)), np.nan)
            # t = end of feature window = start of label window (local, then UTC)
            p["t_local"] = p.wstart + pd.Timedelta(hours=6)
            p["t"] = p.t_local.dt.tz_localize(cfg["LOCAL_TZ"], ambiguous="NaT",
                                              nonexistent="shift_forward").dt.tz_convert("UTC")
            p["wslot"] = p.wstart.dt.hour // 6
            p = time_features(cfg, p, "t", spec["event"])
            out.append(p)
    panel = pd.concat(out, ignore_index=True)
    panel["log_cross"] = np.log1p(panel.n_cross)
    panel["log_trips"] = np.log1p(panel.n_trips)
    return panel[panel.t.notna()]


def main():
    cfg = load_config()
    routes = read_routes(cfg)
    cx = load_clean(cfg)
    labels = list(dict.fromkeys(routes.route))
    # keep only route labels defined by the current ROUTE_DIRECTIONS setting, so the
    # setting can be changed without re-running step 1
    cx = cx[cx.route.isin(labels) & cx.seg.isin(set(routes.seg))]
    per = cx.groupby(["route", "dataset"]).size().unstack(fill_value=0).reindex(labels, fill_value=0)
    active = [r for r in labels if (per.loc[r] > 0).all()]
    dropped = [r for r in labels if r not in active]
    if dropped:
        log(f"WARNING: no data in at least one dataset for {dropped}; these series are dropped")
    wpath(cfg, "active_routes.txt").write_text("\n".join(active))
    log("route series modelled: " + ", ".join(active))
    keep = ["dataset", "route", "trip", "seg", "next_seg", "t_start", "next_t",
            "speed_kph", "length_m", "tt_s", "provider", "trip_mean_speed"]
    cx[keep].to_parquet(wpath(cfg, "crossings.parquet"), index=False)
    p15 = build_panel15(cfg, cx)
    p15.to_parquet(wpath(cfg, "panel15.parquet"), index=False)
    ref = ref_speeds(cfg, cx)
    ref.to_csv(wpath(cfg, "ref_speed.csv"))
    log("reference speeds (kph):\n" + ref.round(1).to_string())
    if ref.isna().any():
        raise ValueError("Missing reference speed for some route; check REF_SPEED_SOURCE")
    p6 = build_panel6h(cfg, cx, ref)
    p6.to_parquet(wpath(cfg, "panel6h.parquet"), index=False)

    # data summary (editor: dataset scale was promised but never reported)
    s = cx.groupby(["dataset", "route"]).agg(crossings=("seg", "size"), trips=("trip", "nunique"),
                                             segs_observed=("seg", "nunique"),
                                             frac_valid_next=("next_seg", lambda x: x.notna().mean()))
    s = s.join(routes.groupby("route").size().rename("segs_listed"), on="route")
    s = s.join(p15.groupby(["dataset", "route"]).obs.mean().rename("frac_15min_bins_observed"))
    s.to_csv(wpath(cfg, "tables", "data_summary.csv"))
    log("data summary:\n" + s.to_string())
    lab = p6[p6.dataset == "affected2020"].groupby("label").size()
    log("2020 6-h label counts (0=none,1=light,2=heavy):\n" + lab.to_string())


if __name__ == "__main__":
    sys.exit(main())
