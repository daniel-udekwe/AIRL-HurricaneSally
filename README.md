# Revision pipeline — TRC-26-03399 resubmission

This pipeline reruns every quantitative result in the manuscript for the four
study routes (A–D, segment lists in `routes/`), using a design that addresses
each point in the editor's decision letter.

## Setup

```
pip install -r requirements.txt
```

Edit `DATA_ROOT` in `config.py` if your INRIX folder is somewhere other than
`C:\Users\dau24\Documents\INRIX files\sally2020`.

## Run order

```
python run_all.py --only 0      # 1. inspect raw files (minutes) - CHECK THE OUTPUT
python run_all.py --from 1 --to 2   # 2. extract + panels (one long scan of the .gz files)
python run_all.py --from 3      # 3. behavior models, forecasting, classification, stats, report
```

Before step 1, check the step 0 output for three things:

1. **Column mapping.** `mapping:` must find TripId, SegmentId, CrossingStartDateUtc
   and CrossingSpeedKph. If not, add the actual header names to
   `COLUMN_CANDIDATES` in `config.py`.
2. **Route coverage.** All four routes should appear under "route matches".
   SegmentIds in the data look like `280673924_5` (way ID + sub-segment); the
   route files list way IDs, so matching uses the part before `_`
   (`SEGMENT_WAY_SEPARATOR`). Consecutive sub-segments of the same way are merged
   into one traversal. Route D has only 2 ways listed, so it will have far fewer
   traversals. Way IDs with a leading `-` are traversals against the digitized
   direction; `ROUTE_DIRECTIONS = "listed"` (default) keeps
   the direction as written in the route files. Divided sections use separate
   ways per carriageway, so "separate"/"merged" are only meaningful if the route
   files list both carriageways. Run `python tools/route_check.py` after step 1
   to check coverage and contiguity. The trips files are not read: their rows have inconsistent field
   counts, and trip speed and provider come from the trajectory file instead.
3. **Timestamp format.** The first-lines printout shows the raw format. ISO strings
   and epoch milliseconds are both handled.

After step 2, read `work/tables/data_summary.csv` and the 6-h label counts printed
in the log before spending hours on step 3–5.

**Smoke test:** `python tools/make_synthetic.py <dir>` builds fake data in the
same layout. Point a copy of `config.py` at it and run
`python run_all.py --quick --config <copy>`. The whole pipeline finishes in a few
minutes. This only checks that the code runs; the numbers are meaningless.

## Compute budget

The full design is about 20 folds × 4 horizons × (9 LSTM feature conditions + RNN)
× 5 seeds, i.e. roughly 4,000 network fits plus about 300 behavior-model fits. One
LSTM fit took about 20 s on one CPU thread with synthetic data. On real data,
expect tens of CPU-hours in total. Steps 3–5 skip outputs that already exist, so
you can:

- stop and resume at any time;
- run several processes in parallel on disjoint folds, e.g.
  `python run_all.py --only 4 --folds r0908 r0909 r0910` in one terminal and
  `--folds r0911 r0912 r0913` in another (run step 3 for those folds first);
- reduce cost by setting `ROLLING["step_hours"] = 48`, using fewer seeds for the
  minor conditions, or dropping conditions from `CONDITIONS`.

Run steps 6–7 once all folds are finished.

## What each output answers

| Editor point | Design change | Outputs |
|---|---|---|
| 1. Landfall not tested out-of-sample | Rolling origins at each local midnight, Sep 8–27, 2020. Each day is forecast by models trained only on data before it. Results are reported by phase (pre-landfall, landfall [L−12 h, L+48 h), recovery) plus the original late window for comparability. Training pools the full 2019 control period with 2020 data before the origin. | `T2`, `T3`, `T3b` (by phase), `fig_heldout_landfall_forecasts`, `fig_mae_by_phase` |
| 2. Gains not attributable to the adversarial component | The same H and Pmax features from four route-choice models over the identical state space and action sets: empirical (smoothed counts), conditional logit, behavior cloning (same policy network, no discriminator), and AIRL-inspired. Also route-ID and topology (log\|A(i)\|) baselines, entropy-only and Pmax-only variants, and a direct AIRL-vs-alternatives table. | `T3_feature_ladder`, `T4_airl_vs_alternatives` |
| 3. No run-to-run variance | Every network (behavior models, LSTM/RNN, MLP) is trained with each seed in `SEEDS`. Tables report mean ± SD and the SD of the per-seed difference. | `seed SD` columns, `T3b`, `*_by_seed.csv` |
| 4. Few minority-class samples | Per-class test counts by phase, training class counts by fold, per-class recall, binary congested F1, PR-AUC, majority-vote confusion matrices. Rolling origins put most congested windows in the test set. | `T5`, `T6`, `T7`, `T7b`, `T7c` |
| 5. Route-clustered re-analysis not done | With 4 routes, a route-clustered bootstrap has only 4 clusters, which is not meaningful. Instead: a Diebold–Mariano test with Newey–West HAC variance and the Harvey–Leybourne–Newbold correction on the seed- and route-averaged loss differential, a day-block bootstrap for CIs (resampling whole days across all routes), and Holm correction within each table and phase. | `p_DM (Holm)`, `p_boot (Holm)`, CI columns |

Added rigor (second round):

- `T10_beyond_route_identity`: route-aware LSTM with vs without behavioral features
  (`route_id+airl`, `route_id+empirical`, `route_id+bc` vs `route_id`), and
  `T10b`: AIRL vs alternatives within the route-aware model.
- `T11_ridge_features`, `T11b_ridge_contrasts`: behavioral features on the linear AR
  model, the strongest model in the acute period.
- `T12_transfer_learning`: training regimes vs training from scratch on 2020
  (`2020_only`): pooled 2019+2020, pretrain on the full 2019 period then fine-tune
  on 2020 (`finetune`), and 2019 only (`pretrain_only`, domain-shift reference).
- `T13_airl_effect_by_regime`: AIRL vs no behavioral features within each regime
  (the 2x2 TL x AIRL analysis, now at every horizon and phase).

Replication and training diagnostics:

- `work/manifest/run_*.json`: written by every `run_all.py` call — full config,
  Python and package versions, platform, input files with sizes, route way lists.
- `work/curves/`: per-epoch training and (chronological) validation loss for every
  LSTM/RNN fit (`forecast/`, with pretrain and fine-tune stages for `finetune`) and
  MLP fit (`classify/`), with the selected (early-stopping) epoch marked. AIRL,
  behavior-cloning and logit training histories are in `work/behavior/<regime>/<fold>/`.
- `T14_training_summary`: median selected epoch and share of fits reaching the
  epoch cap, per model/condition/regime.
- `fig_learning_curves`: curves for the last rolling fold at the figure horizon.
- `SAVE_MODELS = True` in config.py also saves every LSTM/RNN's weights.
- Seeds are fixed per fit and PyTorch deterministic algorithms are enabled.
- Fits run before curve logging existed have no curves; regenerate them for chosen
  folds without touching predictions: `python tools/learning_curves.py --folds r0926`.
  For the MLP, delete `work/classify` and rerun step 5 (minutes).

Extra outputs that answer issues the manuscript itself raised:

- `T8_action_coverage`: the share of test transitions that fall outside the
  training action sets, and the share in landfall bins never seen in training.
- `fig_landfall_conditioned_features`: H and Pmax from the frozen policy as a
  function of landfall bin, with speed bin fixed. This is the falsifiable test
  that Sec. 3.3.3 says is missing.
- `fig_airl_training`: replaces the old Fig. 7.
- `T9_regimes`, if `SECONDARY_REGIMES` is set: 2020-only and pretrain/fine-tune
  versus pooled training. This replaces the old transfer-learning analysis with a
  full three-month pretraining set.
- `headline_numbers.json`: the numbers to quote in the abstract and conclusion.

## Methodological choices to describe in the revised Methods

- **MDP at the way level.** INRIX sub-segments (`<way>_<k>`) are merged into way traversals; "segment" below means a way. The state is (segment, speed bin, landfall bin), where the segment is one on routes A–D. The action is the next segment, which may be off-route. A(i) contains next segments observed from i in training transitions only, with the next crossing starting within 30 min. Self-transitions are excluded. Speed bins are crossing speed divided by the segment's median training speed, with cut points 0.5 / 0.75 / 0.9 / 1.1. Landfall bins are hours to landfall with edges ±12, 24, 48, 96, and 2019 data gets its own "no event" bin.
- **Landfall bins unseen in training** map to an UNK token for the neural and logit models. That token is trained by randomly masking 10% of training bins. The empirical model backs off from (i, sb, lb) to (i, lb) to (i).
- **Route-level features** are the mean of crossing-level H, Pmax and log|A| per route per 15-min bin or 6-h window. Gaps are forward-filled causally, and any remaining gaps get the training mean.
- **Traffic features.** Speed is space-mean speed: total length divided by total travel time. The panel also includes an observed flag, log minutes since the last observation, log crossings, log trips, trip mean speed, sin/cos of hour and day of week, clipped hours-to-landfall, and a pre-landfall flag. The SPI reference speed is the route's space-mean speed over the full 2019 control period.
- **Forecast targets** are observed 15-min speeds only. Unobserved bins are never scored.
- **Early stopping** uses a chronological validation set (the last 10% of training samples by target time) for every network, replacing the random split.
- **Classification.** Features come from the 6-h window (t−6h, t]. The label is the SPI class of (t, t+6h]. A training window is used only if its label window ends by the fold origin.

## Caveats to state in the paper

- Four routes give few congested 6-h windows. Report the counts (`T7`) and lead
  with binary congested F1 and PR-AUC rather than three-class macro-F1 if heavy
  congestion has only a handful of cases.
- Folds before landfall have essentially no congested training examples, so the
  classifiers cannot learn those classes yet (`T7c`). This is a property of the
  forecasting problem, not a bug.
- Route D has only 2 listed segments. Its behavior features depend mostly on
  where trips go after leaving it.
