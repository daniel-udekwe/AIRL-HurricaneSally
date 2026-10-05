"""
Configuration for the revision pipeline (TRC-26-03399 resubmission).

Edit DATA_ROOT and, if 00_inspect reports a mismatch, COLUMN_CANDIDATES.
Everything else has defaults matched to the revision plan.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
DATA_ROOT = Path(r"C:\Users\dau24\Documents\INRIX files\sally2020")
WORK_DIR = HERE / "work"          # all intermediate + final outputs go here
DUCKDB_TEMP = WORK_DIR / "duckdb_tmp"
DUCKDB_MEMORY = "6GB"             # lower if your machine has < 16 GB RAM

ROUTE_FILES = {
    "A": HERE / "routes" / "routeA.txt",
    "B": HERE / "routes" / "routeB.txt",
    "C": HERE / "routes" / "routeC.txt",
    "D": HERE / "routes" / "routeD.txt",
}

# Direction handling. Way IDs with a leading "-" are traversals against the way's
# digitized direction. "separate": each route becomes two series, <R> (direction
# as listed in the route file) and <R>_rev (opposite); "merged": both directions
# pooled into one series; "listed": only the direction as written in the file.
ROUTE_DIRECTIONS = "listed"

# Each dataset folder is searched recursively for data/trajs/*.gz,
# data/trips/*.gz and schema/*Headers.csv, so report IDs need not be listed.
DATASETS = {
    "affected2020": {"dir": "escambia-affected", "start": "2020-08-16", "end": "2020-11-16", "event": True},
    "control2019":  {"dir": "escambia-controlpy", "start": "2019-08-16", "end": "2019-11-16", "event": False},
}

CSV_DELIM = ","

# Logical name -> accepted header names (case-insensitive). Optional columns
# may be missing; required ones raise an error in 00_inspect / 01_extract.
COLUMN_CANDIDATES = {
    "traj": {
        "trip":      (["TripId", "TripID", "trip_id"], True),
        "provider":  (["ProviderId", "ProviderID"], False),
        "traj_idx":  (["TrajIdx", "TrajectoryIdx", "TrajectoryIndex"], False),
        "traj_dist": (["TrajRawDistanceM", "TripRawDistanceM"], False),
        "traj_dur":  (["TrajRawDurationMillis", "TripRawDurationMillis"], False),
        "seg":       (["SegmentId", "SegmentID", "segment_id"], True),
        "seg_idx":   (["SegmentIdx", "SegmentIndex"], False),
        "length":    (["LengthM", "SegmentLengthM"], False),
        "t_start":   (["CrossingStartDateUtc", "CrossingStartDateUTC"], True),
        "t_end":     (["CrossingEndDateUtc", "CrossingEndDateUTC"], False),
        "speed":     (["CrossingSpeedKph", "CrossingSpeedKPH"], True),
    },
}
# SegmentId in the data is "<wayId>_<sub-index>" (e.g. 280673924_5); the route
# files list way IDs. Matching and the route-choice MDP use the part before this
# separator, and consecutive sub-segments of the same way are merged into one
# traversal. Set to None if your route files list full SegmentIds instead.
SEGMENT_WAY_SEPARATOR = "_"

# --------------------------------------------------------------------------
# Event + cleaning
# --------------------------------------------------------------------------
LOCAL_TZ = "America/Chicago"            # Escambia County is on Central time
LANDFALL_UTC = "2020-09-16 09:45"       # Sally landfall, Gulf Shores AL (NHC TCR)
SPEED_MIN_KPH, SPEED_MAX_KPH = 2.0, 160.0
MAX_TRANSITION_GAP_MIN = 30             # next crossing must start within this
HTL_CLIP_H = 168                        # hours-to-landfall feature clipped to +-1 week

# SPI reference speed: "control2019" = route space-mean speed over the full
# same-calendar 2019 period (recommended); "first_week" = Aug 16-22, 2020
# (the original manuscript's choice).
REF_SPEED_SOURCE = "control2019"

# --------------------------------------------------------------------------
# Evaluation design (editor point 1)
# --------------------------------------------------------------------------
# Rolling origins at local midnight; each fold trains on everything before
# its origin and tests on the next `step_hours`.
ROLLING = {"start": "2020-09-08", "end": "2020-09-27", "step_hours": 24}
# Extra frozen-model folds (name, origin, end), local dates. "late" reproduces
# the original manuscript's test window for comparability.
EXTRA_FOLDS = [("late", "2020-10-29", "2020-11-16")]
# Phases relative to landfall (hours). Anything in an EXTRA fold -> its name.
PHASE_LANDFALL_WINDOW_H = (-12, 48)     # [L-12h, L+48h) = "landfall"
PHASE_GROUPS = {
    "acute": ["pre_landfall", "landfall"],
    "rolling_all": ["pre_landfall", "landfall", "recovery"],
}

# --------------------------------------------------------------------------
# Forecasting (short-term)
# --------------------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]                 # editor point 3; 10 is better if time allows
HORIZONS_STEPS = [1, 2, 4, 6]           # x15 min = 15, 30, 60, 90 min
SEQ_LEN = 8
# Feature conditions (editor point 2). Each adds columns to the base inputs.
CONDITIONS = ["none", "route_id", "topology", "empirical", "logit", "bc",
              "airl", "airl_H", "airl_P",
              # route-aware model + behavioral features: value beyond route identity
              "route_id+airl", "route_id+empirical", "route_id+bc"]
# Behavioral features on the linear AR model (strongest model in the acute period)
RIDGE_CONDITIONS = ["none", "route_id", "empirical", "airl",
                    "route_id+empirical", "route_id+airl"]
BASELINE_MODELS = ["persistence", "ridge", "rnn"]
# Training regimes (transfer learning):
#   "pooled"        = full 2019 period + 2020 before the origin, trained jointly (primary)
#   "2020_only"     = from scratch on 2020 before the origin
#   "finetune"      = pretrain on the full 2019 period, fine-tune on 2020 before the origin
#   "pretrain_only" = trained on 2019 only, applied to 2020 (domain-shift reference)
PRIMARY_REGIME = "pooled"
SECONDARY_REGIMES = ["2020_only", "finetune", "pretrain_only"]
SECONDARY_CONDITIONS = ["none", "airl"]

LSTM = dict(hidden=64, lr=1e-3, batch=256, max_epochs=40, patience=5,
            clip=5.0, val_frac=0.1)
FINETUNE = dict(lr=1e-4, max_epochs=50)

# --------------------------------------------------------------------------
# Behavior models (editor point 2)
# --------------------------------------------------------------------------
BEHAVIOR = dict(
    methods=["empirical", "logit", "bc", "airl"],
    epochs=30, lr=1e-3, batch=1024, draws_per_epoch=200_000,
    emb=16, hidden=64, unk_p=0.1, weight_decay=1e-5,
    alpha=5.0,                          # Dirichlet smoothing for "empirical"
)
SB_FIXED_FOR_CURVES = 3                 # speed bin 0.9-1.1 x median ("free flow")

# --------------------------------------------------------------------------
# Long-term classification (editor point 4)
# --------------------------------------------------------------------------
CLASSIFY = dict(models=["mlp", "svm", "knn"],
                class_weights={0: 1.0, 1: 5.0, 2: 6.0},
                mlp_lr=5e-4, mlp_epochs=50, patience=10, batch=256, val_frac=0.1)

# --------------------------------------------------------------------------
# Statistics (editor points 3 and 5)
# --------------------------------------------------------------------------
BOOT_B = 2000
FIG_HORIZON_STEPS = 4
# Save trained LSTM/RNN weights (one small .pt file per fit, a few hundred MB in total)
SAVE_MODELS = False
N_THREADS = 4

# Applied when run with --quick (smoke test).
QUICK_OVERRIDES = dict(
    SEEDS=[0, 1],
    HORIZONS_STEPS=[1, 4],
    ROLLING={"start": "2020-09-13", "end": "2020-09-19", "step_hours": 48},
    LSTM=dict(hidden=16, lr=1e-3, batch=256, max_epochs=2, patience=1, clip=5.0, val_frac=0.1),
    FINETUNE=dict(lr=1e-4, max_epochs=1),
    BEHAVIOR=dict(methods=["empirical", "logit", "bc", "airl"], epochs=2, lr=1e-3,
                  batch=1024, draws_per_epoch=20_000, emb=8, hidden=16,
                  unk_p=0.1, weight_decay=1e-5, alpha=5.0),
    CLASSIFY=dict(models=["mlp", "svm", "knn"], class_weights={0: 1.0, 1: 5.0, 2: 6.0},
                  mlp_lr=5e-4, mlp_epochs=2, patience=1, batch=256, val_frac=0.1),
    BOOT_B=50,
    SECONDARY_REGIMES=["2020_only", "finetune", "pretrain_only"],
)
