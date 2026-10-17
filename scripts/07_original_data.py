# ============================================================================
# ROUND 5: LEARNING FROM THE ORIGINAL DATASET
# Run with:   python scripts/07_original_data.py
#
# THE IDEA
#   Kaggle made this competition's 700,000 passengers with a computer program
#   that learned from a REAL, smaller dataset: "Airline Passenger Satisfaction"
#   (about 130,000 real survey answers). That original data is public.
#
#   Other entrants found that adding the original rows straight into training
#   makes results WORSE (the real and the generated passengers are slightly
#   different). So we use the original data in two safer ways instead:
#
#   1. ORIGINAL ROUTE RATES: for each flight distance (route), the share of
#      REAL passengers who were satisfied. Also per route AND class.
#   2. A "TEACHER" MODEL: a LightGBM trained ONLY on the original passengers.
#      It predicts every competition passenger, and that prediction becomes a
#      new column. It's like asking an experienced teacher for their opinion
#      before marking.
#
#   Both use ONLY the original data's answers, never the competition's, so they
#   cannot leak the competition target.
#
# THE EXPERIMENTS (LightGBM, same folds and settings as every round)
#   v5a = v3c features + original route rates
#   v5b = v5a + teacher prediction
#   v5_blend = rank average of: the best v5 LightGBM, XGBoost on the same
#              features, and your SAVED Round 4 CatBoost (no 70-minute wait).
#
# HOW LONG: about 5-10 minutes in total.
#
# BEFORE RUNNING: put the original dataset's CSV files in data\original\
#   (see ROUND5_GUIDE.md, part 3).
# ============================================================================

import time
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

from common import (load_data, find_feature_columns, make_categories_match,
                    target_to_number, DATA_DIR, OUTPUT_DIR, SUBMISSION_DIR,
                    TARGET, ID_COL)

# ----------------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------------
N_FOLDS = 5
RANDOM_SEED = 42
QUICK_TEST = False        # True = use only 50,000 rows to test the script fast
PREVIOUS_BEST = 0.96065   # your best CV so far (Round 4 rank-average blend)
SMOOTHING = 20            # see add_original_route_rates() for what this means
ORIGINAL_DIR = DATA_DIR / "original"

# Same model settings as every earlier round
LGBM_SETTINGS = {
    "n_estimators": 3000, "learning_rate": 0.05, "num_leaves": 63,
    "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED, "n_jobs": -1, "verbose": -1,
}
XGB_SETTINGS = {
    "n_estimators": 3000, "learning_rate": 0.05, "max_depth": 8,
    "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist",
    "enable_categorical": True, "max_cat_to_onehot": 1,
    "eval_metric": "auc", "early_stopping_rounds": 100,
    "random_state": RANDOM_SEED, "n_jobs": -1,
}

DISTANCE = "Flight Distance"
WORD_COLUMNS = ["Class", "Type of Travel", "Customer Type"]
KEY_RATINGS = ["Online boarding", "Inflight entertainment", "Seat comfort", "Inflight wifi service"]


# ============================================================================
# PART A: THE v3c FEATURES (the same as Rounds 3 and 4)
# ============================================================================

def short(name):
    return name.lower().replace(" ", "_")


def find_rating_columns(table, numeric_cols):
    return [c for c in numeric_cols if table[c].min() >= 0 and table[c].max() <= 5]


def build_v3c_features(raw_train, raw_test):
    n_train = len(raw_train)
    target = raw_train[TARGET].values
    everyone = pd.concat([raw_train.drop(columns=[TARGET]), raw_test], ignore_index=True)

    _, _, numeric = find_feature_columns(raw_train)
    rating_cols = find_rating_columns(raw_train, numeric)

    everyone["distance_count"] = everyone.groupby(DISTANCE)[DISTANCE].transform("size")
    for col in WORD_COLUMNS:
        if col in everyone.columns:
            everyone["count_distance_and_" + short(col)] = (
                everyone.groupby([DISTANCE, col])[DISTANCE].transform("size"))
    shares = {
        "route_share_business_class": ("Class", "Business"),
        "route_share_business_travel": ("Type of Travel", "Business travel"),
        "route_share_loyal": ("Customer Type", "Loyal Customer"),
    }
    for new_name, (col, value) in shares.items():
        if col in everyone.columns:
            is_value = (everyone[col].astype(str) == value).astype(float)
            everyone[new_name] = is_value.groupby(everyone[DISTANCE]).transform("mean")
    if "Age" in everyone.columns:
        everyone["route_mean_age"] = everyone.groupby(DISTANCE)["Age"].transform("mean")
    for rating in KEY_RATINGS:
        if rating in everyone.columns:
            answered = everyone[rating].replace(0, np.nan)
            everyone["route_mean_" + short(rating)] = (
                answered.groupby(everyone[DISTANCE]).transform("mean"))

    train = everyone.iloc[:n_train].copy()
    test = everyone.iloc[n_train:].copy()
    train[TARGET] = target

    features, categorical, _ = find_feature_columns(train)
    categorical = categorical + rating_cols
    train, test = make_categories_match(train, test, categorical)
    return train, test, features, categorical, rating_cols


# ============================================================================
# PART B: THE ORIGINAL DATASET
# ============================================================================

def load_original(competition_features):
    """Read every CSV in data/original and keep only the competition's columns."""
    files = sorted(ORIGINAL_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(
            f"\nI can't find any CSV files in {ORIGINAL_DIR}\n"
            "Download the 'Airline Passenger Satisfaction' dataset (teejmahal20) from Kaggle,\n"
            "unzip it, and put its CSV files in a folder called 'original' inside 'data'.\n"
            "See ROUND5_GUIDE.md, part 3.")

    # The original comes as two files (its own train.csv and test.csv); we use both
    original = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"Original data: {len(original):,} passengers from {len(files)} file(s)")

    if TARGET not in original.columns:
        raise SystemExit(f"The original files have no '{TARGET}' column. Check the files in {ORIGINAL_DIR}")

    # Its answers are words ("satisfied" / "neutral or dissatisfied"); turn them into 1/0
    original[TARGET] = target_to_number(original[TARGET])
    print(f"   satisfied in the original: {original[TARGET].mean():.1%}")

    # Keep only the columns the competition also has (e.g. drop 'Inflight service' and row numbers)
    dropped = [c for c in original.columns if c not in competition_features + [TARGET]]
    missing = [c for c in competition_features if c not in original.columns]
    if dropped:
        print(f"   ignoring columns not in the competition: {', '.join(dropped)}")
    if missing:
        print(f"   competition columns missing from the original: {', '.join(missing)}")
    keep = [c for c in competition_features if c in original.columns]
    return original[keep + [TARGET]]


def add_original_route_rates(train, test, original):
    """
    For each route (flight distance): what share of REAL passengers were satisfied?

    SMOOTHING: a route with only 2 real passengers, both satisfied, would get
    100%, which is far too confident. So we pretend every route also has
    SMOOTHING (20) extra "average" passengers. Then:
        rate = (satisfied on route + 20 x overall rate) / (passengers on route + 20)
    Big routes keep their own rate; tiny routes are pulled toward the average.
    """
    overall = original[TARGET].mean()

    def smoothed_rate(group_cols):
        stats = original.groupby(group_cols)[TARGET].agg(["sum", "count"])
        return (stats["sum"] + SMOOTHING * overall) / (stats["count"] + SMOOTHING), stats["count"]

    route_rate, route_count = smoothed_rate([DISTANCE])
    for table in (train, test):
        table["orig_route_rate"] = table[DISTANCE].map(route_rate).fillna(overall)
        table["orig_route_count"] = table[DISTANCE].map(route_count).fillna(0)

    # The same, but per route AND class (e.g. this distance in Business class)
    if "Class" in original.columns:
        pair_rate, _ = smoothed_rate([DISTANCE, "Class"])
        pair_rate = pair_rate.rename("orig_route_class_rate").reset_index()
        pair_rate["Class"] = pair_rate["Class"].astype(str)
        for table in (train, test):
            lookup = pd.DataFrame({DISTANCE: table[DISTANCE].values,
                                   "Class": table["Class"].astype(str).values})
            table["orig_route_class_rate"] = (
                lookup.merge(pair_rate, on=[DISTANCE, "Class"], how="left")
                ["orig_route_class_rate"].fillna(overall).values)
    return ["orig_route_rate", "orig_route_count", "orig_route_class_rate"]


def same_categories(tables, columns):
    """Give word columns the SAME list of possible values in every table."""
    for col in columns:
        values = sorted(pd.concat([t[col].astype(str).where(t[col].notna()) for t in tables])
                        .dropna().unique())
        kind = pd.CategoricalDtype(categories=values)
        for t in tables:
            t[col] = t[col].astype(str).where(t[col].notna()).astype(kind)


def add_teacher(train, test, original, raw_features, categorical_raw):
    """
    Train a LightGBM on the ORIGINAL passengers only, then let it predict every
    competition passenger. Its prediction becomes a new column.
    """
    orig = original.copy()
    tr = train[raw_features].copy()
    te = test[raw_features].copy()
    cats = [c for c in categorical_raw if c in orig.columns]
    same_categories([orig, tr, te], cats)

    cols = [c for c in raw_features if c in orig.columns]
    # Keep 10% of the original aside to tell the teacher when to stop (early stopping)
    fit_rows, stop_rows = train_test_split(orig, test_size=0.1, stratify=orig[TARGET],
                                           random_state=RANDOM_SEED)
    teacher = lgb.LGBMClassifier(**LGBM_SETTINGS)
    teacher.fit(fit_rows[cols], fit_rows[TARGET],
                eval_X=stop_rows[cols], eval_y=stop_rows[TARGET], eval_metric="auc",
                callbacks=[lgb.early_stopping(100, verbose=False)])

    train["orig_teacher"] = teacher.predict_proba(tr[cols])[:, 1]
    test["orig_teacher"] = teacher.predict_proba(te[cols])[:, 1]

    own = roc_auc_score(stop_rows[TARGET], teacher.predict_proba(stop_rows[cols])[:, 1])
    on_competition = roc_auc_score(train[TARGET], train["orig_teacher"])
    print(f"   teacher: AUC {own:.5f} on held-back REAL passengers, "
          f"{on_competition:.5f} on the competition passengers ({teacher.best_iteration_} trees)")
    return ["orig_teacher"]


# ============================================================================
# PART C: CROSS-VALIDATION (the same folds as every round)
# ============================================================================

def run_cv(model_name, train, test, features):
    X, y, X_test = train[features], train[TARGET], test[features]
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    oof, test_pred, scores = np.zeros(len(train)), np.zeros(len(test)), []

    for fold_number, (tr_rows, va_rows) in enumerate(folds.split(X, y), start=1):
        if model_name == "lightgbm":
            model = lgb.LGBMClassifier(**LGBM_SETTINGS)
            model.fit(X.iloc[tr_rows], y.iloc[tr_rows],
                      eval_X=X.iloc[va_rows], eval_y=y.iloc[va_rows], eval_metric="auc",
                      callbacks=[lgb.early_stopping(100, verbose=False)])
            n_trees = model.best_iteration_
        else:
            model = xgb.XGBClassifier(**XGB_SETTINGS)
            model.fit(X.iloc[tr_rows], y.iloc[tr_rows],
                      eval_set=[(X.iloc[va_rows], y.iloc[va_rows])], verbose=False)
            n_trees = model.best_iteration + 1
        oof[va_rows] = model.predict_proba(X.iloc[va_rows])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / N_FOLDS
        scores.append(roc_auc_score(y.iloc[va_rows], oof[va_rows]))
        print(f"   fold {fold_number}: AUC {scores[-1]:.5f}  ({n_trees} trees)")

    cv = roc_auc_score(y, oof)
    print(f"   CV AUC {cv:.5f}")
    return cv, np.std(scores), oof, test_pred


def to_ranks(values):
    return pd.Series(values).rank(pct=True).values


# ============================================================================
# PART D: RUN EVERYTHING
# ============================================================================

start_time = time.time()
raw_train, raw_test = load_data()
if QUICK_TEST:
    raw_train = raw_train.sample(50_000, random_state=RANDOM_SEED).reset_index(drop=True)
    print("QUICK_TEST is on: using 50,000 rows only. Scores will be lower.")

raw_features, raw_categorical, _ = find_feature_columns(raw_train)
original = load_original(raw_features)

train, test, v3c_features, _, rating_cols = build_v3c_features(raw_train, raw_test)
y = train[TARGET].values

print("\nBuilding features from the original data...")
rate_cols = add_original_route_rates(train, test, original)
teacher_cols = add_teacher(train, test, original, raw_features, raw_categorical + rating_cols)

OUTPUT_DIR.mkdir(exist_ok=True)
SUBMISSION_DIR.mkdir(exist_ok=True)
results = {}   # name -> (cv, spread, oof, test_pred, n_features, note)

experiments = [
    ("v5a_route_rates", v3c_features + rate_cols, "v3c + original route rates"),
    ("v5b_plus_teacher", v3c_features + rate_cols + teacher_cols,
     "v3c + original route rates + teacher"),
]
for name, features, note in experiments:
    print(f"\n=== {name} (LightGBM, {len(features)} features) ===")
    cv, spread, oof, test_pred = run_cv("lightgbm", train, test, features)
    results[name] = (cv, spread, oof, test_pred, len(features), note)

# ----------------------------------------------------------------------------
# The blend: best v5 LightGBM + XGBoost on the same features + saved CatBoost
# ----------------------------------------------------------------------------
best_lgbm = max(["v5a_route_rates", "v5b_plus_teacher"], key=lambda n: results[n][0])
best_features = dict((n, f) for n, f, _ in experiments)[best_lgbm]

print(f"\n=== XGBoost on the {best_lgbm} features ===")
cv, spread, oof, test_pred = run_cv("xgboost", train, test, best_features)
results["v5_xgboost"] = (cv, spread, oof, test_pred, len(best_features),
                         f"XGBoost on {best_lgbm} features")

blend_parts = {"lightgbm": results[best_lgbm], "xgboost": results["v5_xgboost"]}
cat_oof_path = OUTPUT_DIR / "oof_v4_catboost.csv"
cat_test_path = OUTPUT_DIR / "test_pred_v4_catboost.csv"
used_catboost = False
if cat_oof_path.exists() and cat_test_path.exists() and not QUICK_TEST:
    cat_oof = pd.read_csv(cat_oof_path)
    cat_test = pd.read_csv(cat_test_path)
    if (cat_oof[ID_COL].tolist() == train[ID_COL].tolist()
            and cat_test[ID_COL].tolist() == test[ID_COL].tolist()):
        cat_cv = roc_auc_score(y, cat_oof["oof"])
        blend_parts["catboost (Round 4)"] = (cat_cv, 0, cat_oof["oof"].values,
                                             cat_test["pred"].values, 0, "")
        used_catboost = True
        print(f"\nLoaded your saved Round 4 CatBoost (CV AUC {cat_cv:.5f})")
if QUICK_TEST:
    print("\nQuick test: skipping the saved Round 4 CatBoost (it's for the full data)")
elif not used_catboost:
    print("\nSaved Round 4 CatBoost not found (or doesn't match), so blending 2 models only")

blend_oof = np.mean([to_ranks(part[2]) for part in blend_parts.values()], axis=0)
blend_test = np.mean([to_ranks(part[3]) for part in blend_parts.values()], axis=0)
blend_name = "v5_blend_rank_average"
results[blend_name] = (roc_auc_score(y, blend_oof), 0, blend_oof, blend_test,
                       len(best_features),
                       "rank average of " + " + ".join(blend_parts))

print("\nHow similar the blended models are (rank correlation):")
print(pd.DataFrame({k: v[2] for k, v in blend_parts.items()}).corr(method="spearman")
      .round(4).to_string())

# ----------------------------------------------------------------------------
# Save submissions and log every result
# ----------------------------------------------------------------------------
log_path = OUTPUT_DIR / "experiment_log.csv"
for name, (cv, spread, oof, test_pred, n_features, note) in results.items():
    pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_pred}).to_csv(
        SUBMISSION_DIR / f"submission_{name}.csv", index=False)
    if not QUICK_TEST:
        pd.DataFrame({ID_COL: train[ID_COL], "oof": oof}).to_csv(
            OUTPUT_DIR / f"oof_{name}.csv", index=False)
    pd.DataFrame([{
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "experiment": name,
        "rows_used": len(train), "n_features": n_features, "cv_auc": round(cv, 5),
        "fold_std": round(spread, 5) if spread else "", "public_lb": "", "notes": note,
    }]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
print("\n" + "=" * 72)
print("ROUND 5 SUMMARY")
print("=" * 72)
print(f"{'experiment':<26}{'CV AUC':>10}{'vs Round 4 blend (0.96065)':>30}")
print(f"{'v3c LightGBM (reference)':<26}{0.96026:>10.5f}{0.96026 - PREVIOUS_BEST:>+30.5f}")
for name, (cv, *_rest) in results.items():
    print(f"{name:<26}{cv:>10.5f}{cv - PREVIOUS_BEST:>+30.5f}")
print(f"\nBlend = rank average of: {', '.join(blend_parts)}")

best_name = max(results, key=lambda n: results[n][0])
best_cv = results[best_name][0]
if best_cv > PREVIOUS_BEST:
    print(f"\nRECOMMENDED UPLOAD: submissions/submission_{best_name}.csv  (CV AUC {best_cv:.5f})")
else:
    print(f"\nNothing beat your Round 4 blend ({PREVIOUS_BEST}). Don't upload; your Round 4 file is still your best.")
print(f"Finished in {(time.time() - start_time) / 60:.1f} minutes.")
