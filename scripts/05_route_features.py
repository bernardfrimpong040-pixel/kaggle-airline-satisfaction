# ============================================================================
# ROUND 3: ROUTE FEATURES
# Run with:   python scripts/05_route_features.py
#
# WHAT WE LEARNED IN ROUND 2
#   Only one feature gave a real gain: "distance count" (how many passengers
#   share the exact same flight distance). The same distance usually means
#   the same ROUTE, e.g. Cleveland -> Chicago. So this round asks:
#   what else can we learn about each route?
#
# THE 4 EXPERIMENTS (each adds one more group to the one before):
#   v3a = LEAN BASE: original columns + distance count + ratings as categories.
#         We DROP the Round 2 groups that didn't help. If v3a scores about the
#         same as v2e (0.95923), dropping them was the right call.
#   v3b = + pair counts: how many passengers share the same distance AND the
#         same class (and the same distance AND the same trip type, etc.)
#   v3c = + route profile: for each route, the share of business-class
#         passengers, business travellers, loyal customers, the average age,
#         and the average of the 4 strongest ratings.
#   v3d = + route residuals: how THIS passenger's ratings compare with the
#         average on their route. "Online boarding 2 on a route where people
#         usually give 4" is a strong sign of an unhappy passenger.
#
# NONE of these features use the satisfaction answers, so none can leak
# the target. That is why it's safe to compute them on train AND test together.
#
# Everything else (folds, seed, model settings) is exactly the same as before.
# HOW LONG: about 5-10 minutes per experiment, so 20-40 minutes in total.
# ============================================================================

import time
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from common import (load_data, find_feature_columns, make_categories_match,
                    OUTPUT_DIR, SUBMISSION_DIR, TARGET, ID_COL)

# ----------------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------------
N_FOLDS = 5
RANDOM_SEED = 42
QUICK_TEST = False      # True = use only 50,000 rows to test the script fast
PREVIOUS_BEST = 0.95923 # your best CV so far (v2e), for comparison

# Same model settings as Rounds 1 and 2 (do NOT change them in this round)
MODEL_SETTINGS = {
    "n_estimators": 3000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

EXPERIMENTS = [
    ("v3a_lean_base",       ["base"]),
    ("v3b_plus_pair_counts", ["base", "pair_counts"]),
    ("v3c_plus_route_profile", ["base", "pair_counts", "route_profile"]),
    ("v3d_plus_route_residuals", ["base", "pair_counts", "route_profile", "route_residuals"]),
]

# Column names in the competition data
DISTANCE = "Flight Distance"
WORD_COLUMNS = ["Class", "Type of Travel", "Customer Type"]
# The 4 strongest ratings from your first-look report
KEY_RATINGS = ["Online boarding", "Inflight entertainment", "Seat comfort", "Inflight wifi service"]


# ============================================================================
# PART A: THE FEATURE GROUPS
# All of them work on "everyone" = train and test stacked into one table,
# so that route statistics use every passenger we have.
# ============================================================================

def find_rating_columns(table, numeric_cols):
    """The service ratings are the number columns whose values only go from 0 to 5."""
    return [c for c in numeric_cols if table[c].min() >= 0 and table[c].max() <= 5]


def add_distance_count(everyone):
    """How many passengers share this exact flight distance (from Round 2)."""
    # groupby(...).transform("size") puts each group's size on every row of that group
    everyone["distance_count"] = everyone.groupby(DISTANCE)[DISTANCE].transform("size")
    return everyone


def add_pair_counts(everyone):
    """How many passengers share this distance AND this class / trip type / customer type."""
    for col in WORD_COLUMNS:
        if col in everyone.columns:
            new_name = "count_distance_and_" + col.lower().replace(" ", "_")
            everyone[new_name] = everyone.groupby([DISTANCE, col])[DISTANCE].transform("size")
    return everyone


def add_route_profile(everyone):
    """Describe each route by the kind of passengers who fly it."""
    route = everyone.groupby(DISTANCE)

    # Share of passengers on this route who are in Business class, etc.
    # (x == "Business") is True/False; the mean of True/False is the share of True.
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
        everyone["route_mean_age"] = route["Age"].transform("mean")

    # Average of the strongest ratings on this route (0 = not answered, so left out)
    for rating in KEY_RATINGS:
        if rating in everyone.columns:
            answered = everyone[rating].replace(0, np.nan)
            everyone["route_mean_" + short(rating)] = (
                answered.groupby(everyone[DISTANCE]).transform("mean"))
    return everyone


def add_route_residuals(everyone):
    """This passenger's rating MINUS the average rating on their route."""
    for rating in KEY_RATINGS:
        route_col = "route_mean_" + short(rating)
        if rating in everyone.columns and route_col in everyone.columns:
            mine = everyone[rating].replace(0, np.nan)   # unanswered stays empty
            everyone["vs_route_" + short(rating)] = mine - everyone[route_col]
    return everyone


def short(name):
    """'Online boarding' -> 'online_boarding' (column names without spaces)."""
    return name.lower().replace(" ", "_")


def build_features(raw_train, raw_test, groups):
    """Stack train and test, add the chosen feature groups, then split them apart again."""
    n_train = len(raw_train)
    target = raw_train[TARGET].values

    # Stack: train rows first, then test rows. ignore_index gives rows 0, 1, 2, ...
    everyone = pd.concat([raw_train.drop(columns=[TARGET]), raw_test], ignore_index=True)

    _, _, numeric = find_feature_columns(raw_train)
    rating_cols = find_rating_columns(raw_train, numeric)

    if "base" in groups:
        everyone = add_distance_count(everyone)
    if "pair_counts" in groups:
        everyone = add_pair_counts(everyone)
    if "route_profile" in groups:
        everyone = add_route_profile(everyone)
    if "route_residuals" in groups:
        everyone = add_route_residuals(everyone)

    # Split back: the first n_train rows are train, the rest are test
    train = everyone.iloc[:n_train].copy()
    test = everyone.iloc[n_train:].copy()
    train[TARGET] = target

    # Sort columns into word and number columns, then (base) treat ratings as categories
    features, categorical, _ = find_feature_columns(train)
    if "base" in groups:
        categorical = categorical + rating_cols

    train, test = make_categories_match(train, test, categorical)
    return train, test, features


# ============================================================================
# PART B: CROSS-VALIDATION (the same method as Rounds 1 and 2)
# ============================================================================

def run_cross_validation(train, test, features):
    X, y, X_test = train[features], train[TARGET], test[features]
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    oof_predictions = np.zeros(len(train))
    test_predictions = np.zeros(len(test))
    fold_scores = []

    for fold_number, (train_rows, valid_rows) in enumerate(folds.split(X, y), start=1):
        model = lgb.LGBMClassifier(**MODEL_SETTINGS)
        model.fit(
            X.iloc[train_rows], y.iloc[train_rows],
            eval_X=X.iloc[valid_rows], eval_y=y.iloc[valid_rows],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        valid_pred = model.predict_proba(X.iloc[valid_rows])[:, 1]
        oof_predictions[valid_rows] = valid_pred
        fold_scores.append(roc_auc_score(y.iloc[valid_rows], valid_pred))
        test_predictions += model.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f"   fold {fold_number}: AUC {fold_scores[-1]:.5f}  ({model.best_iteration_} trees)")

    importance = pd.Series(model.booster_.feature_importance("gain"), index=features)
    importance = (importance / importance.sum()).sort_values(ascending=False)

    cv_score = roc_auc_score(y, oof_predictions)
    return cv_score, np.std(fold_scores), oof_predictions, test_predictions, importance


# ============================================================================
# PART C: RUN ALL EXPERIMENTS
# ============================================================================

start_time = time.time()
raw_train, raw_test = load_data()
if QUICK_TEST:
    raw_train = raw_train.sample(50_000, random_state=RANDOM_SEED).reset_index(drop=True)
    print("QUICK_TEST is on: using 50,000 rows only. Scores will be lower.")

OUTPUT_DIR.mkdir(exist_ok=True)
SUBMISSION_DIR.mkdir(exist_ok=True)
log_path = OUTPUT_DIR / "experiment_log.csv"
results = []

for name, groups in EXPERIMENTS:
    print(f"\n=== {name}: {', '.join(groups)} ===")
    train, test, features = build_features(raw_train, raw_test, groups)
    print(f"   {len(features)} features")

    cv, spread, oof, test_pred, importance = run_cross_validation(train, test, features)
    print(f"   CV AUC {cv:.5f}   (change vs previous best: {cv - PREVIOUS_BEST:+.5f})")
    results.append((name, len(features), cv, spread))

    # Show the 5 most useful features, so we can see if the new ones are used
    print("   top 5 features: " + ", ".join(importance.head(5).index))

    pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_pred}).to_csv(
        SUBMISSION_DIR / f"submission_{name}.csv", index=False)
    pd.DataFrame({ID_COL: train[ID_COL], "oof": oof}).to_csv(
        OUTPUT_DIR / f"oof_{name}.csv", index=False)
    importance.rename("share_of_gain").to_csv(OUTPUT_DIR / f"importance_{name}.csv")

    pd.DataFrame([{
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "experiment": name,
        "rows_used": len(train),
        "n_features": len(features),
        "cv_auc": round(cv, 5),
        "fold_std": round(spread, 5),
        "public_lb": "",
        "notes": "raw + distance count + ratings as categories + " + " + ".join(groups[1:]) if len(groups) > 1
                 else "raw + distance count + ratings as categories (lean base)",
    }]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)

# ----------------------------------------------------------------------------
# Summary table
# ----------------------------------------------------------------------------
print("\n" + "=" * 78)
print("ROUND 3 SUMMARY (each row adds one more feature group)")
print("=" * 78)
print(f"{'experiment':<28}{'features':>9}{'CV AUC':>10}{'± folds':>10}{'vs row above':>14}")
print(f"{'v2e (previous best)':<28}{35:>9}{PREVIOUS_BEST:>10.5f}{'':>10}{'':>14}")
previous = PREVIOUS_BEST
for name, n, cv, spread in results:
    print(f"{name:<28}{n:>9}{cv:>10.5f}{spread:>10.5f}{cv - previous:>+14.5f}")
    previous = cv

best = max(results, key=lambda row: row[2])
print(f"\nBest experiment: {best[0]} (CV AUC {best[2]:.5f}, "
      f"{best[2] - PREVIOUS_BEST:+.5f} vs previous best)")
if best[2] > PREVIOUS_BEST:
    print(f"Upload: submissions/submission_{best[0]}.csv  (check it first with 03_check_submission.py)")
else:
    print("No experiment beat v2e. Don't upload; your v2e submission is still your best.")
print(f"Finished in {(time.time() - start_time) / 60:.1f} minutes.")
