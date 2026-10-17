# ============================================================================
# ROUND 2: NEW FEATURES ("feature engineering")
# Run with:   python scripts/04_feature_experiments.py
#
# WHAT IS A NEW FEATURE?
#   A new column we build from the existing columns. Example: the AVERAGE of
#   all 13 service ratings. The model could, in theory, work this out by
#   itself, but handing it the answer directly often helps.
#
# HOW THIS SCRIPT TESTS THEM FAIRLY
#   It runs 5 experiments. Each one ADDS one more group of features to the
#   one before:
#       v2a = baseline + rating summaries
#       v2b = v2a + delay features
#       v2c = v2b + passenger "segment"
#       v2d = v2c + flight distance counts
#       v2e = v2d + ratings treated as categories
#   Everything else (folds, seed, model settings) stays EXACTLY the same as
#   the baseline. So if the score changes, we know the features caused it.
#   Scientists call this "changing one thing at a time".
#
# HOW LONG: about 5-10 minutes PER experiment, so 30-50 minutes in total.
#   You can leave it running. Set QUICK_TEST = True to check it works first.
#
# AT THE END: a table comparing every experiment with your baseline (0.95878).
#   Only upload the best one to Kaggle.
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
N_FOLDS = 5             # same as baseline
RANDOM_SEED = 42        # same as baseline, so the folds are identical
QUICK_TEST = False      # True = use only 50,000 rows to test the script fast
BASELINE_CV = 0.95878   # your Round 1 score, for comparison

# Same model settings as the baseline (do NOT change them in this round)
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

# The 5 experiments: (name, list of feature groups switched on)
EXPERIMENTS = [
    ("v2a_rating_stats",        ["rating_stats"]),
    ("v2b_plus_delays",         ["rating_stats", "delays"]),
    ("v2c_plus_segment",        ["rating_stats", "delays", "segment"]),
    ("v2d_plus_distance_count", ["rating_stats", "delays", "segment", "distance_count"]),
    ("v2e_plus_ratings_as_cat", ["rating_stats", "delays", "segment", "distance_count",
                                 "ratings_as_categories"]),
]

# Column names in the competition data (we check they exist before using them)
DEP_DELAY = "Departure Delay in Minutes"
ARR_DELAY = "Arrival Delay in Minutes"
DISTANCE = "Flight Distance"
DIGITAL_RATINGS = ["Online boarding", "Inflight wifi service", "Ease of Online booking"]


# ============================================================================
# PART A: THE FEATURE GROUPS
# Each function takes a table and adds new columns to it.
# ============================================================================

def find_rating_columns(table, numeric_cols):
    """The service ratings are the number columns whose values only go from 0 to 5."""
    return [c for c in numeric_cols if table[c].min() >= 0 and table[c].max() <= 5]


def add_rating_stats(table, rating_cols):
    """Summaries of each passenger's 13 ratings."""
    ratings = table[rating_cols]

    # In this survey a 0 means "not answered", not "terrible".
    # So for averages we hide the 0s (turn them into empty cells).
    answered = ratings.replace(0, np.nan)

    table["rating_mean"] = answered.mean(axis=1)      # axis=1 means "across the row"
    table["rating_min"] = answered.min(axis=1)        # their worst score
    table["rating_max"] = answered.max(axis=1)        # their best score
    table["rating_spread"] = answered.std(axis=1)     # how much their scores vary
    table["n_not_answered"] = (ratings == 0).sum(axis=1)
    table["n_top_marks"] = (ratings == 5).sum(axis=1)
    table["n_low_marks"] = ratings.isin([1, 2]).sum(axis=1)

    # Your first-look report showed Online boarding was the strongest column,
    # so we also average the three "digital experience" ratings.
    digital = [c for c in DIGITAL_RATINGS if c in table.columns]
    if digital:
        table["digital_mean"] = table[digital].replace(0, np.nan).mean(axis=1)
    return table


def add_delay_features(table):
    """Delays on their own were weak (AUC 0.509). Combinations might do better."""
    if DEP_DELAY not in table.columns or ARR_DELAY not in table.columns:
        print("   (delay columns not found, skipping delay features)")
        return table

    departure = table[DEP_DELAY]
    # Fill the 292 missing arrival delays with the departure delay (a sensible guess)
    arrival = table[ARR_DELAY].fillna(departure)

    table["total_delay"] = departure + arrival
    table["delay_made_up"] = departure - arrival       # positive = the plane caught up
    table["late_over_15min"] = (arrival > 15).astype(int)
    if DISTANCE in table.columns:
        # A 60-minute delay feels worse on a short flight than on a long one
        table["delay_per_1000_miles"] = arrival / (table[DISTANCE] / 1000)
    return table


def add_segment(table):
    """
    Combine Class + Type of Travel + Customer Type into one 'segment' column,
    e.g. "Eco | Personal Travel | Loyal Customer".
    Your report showed these three matter a lot; together they may matter more.
    """
    parts = [c for c in ["Class", "Type of Travel", "Customer Type"] if c in table.columns]
    if len(parts) < 2:
        print("   (segment columns not found, skipping)")
        return table
    table["segment"] = table[parts].astype(str).agg(" | ".join, axis=1)
    return table


def add_distance_count(train, test):
    """
    How many passengers (in train AND test together) share the exact same
    flight distance. The same distance usually means the same route, so this
    tells the model how common or rare a route is.
    This uses NO satisfaction answers, so it cannot leak the target.
    """
    if DISTANCE not in train.columns:
        print("   (flight distance not found, skipping)")
        return train, test
    counts = pd.concat([train[DISTANCE], test[DISTANCE]]).value_counts()
    train["distance_count"] = train[DISTANCE].map(counts)
    test["distance_count"] = test[DISTANCE].map(counts)
    return train, test


def build_features(raw_train, raw_test, groups):
    """Start from fresh copies of the data and add the chosen feature groups."""
    train = raw_train.copy()     # .copy() so we never change the original data
    test = raw_test.copy()

    _, _, numeric = find_feature_columns(train)
    rating_cols = find_rating_columns(train, numeric)

    if "rating_stats" in groups:
        train = add_rating_stats(train, rating_cols)
        test = add_rating_stats(test, rating_cols)
    if "delays" in groups:
        train = add_delay_features(train)
        test = add_delay_features(test)
    if "segment" in groups:
        train = add_segment(train)
        test = add_segment(test)
    if "distance_count" in groups:
        train, test = add_distance_count(train, test)

    # Now sort all columns (old + new) into word columns and number columns
    features, categorical, _ = find_feature_columns(train)

    # Optional: treat each rating (0-5) as a category instead of a number.
    # Then the model can treat "0 = not answered" as its own thing, rather
    # than as "one less than 1".
    if "ratings_as_categories" in groups:
        categorical = categorical + rating_cols

    train, test = make_categories_match(train, test, categorical)
    return train, test, features


# ============================================================================
# PART B: CROSS-VALIDATION (the same method as 02_baseline_model.py)
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
    raw_train = raw_train.sample(50_000, random_state=RANDOM_SEED)
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
    gain = cv - BASELINE_CV
    print(f"   CV AUC {cv:.5f}   (change vs baseline: {gain:+.5f})")
    results.append((name, len(features), cv, spread, gain))

    # Save this experiment's submission, out-of-fold predictions and importance
    pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_pred}).to_csv(
        SUBMISSION_DIR / f"submission_{name}.csv", index=False)
    pd.DataFrame({ID_COL: train[ID_COL], "oof": oof}).to_csv(
        OUTPUT_DIR / f"oof_{name}.csv", index=False)
    importance.rename("share_of_gain").to_csv(OUTPUT_DIR / f"importance_{name}.csv")

    # Add a line to the experiment log
    pd.DataFrame([{
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "experiment": name,
        "rows_used": len(train),
        "n_features": len(features),
        "cv_auc": round(cv, 5),
        "fold_std": round(spread, 5),
        "public_lb": "",
        "notes": "baseline + " + " + ".join(groups),
    }]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)

# ----------------------------------------------------------------------------
# Summary table
# ----------------------------------------------------------------------------
print("\n" + "=" * 72)
print("ROUND 2 SUMMARY (each row adds one more feature group)")
print("=" * 72)
print(f"{'experiment':<26}{'features':>9}{'CV AUC':>10}{'± folds':>10}{'vs baseline':>13}")
print(f"{'baseline (Round 1)':<26}{21:>9}{BASELINE_CV:>10.5f}{'':>10}{'':>13}")
for name, n, cv, spread, gain in results:
    print(f"{name:<26}{n:>9}{cv:>10.5f}{spread:>10.5f}{gain:>+13.5f}")

best = max(results, key=lambda row: row[2])
print(f"\nBest experiment: {best[0]} (CV AUC {best[2]:.5f})")
print(f"Upload: submissions/submission_{best[0]}.csv  (check it first with 03_check_submission.py)")
print("\nRule of thumb: a change smaller than the '± folds' number may just be luck.")
print(f"Finished in {(time.time() - start_time) / 60:.1f} minutes.")
