# ============================================================================
# ROUND 4: THREE MODELS AND A BLEND
# Run with:   python scripts/06_model_blend.py
#
# THE IDEA
#   So far we've used one model: LightGBM. Two other popular models are
#   XGBoost and CatBoost. All three build many decision trees, but each
#   builds them in a slightly different way, so they make slightly
#   DIFFERENT mistakes.
#
#   If we average their predictions, some of those mistakes cancel out.
#   This is called BLENDING (or ensembling). It's how most Kaggle
#   competitions are won.
#
#   Think of three teachers marking the same exam. Each has their own small
#   biases, but the average of their three marks is usually fairer than any
#   one teacher's mark.
#
# WHAT THIS SCRIPT DOES
#   1. Builds the v3c features (your best so far: lean base + pair counts
#      + route profile).
#   2. Trains LightGBM, XGBoost and CatBoost, each with the SAME 5 folds.
#   3. Tries three ways of blending them and compares everything.
#
# HOW LONG: roughly 10 minutes for LightGBM, 10-20 for XGBoost and 20-40 for
#   CatBoost: up to about an hour in total. You can leave it running.
#   If it stops part-way (laptop sleeps, etc.), just run it again: models that
#   already finished are loaded from disk instead of retrained.
# ============================================================================

import time
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from common import (load_data, find_feature_columns, make_categories_match,
                    OUTPUT_DIR, SUBMISSION_DIR, TARGET, ID_COL)

# ----------------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------------
N_FOLDS = 5
RANDOM_SEED = 42
QUICK_TEST = False       # True = use only 50,000 rows to test the script fast
PREVIOUS_BEST = 0.96026  # your best CV so far (v3c), for comparison

# Each model's settings. LightGBM's are the same as every earlier round,
# so its score should match v3c (0.96026).
LGBM_SETTINGS = {
    "n_estimators": 3000, "learning_rate": 0.05, "num_leaves": 63,
    "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED, "n_jobs": -1, "verbose": -1,
}
XGB_SETTINGS = {
    "n_estimators": 3000, "learning_rate": 0.05, "max_depth": 8,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "tree_method": "hist",          # the fast way of building trees
    "enable_categorical": True,     # let XGBoost use our word columns directly
    "max_cat_to_onehot": 1,
    "eval_metric": "auc", "early_stopping_rounds": 100,
    "random_state": RANDOM_SEED, "n_jobs": -1,
}
CAT_SETTINGS = {
    "iterations": 3000, "learning_rate": 0.08, "depth": 7,
    "eval_metric": "AUC",
    "od_type": "Iter", "od_wait": 100,   # CatBoost's name for early stopping
    "random_seed": RANDOM_SEED, "thread_count": -1, "verbose": 0,
}

# Column names and lists (the same as Round 3)
DISTANCE = "Flight Distance"
WORD_COLUMNS = ["Class", "Type of Travel", "Customer Type"]
KEY_RATINGS = ["Online boarding", "Inflight entertainment", "Seat comfort", "Inflight wifi service"]


# ============================================================================
# PART A: THE v3c FEATURES (copied from 05_route_features.py)
# ============================================================================

def short(name):
    return name.lower().replace(" ", "_")


def find_rating_columns(table, numeric_cols):
    return [c for c in numeric_cols if table[c].min() >= 0 and table[c].max() <= 5]


def build_v3c_features(raw_train, raw_test):
    """Lean base + pair counts + route profile, exactly as in Round 3's v3c."""
    n_train = len(raw_train)
    target = raw_train[TARGET].values
    everyone = pd.concat([raw_train.drop(columns=[TARGET]), raw_test], ignore_index=True)

    _, _, numeric = find_feature_columns(raw_train)
    rating_cols = find_rating_columns(raw_train, numeric)

    # Lean base: distance count
    everyone["distance_count"] = everyone.groupby(DISTANCE)[DISTANCE].transform("size")

    # Pair counts
    for col in WORD_COLUMNS:
        if col in everyone.columns:
            everyone["count_distance_and_" + short(col)] = (
                everyone.groupby([DISTANCE, col])[DISTANCE].transform("size"))

    # Route profile
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
    categorical = categorical + rating_cols      # ratings as categories (lean base)
    train, test = make_categories_match(train, test, categorical)
    return train, test, features, categorical


# ============================================================================
# PART B: ONE FUNCTION PER MODEL
# Each one trains on 4 folds, predicts the 5th fold and the test set.
# ============================================================================

def fit_lightgbm(X_tr, y_tr, X_va, y_va, X_test, categorical):
    model = lgb.LGBMClassifier(**LGBM_SETTINGS)
    model.fit(X_tr, y_tr, eval_X=X_va, eval_y=y_va, eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    return (model.predict_proba(X_va)[:, 1], model.predict_proba(X_test)[:, 1],
            model.best_iteration_)


def fit_xgboost(X_tr, y_tr, X_va, y_va, X_test, categorical):
    model = xgb.XGBClassifier(**XGB_SETTINGS)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    # With early stopping, predict_proba automatically uses the best number of trees
    return (model.predict_proba(X_va)[:, 1], model.predict_proba(X_test)[:, 1],
            model.best_iteration + 1)


def as_text(table, categorical):
    """CatBoost wants word columns as plain text, with no empty cells."""
    table = table.copy()
    for col in categorical:
        table[col] = table[col].astype(str)      # empty cells become the text "nan"
    return table


def fit_catboost(X_tr, y_tr, X_va, y_va, X_test, categorical):
    model = CatBoostClassifier(**CAT_SETTINGS)
    model.fit(as_text(X_tr, categorical), y_tr, cat_features=categorical,
              eval_set=(as_text(X_va, categorical), y_va), use_best_model=True)
    return (model.predict_proba(as_text(X_va, categorical))[:, 1],
            model.predict_proba(as_text(X_test, categorical))[:, 1],
            model.get_best_iteration() + 1)


MODELS = {
    "lightgbm": fit_lightgbm,
    "xgboost": fit_xgboost,
    "catboost": fit_catboost,
}


def run_cross_validation(model_name, fit_function, train, test, features, categorical):
    """5-fold CV for one model. Returns out-of-fold and test predictions."""
    oof_path = OUTPUT_DIR / f"oof_v4_{model_name}.csv"
    test_path = OUTPUT_DIR / f"test_pred_v4_{model_name}.csv"

    # RESUME: if this model already finished in an earlier run, load its results.
    # We only reuse them if they are for exactly the same passengers (same ids,
    # same order); otherwise we retrain.
    if oof_path.exists() and test_path.exists() and not QUICK_TEST:
        saved_oof = pd.read_csv(oof_path)
        saved_test = pd.read_csv(test_path)
        same_train = saved_oof[ID_COL].tolist() == train[ID_COL].tolist()
        same_test = saved_test[ID_COL].tolist() == test[ID_COL].tolist()
        if same_train and same_test:
            print(f"   found saved results for {model_name}, loading them (delete the files to retrain)")
            return saved_oof["oof"].values, saved_test["pred"].values
        print(f"   saved results for {model_name} don't match this data, retraining")

    X, y, X_test = train[features], train[TARGET], test[features]
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))

    for fold_number, (tr_rows, va_rows) in enumerate(folds.split(X, y), start=1):
        fold_start = time.time()
        va_pred, te_pred, n_trees = fit_function(
            X.iloc[tr_rows], y.iloc[tr_rows], X.iloc[va_rows], y.iloc[va_rows],
            X_test, categorical)
        oof[va_rows] = va_pred
        test_pred += te_pred / N_FOLDS
        print(f"   fold {fold_number}: AUC {roc_auc_score(y.iloc[va_rows], va_pred):.5f}"
              f"  ({n_trees} trees, {(time.time() - fold_start) / 60:.1f} min)")

    # Save straight away, so a crash later doesn't lose this model's work.
    # (A quick test is never saved, so it can't be mixed up with the real run.)
    if not QUICK_TEST:
        pd.DataFrame({ID_COL: train[ID_COL], "oof": oof}).to_csv(oof_path, index=False)
        pd.DataFrame({ID_COL: test[ID_COL], "pred": test_pred}).to_csv(test_path, index=False)
    return oof, test_pred


# ============================================================================
# PART C: BLENDING
# ============================================================================

def to_ranks(predictions):
    """
    Turn predictions into ranks between 0 and 1 (0 = lowest, 1 = highest).
    AUC only cares about the ORDER of the predictions, not their exact values.
    Different models spread their probabilities differently, so averaging
    ranks is often fairer than averaging the raw probabilities.
    """
    return pd.Series(predictions).rank(pct=True).values


def best_weights(oof_by_model, y):
    """
    Try every combination of weights in steps of 0.1 that add up to 1
    (e.g. 0.5 LightGBM + 0.3 XGBoost + 0.2 CatBoost) and keep the best.
    """
    names = list(oof_by_model)
    ranked = {name: to_ranks(oof_by_model[name]) for name in names}
    steps = [i / 10 for i in range(11)]
    best = (None, -1.0)
    for weights in product(steps, repeat=len(names)):
        if abs(sum(weights) - 1) > 1e-9:
            continue                                   # weights must add up to 1
        blend = sum(w * ranked[name] for w, name in zip(weights, names))
        score = roc_auc_score(y, blend)
        if score > best[1]:
            best = (dict(zip(names, weights)), score)
    return best


# ============================================================================
# PART D: RUN EVERYTHING
# ============================================================================

start_time = time.time()
raw_train, raw_test = load_data()
if QUICK_TEST:
    raw_train = raw_train.sample(50_000, random_state=RANDOM_SEED).reset_index(drop=True)
    print("QUICK_TEST is on: using 50,000 rows only. Scores will be lower.")

OUTPUT_DIR.mkdir(exist_ok=True)
SUBMISSION_DIR.mkdir(exist_ok=True)

train, test, features, categorical = build_v3c_features(raw_train, raw_test)
y = train[TARGET].values
print(f"Using the v3c features: {len(features)} columns")

oof_by_model, test_by_model, results = {}, {}, []
for model_name, fit_function in MODELS.items():
    print(f"\n=== {model_name} ===")
    oof, test_pred = run_cross_validation(model_name, fit_function, train, test,
                                          features, categorical)
    oof_by_model[model_name] = oof
    test_by_model[model_name] = test_pred
    cv = roc_auc_score(y, oof)
    print(f"   CV AUC {cv:.5f}")
    results.append((f"v4_{model_name}", cv, test_pred))

# How similar are the models? (1.0 = identical rankings; lower = more different)
print("\nHow similar the models' predictions are (rank correlation):")
similarity = pd.DataFrame(oof_by_model).corr(method="spearman")
print(similarity.round(4).to_string())

# Blend 1: simple average of probabilities
avg = np.mean([oof_by_model[m] for m in MODELS], axis=0)
avg_test = np.mean([test_by_model[m] for m in MODELS], axis=0)
results.append(("v4_blend_average", roc_auc_score(y, avg), avg_test))

# Blend 2: simple average of ranks
rank_avg = np.mean([to_ranks(oof_by_model[m]) for m in MODELS], axis=0)
rank_avg_test = np.mean([to_ranks(test_by_model[m]) for m in MODELS], axis=0)
results.append(("v4_blend_rank_average", roc_auc_score(y, rank_avg), rank_avg_test))

# Blend 3: best weights (searched on the out-of-fold predictions)
weights, _ = best_weights(oof_by_model, y)
weighted = sum(w * to_ranks(oof_by_model[m]) for m, w in weights.items())
weighted_test = sum(w * to_ranks(test_by_model[m]) for m, w in weights.items())
weight_text = ", ".join(f"{m} {w:.1f}" for m, w in weights.items())
results.append(("v4_blend_weighted", roc_auc_score(y, weighted), weighted_test))

# Save every submission and log every result
log_path = OUTPUT_DIR / "experiment_log.csv"
for name, cv, test_pred in results:
    pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_pred}).to_csv(
        SUBMISSION_DIR / f"submission_{name}.csv", index=False)
    note = f"v3c features; {name.replace('v4_', '')}"
    if name == "v4_blend_weighted":
        note += f" ({weight_text})"
    pd.DataFrame([{
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "experiment": name, "rows_used": len(train), "n_features": len(features),
        "cv_auc": round(cv, 5), "fold_std": "", "public_lb": "", "notes": note,
    }]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)

# ----------------------------------------------------------------------------
# Summary table
# ----------------------------------------------------------------------------
print("\n" + "=" * 66)
print("ROUND 4 SUMMARY")
print("=" * 66)
print(f"{'experiment':<26}{'CV AUC':>10}{'vs v3c (0.96026)':>20}")
for name, cv, _ in results:
    print(f"{name:<26}{cv:>10.5f}{cv - PREVIOUS_BEST:>+20.5f}")
print(f"\nBest weights found: {weight_text}")

# Recommend a file. The weighted blend chose its weights by looking at the
# out-of-fold answers, so its score is a little optimistic. We only prefer it
# over the plain rank average if it is clearly better (by more than 0.00005).
scores = {name: cv for name, cv, _ in results}
blends = ["v4_blend_average", "v4_blend_rank_average"]
choice = max(blends, key=lambda n: scores[n])
if scores["v4_blend_weighted"] - scores[choice] > 0.00005:
    choice = "v4_blend_weighted"
single_best = max((n for n in scores if n not in blends + ["v4_blend_weighted"]),
                  key=lambda n: scores[n])
if scores[single_best] > scores[choice]:
    choice = single_best

print(f"\nRECOMMENDED UPLOAD: submissions/submission_{choice}.csv  (CV AUC {scores[choice]:.5f})")
if scores[choice] <= PREVIOUS_BEST:
    print("This does not beat v3c, so you may prefer not to upload it.")
print(f"Finished in {(time.time() - start_time) / 60:.1f} minutes.")
