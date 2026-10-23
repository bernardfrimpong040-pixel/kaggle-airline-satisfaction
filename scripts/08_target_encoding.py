# ============================================================================
# ROUND 6: TARGET ENCODING (DONE SAFELY) AND A HILL-CLIMBING BLEND
# Run with:   python scripts/08_target_encoding.py
#
# PART 1: TARGET ENCODING
#   For each flight distance (route), what share of COMPETITION passengers
#   were satisfied? That number is called a "target encoding", because it is
#   built from the target (the satisfaction answers).
#
#   DANGER: if a passenger's own answer is used to build their own encoding,
#   the model sees a hint of the answer. That's DATA LEAKAGE: the score looks
#   great on our laptop but drops on Kaggle.
#
#   THE SAFE WAY (what we do):
#     Inside each of our 5 folds:
#       - the 4 training folds get encodings from an INNER split, so no
#         passenger's own answer is ever used for their own value
#         (scikit-learn's TargetEncoder does this for us with fit_transform)
#       - the held-out fold and the test set get encodings learned ONLY from
#         the 4 training folds (TargetEncoder.transform)
#     So the held-out fold is never touched by its own answers. Honest score.
#
#   THE LEAKY WAY (a DEMO, never uploaded):
#     Learn the encodings once from ALL training answers, then cross-validate.
#     Every held-out passenger's answer helped build their own encoding.
#     Watch the CV score: it rises, but it's a lie. This is exactly the kind of
#     flaw the North Hires job asks you to catch.
#
# PART 2: HILL-CLIMBING BLEND
#   You now have many saved models (Rounds 4, 5 and 6). Instead of averaging
#   a fixed three, we let the computer build the best mix:
#     start with the best single model, then repeatedly add whichever model
#     (repeats allowed) improves the blend's CV the most; stop when nothing helps.
#
# HOW LONG: about 10-15 minutes. Needs data\original\ (from Round 5).
# ============================================================================

import time
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import TargetEncoder
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
RUN_LEAKY_DEMO = True     # False = skip the leakage demo (saves about 2 minutes)
PREVIOUS_BEST = 0.96090   # your best CV so far (Round 5 blend)
SMOOTHING = 20
ORIGINAL_DIR = DATA_DIR / "original"
MAX_BLEND_STEPS = 25      # the most models the hill-climbing blend may add

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

# The groups we target-encode: (new column name, columns that define the group)
TE_GROUPS = [
    ("te_distance", ["Flight Distance"]),
    ("te_age", ["Age"]),
    ("te_distance_class", ["Flight Distance", "Class"]),
    ("te_distance_travel", ["Flight Distance", "Type of Travel"]),
    ("te_distance_customer", ["Flight Distance", "Customer Type"]),
    ("te_age_class", ["Age", "Class"]),
]

# Saved models from earlier rounds that the blend may use: name -> where its
# out-of-fold predictions and test predictions are saved
EARLIER_MODELS = {
    "v4_lightgbm": ("oof_v4_lightgbm.csv", "test_pred_v4_lightgbm.csv"),
    "v4_xgboost": ("oof_v4_xgboost.csv", "test_pred_v4_xgboost.csv"),
    "v4_catboost": ("oof_v4_catboost.csv", "test_pred_v4_catboost.csv"),
    "v5b_plus_teacher": ("oof_v5b_plus_teacher.csv", "submission_v5b_plus_teacher.csv"),
    "v5_xgboost": ("oof_v5_xgboost.csv", "submission_v5_xgboost.csv"),
}


# ============================================================================
# PART A: FEATURES FROM ROUNDS 3 AND 5 (copied, unchanged)
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
    return train, test, features, rating_cols


def load_original(competition_features):
    files = sorted(ORIGINAL_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"\nI can't find any CSV files in {ORIGINAL_DIR}\n"
                         "This round needs the original dataset from Round 5 (see ROUND5_GUIDE.md).")
    original = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    original[TARGET] = target_to_number(original[TARGET])
    keep = [c for c in competition_features if c in original.columns]
    print(f"Original data: {len(original):,} passengers")
    return original[keep + [TARGET]]


def add_original_route_rates(train, test, original):
    overall = original[TARGET].mean()

    def smoothed_rate(group_cols):
        stats = original.groupby(group_cols)[TARGET].agg(["sum", "count"])
        return (stats["sum"] + SMOOTHING * overall) / (stats["count"] + SMOOTHING), stats["count"]

    route_rate, route_count = smoothed_rate([DISTANCE])
    for table in (train, test):
        table["orig_route_rate"] = table[DISTANCE].map(route_rate).fillna(overall)
        table["orig_route_count"] = table[DISTANCE].map(route_count).fillna(0)
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
    for col in columns:
        values = sorted(pd.concat([t[col].astype(str).where(t[col].notna()) for t in tables])
                        .dropna().unique())
        kind = pd.CategoricalDtype(categories=values)
        for t in tables:
            t[col] = t[col].astype(str).where(t[col].notna()).astype(kind)


def add_teacher(train, test, original, raw_features, categorical_raw):
    orig = original.copy()
    tr = train[raw_features].copy()
    te = test[raw_features].copy()
    same_categories([orig, tr, te], [c for c in categorical_raw if c in orig.columns])
    cols = [c for c in raw_features if c in orig.columns]
    fit_rows, stop_rows = train_test_split(orig, test_size=0.1, stratify=orig[TARGET],
                                           random_state=RANDOM_SEED)
    teacher = lgb.LGBMClassifier(**LGBM_SETTINGS)
    teacher.fit(fit_rows[cols], fit_rows[TARGET],
                eval_X=stop_rows[cols], eval_y=stop_rows[TARGET], eval_metric="auc",
                callbacks=[lgb.early_stopping(100, verbose=False)])
    train["orig_teacher"] = teacher.predict_proba(tr[cols])[:, 1]
    test["orig_teacher"] = teacher.predict_proba(te[cols])[:, 1]
    return ["orig_teacher"]


# ============================================================================
# PART B: TARGET ENCODING
# ============================================================================

def make_group_keys(raw_table):
    """
    One text column per group, e.g. "1235|Business" for distance 1235 in
    Business class. TargetEncoder treats each different text as one group.
    """
    keys = pd.DataFrame(index=range(len(raw_table)))
    for new_name, cols in TE_GROUPS:
        keys[new_name] = raw_table[cols].astype(str).agg("|".join, axis=1).values
    return keys


def honest_encodings(keys, keys_test, y, folds):
    """
    The SAFE way. For each outer fold, returns the encodings for its training
    rows, its held-out rows and the test set, never using a held-out answer.
    """
    names = [n for n, _ in TE_GROUPS]
    per_fold = []
    for tr_rows, va_rows in folds:
        encoder = TargetEncoder(
            target_type="binary",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED))
        # fit_transform = learn from the 4 training folds, with an INNER split
        # so no training passenger's own answer builds their own value
        tr_enc = encoder.fit_transform(keys.iloc[tr_rows], y[tr_rows])
        # transform = apply what was learned to rows it did NOT learn from
        va_enc = encoder.transform(keys.iloc[va_rows])
        te_enc = encoder.transform(keys_test)
        per_fold.append((pd.DataFrame(tr_enc, columns=names),
                         pd.DataFrame(va_enc, columns=names),
                         pd.DataFrame(te_enc, columns=names)))
    return per_fold


def leaky_encodings(keys, keys_test, y, folds):
    """
    The WRONG way, for the demo only. Learn from ALL training answers at once,
    then hand those values to every fold. Held-out passengers' own answers
    went into their own encodings.
    """
    names = [n for n, _ in TE_GROUPS]
    encoder = TargetEncoder(target_type="binary")
    encoder.fit(keys, y)                     # learns from every answer...
    all_enc = pd.DataFrame(encoder.transform(keys), columns=names)   # ...then reuses them
    te_enc = pd.DataFrame(encoder.transform(keys_test), columns=names)
    return [(all_enc.iloc[tr].reset_index(drop=True), all_enc.iloc[va].reset_index(drop=True), te_enc)
            for tr, va in folds]


# ============================================================================
# PART C: CROSS-VALIDATION WITH PER-FOLD ENCODINGS
# ============================================================================

def run_cv(model_name, train, test, features, folds, encodings=None):
    X, y, X_test = train[features], train[TARGET], test[features]
    oof, test_pred, scores = np.zeros(len(train)), np.zeros(len(test)), []

    for fold_number, (tr_rows, va_rows) in enumerate(folds, start=1):
        X_tr = X.iloc[tr_rows].reset_index(drop=True)
        X_va = X.iloc[va_rows].reset_index(drop=True)
        X_te = X_test.reset_index(drop=True)
        if encodings is not None:
            tr_enc, va_enc, te_enc = encodings[fold_number - 1]
            # Glue the encoding columns onto the right of the other features
            X_tr = pd.concat([X_tr, tr_enc], axis=1)
            X_va = pd.concat([X_va, va_enc], axis=1)
            X_te = pd.concat([X_te, te_enc], axis=1)
        y_tr, y_va = y.iloc[tr_rows].values, y.iloc[va_rows].values

        if model_name == "lightgbm":
            model = lgb.LGBMClassifier(**LGBM_SETTINGS)
            model.fit(X_tr, y_tr, eval_X=X_va, eval_y=y_va, eval_metric="auc",
                      callbacks=[lgb.early_stopping(100, verbose=False)])
            n_trees = model.best_iteration_
        else:
            model = xgb.XGBClassifier(**XGB_SETTINGS)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            n_trees = model.best_iteration + 1
        oof[va_rows] = model.predict_proba(X_va)[:, 1]
        test_pred += model.predict_proba(X_te)[:, 1] / N_FOLDS
        scores.append(roc_auc_score(y_va, oof[va_rows]))
        print(f"   fold {fold_number}: AUC {scores[-1]:.5f}  ({n_trees} trees)")

    cv = roc_auc_score(y, oof)
    print(f"   CV AUC {cv:.5f}")
    return cv, np.std(scores), oof, test_pred


# ============================================================================
# PART D: HILL-CLIMBING BLEND
# ============================================================================

def to_ranks(values):
    return pd.Series(values).rank(pct=True).values


def load_earlier_models(train, test):
    """Load saved predictions from earlier rounds, if they match this data."""
    found = {}
    for name, (oof_file, test_file) in EARLIER_MODELS.items():
        oof_path = OUTPUT_DIR / oof_file
        test_path = (OUTPUT_DIR / test_file if test_file.startswith("test_pred")
                     else SUBMISSION_DIR / test_file)
        if not (oof_path.exists() and test_path.exists()):
            print(f"   {name}: not found, skipping")
            continue
        oof_table, test_table = pd.read_csv(oof_path), pd.read_csv(test_path)
        if (oof_table[ID_COL].tolist() != train[ID_COL].tolist()
                or test_table[ID_COL].tolist() != test[ID_COL].tolist()):
            print(f"   {name}: saved file doesn't match this data, skipping")
            continue
        test_values = test_table["pred"] if "pred" in test_table else test_table[TARGET]
        found[name] = (oof_table["oof"].values, test_values.values)
        print(f"   {name}: loaded (CV AUC {roc_auc_score(train[TARGET], found[name][0]):.5f})")
    return found


def hill_climb(models, y):
    """
    Start with the best single model. Each step, try adding every model
    (repeats allowed) and keep the one that raises the blend's CV the most.
    Stop when no model helps. Returns how many times each model was picked.
    """
    ranked = {name: to_ranks(oof) for name, (oof, _) in models.items()}
    best_start = max(ranked, key=lambda n: roc_auc_score(y, ranked[n]))
    picks = [best_start]
    total = ranked[best_start].copy()
    best_score = roc_auc_score(y, total)
    print(f"   start: {best_start} ({best_score:.5f})")

    for step in range(1, MAX_BLEND_STEPS + 1):
        trial = {name: roc_auc_score(y, (total + r) / (len(picks) + 1))
                 for name, r in ranked.items()}
        name = max(trial, key=trial.get)
        if trial[name] <= best_score + 1e-6:
            print("   no model improves the blend any more; stopping")
            break
        picks.append(name)
        total += ranked[name]
        best_score = trial[name]
        print(f"   step {step}: add {name:<22} blend CV {best_score:.5f}")

    counts = pd.Series(picks).value_counts()
    weights = (counts / counts.sum()).to_dict()
    return weights, best_score


# ============================================================================
# PART E: RUN EVERYTHING
# ============================================================================

start_time = time.time()
raw_train, raw_test = load_data()
if QUICK_TEST:
    raw_train = raw_train.sample(50_000, random_state=RANDOM_SEED).reset_index(drop=True)
    print("QUICK_TEST is on: using 50,000 rows only. Scores will be lower.")

raw_features, raw_categorical, _ = find_feature_columns(raw_train)
original = load_original(raw_features)

# Group keys are made from the RAW values (before the category conversion)
keys = make_group_keys(raw_train)
keys_test = make_group_keys(raw_test)

train, test, v3c_features, rating_cols = build_v3c_features(raw_train, raw_test)
rate_cols = add_original_route_rates(train, test, original)
teacher_cols = add_teacher(train, test, original, raw_features, raw_categorical + rating_cols)
v5b_features = v3c_features + rate_cols + teacher_cols
y = train[TARGET].values

splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
folds = list(splitter.split(train[v5b_features], y))   # the same folds as every round

print("\nBuilding the target encodings inside each fold (the safe way)...")
safe_enc = honest_encodings(keys, keys_test, y, folds)

OUTPUT_DIR.mkdir(exist_ok=True)
SUBMISSION_DIR.mkdir(exist_ok=True)
results = {}

print(f"\n=== v6a_target_encoding (LightGBM, {len(v5b_features) + len(TE_GROUPS)} features) ===")
results["v6a_target_encoding"] = run_cv("lightgbm", train, test, v5b_features, folds, safe_enc)

print(f"\n=== v6_xgboost (XGBoost, same features) ===")
results["v6_xgboost"] = run_cv("xgboost", train, test, v5b_features, folds, safe_enc)

leaky_cv = None
if RUN_LEAKY_DEMO:
    print("\n=== LEAKAGE DEMO: the same LightGBM, but encodings learned the WRONG way ===")
    leaky = run_cv("lightgbm", train, test, v5b_features, folds,
                   leaky_encodings(keys, keys_test, y, folds))
    leaky_cv = leaky[0]

# Save this round's models
for name, (cv, spread, oof, test_pred) in results.items():
    pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_pred}).to_csv(
        SUBMISSION_DIR / f"submission_{name}.csv", index=False)
    if not QUICK_TEST:
        pd.DataFrame({ID_COL: train[ID_COL], "oof": oof}).to_csv(
            OUTPUT_DIR / f"oof_{name}.csv", index=False)

# ----------------------------------------------------------------------------
# Hill-climbing blend over every model we have
# ----------------------------------------------------------------------------
print("\nLoading saved models from earlier rounds...")
blend_models = {} if QUICK_TEST else load_earlier_models(train, test)
if QUICK_TEST:
    print("   (quick test: earlier models are for the full data, so they're skipped)")
for name, (cv, spread, oof, test_pred) in results.items():
    blend_models[name] = (oof, test_pred)

print("\nHill-climbing blend:")
weights, blend_cv = hill_climb(blend_models, y)
blend_test = sum(w * to_ranks(blend_models[n][1]) for n, w in weights.items())
blend_oof = sum(w * to_ranks(blend_models[n][0]) for n, w in weights.items())
weight_text = ", ".join(f"{n} {w:.2f}" for n, w in sorted(weights.items(), key=lambda kv: -kv[1]))
results["v6_hill_climb_blend"] = (roc_auc_score(y, blend_oof), 0, blend_oof, blend_test)
pd.DataFrame({ID_COL: test[ID_COL], TARGET: blend_test}).to_csv(
    SUBMISSION_DIR / "submission_v6_hill_climb_blend.csv", index=False)

# Log everything
log_path = OUTPUT_DIR / "experiment_log.csv"
notes = {
    "v6a_target_encoding": "v5b features + fold-safe target encoding (LightGBM)",
    "v6_xgboost": "v5b features + fold-safe target encoding (XGBoost)",
    "v6_hill_climb_blend": "hill-climbing rank blend: " + weight_text,
}
for name, (cv, spread, _, _) in results.items():
    pd.DataFrame([{
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "experiment": name,
        "rows_used": len(train), "n_features": len(v5b_features) + len(TE_GROUPS),
        "cv_auc": round(cv, 5), "fold_std": round(spread, 5) if spread else "",
        "public_lb": "", "notes": notes[name],
    }]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
print("\n" + "=" * 72)
print("ROUND 6 SUMMARY")
print("=" * 72)
print(f"{'experiment':<26}{'CV AUC':>10}{'vs Round 5 blend (0.96090)':>30}")
print(f"{'v5b LightGBM (reference)':<26}{0.96046:>10.5f}{0.96046 - PREVIOUS_BEST:>+30.5f}")
for name, (cv, *_rest) in results.items():
    print(f"{name:<26}{cv:>10.5f}{cv - PREVIOUS_BEST:>+30.5f}")
print(f"\nBlend weights: {weight_text}")

if leaky_cv is not None:
    honest_cv = results["v6a_target_encoding"][0]
    print("\nLEAKAGE DEMO (same model, same features, only the encoding method differs):")
    print(f"   honest encodings  CV AUC {honest_cv:.5f}")
    print(f"   leaky encodings   CV AUC {leaky_cv:.5f}   ({leaky_cv - honest_cv:+.5f})")
    print("   The leaky score is NOT real. Never upload it; it would score worse on Kaggle.")

best_name = max(results, key=lambda n: results[n][0])
best_cv = results[best_name][0]
if best_cv > PREVIOUS_BEST:
    print(f"\nRECOMMENDED UPLOAD: submissions/submission_{best_name}.csv  (CV AUC {best_cv:.5f})")
else:
    print(f"\nNothing beat your Round 5 blend ({PREVIOUS_BEST}). Don't upload; your Round 5 file is still your best.")
print(f"Finished in {(time.time() - start_time) / 60:.1f} minutes.")
