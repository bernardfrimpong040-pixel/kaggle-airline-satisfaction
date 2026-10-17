# ============================================================================
# STEP 1b: OUR FIRST MODEL (the "baseline")
# Run with:   python scripts/02_baseline_model.py
#
# WHAT THIS SCRIPT DOES
#   1. Splits the training passengers into 5 equal groups ("folds").
#   2. Five times: trains a model on 4 groups and tests it on the 5th group,
#      which the model has never seen. This is called CROSS-VALIDATION (CV).
#   3. Reports the average score. This tells us how well the model will do on
#      NEW passengers, so we can trust it BEFORE uploading to Kaggle.
#   4. Uses the 5 models to predict the test passengers and writes the
#      submission file for Kaggle.
#
# WHY NOT JUST TRAIN AND SCORE ON THE SAME DATA?
#   That is like letting a student see the exam answers, then grading them on
#   the same exam. The score looks great but means nothing. Scoring each fold
#   on passengers the model never saw is the honest way.
#
# THE MODEL: LightGBM
#   It builds hundreds of small decision trees one after another. Each new
#   tree tries to fix the mistakes of the trees before it ("gradient boosting").
#   It is the most common winning tool for table data on Kaggle.
#
# HOW LONG: about 3-10 minutes on a normal laptop for 700,000 rows.
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
# SETTINGS (change these, not the code below)
# ----------------------------------------------------------------------------
EXPERIMENT_NAME = "baseline_lgbm"   # a short name for this run, used in file names
N_FOLDS = 5                         # how many groups to split the data into
RANDOM_SEED = 42                    # fixed seed = same result every time you run it
QUICK_TEST = False                  # True = use only 50,000 rows to test the script fast

# LightGBM's settings ("hyperparameters"). These are sensible starting values.
MODEL_SETTINGS = {
    "n_estimators": 3000,       # the most trees it is allowed to build
    "learning_rate": 0.05,      # how big a step each tree takes (smaller = slower but careful)
    "num_leaves": 63,           # how detailed each tree can be
    "subsample": 0.8,           # each tree sees a random 80% of rows...
    "subsample_freq": 1,        # ...re-drawn for every tree
    "colsample_bytree": 0.8,    # ...and a random 80% of columns
    "random_state": RANDOM_SEED,
    "n_jobs": -1,               # use all the computer's processor cores
    "verbose": -1,              # keep LightGBM quiet
}

# ----------------------------------------------------------------------------
# 1. Load and prepare the data
# ----------------------------------------------------------------------------
start_time = time.time()
train, test = load_data()

if QUICK_TEST:
    train = train.sample(50_000, random_state=RANDOM_SEED)
    print("QUICK_TEST is on: using 50,000 rows only. Scores will be a bit lower.")

features, categorical, numeric = find_feature_columns(train)
train, test = make_categories_match(train, test, categorical)

X = train[features]          # X = the information we use to predict
y = train[TARGET]            # y = the answer (1 satisfied, 0 not)
X_test = test[features]

print(f"Using {len(features)} features on {len(train):,} passengers.")

# ----------------------------------------------------------------------------
# 2. Cross-validation
# ----------------------------------------------------------------------------
# "Stratified" means every fold gets the same share of satisfied passengers
# (about 44%), so each fold is a fair mini-version of the whole data.
folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

# oof = "out-of-fold". For every training passenger we store the prediction
# made by the one model that did NOT see that passenger.
oof_predictions = np.zeros(len(train))
test_predictions = np.zeros(len(test))
fold_scores = []

for fold_number, (train_rows, valid_rows) in enumerate(folds.split(X, y), start=1):
    print(f"\n--- Fold {fold_number} of {N_FOLDS} ---")

    # Pick out the rows for this fold
    X_train, y_train = X.iloc[train_rows], y.iloc[train_rows]
    X_valid, y_valid = X.iloc[valid_rows], y.iloc[valid_rows]

    model = lgb.LGBMClassifier(**MODEL_SETTINGS)

    # EARLY STOPPING: after each new tree, check the score on the held-out
    # fold. If it hasn't improved for 100 trees, stop. This prevents the model
    # from memorising the training rows ("overfitting").
    model.fit(
        X_train, y_train,
        eval_X=X_valid, eval_y=y_valid,   # the held-out fold to watch
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    # predict_proba gives two columns: [chance of 0, chance of 1].
    # [:, 1] keeps the chance of 1 = chance the passenger is satisfied.
    valid_pred = model.predict_proba(X_valid)[:, 1]
    oof_predictions[valid_rows] = valid_pred

    score = roc_auc_score(y_valid, valid_pred)
    fold_scores.append(score)
    print(f"Trees used: {model.best_iteration_}   AUC on unseen fold: {score:.5f}")

    # Each fold's model also predicts the test set. We average the 5 later.
    test_predictions += model.predict_proba(X_test)[:, 1] / N_FOLDS

# ----------------------------------------------------------------------------
# 3. The overall score
# ----------------------------------------------------------------------------
cv_score = roc_auc_score(y, oof_predictions)   # one AUC over ALL passengers
spread = np.std(fold_scores)                    # how much the folds disagree

print("\n" + "=" * 60)
print(f"CROSS-VALIDATION AUC: {cv_score:.5f}   (folds differ by about ±{spread:.5f})")
print("=" * 60)
print("This is our honest estimate of the Kaggle score.")

# ----------------------------------------------------------------------------
# 4. Which features mattered most? (from the last fold's model)
# ----------------------------------------------------------------------------
# "gain" = how much each feature improved the trees in total
importance = pd.Series(model.booster_.feature_importance("gain"), index=features)
importance = (importance / importance.sum()).sort_values(ascending=False)
print("\nTop 10 features (share of total importance):")
for name, share in importance.head(10).items():
    print(f"   {name:<36} {share:6.1%}")

# ----------------------------------------------------------------------------
# 5. Save everything
# ----------------------------------------------------------------------------
OUTPUT_DIR.mkdir(exist_ok=True)
SUBMISSION_DIR.mkdir(exist_ok=True)

# 5a. The Kaggle submission: one row per test passenger, columns id and satisfaction
submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_predictions})
submission_path = SUBMISSION_DIR / f"submission_{EXPERIMENT_NAME}.csv"
submission.to_csv(submission_path, index=False)

# 5b. The out-of-fold predictions (we'll use them later for ensembling)
pd.DataFrame({ID_COL: train[ID_COL], "oof": oof_predictions}).to_csv(
    OUTPUT_DIR / f"oof_{EXPERIMENT_NAME}.csv", index=False)

# 5c. Feature importance table
importance.rename("share_of_gain").to_csv(OUTPUT_DIR / f"importance_{EXPERIMENT_NAME}.csv")

# 5d. Add one line to the EXPERIMENT LOG. Every run gets recorded, so you can
#     show exactly how your score improved over time. Kaggle grandmasters do this.
log_path = OUTPUT_DIR / "experiment_log.csv"
log_row = pd.DataFrame([{
    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "experiment": EXPERIMENT_NAME,
    "rows_used": len(train),
    "n_features": len(features),
    "cv_auc": round(cv_score, 5),
    "fold_std": round(spread, 5),
    "public_lb": "",          # you fill this in after uploading to Kaggle
    "notes": "LightGBM, raw features, native categories, early stopping",
}])
log_row.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)

minutes = (time.time() - start_time) / 60
print(f"\nSaved submission:  {submission_path}")
print(f"Logged this run in: {log_path}")
print(f"Finished in {minutes:.1f} minutes.")
print("\nNext: python scripts/03_check_submission.py")
