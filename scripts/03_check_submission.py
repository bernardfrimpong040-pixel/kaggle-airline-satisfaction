# ============================================================================
# STEP 1c: CHECK THE SUBMISSION FILE BEFORE UPLOADING
# Run with:   python scripts/03_check_submission.py
#   or, to check one particular file:
#             python scripts/03_check_submission.py submissions/submission_baseline_lgbm.csv
#
# Kaggle allows only a few uploads per day, so we never waste one on a broken
# file. This script runs 7 checks. Each one prints PASS or FAIL.
#
# This is also a small preview of the job you are applying for: writing
# automatic checks that catch mistakes before anyone else sees them.
# ============================================================================

import sys
import pandas as pd
from common import DATA_DIR, SUBMISSION_DIR, TARGET, ID_COL

# ----------------------------------------------------------------------------
# Which file to check?
# ----------------------------------------------------------------------------
if len(sys.argv) > 1:
    # You typed a file name after the script name
    from pathlib import Path
    file_to_check = Path(sys.argv[1])
else:
    # Otherwise check the newest .csv in the submissions folder
    csv_files = sorted(SUBMISSION_DIR.glob("*.csv"), key=lambda f: f.stat().st_mtime)
    if not csv_files:
        raise SystemExit("No submission files found. Run 02_baseline_model.py first.")
    file_to_check = csv_files[-1]

print(f"Checking: {file_to_check}\n")

submission = pd.read_csv(file_to_check)
test = pd.read_csv(DATA_DIR / "test.csv", usecols=[ID_COL])   # we only need the ids

# ----------------------------------------------------------------------------
# The checks. Each is (description, True/False result).
# ----------------------------------------------------------------------------
checks = []

# 1. Exactly the two columns Kaggle expects, in the right order
checks.append(("Columns are exactly: id, satisfaction",
               list(submission.columns) == [ID_COL, TARGET]))

# 2. One row per test passenger
checks.append((f"Row count matches test.csv ({len(test):,} rows)",
               len(submission) == len(test)))

# 3. No passenger appears twice
checks.append(("No duplicate ids",
               not submission[ID_COL].duplicated().any()))

# 4. The ids are exactly the test ids (none missing, none extra)
checks.append(("Ids are exactly the test-set ids",
               set(submission[ID_COL]) == set(test[ID_COL])))

# 5. No empty predictions
has_target = TARGET in submission.columns
checks.append(("No missing predictions",
               has_target and not submission[TARGET].isna().any()))

# 6. Probabilities must be between 0 and 1
checks.append(("All predictions are between 0 and 1",
               has_target and submission[TARGET].between(0, 1).all()))

# 7. The model isn't predicting the same number for everyone.
#    A constant prediction scores AUC 0.5 (useless), and usually means a bug.
checks.append(("Predictions vary (not all the same value)",
               has_target and submission[TARGET].nunique() > 1))

# ----------------------------------------------------------------------------
# Print the results
# ----------------------------------------------------------------------------
for description, passed in checks:
    print(f"   [{'PASS' if passed else 'FAIL'}]  {description}")

failures = sum(1 for _, passed in checks if not passed)
print()
if failures == 0:
    print("All 7 checks passed. This file is safe to upload to Kaggle.")
else:
    print(f"{failures} check(s) FAILED. Do not upload this file yet.")
    sys.exit(1)   # a non-zero exit code tells other programs "something went wrong"
