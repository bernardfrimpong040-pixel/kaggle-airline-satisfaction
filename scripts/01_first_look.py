# ============================================================================
# STEP 1a: FIRST LOOK AT THE DATA
# Run with:   python scripts/01_first_look.py
#
# Before building any model, a good data scientist looks at the data:
#   - How big is it?
#   - What columns are there?
#   - Is anything missing?
#   - How many passengers were satisfied?
#   - Which columns seem to matter most?
# Everything printed here is also saved to outputs/01_first_look.txt
# so you can copy numbers into your README later.
# ============================================================================

from sklearn.metrics import roc_auc_score   # the competition's scoring method
from common import (load_data, find_feature_columns, OUTPUT_DIR, TARGET)

# ----------------------------------------------------------------------------
# Small trick: print to the screen AND save to a text file at the same time
# ----------------------------------------------------------------------------
OUTPUT_DIR.mkdir(exist_ok=True)                 # make the outputs folder if needed
report_file = open(OUTPUT_DIR / "01_first_look.txt", "w", encoding="utf-8")

def show(text=""):
    print(text)                    # to the screen
    report_file.write(text + "\n") # to the file


# ----------------------------------------------------------------------------
# 1. Load the data
# ----------------------------------------------------------------------------
train, test = load_data()
features, categorical, numeric = find_feature_columns(train)

show("=" * 70)
show("1. SIZE OF THE DATA")
show("=" * 70)
show(f"Training rows (passengers WITH the answer):    {len(train):,}")
show(f"Test rows     (passengers WITHOUT the answer): {len(test):,}")
show(f"Number of feature columns: {len(features)}")

# ----------------------------------------------------------------------------
# 2. The columns
# ----------------------------------------------------------------------------
show()
show("=" * 70)
show("2. THE COLUMNS")
show("=" * 70)
show(f"Word (categorical) columns ({len(categorical)}):")
for col in categorical:
    # .unique() lists the different values in the column
    values = ", ".join(str(v) for v in train[col].dropna().unique()[:6])
    show(f"   - {col}:  {values}")

show(f"Number columns ({len(numeric)}):")
for col in numeric:
    show(f"   - {col}:  smallest {train[col].min()}, largest {train[col].max()}")

# ----------------------------------------------------------------------------
# 3. Missing values
# ----------------------------------------------------------------------------
show()
show("=" * 70)
show("3. MISSING VALUES (empty cells)")
show("=" * 70)
missing = train[features].isna().sum()        # count empty cells per column
missing = missing[missing > 0]                 # keep only columns with some
if len(missing) == 0:
    show("No missing values in the training data.")
else:
    for col, count in missing.items():
        show(f"   - {col}: {count:,} missing ({count / len(train):.3%} of rows)")

# ----------------------------------------------------------------------------
# 4. The target: how many were satisfied?
# ----------------------------------------------------------------------------
show()
show("=" * 70)
show("4. THE TARGET (what we predict)")
show("=" * 70)
rate = train[TARGET].mean()   # the average of 1s and 0s = share of 1s
show(f"Satisfied passengers: {rate:.1%}")
show(f"Not satisfied:        {1 - rate:.1%}")
show("Both groups are large, so the data is reasonably balanced.")

# ----------------------------------------------------------------------------
# 5. Satisfaction rate inside each group of each word column
# ----------------------------------------------------------------------------
show()
show("=" * 70)
show("5. SATISFACTION RATE BY GROUP")
show("=" * 70)
for col in categorical:
    show(f"{col}:")
    # groupby = split passengers into groups, then take the satisfied share
    by_group = train.groupby(col, observed=True)[TARGET].agg(["mean", "size"])
    for group, row in by_group.iterrows():
        show(f"   {str(group):<28} {row['mean']:6.1%} satisfied   ({int(row['size']):,} passengers)")

# ----------------------------------------------------------------------------
# 6. How useful is each number column ON ITS OWN?
# ----------------------------------------------------------------------------
# We use the competition's own score, ROC AUC, on one column at a time.
#   AUC = 0.5  -> the column is useless (like flipping a coin)
#   AUC = 1.0  -> the column alone separates satisfied from not satisfied perfectly
# If a column works "backwards" (higher value = LESS satisfied), its AUC is
# below 0.5, so we report max(AUC, 1 - AUC) to measure strength either way.
show()
show("=" * 70)
show("6. STRENGTH OF EACH NUMBER COLUMN ON ITS OWN (single-column AUC)")
show("=" * 70)
strengths = []
for col in numeric:
    values = train[col].fillna(train[col].median())   # fill gaps so AUC can run
    auc = roc_auc_score(train[TARGET], values)
    strengths.append((col, max(auc, 1 - auc)))

# Sort from strongest to weakest
strengths.sort(key=lambda pair: pair[1], reverse=True)
for col, auc in strengths:
    bar = "#" * int((auc - 0.5) * 100)   # a little text bar chart
    show(f"   {col:<36} {auc:.3f}  {bar}")

show()
show("Saved this report to outputs/01_first_look.txt")
report_file.close()
