# Round 3 Guide: Route Features

**Goal:** Build on Round 2's one real gain (flight distance count, a stand-in for the route) by learning more about each route.
**Script:** `scripts/05_route_features.py`
**Result:** CV AUC **0.96026**, public score **0.95999**, rank **993** (up from 1,130)

---

## Part 1: The idea

In Round 2, only one new column clearly helped: **distance count**, how many passengers share
the exact same flight distance. Passengers with the same flight distance are usually on the
same **route** (for example, Cleveland → Chicago).

So Round 3 asks: **what else can we learn about each route?**

We also **dropped** the Round 2 groups that didn't help (rating summaries, delay features,
segment), because columns that carry no new information just add noise.

## Part 2: The 4 experiments

Each experiment adds one more group to the one before:

| Experiment | Feature group added | What it is | Example |
|---|---|---|---|
| v3a | Lean base | Original 21 columns + distance count + ratings as categories | 22 columns instead of 35 |
| v3b | Pair counts | Passengers sharing the same distance **and** the same class / trip type / customer type | "120 passengers flew this distance in Business class" |
| v3c | Route profile | For each route: share in Business class, share travelling for business, share of loyal customers, average age, and average of the 4 strongest ratings | "On this route, 48% fly Business class and online boarding averages 2.8" |
| v3d | Route residuals | Each passenger's rating minus the route's average for that rating | "This passenger rated online boarding 2.2 points above usual for the route" |

**None of these features use the satisfaction answers**, so none can leak the target. That's
why it's safe to compute them on train and test together.

Everything else stayed the same as before: the same 5 folds, the same seed (42), and the same model settings.

## Part 3: How to run it

1. Put `05_route_features.py` in the `scripts` folder.
2. In the VS Code terminal (with `(.venv)` showing):
   ```powershell
   python scripts/05_route_features.py
   ```
   It takes about 5–10 minutes per experiment, so 20–40 minutes in total.

   > To test it quickly first, open the script, change `QUICK_TEST = False` to `QUICK_TEST = True`,
   > run it (about 1 minute), then change it back.

3. At the end it prints a **ROUND 3 SUMMARY** table. The last column, **"vs row above"**,
   shows directly what each group added.
4. The script tells you which file to upload, **only if** it beats your previous best (v2e, 0.95923).
5. Check the file, then upload it:
   ```powershell
   python scripts/03_check_submission.py submissions/submission_v3c_plus_route_profile.csv
   ```
   - **File to upload:** `submissions\submission_v3c_plus_route_profile.csv`
   - **Description on Kaggle:** `v3c: lean base + pair counts + route profile, CV AUC 0.96026`
6. Type the public score into `outputs\experiment_log.csv` (v3c row, `public_lb` column).

## Part 4: The results

```
experiment                   features    CV AUC   ± folds  vs row above
v2e (previous best)                35   0.95923
v3a_lean_base                      22   0.95947   0.00068      +0.00024
v3b_plus_pair_counts               25   0.95961   0.00061      +0.00014
v3c_plus_route_profile             33   0.96026   0.00060      +0.00065
v3d_plus_route_residuals           37   0.96014   0.00054      -0.00012
```

| Added | Change | Verdict |
|---|---:|---|
| Lean base (13 noisy columns removed) | 0.95947 − 0.95923 = **+0.00024** | Fewer columns, better score |
| Pair counts | 0.95961 − 0.95947 = +0.00014 | Small gain |
| **Route profile** | 0.96026 − 0.95961 = **+0.00065** | **Biggest gain of the round** |
| Route residuals | 0.96014 − 0.96026 = −0.00012 | No help |

**Kaggle confirmed it:**

| | v2e (Round 2) | v3c (Round 3) | Gain |
|---|---:|---:|---:|
| CV AUC | 0.95923 | 0.96026 | 0.96026 − 0.95923 = +0.00103 |
| Public score | 0.95882 | 0.95999 | 0.95999 − 0.95882 = +0.00117 |
| Rank | 1,130 | 993 | up 1,130 − 993 = 137 places |

CV predicted a gain of +0.00103 and Kaggle gave +0.00117: close agreement again. The gap between
CV and public score was 0.96026 − 0.95999 = 0.00027, in line with earlier rounds (0.00042 and 0.00041).

## Part 5: What we learned

1. **Removing useless columns can improve a model.** Dropping 13 columns from Round 2 *raised*
   the score. Extra columns that carry no new information give the trees more ways to fit
   random noise.
2. **The route profile was the real win.** The model already knew about each passenger. Now it
   also knows what kind of passengers usually fly that route. A business traveller on a mostly-holiday
   route may have a different experience from one on a busy business route.
3. **Differences the model can compute itself don't help.** Route residuals are just
   "rating − route average", and the model already had both numbers. It's the same lesson as
   Round 2's rating summaries: give the model **new information**, not rearrangements of what it has.
4. **Our validation is still trustworthy.** Three rounds in a row, CV has predicted the Kaggle result closely.

**Progress so far:**

| Round | CV AUC | Public score | Rank |
|---|---:|---:|---:|
| 1. Baseline | 0.95878 | 0.95836 | 1,361 |
| 2. Feature engineering | 0.95923 | 0.95882 | 1,130 |
| 3. Route features | 0.96026 | 0.95999 | 993 |
| **Total gain** | **+0.00148** | **+0.00163** | **up 368 places** |

---

## New words in this round

| Word | Plain meaning |
|---|---|
| **Lean feature set** | Only the columns that earned their place; the unhelpful ones are removed |
| **Route profile** | A description of a route built from all its passengers, e.g. "48% Business class" |
| **Residual** | A value minus its expected (average) value, e.g. "2 points above the route average" |
| **Aggregate / group statistic** | A number computed over a group, like the mean age of everyone on a route |
| **Share** | The fraction of a group with some property; the mean of a True/False column |
