# Predicting Airline Passenger Satisfaction (Kaggle Playground S6E10)

- **Author:** Bernard Frimpong ([GitHub](https://github.com/bernardfrimpong040-pixel) · [LinkedIn](https://www.linkedin.com/in/bernard-frimpong-681008222))
- **Competition:** [Kaggle Playground Series, Season 6 Episode 10](https://www.kaggle.com/competitions/playground-series-s6e10), October 2026
- **Task:** Binary classification. Predict the probability that a passenger is satisfied.
- **Metric:** ROC AUC
- **Tools:** Python, pandas, scikit-learn, LightGBM

## Results

| Version | What changed | CV AUC | Public LB | Rank |
|---|---|---:|---:|---:|
| v1 | Baseline LightGBM, raw features | 0.95878 ± 0.00058 | 0.95836 | 1,361 (17 Oct 2026) |
| v2a | + rating summaries | 0.95861 ± 0.00053 | — | — |
| v2b | + delay features | 0.95846 ± 0.00056 | — | — |
| v2c | + passenger segment | 0.95856 ± 0.00053 | — | — |
| v2d | + flight distance count | 0.95916 ± 0.00059 | — | — |
| **v2e** | **+ ratings as categories** | **0.95923 ± 0.00061** | **0.95882** | **1,130 (17 Oct 2026)** |
| v3a | Lean base: raw + distance count + ratings as categories | 0.95947 ± 0.00068 | — | — |
| v3b | + pair counts | 0.95961 ± 0.00061 | — | — |
| **v3c** | **+ route profile** | **0.96026 ± 0.00060** | **0.95999** | **993 (17 Oct 2026)** |
| v3d | + route residuals | 0.96014 ± 0.00054 | — | — |

Only the best experiment of each round is uploaded to Kaggle; the others are judged on CV alone.

**Validation check:** the CV score has tracked the public leaderboard closely in every upload:

| Upload | CV AUC | Public LB | Gap |
|---|---:|---:|---:|
| v1 baseline | 0.95878 | 0.95836 | 0.00042 |
| v2e | 0.95923 | 0.95882 | 0.00041 |
| v3c | 0.96026 | 0.95999 | 0.00027 |
| **Total improvement** | **+0.00148** | **+0.00163** | |

CV predicted each round's gain closely (Round 2: +0.00045 CV vs +0.00046 public;
Round 3: +0.00103 CV vs +0.00117 public), so the 5-fold cross-validation is a trustworthy
guide. Model decisions are made on CV, saving Kaggle submissions for real improvements.

Public leaderboard ranks change daily as other teams submit, and the final ranking uses the
private leaderboard, revealed after the deadline (31 Oct 2026).

Every run is recorded automatically in `outputs/experiment_log.csv`.

## The data

| Split | Rows | Columns |
|---|---:|---|
| Train | 699,635 | 21 features + target |
| Test | 299,844 | 21 features |

- **4 categorical columns:** Gender, Customer Type, Type of Travel, Class
- **13 service ratings (0–5):** e.g. Online boarding, Seat comfort, Inflight entertainment, Cleanliness
- **4 numeric columns:** Age, Flight Distance, Departure Delay in Minutes, Arrival Delay in Minutes
- **Target:** 44.4% of passengers satisfied, 55.6% not satisfied

## Key findings

- **Trip purpose and class dominate.** Business travellers were 58.4% satisfied vs 9.7% for
  personal travel; Business class 72.5% vs 16.8% in Economy. Loyal customers were 49.5% satisfied
  vs 20.1% for disloyal customers. Gender barely matters (43.8% vs 44.9%).
- **Digital experience matters most among the ratings.** Online boarding alone gives AUC 0.840,
  followed by inflight entertainment (0.745) and seat comfort (0.732).
- **Delays are weak on their own** (departure delay AUC 0.509, close to random), suggesting
  passengers forgive delays when the service is good.
- **Data quality is high:** only 292 missing values (0.042%), all in Arrival Delay in Minutes.
  Baggage handling is the only rating that never takes the value 0.

## Round 2 findings

Each experiment adds one feature group to the one before, so the change from the previous row
shows what that group contributed:

| Feature group | Change in CV AUC | Verdict |
|---|---:|---|
| Rating summaries | −0.00017 | No help |
| Delay features | −0.00015 | No help |
| Passenger segment | +0.00010 | No clear effect |
| **Flight distance count** | **+0.00060** | **Real gain** |
| Ratings as categories | +0.00007 | Small gain |

- **Summaries and combinations of existing columns did not help.** Gradient-boosted trees can
  already build averages and interactions from the raw columns, so these features added noise
  rather than new information.
- **Flight distance count gave the only clear gain.** Counting how many passengers share each
  exact flight distance acts as a proxy for the route, which the raw columns do not capture.
- **Next round:** drop the groups that did not help and build more route features from
  flight distance.

## Round 3 findings

| Feature group | Change in CV AUC | Verdict |
|---|---:|---|
| Lean base (13 unhelpful Round 2 columns removed) | +0.00024 | Fewer columns, better score |
| Pair counts | +0.00014 | Small gain |
| **Route profile** | **+0.00065** | **Biggest gain of the round** |
| Route residuals | −0.00012 | No help |

- **Removing useless columns improved the model.** Columns with no new information give the
  trees more chances to fit noise.
- **Describing each route helped most.** Knowing what kind of passengers usually fly a route
  (share in Business class, share travelling for business, typical ratings) adds information
  the raw columns do not contain.
- **Differences the model can compute itself did not help.** Route residuals are rating minus
  route average, and the model already had both, the same lesson as Round 2's summaries.

## Approach

### 1. Look before modelling (`scripts/01_first_look.py`)
Size, column types, missing values, target balance, satisfaction rate by group,
and the strength of each numeric column on its own (single-column AUC).

### 2. Honest validation (`scripts/02_baseline_model.py`)
5-fold stratified cross-validation with a fixed seed. Every decision is based on
the out-of-fold AUC, not the public leaderboard, which uses only part of the
test set and can mislead.

### 3. Baseline model
LightGBM gradient boosting, using the categorical columns directly (stored as a shared
category type across train and test) and early stopping on each held-out fold to
prevent overfitting. Test predictions are the average of the 5 fold models.

### 4. Verified submissions (`scripts/03_check_submission.py`)
Seven automatic checks before any upload: column names, row count, duplicate ids,
exact id match with the test set, no missing values, probabilities in [0, 1], and
non-constant predictions.

### 5. Round 2: feature engineering (`scripts/04_feature_experiments.py`)
Five experiments, each adding one more group of new features, with folds, seed and model
settings kept identical to the baseline so any change in score comes from the features alone:

| Experiment | Feature group added | Idea |
|---|---|---|
| v2a | Rating summaries | Mean, min, max and spread of the 13 ratings (0 treated as "not answered"), counts of top, low and missing marks, and a digital-experience average |
| v2b | Delay features | Total delay, time made up in the air, late-over-15-minutes flag, delay per 1,000 miles |
| v2c | Passenger segment | Class × Type of Travel × Customer Type combined into one column |
| v2d | Distance count | How many passengers share each exact flight distance (a proxy for route), built without the target so it cannot leak |
| v2e | Ratings as categories | Each 0–5 rating treated as a category, so "0 = not answered" is not read as "one less than 1" |

### 6. Round 3: route features (`scripts/05_route_features.py`)
Round 2 showed that flight distance count, a proxy for the route, was the only feature group
with a clear gain. Round 3 drops the groups that did not help and builds more route features.
None of them use the target, so they cannot leak it:

| Experiment | Feature group added | Idea |
|---|---|---|
| v3a | Lean base | Original columns + distance count + ratings as categories |
| v3b | Pair counts | Passengers sharing the same distance *and* the same class, trip type or customer type |
| v3c | Route profile | Per route: share of Business class, business travel and loyal customers, mean age, and mean of the 4 strongest ratings |
| v3d | Route residuals | Each passenger's rating minus the route average, e.g. "online boarding 1.5 points below usual for this route" |

## Project structure

```
kaggle-airline-satisfaction/
├── data/                        # Kaggle files (not on GitHub, per competition rules)
├── outputs/
│   ├── 01_first_look.txt        # data exploration report
│   ├── experiment_log.csv       # every run: date, CV AUC, public score, notes
│   └── importance_*.csv         # feature importance per experiment
├── scripts/
│   ├── common.py                # shared loading and preparation helpers
│   ├── 01_first_look.py         # data exploration
│   ├── 02_baseline_model.py     # 5-fold CV LightGBM + submission file
│   ├── 03_check_submission.py   # 7 checks before uploading
│   ├── 04_feature_experiments.py # Round 2: 5 feature experiments
│   └── 05_route_features.py     # Round 3: 4 route-feature experiments
├── submissions/                 # files uploaded to Kaggle (not on GitHub)
├── requirements.txt             # pinned library versions
├── SETUP_STEP1.md               # Round 1 guide: setup, baseline, first upload
├── ROUND2_GUIDE.md              # Round 2 guide: feature experiments and results
└── ROUND3_GUIDE.md              # Round 3 guide: route features and results
```

## Step-by-step guides

Each round has its own guide with the steps, the results and what was learned:

| Round | Guide | Result |
|---|---|---|
| 1 | [SETUP_STEP1.md](SETUP_STEP1.md): setup, data exploration, baseline model | CV 0.95878, public 0.95836 |
| 2 | [ROUND2_GUIDE.md](ROUND2_GUIDE.md): feature engineering experiments | CV 0.95923, public 0.95882 |
| 3 | [ROUND3_GUIDE.md](ROUND3_GUIDE.md): route features | CV 0.96026, public 0.95999 |

## Reproducing the results

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# put train.csv and test.csv from Kaggle into data/
python scripts/01_first_look.py
python scripts/02_baseline_model.py
python scripts/03_check_submission.py
python scripts/04_feature_experiments.py
python scripts/05_route_features.py
```

All randomness is seeded (`RANDOM_SEED = 42`) and library versions are pinned in
`requirements.txt`, so the same code gives the same CV score.

The competition data is not included, as Kaggle's rules require. Download it from the competition page.
