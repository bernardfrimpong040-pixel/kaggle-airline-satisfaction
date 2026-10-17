# Predicting Airline Passenger Satisfaction (Kaggle Playground S6E10)

- **Author:** Bernard Frimpong ([GitHub](https://github.com/bernardfrimpong040-pixel) · [LinkedIn](https://www.linkedin.com/in/bernard-frimpong-681008222))
- **Competition:** [Kaggle Playground Series, Season 6 Episode 10](https://www.kaggle.com/competitions/playground-series-s6e10), October 2026
- **Task:** Binary classification. Predict the probability that a passenger is satisfied.
- **Metric:** ROC AUC
- **Tools:** Python, pandas, scikit-learn, LightGBM

## Results

| Version | What changed | CV AUC | Public LB | Rank |
|---|---|---:|---:|---:|
| v1 | Baseline LightGBM, raw features | _fill in_ | 0.95836 | _fill in_ |

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
│   └── 03_check_submission.py   # 7 checks before uploading
├── submissions/                 # files uploaded to Kaggle (not on GitHub)
├── requirements.txt             # pinned library versions
└── SETUP_STEP1.md               # step-by-step setup guide
```

## Reproducing the results

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# put train.csv and test.csv from Kaggle into data/
python scripts/01_first_look.py
python scripts/02_baseline_model.py
python scripts/03_check_submission.py
```

All randomness is seeded (`RANDOM_SEED = 42`) and library versions are pinned in
`requirements.txt`, so the same code gives the same CV score.

The competition data is not included, as Kaggle's rules require. Download it from the competition page.
