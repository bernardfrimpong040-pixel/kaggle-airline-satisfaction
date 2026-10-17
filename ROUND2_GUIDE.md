# Round 2 Guide: Feature Engineering

**Goal:** Improve the baseline (CV AUC 0.95878, public score 0.95836) by giving the model new columns built from the existing ones.
**Script:** `scripts/04_feature_experiments.py`
**Result:** CV AUC **0.95923**, public score **0.95882**, rank **1,130** (up from 1,361)

---

## Part 1: The idea

A **feature** is a column the model uses to predict. **Feature engineering** means building
new columns from the existing ones. For example, the average of a passenger's 13 service
ratings is a new column made from 13 old ones.

The model could, in theory, work these out by itself. Sometimes handing them over directly
helps. Sometimes it doesn't. The only way to know is to test.

## Part 2: How we tested fairly

We ran **5 experiments**. Each one **adds one more feature group** to the one before:

| Experiment | Feature group added | What it is |
|---|---|---|
| v2a | Rating summaries | Average, worst, best and spread of the 13 ratings (0 = "not answered" is left out); counts of top marks (5), low marks (1–2) and unanswered (0); average of the 3 digital ratings |
| v2b | Delay features | Total delay; time made up in the air (departure minus arrival delay); late-over-15-minutes flag; delay per 1,000 miles |
| v2c | Passenger segment | Class + Type of Travel + Customer Type in one column, e.g. `Eco \| Personal Travel \| Loyal Customer` |
| v2d | Flight distance count | How many passengers (train and test together) share each exact flight distance |
| v2e | Ratings as categories | Each 0–5 rating treated as a label instead of a number |

**Everything else stayed exactly the same as the baseline:** the same 5 folds, the same seed (42),
and the same model settings. So any change in score was caused by the features alone.
Scientists call this **changing one thing at a time**.

## Part 3: How to run it

1. Put `04_feature_experiments.py` in the `scripts` folder.
2. In the VS Code terminal (with `(.venv)` showing):
   ```powershell
   python scripts/04_feature_experiments.py
   ```
   It takes about 5–10 minutes per experiment, so 30–50 minutes in total. You can leave it running.

   > To test it quickly first, open the script, change `QUICK_TEST = False` to `QUICK_TEST = True`,
   > run it (about 1 minute), then change it back.

3. At the end it prints a **ROUND 2 SUMMARY** table and the name of the best experiment.
4. Check the best file, then upload it to Kaggle:
   ```powershell
   python scripts/03_check_submission.py submissions/submission_v2e_plus_ratings_as_cat.csv
   ```
   Description on Kaggle: `v2e: distance count + ratings as categories, CV AUC 0.95923`
5. Type the public score into `outputs\experiment_log.csv` (v2e row, `public_lb` column).

## Part 4: The results

```
experiment                 features    CV AUC   ± folds  vs baseline
baseline (Round 1)               21   0.95878
v2a_rating_stats                 29   0.95861   0.00053     -0.00017
v2b_plus_delays                  33   0.95846   0.00056     -0.00032
v2c_plus_segment                 34   0.95856   0.00053     -0.00022
v2d_plus_distance_count          35   0.95916   0.00059     +0.00038
v2e_plus_ratings_as_cat          35   0.95923   0.00061     +0.00045
```

The **"vs baseline"** column compares each row with the baseline. To see what **one group** did,
compare each row with **the row above it**:

| Added | CV AUC | Change from row above | Verdict |
|---|---:|---:|---|
| Baseline | 0.95878 | | |
| Rating summaries | 0.95861 | 0.95861 − 0.95878 = −0.00017 | No help |
| Delay features | 0.95846 | 0.95846 − 0.95861 = −0.00015 | No help |
| Segment | 0.95856 | 0.95856 − 0.95846 = +0.00010 | No clear effect |
| **Distance count** | 0.95916 | 0.95916 − 0.95856 = **+0.00060** | **Real gain** |
| Ratings as categories | 0.95923 | 0.95923 − 0.95916 = +0.00007 | Small gain |

**Kaggle confirmed it:**

| | Baseline | v2e | Gain |
|---|---:|---:|---:|
| CV AUC | 0.95878 | 0.95923 | +0.00045 |
| Public score | 0.95836 | 0.95882 | +0.00046 |
| Rank | 1,361 | 1,130 | up 231 places |

CV predicted the gain (+0.00045) almost exactly (+0.00046 on Kaggle).

## Part 5: What we learned

1. **Summaries and combinations of existing columns didn't help.** LightGBM builds hundreds of
   trees that can already combine columns. Handing it an average it could build itself adds no new
   information, just noise.
2. **Flight distance count was the real win.** Passengers with the exact same flight distance are
   usually on the same route. Counting them tells the model how common each route is, which
   none of the original columns say directly.
3. **Distance count can't leak the answer.** It's built only from flight distances, never from the
   satisfaction column. That's why it's safe to count train and test together.
4. **Our validation is trustworthy.** CV and Kaggle moved by the same amount, so we can test ideas
   on the laptop and save Kaggle uploads for real improvements.

**Next (Round 3):** drop the groups that didn't help, and build more route features from flight distance.

---

## New words in this round

| Word | Plain meaning |
|---|---|
| **Feature engineering** | Building new columns from existing ones to help the model |
| **Feature group** | A set of related new columns tested together, e.g. all the delay features |
| **Cumulative experiments** | Each experiment keeps everything from the one before and adds one thing |
| **Noise** | Random ups and downs in a score that aren't caused by a real improvement |
| **Proxy** | A column that stands in for something we can't see directly. Distance count is a proxy for the route |
| **Count encoding** | Replacing a value with how often it appears in the data |
| **Data leakage** | When a feature secretly contains the answer, so the score looks great but fails on new data. Features built without the target, like distance count, can't leak it |
