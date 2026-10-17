# Round 5 Guide: Learning from the Original Dataset

**Goal:** Use the real dataset this competition was generated from to give the model new information.
**Script:** `scripts/07_original_data.py`
**Starting point:** Round 4 blend, CV AUC 0.96065, public score 0.96025, rank 945
**Result:** CV AUC **0.96090**, public score **0.96049**, rank **863** (up from 945)

---

## Part 1: The idea

Kaggle made this competition's 700,000 passengers with a computer program that learned from a
**real, smaller dataset**: *Airline Passenger Satisfaction* (about 130,000 real survey answers),
which is public on Kaggle.

Other entrants found that **adding the original rows straight into training made results worse**:
the real passengers and the generated ones are slightly different. So this round uses the original
data in two safer ways:

| Feature | What it is | Example |
|---|---|---|
| **Original route rates** | For each flight distance, the share of **real** passengers who were satisfied; also per distance **and** class; and how many real passengers flew that distance | "On this route, 61% of real passengers were satisfied" |
| **Teacher prediction** | A LightGBM trained **only** on the real passengers predicts every competition passenger. Its prediction becomes a new column | "The teacher thinks this passenger is 87% likely to be satisfied" |

**Neither uses the competition's answers**, only the original dataset's, so neither can leak
the competition target.

**Smoothing:** a route with only 2 real passengers, both satisfied, would get a 100% rate, which
is far too confident. So every route is treated as if it also had 20 "average" passengers:

```
rate = (satisfied real passengers on route + 20 × overall rate) / (real passengers on route + 20)
```

Big routes keep their own rate; tiny routes are pulled toward the average.

## Part 2: The experiments

All use the same 5 folds, seed (42) and model settings as every earlier round.

| Experiment | What it is |
|---|---|
| v5a | LightGBM on v3c features + original route rates |
| v5b | v5a + teacher prediction |
| v5 XGBoost | XGBoost on the better of v5a / v5b |
| **v5 blend** | Rank average of the better v5 LightGBM + v5 XGBoost + **your saved Round 4 CatBoost** |

Reusing the saved CatBoost results means **no 70-minute wait**: the whole round takes about 5–10 minutes.

## Part 3: How to run it

**1. Check the rules.** On the competition page, open **Rules** and find the section on
**external data**. Playground competitions normally allow publicly available data, and this
dataset is openly discussed by competitors, but confirm it before uploading.

**2. Download the original dataset.**
1. Go to https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction
2. Click **Download** (top right). You get a zip file.
3. Unzip it. It contains two files, also called `train.csv` and `test.csv`.

**3. Put them in a NEW folder so they don't replace the competition files.**
Inside your `data` folder, make a folder called `original`, and put both files there:

```
kaggle-airline-satisfaction\
    data\
        train.csv               <- competition (don't touch)
        test.csv                <- competition (don't touch)
        sample_submission.csv
        original\
            train.csv           <- original dataset
            test.csv            <- original dataset
```

> The original dataset's own `test.csv` **has answers too**, so the script uses both files:
> about 130,000 real passengers in total.

**4. Run the script** (with `(.venv)` showing):
```powershell
python scripts/07_original_data.py
```
✅ *You should see:*
- `Original data: 129,880 passengers from 2 file(s)` (approximately)
- `ignoring columns not in the competition: ...`, which should include **Inflight service** (the
  original has 14 ratings; the competition has 13)
- a **teacher** line with two AUC scores
- a **ROUND 5 SUMMARY** table and either a **RECOMMENDED UPLOAD** line or a message saying not to upload

**5. Upload only if it says so.** The script compares everything with your Round 4 blend (0.96065).
The data folder is in `.gitignore`, so the original dataset won't go to GitHub either.
   ```powershell
   python scripts/03_check_submission.py submissions/submission_v5_blend_rank_average.csv
   ```
   - **File to upload:** `submissions\submission_v5_blend_rank_average.csv`
   - **Description on Kaggle:** `v5: rank-average blend of LightGBM + XGBoost (original-data teacher features) + Round 4 CatBoost, CV AUC 0.96090`

**6. Update your final selection.** On the Submissions tab, tick **v5_blend_rank_average** and
**v4_blend_rank_average** as your 2 final submissions.

## Part 4: The results

The run took **4.4 minutes**.

```
Original data: 129,880 passengers from 2 file(s)
   satisfied in the original: 43.4%
   ignoring columns not in the competition: Unnamed: 0, id, Inflight service
   teacher: AUC 0.99480 on held-back REAL passengers, 0.95475 on the competition passengers (369 trees)
```

| Experiment | CV AUC | Change | Verdict |
|---|---:|---:|---|
| v3c LightGBM (start) | 0.96026 | | |
| v5a: + original route rates | 0.96028 | 0.96028 − 0.96026 = +0.00002 | No real effect |
| v5b: + teacher | 0.96046 | 0.96046 − 0.96028 = **+0.00018** | **Real gain** |
| v5 XGBoost (v5b features) | 0.96053 | | Best single model so far |
| **v5 blend** | **0.96090** | 0.96090 − 0.96065 = **+0.00025** vs Round 4 blend | **New best** |

**How similar the blended models are:**

|  | LightGBM | XGBoost | CatBoost (Round 4) |
|---|---:|---:|---:|
| LightGBM | 1.0000 | 0.9832 | 0.9663 |
| XGBoost | 0.9832 | 1.0000 | 0.9664 |
| CatBoost (Round 4) | 0.9663 | 0.9664 | 1.0000 |

**Kaggle confirmed it:**

| | Round 4 blend | Round 5 blend | Gain |
|---|---:|---:|---:|
| CV AUC | 0.96065 | 0.96090 | 0.96090 − 0.96065 = +0.00025 |
| Public score | 0.96025 | 0.96049 | 0.96049 − 0.96025 = +0.00024 |
| Rank | 945 | 863 | up 945 − 863 = 82 places |

CV predicted +0.00025 and Kaggle gave +0.00024, almost exactly. The gap between CV and public
score was 0.96090 − 0.96049 = 0.00041, in line with every earlier round.

## Part 5: What we learned

1. **The original route rates added nothing** (+0.00002). Round 3's route profile already told the
   model about each route, so the real passengers' rates were not new information.
2. **The teacher helped** (+0.00018). Its two scores explain why: it scored **0.99480** on real
   passengers but only **0.95475** on the generated ones. The real data is much cleaner and more
   predictable than the generated data. That's why adding real rows to training hurts (the two
   behave differently), but the teacher's *opinion* still carries useful knowledge.
3. **Mixing models trained on different features adds diversity.** CatBoost was trained on the
   v3c features (no teacher), while LightGBM and XGBoost had the teacher. Its similarity with them
   fell from about 0.975 (Round 4) to 0.966, so the blend cancelled more mistakes.
4. **Reusing saved work saves time.** Loading Round 4's CatBoost predictions avoided a 70-minute
   retrain; the whole round took 4.4 minutes.

**Progress so far:**

| Round | CV AUC | Public score | Rank |
|---|---:|---:|---:|
| 1. Baseline | 0.95878 | 0.95836 | 1,361 |
| 2. Feature engineering | 0.95923 | 0.95882 | 1,130 |
| 3. Route features | 0.96026 | 0.95999 | 993 |
| 4. Three-model blend | 0.96065 | 0.96025 | 945 |
| 5. Original-data teacher | 0.96090 | 0.96049 | 863 |
| **Total gain** | **+0.00212** | **+0.00213** | **up 498 places** |

---

## New words in this round

| Word | Plain meaning |
|---|---|
| **Original (source) dataset** | The real data that the competition's generated data was learned from |
| **Synthetic data** | Data made by a computer program to look like real data |
| **External data** | Data from outside the competition; allowed only if the rules permit it |
| **Smoothing** | Pulling a rate from a small group toward the overall average, so tiny groups don't give extreme values |
| **Teacher model** | A model trained on one dataset whose predictions become a feature for another model |
| **Distribution shift** | When two datasets behave differently, e.g. real vs generated passengers |
