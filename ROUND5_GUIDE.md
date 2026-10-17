# Round 5 Guide: Learning from the Original Dataset

**Goal:** Use the real dataset this competition was generated from to give the model new information.
**Script:** `scripts/07_original_data.py`
**Starting point:** Round 4 blend, CV AUC 0.96065, public score 0.96025, rank 945

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
