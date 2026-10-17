# Round 4 Guide: Three Models and a Blend

**Goal:** Train three different models on the v3c features and blend their predictions.
**Script:** `scripts/06_model_blend.py`
**Result:** CV AUC **0.96065**, public score **0.96025**, rank **945** (up from 993)

---

## Part 1: The idea

Rounds 1–3 used one model: **LightGBM**. Two other popular models are **XGBoost** and
**CatBoost**. All three build hundreds of decision trees, but each builds them in a slightly
different way, so they make slightly **different mistakes**.

If we average their predictions, some of those mistakes cancel out. This is called
**blending** (or ensembling), and it's how most Kaggle competitions are won.

> Think of three teachers marking the same exam. Each has small biases of their own, but the
> average of their three marks is usually fairer than any one teacher's mark.

## Part 2: What the script does

1. Builds the **v3c features** (Round 3's best: lean base + pair counts + route profile).
2. Trains **LightGBM, XGBoost and CatBoost**, each on the **same 5 folds**, so their
   out-of-fold predictions can be compared and blended fairly.
3. Measures **how similar** the three models' predictions are.
4. Tries **three blends**:

| Blend | How it works |
|---|---|
| Average | The mean of the three models' probabilities |
| Rank average | Each model's predictions are turned into ranks (0 = lowest, 1 = highest), then averaged. AUC only cares about the **order** of predictions, so this stops one model's probability scale from dominating |
| Weighted | A rank blend that tries every set of weights in steps of 0.1 that add up to 1, and keeps the best. Because the weights are chosen on the same data they're scored on, the score is a little optimistic, so it's only preferred if it's **clearly** better |

**Resume feature:** each model's predictions are saved as soon as it finishes. If the run
stops part-way, running it again reloads the finished models instead of retraining them.

## Part 3: How to run it

1. Install the two new libraries (with `(.venv)` showing):
   ```powershell
   pip install -r requirements.txt
   ```
2. Run the script:
   ```powershell
   python scripts/06_model_blend.py
   ```
   It took **73 minutes** in total. LightGBM took about 1 minute, XGBoost about 3, and
   **CatBoost about 69** (6.7 to 25 minutes per fold). Keep the laptop plugged in and awake.
   The script prints a line only when a whole fold finishes, so long gaps during CatBoost are normal.
3. At the end it prints a **ROUND 4 SUMMARY** and a **RECOMMENDED UPLOAD** line.
4. Check the file, then upload it:
   ```powershell
   python scripts/03_check_submission.py submissions/submission_v4_blend_rank_average.csv
   ```
   - **File to upload:** `submissions\submission_v4_blend_rank_average.csv`
   - **Description on Kaggle:** `v4: rank-average blend of LightGBM + XGBoost + CatBoost on v3c features, CV AUC 0.96065`
5. Type the public score into `outputs\experiment_log.csv` (v4_blend_rank_average row, `public_lb` column).

## Part 4: The results

**Each model on its own:**

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | CV AUC | Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| LightGBM | 0.96053 | 0.95910 | 0.96083 | 0.96040 | 0.96045 | 0.96026 | ~1 min |
| XGBoost | 0.96049 | 0.95919 | 0.96105 | 0.96035 | 0.96020 | 0.96025 | ~3 min |
| CatBoost | 0.96029 | 0.95919 | 0.96114 | 0.96051 | 0.96026 | 0.96027 | ~69 min |

Every model found fold 2 hardest and fold 3 easiest. The folds are random groups of
passengers, and some groups are simply easier to predict than others.

**How similar the models are** (rank correlation; 1.0000 = identical ordering):

|  | LightGBM | XGBoost | CatBoost |
|---|---:|---:|---:|
| LightGBM | 1.0000 | 0.9816 | 0.9750 |
| XGBoost | 0.9816 | 1.0000 | 0.9749 |
| CatBoost | 0.9750 | 0.9749 | 1.0000 |

**The blends:**

| Experiment | CV AUC | vs v3c (0.96026) |
|---|---:|---:|
| Average | 0.96063 | +0.00037 |
| **Rank average** | **0.96065** | **+0.00039** |
| Weighted (LightGBM 0.3, XGBoost 0.3, CatBoost 0.4) | 0.96065 | +0.00039 |

**Kaggle confirmed it:**

| | v3c (Round 3) | v4 rank average | Gain |
|---|---:|---:|---:|
| CV AUC | 0.96026 | 0.96065 | 0.96065 − 0.96026 = +0.00039 |
| Public score | 0.95999 | 0.96025 | 0.96025 − 0.95999 = +0.00026 |
| Rank | 993 | 945 | up 993 − 945 = 48 places |

The public score landed in the predicted range (0.9602 to 0.9604). The gap between CV and
public score was 0.96065 − 0.96025 = 0.00040, in line with earlier rounds (0.00027 to 0.00042).

## Part 5: What we learned

1. **The three models were equally good alone.** Their CV scores were within 0.00002 of each
   other. CatBoost took about 69 minutes and was no better than LightGBM's 1 minute.
2. **Together they beat every single model.** The blend scored 0.96065 − 0.96027 = **+0.00038**
   above the best single model. Equally good models can still help each other, as long as they
   make **different** mistakes.
3. **Different matters more than better.** CatBoost was the most different model (0.9750 and
   0.9749 with the others), so the weight search gave it the largest share (0.4).
4. **Simple beat clever.** The weighted blend scored exactly the same as the plain rank average.
   When two options score the same, the one with no tuned settings is safer, because there's
   nothing that could have been fitted to noise.
5. **Our validation is still trustworthy.** Four rounds in a row, CV has predicted the Kaggle
   result closely.

**Progress so far:**

| Round | CV AUC | Public score | Rank |
|---|---:|---:|---:|
| 1. Baseline | 0.95878 | 0.95836 | 1,361 |
| 2. Feature engineering | 0.95923 | 0.95882 | 1,130 |
| 3. Route features | 0.96026 | 0.95999 | 993 |
| 4. Three-model blend | 0.96065 | 0.96025 | 945 |
| **Total gain** | **+0.00187** | **+0.00189** | **up 416 places** |

---

## New words in this round

| Word | Plain meaning |
|---|---|
| **XGBoost** | Another gradient-boosting library; grows trees level by level |
| **CatBoost** | Another gradient-boosting library; especially good with word (categorical) columns, but slower |
| **Blending / ensembling** | Combining several models' predictions into one, usually by averaging |
| **Rank** | A prediction's position in the sorted list, scaled from 0 (lowest) to 1 (highest) |
| **Rank correlation** | How similarly two models order the passengers; 1.0 means identical order |
| **Diversity** | How differently models make mistakes. More diversity means a bigger gain from blending |
| **Resume** | Picking up an interrupted run from where it stopped, using saved results |
