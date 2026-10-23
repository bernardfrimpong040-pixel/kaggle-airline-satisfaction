# Round 6 Guide: Target Encoding (Done Safely) and a Hill-Climbing Blend

**Goal:** Add target-encoded features without leaking the answer, then build the best possible blend from every saved model.
**Script:** `scripts/08_target_encoding.py`
**Starting point:** Round 5 blend, CV AUC 0.96090, public score 0.96049, rank 863 (17 Oct)
**Result:** CV AUC **0.96107**, public score **0.96056**, rank **1,079** (22 Oct; see Part 5 for why the rank fell)

---

## Part 1: Target encoding

**Target encoding** replaces a group with the share of its passengers who were satisfied. For
example, every passenger who flew 1,235 miles in Business class gets one number: the satisfaction
rate of that group in the **competition** training data.

This round encodes 6 groups:

| New column | Group |
|---|---|
| `te_distance` | Flight distance (route) |
| `te_age` | Age |
| `te_distance_class` | Route + class |
| `te_distance_travel` | Route + type of travel |
| `te_distance_customer` | Route + customer type |
| `te_age_class` | Age + class |

### The danger: data leakage

Target encoding is built from the **answers**. If a passenger's own answer helps build their own
number, the model gets a hint of the answer it's supposed to predict. The score looks great on
the laptop, then drops on Kaggle. This is **data leakage**, one of the most common mistakes in
data science.

### The safe way (what we use)

Inside **each** of the 5 folds:

| Rows | Where their encoding comes from |
|---|---|
| The 4 training folds | An **inner** split of those 4 folds, so no passenger's own answer builds their own value (`TargetEncoder.fit_transform`) |
| The held-out fold | Learned **only** from the 4 training folds (`TargetEncoder.transform`) |
| The test set | Learned **only** from the 4 training folds |

The held-out fold's answers are never used to build its own encodings, so its score stays honest.

### The leaky way (a demo, never uploaded)

The script also runs the **wrong** way: learn the encodings once from **all** training answers,
then cross-validate. Every held-out passenger's answer helped build their own encoding. The CV
score rises, **but the rise is fake**. The script prints both scores side by side.

### How we verified the safe version

Before sending the script, these checks were run on practice data:

| Check | Safe version | Leaky version |
|---|---|---|
| Flip **every** held-out answer: do the held-out encodings change? | **No** (unchanged) | Yes (they changed) |
| Flip **every** held-out answer: do the test encodings change? | **No** (unchanged) | |
| Flip **one** training passenger's answer: does **their own** encoding change? (inner split held fixed) | **No**: 0 of 50 changed | |

## Part 2: Hill-climbing blend

You now have many saved models:

| From | Models |
|---|---|
| Round 4 | LightGBM, XGBoost, CatBoost |
| Round 5 | LightGBM with teacher, XGBoost with teacher |
| Round 6 | LightGBM with target encoding, XGBoost with target encoding |

Instead of averaging a fixed three, the computer builds the best mix:

1. Start with the best single model.
2. Try adding each model (repeats allowed) and keep the one that raises the blend's CV the most.
3. Repeat until no model helps.

A model picked twice gets twice the weight.

## Part 3: How to run it

This round needs the `data\original\` folder from Round 5, and the saved Round 4 and 5 files
in `outputs\` and `submissions\` (already there if you ran those rounds).

```powershell
python scripts/08_target_encoding.py
```

It takes about **10–15 minutes**. To skip the leakage demo and save about 2 minutes, open the
script and set `RUN_LEAKY_DEMO = False`.

✅ *You should see:*
- 5 earlier models listed as **loaded**
- the hill-climbing steps
- a **ROUND 6 SUMMARY**, a **LEAKAGE DEMO** comparison, and either a **RECOMMENDED UPLOAD** line or
  a message saying not to upload

**Never upload a leaky result.** The script never saves or recommends one.

**Upload:**
```powershell
python scripts/03_check_submission.py submissions/submission_v6_hill_climb_blend.csv
```
- **File to upload:** `submissions\submission_v6_hill_climb_blend.csv`
- **Description on Kaggle:** `v6: hill-climbing rank blend (v6 LightGBM + v6 XGBoost with fold-safe target encoding, v5 XGBoost, v4 XGBoost, v4 CatBoost), CV AUC 0.96107`

**Final selection:** on the Submissions tab, tick **v6_hill_climb_blend** and **v5_blend_rank_average**.

## Part 4: The results

The run took **6.4 minutes**.

**Target encoding on single models:**

| Model | Round 5 | With target encoding | Gain |
|---|---:|---:|---:|
| LightGBM | 0.96046 | 0.96062 | 0.96062 − 0.96046 = +0.00016 |
| XGBoost | 0.96053 | **0.96072** | 0.96072 − 0.96053 = +0.00019 |

| Fold | v6 LightGBM | v6 XGBoost |
|---|---:|---:|
| 1 | 0.96066 | 0.96090 |
| 2 | 0.95958 | 0.95981 |
| 3 | 0.96126 | 0.96132 |
| 4 | 0.96090 | 0.96079 |
| 5 | 0.96074 | 0.96080 |

**Leakage demo** (same LightGBM, same features, only the encoding method differs):

| Encodings | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | CV AUC |
|---|---:|---:|---:|---:|---:|---:|
| Honest | 0.96066 | 0.95958 | 0.96126 | 0.96090 | 0.96074 | 0.96062 |
| **Leaky** | 0.96189 | 0.96108 | 0.96265 | 0.96221 | 0.96204 | **0.96196** |

The leaky version "wins" on **every** fold, by 0.96196 − 0.96062 = **+0.00134** overall. That
fake gain is about 8 times the real gain from target encoding (+0.00016).

**Hill-climbing blend:**

```
start: v6_xgboost (0.96072)
step 1: add v4_catboost            blend CV 0.96097
step 2: add v6a_target_encoding    blend CV 0.96104
step 3: add v5_xgboost             blend CV 0.96105
step 4: add v4_xgboost             blend CV 0.96106
step 5: add v6_xgboost             blend CV 0.96107
step 6: add v4_catboost            blend CV 0.96107
step 7: add v6a_target_encoding    blend CV 0.96107
```

Final weights: v6 XGBoost 0.25, v4 CatBoost 0.25, v6 LightGBM 0.25, v5 XGBoost 0.12, v4 XGBoost 0.12.

**Kaggle:**

| | Round 5 blend | Round 6 blend | Change |
|---|---:|---:|---:|
| CV AUC | 0.96090 | 0.96107 | 0.96107 − 0.96090 = +0.00017 |
| Public score | 0.96049 | 0.96056 | 0.96056 − 0.96049 = +0.00007 |
| Rank | 863 (17 Oct) | 1,079 (22 Oct) | down 1,079 − 863 = 216 places |

## Part 5: What we learned

1. **Fold-safe target encoding gives a small, real gain** (+0.00016 LightGBM, +0.00019 XGBoost).
2. **Leakage looks like success.** The leaky version beat the honest one on every fold. Nothing
   in the CV output warned that it was fake; only knowing *how* the features were built reveals it.
   That's why tests that check **how** a solution was built matter as much as tests on its score.
3. **Hill climbing gains shrink fast.** Steps 1–2 gave +0.00032; steps 3–7 together gave only
   +0.00003, which is close to noise. Weights chosen on the same data they're scored on are
   slightly optimistic, which partly explains why Kaggle confirmed only +0.00007 of the +0.00017.
4. **The CV–public gap widened** to 0.96107 − 0.96056 = 0.00051 (previous rounds: 0.00027–0.00042).
   We're now at the point where improvements are about the size of the noise, which is a sign
   the competition is close to its limit for this approach.
5. **Score up, rank down.** The public score improved, but the rank fell from 863 to 1,079,
   because over the 5 days between submissions, many other teams improved too. Rank depends on
   everyone else; the score is the fair measure of our own progress.

**Progress so far:**

| Round | CV AUC | Public score | Rank (date) |
|---|---:|---:|---:|
| 1. Baseline | 0.95878 | 0.95836 | 1,361 (17 Oct) |
| 2. Feature engineering | 0.95923 | 0.95882 | 1,130 (17 Oct) |
| 3. Route features | 0.96026 | 0.95999 | 993 (17 Oct) |
| 4. Three-model blend | 0.96065 | 0.96025 | 945 (17 Oct) |
| 5. Original-data teacher | 0.96090 | 0.96049 | 863 (17 Oct) |
| 6. Target encoding + hill climbing | 0.96107 | 0.96056 | 1,079 (22 Oct) |
| **Total gain** | **+0.00229** | **+0.00220** | |

---

## New words in this round

| Word | Plain meaning |
|---|---|
| **Target encoding** | Replacing a group with the share of its passengers who had the answer "satisfied" |
| **Cross-fitting** | Building each row's value only from *other* rows, so its own answer is never used |
| **Data leakage** | When information about the answer sneaks into the features; the score looks great but isn't real |
| **Hill climbing** | Building a blend step by step, adding whichever model helps most each time |
| **Diminishing returns** | Each extra effort brings a smaller gain than the one before |
