# Step 1 Guide: Your First Kaggle Competition

**Competition:** Kaggle Playground Series, Season 6 Episode 10, *Predicting Airline Satisfaction*
**Link:** https://www.kaggle.com/competitions/playground-series-s6e10
**Deadline:** Saturday **October 31, 2026**, 11:59 PM UTC (that is **7:59 PM in Ohio**)
**Upload limit:** 10 submissions per day

**The task:** For each passenger, predict the *probability* (a number from 0 to 1) that they were satisfied with their flight.
**The score:** ROC AUC. 0.5 = guessing, 1.0 = perfect. Strong entries score about 0.96.

---

## Part 1: Create your Kaggle account (5 minutes)

1. Go to https://www.kaggle.com and click **Register**. Signing in with Google is easiest.
2. Click your profile picture (top right), then **Settings**.
3. Find **Phone verification** and verify your phone number. Kaggle requires this before you can submit to competitions.
4. Add your real name, a photo and your LinkedIn link to your profile. Recruiters will look at it.

## Part 2: Join the competition (2 minutes)

1. Open https://www.kaggle.com/competitions/playground-series-s6e10
2. Click **Join Competition** and accept the rules.

## Part 3: Download the data (3 minutes)

1. On the competition page, click the **Data** tab.
2. Scroll down and click **Download All**. You get a zip file in your Downloads folder.
3. Unzip it. You should see three files: `train.csv`, `test.csv`, `sample_submission.csv`.
4. You will move these into the project's `data` folder in Part 4.

> **Important:** Kaggle's rules don't allow re-sharing the data. The `.gitignore`
> file already stops the `data` folder from going to GitHub.

## Part 4: Set up the project on your laptop (10 minutes)

1. Unzip `kaggle-airline-satisfaction.zip` into your Documents folder.
2. Make a folder called `data` inside it, and move the three CSV files into it:
   ```
   kaggle-airline-satisfaction\
       data\
           train.csv
           test.csv
           sample_submission.csv
       scripts\
       README.md
       ...
   ```
3. Open the `kaggle-airline-satisfaction` folder in VS Code (**File → Open Folder**).
4. Open the terminal (**Terminal → New Terminal**) and create the virtual environment.
   Use `py -3.13`, not plain `python`, so Windows doesn't pick up the old Anaconda Python:
   ```powershell
   py -3.13 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python --version
   ```
   ✅ *You should see:* `Python 3.13.x`, and `(.venv)` at the start of the terminal line.
5. Install the libraries:
   ```powershell
   pip install -r requirements.txt
   ```

## Part 5: Run the three scripts

**5a. Look at the data**
```powershell
python scripts/01_first_look.py
```
✅ *You should see:* about **699,635** training rows, **299,844** test rows, roughly **44%** satisfied,
and only `Arrival Delay in Minutes` with a few hundred missing values.

**5b. Train the model** (takes about 3 to 10 minutes)
```powershell
python scripts/02_baseline_model.py
```
✅ *You should see:* five folds, each with an AUC, then a **CROSS-VALIDATION AUC** of about **0.958**.

> If you want to test that everything works first, open `scripts/02_baseline_model.py`,
> change `QUICK_TEST = False` to `QUICK_TEST = True`, run it (under a minute), then change it back.

**5c. Check the submission file**
```powershell
python scripts/03_check_submission.py
```
✅ *You should see:* seven **PASS** lines and *"This file is safe to upload to Kaggle."*

## Part 6: Upload to Kaggle

1. On the competition page, click **Submit Prediction**.
2. Upload `submissions\submission_baseline_lgbm.csv`.
3. In the description box, type: `Baseline LightGBM, 5-fold CV AUC 0.95xx` (your real number).
4. Click **Submit**. After a minute you get a **Public Score**.
5. Open `outputs\experiment_log.csv` in Excel or VS Code and type that score in the `public_lb` column.

✅ *What to expect:* your Public Score should be within about 0.001 of your CV score.
If it is, your cross-validation is trustworthy, which is the most important skill this step teaches.

**Then send me:** your CV AUC, your Public Score, your leaderboard rank, and the output of `01_first_look.py`.

## Part 7: Put it on GitHub

1. On GitHub, create a new **public** repository called `kaggle-airline-satisfaction` (no README, it already has one).
2. In the VS Code terminal:
   ```powershell
   git init
   git add .
   git commit -m "Step 1: first look, baseline LightGBM, submission checker"
   git branch -M main
   git remote add origin https://github.com/bernardfrimpong040-pixel/kaggle-airline-satisfaction.git
   git push -u origin main
   ```
3. Check on GitHub that there is **no `data` folder**. If there is, tell me before doing anything else.

---

## Words you'll hear (glossary)

| Word | Plain meaning |
|---|---|
| **Feature** | A column the model uses to predict, e.g. Age or Seat comfort |
| **Target** | The column we predict: satisfaction |
| **Fold** | One of the 5 equal groups we split the training data into |
| **Cross-validation (CV)** | Train on 4 folds, test on the 5th, repeat 5 times. An honest score |
| **Overfitting** | The model memorises the training rows instead of learning patterns. It looks great on training data and fails on new data |
| **Early stopping** | Stop adding trees once the score on unseen rows stops improving. Prevents overfitting |
| **Out-of-fold (OOF) prediction** | The prediction for each training row from the model that didn't see that row |
| **ROC AUC** | The chance that a random satisfied passenger gets a higher prediction than a random unsatisfied one |
| **Public leaderboard** | Your score on part of the test set, shown during the competition |
| **Private leaderboard** | Your score on the rest of the test set, revealed after the deadline. **This is the final ranking** |
| **Seed** | A fixed number for the randomness, so you get the same result every time you run the script |
