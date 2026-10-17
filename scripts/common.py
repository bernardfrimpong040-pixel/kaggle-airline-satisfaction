# ============================================================================
# common.py  —  small helper functions that the other scripts share
#
# You never run this file directly. The other scripts "import" it, which
# means they borrow the functions written here. Keeping them in one place
# means that if we fix something, we fix it once.
# ============================================================================

from pathlib import Path   # Path makes file locations work on Windows AND Mac/Linux
import pandas as pd         # pandas = the "Excel inside Python" library

# ----------------------------------------------------------------------------
# Folder locations
# ----------------------------------------------------------------------------
# __file__ is the location of THIS file (scripts/common.py).
# .parent is the folder it sits in (scripts), and .parent again is the
# project folder. So PROJECT_DIR always points at the project, no matter
# which folder you run the script from.
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"                # where Kaggle's files go
OUTPUT_DIR = PROJECT_DIR / "outputs"           # results, logs and tables
SUBMISSION_DIR = PROJECT_DIR / "submissions"   # files we upload to Kaggle

# The column we are trying to predict, and the ID column
TARGET = "satisfaction"
ID_COL = "id"


def load_data():
    """Read train.csv and test.csv and return them as two tables."""

    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"

    # If the files are missing, stop with a clear message instead of a scary error
    for path in [train_path, test_path]:
        if not path.exists():
            raise SystemExit(
                f"\nI can't find {path}\n"
                "Download the competition files from Kaggle and put "
                "train.csv and test.csv inside the 'data' folder.\n"
                "See SETUP_STEP1.md, part 3."
            )

    print("Reading the data files (this can take 10-30 seconds)...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Turn the target into 1 (satisfied) and 0 (not satisfied).
    # Kaggle may store it as True/False, or as words. This handles both.
    train[TARGET] = target_to_number(train[TARGET])

    return train, test


def target_to_number(column):
    """Convert the satisfaction column to 1 = satisfied, 0 = not satisfied."""

    # Make everything lower-case text, e.g. True -> "true", "Satisfied" -> "satisfied"
    as_text = column.astype(str).str.strip().str.lower()

    # These words mean "satisfied"
    yes_words = ["true", "1", "1.0", "satisfied"]

    # .isin(...) gives True/False for each row; .astype(int) turns that into 1/0
    return as_text.isin(yes_words).astype(int)


def find_feature_columns(train):
    """
    Split the columns into two groups:
      - categorical columns: words, like Gender = "Male" / "Female"
      - numeric columns: numbers, like Age or a 0-5 rating
    The ID and the target are NOT features, so they are left out.
    """
    features = [c for c in train.columns if c not in (ID_COL, TARGET)]

    # A column is categorical if pandas did not read it as numbers
    categorical = [c for c in features
                   if not pd.api.types.is_numeric_dtype(train[c])
                   or pd.api.types.is_bool_dtype(train[c])]
    numeric = [c for c in features if c not in categorical]

    return features, categorical, numeric


def make_categories_match(train, test, categorical_cols):
    """
    LightGBM can use word columns directly, but only if they are stored as the
    pandas "category" type, with the SAME list of possible values in train and
    test. Otherwise "Male" might be code 0 in train and code 1 in test.
    """
    for col in categorical_cols:
        # Every value that appears in either table, sorted so the order is fixed
        all_values = sorted(pd.concat([train[col], test[col]]).dropna().astype(str).unique())
        kind = pd.CategoricalDtype(categories=all_values)
        train[col] = train[col].astype(str).where(train[col].notna()).astype(kind)
        test[col] = test[col].astype(str).where(test[col].notna()).astype(kind)
    return train, test
