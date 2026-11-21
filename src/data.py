"""Loading, cleaning and splitting the UCI Heart Disease data (all four sites)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_RAW, RANDOM_STATE, TEST_SIZE

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

SITE_FILES = {
    "Cleveland": "processed.cleveland.data",
    "Hungary": "processed.hungarian.data",
    "Switzerland": "processed.switzerland.data",
    "Long Beach VA": "processed.va.data",
}
SITES = list(SITE_FILES)

# ca (number of major vessels coloured by fluoroscopy, 0-3) is an ordered count,
# so it is treated as numeric rather than one-hot encoded.
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
TARGET = "target"

FEATURE_LABELS = {
    "age": "Age (years)",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl",
    "restecg": "Resting ECG",
    "thalach": "Max heart rate achieved (bpm)",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (oldpeak)",
    "slope": "Slope of peak exercise ST",
    "ca": "Major vessels coloured (0-3)",
    "thal": "Thallium stress test",
    "target": "Heart disease",
}

CATEGORY_LABELS = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"},
    "fbs": {0: "No", 1: "Yes"},
    "restecg": {0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"},
    "exang": {0: "No", 1: "Yes"},
    "slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    "thal": {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"},
    "target": {0: "No disease", 1: "Disease"},
}


def load_raw(raw_dir=DATA_RAW):
    """Read the four processed site files exactly as published ('?' = missing)."""
    frames = []
    for site, fname in SITE_FILES.items():
        df = pd.read_csv(raw_dir / fname, header=None, names=COLUMNS, na_values="?")
        df.insert(0, "site", site)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean(df):
    """Mark physiologically impossible zeros as missing and build the binary target.

    * chol == 0 (all of Switzerland and part of Long Beach VA) is a missing-value
      placeholder, not a real cholesterol reading.
    * trestbps == 0 (one Long Beach VA record) is likewise impossible.
    * num is the angiographic diagnosis (0 = <50% narrowing, 1-4 = >50% narrowing);
      following the dataset documentation and prior work it is binarised to
      presence (1) vs absence (0) of disease.
    """
    df = df.copy()
    df["chol"] = df["chol"].mask(df["chol"] == 0)
    df["trestbps"] = df["trestbps"].mask(df["trestbps"] == 0)
    df[TARGET] = (df["num"] > 0).astype(int)
    for col in FEATURES + ["num"]:
        df[col] = df[col].astype(float)
    return df


def load_dataset(raw_dir=DATA_RAW):
    """Cleaned 920-row dataframe with columns: site, 13 features, num, target."""
    return clean(load_raw(raw_dir))


def split_data(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Hold-out split stratified jointly on site and target.

    Joint stratification keeps each hospital's share and disease prevalence the
    same in train and test, which matters because prevalence differs a lot by site.
    """
    strata = df["site"] + "_" + df[TARGET].astype(str)
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=strata, random_state=random_state
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_xy(df, features=None):
    features = FEATURES if features is None else features
    return df[features].copy(), df[TARGET].to_numpy()


def missing_summary(df):
    """Percentage of missing values per feature, overall and by site."""
    overall = df[FEATURES].isna().mean().mul(100).rename("All sites")
    by_site = df.groupby("site", sort=False)[FEATURES].apply(lambda g: g.isna().mean().mul(100)).T
    return pd.concat([by_site, overall], axis=1).round(1)
