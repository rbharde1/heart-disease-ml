"""Preprocessing pipelines.

All imputation, encoding and scaling is fitted inside an sklearn Pipeline so that
statistics come only from the training folds (no test-set leakage).
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES

SPARSE_FEATURES = ["ca", "slope", "thal"]  # >30% missing across the pooled data

STRATEGIES = {
    "indicator": "Median/mode imputation + missing-value indicators, "
                 "missing category kept for categorical features (default)",
    "simple": "Median/mode imputation only (no missingness information)",
    "drop_sparse": "Drop ca, slope, thal (>30% missing), median/mode impute the rest",
}


def _num_pipe(add_indicator, scale):
    steps = [("impute", SimpleImputer(strategy="median", add_indicator=add_indicator))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def _cat_pipe(missing_as_category):
    if missing_as_category:
        imputer = SimpleImputer(strategy="constant", fill_value=-1)
    else:
        imputer = SimpleImputer(strategy="most_frequent")
    return Pipeline([
        ("impute", imputer),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])


def build_preprocessor(strategy="indicator", scale=True, include_site=False):
    """Return an unfitted ColumnTransformer for the chosen missing-data strategy.

    strategy     : 'indicator' (default), 'simple' or 'drop_sparse' (see STRATEGIES)
    scale        : standardise numeric features (needed by LR/SVM, harmless for RF)
    include_site : one-hot encode the hospital of origin as an extra feature
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {list(STRATEGIES)}")

    numeric, binary, categorical = list(NUMERIC_FEATURES), list(BINARY_FEATURES), list(CATEGORICAL_FEATURES)
    if strategy == "drop_sparse":
        numeric = [c for c in numeric if c not in SPARSE_FEATURES]
        categorical = [c for c in categorical if c not in SPARSE_FEATURES]

    use_indicator = strategy == "indicator"
    transformers = [
        ("num", _num_pipe(add_indicator=use_indicator, scale=scale), numeric),
        ("bin", SimpleImputer(strategy="most_frequent"), binary),
        ("cat", _cat_pipe(missing_as_category=use_indicator), categorical),
    ]
    if include_site:
        transformers.append(("site", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["site"]))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def feature_names(fitted_pipeline):
    """Output feature names of the fitted 'prep' step of a model pipeline."""
    return list(fitted_pipeline.named_steps["prep"].get_feature_names_out())
