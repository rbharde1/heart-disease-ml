"""Model definitions, hyperparameter grids and tuning."""
import json

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from config import CV_FOLDS, RANDOM_STATE, RESULTS_DIR
from preprocess import build_preprocessor

MODEL_NAMES = ["Logistic Regression", "SVM (RBF)", "Random Forest"]
BASELINE_NAME = "Majority baseline"
TUNING_SCORING = "roc_auc"
BEST_PARAMS_PATH = RESULTS_DIR / "best_params.json"


def get_model_specs(random_state=RANDOM_STATE):
    """name -> (unfitted estimator, param grid for the 'clf' step)."""
    return {
        "Logistic Regression": (
            LogisticRegression(solver="liblinear", max_iter=5000, random_state=random_state),
            # l1_ratio=0 -> L2 (ridge) penalty, l1_ratio=1 -> L1 (lasso) penalty
            {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100], "clf__l1_ratio": [0.0, 1.0]},
        ),
        "SVM (RBF)": (
            # Platt scaling (sigmoid fitted on internal 5-fold CV predictions) turns the
            # SVM margin into a probability, as SVC(probability=True) used to do.
            CalibratedClassifierCV(SVC(kernel="rbf"), method="sigmoid", cv=5, ensemble=False),
            {
                "clf__estimator__C": [0.1, 0.5, 1, 5, 10, 50],
                "clf__estimator__gamma": ["scale", 0.001, 0.01, 0.05, 0.1],
            },
        ),
        "Random Forest": (
            # n_jobs=1: every caller (GridSearchCV, cross_validate, learning_curve, ...)
            # already parallelises over folds/candidates. A multi-threaded forest inside
            # those workers over-subscribes the CPU and makes joblib emit spurious
            # warnings. The fitted trees do not depend on n_jobs (random_state is fixed).
            RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=random_state),
            {
                "clf__max_depth": [None, 4, 6, 10],
                "clf__min_samples_leaf": [1, 3, 5, 10],
                "clf__max_features": ["sqrt", 0.5],
            },
        ),
    }


def make_pipeline(estimator, preprocessor=None):
    preprocessor = build_preprocessor() if preprocessor is None else preprocessor
    return Pipeline([("prep", preprocessor), ("clf", clone(estimator))])


def make_baseline(preprocessor=None):
    return make_pipeline(DummyClassifier(strategy="most_frequent"), preprocessor)


def cv_splitter(n_splits=CV_FOLDS, random_state=RANDOM_STATE):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def tune_model(name, X, y, preprocessor=None, scoring=TUNING_SCORING, cv=None,
               random_state=RANDOM_STATE, n_jobs=-1):
    """Grid-search one model with stratified k-fold CV on (X, y); returns fitted GridSearchCV."""
    estimator, grid = get_model_specs(random_state)[name]
    search = GridSearchCV(
        make_pipeline(estimator, preprocessor), grid, scoring=scoring,
        cv=cv_splitter(random_state=random_state) if cv is None else cv,
        n_jobs=n_jobs, refit=True,
    )
    return search.fit(X, y)


def build_tuned_pipeline(name, params, preprocessor=None, random_state=RANDOM_STATE):
    """Unfitted pipeline for `name` with already-chosen hyperparameters."""
    estimator, _ = get_model_specs(random_state)[name]
    return make_pipeline(estimator, preprocessor).set_params(**params)


def get_best_params(X_train=None, y_train=None, path=BEST_PARAMS_PATH, refresh=False):
    """Best hyperparameters from the main tuning run (cached in results/best_params.json).

    Experiments reuse these so every analysis uses the same tuned models. If the
    cache is missing (or refresh=True), the models are tuned on (X_train, y_train)
    with the default preprocessing and the result is written to the cache.
    """
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    if X_train is None:
        raise ValueError("best_params.json not found: pass X_train/y_train to tune")
    best = {}
    for name in MODEL_NAMES:
        search = tune_model(name, X_train, y_train)
        best[name] = {k: v for k, v in search.best_params_.items()}
    path.write_text(json.dumps(best, indent=2))
    return best
