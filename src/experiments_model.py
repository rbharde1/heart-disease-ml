"""Experiments on model behaviour (M1-M6).

M1  Learning curves            - is each model limited by bias or by variance?
M2  Validation curves          - how sensitive is each model to its key hyperparameters?
M3  Feature importance         - LR coefficients, RF impurity importance and test-set
                                 permutation importance, plus rank agreement between them
M4  Decision threshold         - thresholds for a screening use case, chosen on
                                 out-of-fold training predictions, then applied to the test set
M5  Calibration                - reliability diagrams and Brier scores on the test set
M6  Error analysis             - which test patients are misclassified, and what they have in common

Every model uses the tuned hyperparameters cached in results/best_params.json and the
default preprocessing ('indicator' strategy). Anything that is *chosen* here (decision
thresholds) is chosen with cross-validation on the training set only; the held-out test
set is used purely to report final numbers.

Outputs: figures/exp_model_*.png, results/exp_model_*.csv and results/exp_model_results.json

Run from the project root:  python src/experiments_model.py
"""
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import (
    RepeatedStratifiedKFold, cross_val_predict, learning_curve, validation_curve,
)

from config import CV_FOLDS, RANDOM_STATE, RESULTS_DIR
from data import (
    BINARY_FEATURES, CATEGORY_LABELS, FEATURE_LABELS, FEATURES, NUMERIC_FEATURES, SITES,
    get_xy, load_dataset, split_data,
)
from evaluation import compute_metrics, predict_scores, save_json
from models import MODEL_NAMES, build_tuned_pipeline, cv_splitter, get_best_params
from plotting import (
    CLASS_COLORS, CLASS_NAMES, DIVERGING_CMAP, MODEL_COLORS, NEUTRAL, TEXT_PRIMARY,
    TEXT_SECONDARY, savefig,
)
from preprocess import feature_names

# Imported after plotting so the shared backend and style are already in place.
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

PREFIX = "exp_model_"

# 5-fold x 3 repeats = 15 fits per curve point: repeating smooths out the noise of a
# single 5-fold split on only 736 patients while keeping the runtime to a few minutes.
CURVE_REPEATS = 3

# Screening set-up for M4: missing a diseased patient (FN) is taken to be five times as
# costly as an unnecessary follow-up test (FP).
COST_FN, COST_FP = 5.0, 1.0
TARGET_SENSITIVITY = 0.90

N_PERM_REPEATS = 30
N_CAL_BINS = 8          # quantile bins of ~23 test patients each
N_BOOT = 2000

SHORT_MODEL = {"Logistic Regression": "LR", "SVM (RBF)": "SVM", "Random Forest": "RF"}
BINARY_LABELS = {
    "sex": "Male sex",
    "fbs": "Fasting blood sugar > 120 mg/dl",
    "exang": "Exercise-induced angina",
}


# ----------------------------------------------------------------------------- helpers
def curve_cv():
    """Repeated stratified 5-fold CV used for the learning and validation curves."""
    return RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CURVE_REPEATS,
                                   random_state=RANDOM_STATE)


def rounded(obj, nd=4):
    """Recursively round floats so the saved results are easy to quote."""
    if isinstance(obj, dict):
        # p-values can be tiny, so they keep 3 significant figures instead of nd decimals.
        return {k: float(f"{v:.3g}") if k == "p_value" else rounded(v, nd)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [rounded(v, nd) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if np.isnan(obj) else round(float(obj), nd)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def save_csv(df, name, index=False):
    decimals = {c: 4 for c in df.columns if c != "p_value"}
    df.round(decimals).to_csv(RESULTS_DIR / f"{PREFIX}{name}.csv", index=index)


def base_label(col):
    """Feature label without its unit, e.g. 'Serum cholesterol' (used inside longer names)."""
    return FEATURE_LABELS[col].split(" (")[0]


def readable_encoded_name(encoded):
    """Map an encoded column name to (readable label, original raw column).

    Handles the three kinds of columns the preprocessor creates: scaled numeric/binary
    features, missing-value indicators and one-hot levels (level -1 = 'missing').
    """
    if encoded.startswith("missingindicator_"):
        col = encoded[len("missingindicator_"):]
        return f"{base_label(col)} missing", col
    if encoded in NUMERIC_FEATURES:
        return FEATURE_LABELS[encoded], encoded
    if encoded in BINARY_FEATURES:
        return BINARY_LABELS[encoded], encoded
    col, level = encoded.rsplit("_", 1)
    level = int(float(level))
    level_name = "missing" if level == -1 else CATEGORY_LABELS[col][level]
    return f"{base_label(col)}: {level_name}", col


def suptitle(fig, text):
    fig.suptitle(text, fontsize=10.5, fontweight="bold")


def single_threaded(pipe):
    """Run the random forest on one core when the caller already parallelises over folds.

    learning_curve, validation_curve, cross_val_predict and permutation_importance spread
    their fits over all cores; letting the random forest open its own thread pool inside
    every worker over-subscribes the CPU and triggers spurious joblib/sklearn warnings.
    Results are identical because the forest's randomness does not depend on n_jobs.
    """
    if isinstance(pipe.named_steps["clf"], RandomForestClassifier):
        pipe.set_params(clf__n_jobs=1)
    return pipe


def fit_final_models(X, y, best):
    """Tuned pipelines refitted on the whole training set (used for all test-set results)."""
    return {name: build_tuned_pipeline(name, best[name]).fit(X, y) for name in MODEL_NAMES}


# ------------------------------------------------------------------ M1 learning curves
def run_learning_curves(X, y, best):
    """Training and CV ROC-AUC for 10%..100% of the CV-training folds."""
    fractions = np.linspace(0.1, 1.0, 10)
    rows = []
    for name in MODEL_NAMES:
        sizes, train_sc, val_sc = learning_curve(
            single_threaded(build_tuned_pipeline(name, best[name])), X, y,
            train_sizes=fractions,
            cv=curve_cv(), scoring="roc_auc", shuffle=True,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        for frac, n, tr, va in zip(fractions, sizes, train_sc, val_sc):
            rows.append({
                "model": name, "train_fraction": frac, "n_train": int(n),
                "train_auc_mean": tr.mean(), "train_auc_std": tr.std(),
                "cv_auc_mean": va.mean(), "cv_auc_std": va.std(),
            })
    return pd.DataFrame(rows)


def summarise_learning_curves(lc):
    summary = {}
    for name in MODEL_NAMES:
        d = lc[lc.model == name].reset_index(drop=True)
        full, half = d.iloc[-1], d.iloc[4]  # 100% and 50% of the training folds
        summary[name] = {
            "n_train_max": int(full.n_train),
            "train_auc_at_max": full.train_auc_mean,
            "cv_auc_at_max": full.cv_auc_mean,
            "cv_auc_std_at_max": full.cv_auc_std,
            "train_cv_gap_at_max": full.train_auc_mean - full.cv_auc_mean,
            "n_train_smallest": int(d.n_train.iloc[0]),
            "cv_auc_at_smallest": d.cv_auc_mean.iloc[0],
            "cv_auc_at_50pct": half.cv_auc_mean,
            "cv_auc_gain_50_to_100pct": full.cv_auc_mean - half.cv_auc_mean,
        }
    return summary


def plot_learning_curves(lc, summary):
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.8), sharey=True, layout="constrained")
    for ax, name in zip(axes, MODEL_NAMES):
        d, color = lc[lc.model == name], MODEL_COLORS[name]
        for kind, style, alpha in (("train", "--", 0.12), ("cv", "-", 0.25)):
            mean, std = d[f"{kind}_auc_mean"], d[f"{kind}_auc_std"]
            ax.fill_between(d.n_train, mean - std, mean + std, color=color, alpha=alpha, lw=0)
            ax.plot(d.n_train, mean, style, color=color, lw=1.6,
                    marker="o" if kind == "cv" else None, ms=3)
        gap = summary[name]["train_cv_gap_at_max"]
        ax.text(0.97, 0.05, f"gap at n={summary[name]['n_train_max']}: {gap:.3f}",
                transform=ax.transAxes, ha="right", fontsize=8, color=TEXT_SECONDARY)
        ax.set_title(name)
        ax.set_xlabel("Training patients")
    axes[0].set_ylabel("ROC-AUC")
    handles = [
        Line2D([], [], color=TEXT_SECONDARY, ls="--", label="Training score"),
        Line2D([], [], color=TEXT_SECONDARY, marker="o", ms=3, label="Cross-validation score"),
        Patch(color=TEXT_SECONDARY, alpha=0.25, label="± 1 SD"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    suptitle(fig, f"Learning curves ({CV_FOLDS}-fold CV × {CURVE_REPEATS} repeats)")
    return savefig(fig, f"{PREFIX}learning_curves")


# ---------------------------------------------------------------- M2 validation curves
def complexity_specs(best):
    """(key, model, parameter, values, x label, axis type) for each validation curve."""
    svm_c = best["SVM (RBF)"]["clf__estimator__C"]
    svm_gamma = best["SVM (RBF)"]["clf__estimator__gamma"]
    return [
        ("lr_C", "Logistic Regression", "clf__C",
         [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100, 1000],
         "C (inverse L2 penalty)", "log"),
        ("svm_C", "SVM (RBF)", "clf__estimator__C",
         [0.01, 0.03, 0.1, 0.3, 1, 2, 5, 10, 30, 100, 300, 1000],
         f"C (gamma = {svm_gamma})", "log"),
        ("svm_gamma", "SVM (RBF)", "clf__estimator__gamma",
         [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1, 3],
         f"gamma (C = {svm_c})", "log"),
        ("rf_depth", "Random Forest", "clf__max_depth",
         [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, None],
         "Maximum tree depth (∞ = no limit)", "category"),
        ("rf_trees", "Random Forest", "clf__n_estimators",
         [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000],
         "Number of trees", "log"),
    ]


def run_validation_curves(X, y, best):
    rows = []
    for key, name, param, values, _, _ in complexity_specs(best):
        train_sc, val_sc = validation_curve(
            single_threaded(build_tuned_pipeline(name, best[name])), X, y, param_name=param,
            param_range=values, cv=curve_cv(), scoring="roc_auc", n_jobs=-1,
        )
        for value, tr, va in zip(values, train_sc, val_sc):
            rows.append({
                "curve": key, "model": name, "parameter": param,
                "value": "None" if value is None else value,
                "train_auc_mean": tr.mean(), "train_auc_std": tr.std(),
                "cv_auc_mean": va.mean(), "cv_auc_std": va.std(),
            })
    return pd.DataFrame(rows)


def tuned_value(best, name, param):
    """Value the main grid search chose; RF uses 500 trees (fixed, not tuned)."""
    if param == "clf__n_estimators":
        return build_tuned_pipeline(name, best[name]).get_params()[param]
    value = best[name][param]
    return "None" if value is None else value


def summarise_validation_curves(vc, best):
    summary = {}
    for key, name, param, _, _, _ in complexity_specs(best):
        d = vc[vc.curve == key].reset_index(drop=True)
        top = d.loc[d.cv_auc_mean.idxmax()]
        tuned = tuned_value(best, name, param)
        at_tuned = d[d.value.astype(str) == str(tuned)].iloc[0]
        summary[key] = {
            "model": name, "parameter": param,
            "cv_best_value": top.value, "cv_best_auc": top.cv_auc_mean,
            "tuned_value": tuned, "cv_auc_at_tuned": at_tuned.cv_auc_mean,
            "train_auc_at_tuned": at_tuned.train_auc_mean,
            "cv_auc_min": d.cv_auc_mean.min(), "value_at_cv_min": d.loc[d.cv_auc_mean.idxmin(), "value"],
            "cv_auc_range": d.cv_auc_mean.max() - d.cv_auc_mean.min(),
        }
    return summary


def plot_validation_curves(vc, best):
    fig = plt.figure(figsize=(6.5, 4.5), layout="constrained")
    gs = GridSpec(2, 6, figure=fig)
    slots = {"lr_C": gs[0, 0:2], "svm_C": gs[0, 2:4], "svm_gamma": gs[0, 4:6],
             "rf_depth": gs[1, 0:3], "rf_trees": gs[1, 3:6]}
    titles = {"lr_C": "LR: regularisation", "svm_C": "SVM: C", "svm_gamma": "SVM: gamma",
              "rf_depth": "RF: tree depth", "rf_trees": "RF: forest size"}
    for key, name, param, values, xlabel, scale in complexity_specs(best):
        ax = fig.add_subplot(slots[key])
        d, color = vc[vc.curve == key], MODEL_COLORS[name]
        if scale == "category":
            x = np.arange(len(values))
            ax.set_xticks(x, ["∞" if v is None else str(v) for v in values])
        else:
            x = np.asarray(values, dtype=float)
            ax.set_xscale("log")
        for kind, style, alpha in (("train", "--", 0.12), ("cv", "-", 0.25)):
            mean, std = d[f"{kind}_auc_mean"].to_numpy(), d[f"{kind}_auc_std"].to_numpy()
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=alpha, lw=0)
            ax.plot(x, mean, style, color=color, lw=1.5,
                    marker="o" if kind == "cv" else None, ms=2.8)
        tuned = tuned_value(best, name, param)
        tuned_x = x[[str(v) for v in d.value].index(str(tuned))]
        ax.axvline(tuned_x, color=TEXT_PRIMARY, ls=":", lw=1.2)
        ax.set_title(titles[key])
        ax.set_xlabel(xlabel)
        if key in ("lr_C", "rf_depth"):
            ax.set_ylabel("ROC-AUC")
    handles = [
        Line2D([], [], color=TEXT_SECONDARY, ls="--", label="Training score"),
        Line2D([], [], color=TEXT_SECONDARY, marker="o", ms=3, label="Cross-validation score"),
        Line2D([], [], color=TEXT_PRIMARY, ls=":", lw=1.2, label="Tuned value"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    suptitle(fig, "Validation curves: ROC-AUC vs model complexity "
                  f"({CV_FOLDS}-fold CV × {CURVE_REPEATS})")
    return savefig(fig, f"{PREFIX}validation_curves")


# ----------------------------------------------------------------- M3 feature importance
def lr_coefficient_table(lr_pipe, X):
    """LR coefficients on the scale the model sees, plus a per-1-SD version.

    Numeric features and missing indicators are standardised inside the pipeline, so their
    coefficient is already 'per 1 SD'. Binary and one-hot columns are 0/1, so their raw
    coefficient is 'present vs absent'; multiplying every coefficient by the SD of its
    column puts all of them on the same per-1-SD scale and makes them comparable.
    """
    names = feature_names(lr_pipe)
    Xt = lr_pipe.named_steps["prep"].transform(X)
    coef = lr_pipe.named_steps["clf"].coef_.ravel()
    sd = Xt.std(axis=0)
    labels, raw_cols = zip(*(readable_encoded_name(n) for n in names))
    table = pd.DataFrame({
        "encoded_feature": names, "label": labels, "raw_feature": raw_cols,
        "column_sd": sd, "coefficient": coef, "std_coefficient": coef * sd,
        "odds_ratio_per_unit": np.exp(coef), "odds_ratio_per_sd": np.exp(coef * sd),
    })
    order = table.std_coefficient.abs().sort_values(ascending=False).index
    return table.loc[order].reset_index(drop=True)


def rf_importance_table(rf_pipe):
    names = feature_names(rf_pipe)
    labels, raw_cols = zip(*(readable_encoded_name(n) for n in names))
    table = pd.DataFrame({
        "encoded_feature": names, "label": labels, "raw_feature": raw_cols,
        "impurity_importance": rf_pipe.named_steps["clf"].feature_importances_,
    })
    return table.sort_values("impurity_importance", ascending=False).reset_index(drop=True)


def run_permutation_importance(final, X_test, y_test):
    """Drop in test ROC-AUC when one *raw* column is shuffled (same 13 inputs for all models)."""
    rows = []
    for name, model in final.items():
        res = permutation_importance(single_threaded(model), X_test, y_test, scoring="roc_auc",
                                     n_repeats=N_PERM_REPEATS, random_state=RANDOM_STATE,
                                     n_jobs=-1)
        for col, m, s in zip(X_test.columns, res.importances_mean, res.importances_std):
            rows.append({"model": name, "feature": col, "label": FEATURE_LABELS[col],
                         "importance_mean": m, "importance_std": s})
    return pd.DataFrame(rows)


def rank_agreement(lr_table, rf_table, perm):
    """Spearman correlation between the importance rankings of the 13 raw features."""
    vectors = {
        "LR |coef| (sum per feature)": lr_table.groupby("raw_feature").std_coefficient
        .apply(lambda s: s.abs().sum()),
        "RF impurity (sum per feature)": rf_table.groupby("raw_feature").impurity_importance.sum(),
    }
    for name in MODEL_NAMES:
        vectors[f"Permutation {SHORT_MODEL[name]}"] = (
            perm[perm.model == name].set_index("feature").importance_mean)
    vec = pd.DataFrame(vectors).reindex(FEATURES)
    keys = list(vec.columns)
    rho = pd.DataFrame(index=keys, columns=keys, dtype=float)
    pairs = []
    for i, a in enumerate(keys):
        for b in keys[i:]:
            res = spearmanr(vec[a], vec[b])
            rho.loc[a, b] = rho.loc[b, a] = res.statistic
            if a != b:
                pairs.append({"a": a, "b": b, "spearman_rho": res.statistic, "p_value": res.pvalue})
    return vec, rho, pd.DataFrame(pairs)


def plot_lr_coefficients(lr_table, top=20):
    d = lr_table.head(top).iloc[::-1]
    pos_color, neg_color = DIVERGING_CMAP(0.82), DIVERGING_CMAP(0.15)
    colors = [pos_color if c > 0 else neg_color for c in d.std_coefficient]
    fig, ax = plt.subplots(figsize=(6.5, 4.5), layout="constrained")
    y = np.arange(len(d))
    ax.barh(y, d.std_coefficient, color=colors, height=0.7)
    ax.axvline(0, color=TEXT_SECONDARY, lw=0.8)
    span = d.std_coefficient.abs().max()
    for yi, c, odds in zip(y, d.std_coefficient, d.odds_ratio_per_sd):
        ax.text(c + np.sign(c) * 0.02 * span, yi, f"OR {odds:.2f}", va="center",
                ha="left" if c > 0 else "right", fontsize=7, color=TEXT_SECONDARY)
    ax.set_yticks(y, d.label, fontsize=8)
    ax.set_xlim(-1.3 * span, 1.3 * span)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Standardised coefficient (change in log-odds per 1 SD)")
    ax.set_title(f"Logistic regression: {top} largest standardised coefficients "
                 f"(of {len(lr_table)})")
    ax.legend(handles=[Patch(color=pos_color, label="Raises odds of disease"),
                       Patch(color=neg_color, label="Lowers odds of disease")],
              loc="upper left")
    return savefig(fig, f"{PREFIX}lr_coefficients")


def plot_rf_importance(rf_table, top=15):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.6), layout="constrained",
                                   gridspec_kw={"width_ratios": [1, 1]})
    color = MODEL_COLORS["Random Forest"]
    d = rf_table.head(top).iloc[::-1]
    ax1.barh(d.label, d.impurity_importance, color=color, height=0.7)
    ax1.set_title(f"(a) Encoded columns (top {top})")
    agg = (rf_table.groupby("raw_feature").impurity_importance.sum()
           .sort_values().rename(index=FEATURE_LABELS))
    ax2.barh(agg.index, agg.values, color=color, height=0.7)
    ax2.set_title("(b) Summed per original feature")
    for ax in (ax1, ax2):
        ax.tick_params(axis="y", labelsize=7.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Mean decrease in impurity")
    suptitle(fig, "Random forest impurity-based feature importance")
    return savefig(fig, f"{PREFIX}rf_importance")


def plot_permutation_importance(perm):
    order = (perm.groupby("feature").importance_mean.mean().sort_values().index.tolist())
    fig, ax = plt.subplots(figsize=(6.5, 4.5), layout="constrained")
    height = 0.26
    base = np.arange(len(order))
    for k, name in enumerate(MODEL_NAMES):
        d = perm[perm.model == name].set_index("feature").loc[order]
        # Reverse the offset so the first model sits on top within each group.
        offset = (1 - k) * height
        ax.barh(base + offset, d.importance_mean, height=height, xerr=d.importance_std,
                color=MODEL_COLORS[name], label=name,
                error_kw={"elinewidth": 0.7, "capsize": 0, "ecolor": TEXT_SECONDARY})
    ax.axvline(0, color=TEXT_SECONDARY, lw=0.8)
    ax.set_yticks(base, [FEATURE_LABELS[f] for f in order], fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Drop in test ROC-AUC when the feature is shuffled (mean ± 1 SD)")
    ax.set_title(f"Permutation importance on the test set ({N_PERM_REPEATS} repeats)")
    ax.legend(loc="lower right")
    return savefig(fig, f"{PREFIX}permutation_importance")


# ------------------------------------------------------------------ M4 decision threshold
def threshold_metrics(y, p, thresholds):
    """Sensitivity, specificity, precision and expected cost for many thresholds at once."""
    y = np.asarray(y).astype(bool)
    pred = p[None, :] >= np.asarray(thresholds)[:, None]
    tp = (pred & y).sum(axis=1)
    fp = (pred & ~y).sum(axis=1)
    fn = y.sum() - tp
    tn = (~y).sum() - fp
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
    return pd.DataFrame({
        "threshold": thresholds, "sensitivity": tp / y.sum(), "specificity": tn / (~y).sum(),
        "precision": precision, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "cost_per_patient": (COST_FN * fn + COST_FP * fp) / len(y),
    })


def choose_thresholds(y, p):
    """Pick the two screening thresholds from out-of-fold training predictions only.

    Candidates are the midpoints between consecutive distinct OOF scores, so every
    candidate gives a different confusion matrix and sits in the middle of its interval.
      (i)  highest threshold with sensitivity >= 0.90 -> the best specificity at that
           sensitivity (sensitivity can only fall and specificity only rise as t grows);
      (ii) threshold with the lowest expected cost 5*FN + 1*FP; ties go to the lower
           threshold, i.e. towards higher sensitivity, which suits screening.
    """
    s = np.unique(p)
    candidates = np.r_[0.0, (s[:-1] + s[1:]) / 2, 1.0]
    table = threshold_metrics(y, p, candidates)
    ok = table[table.sensitivity >= TARGET_SENSITIVITY]
    sens_row = ok.loc[ok.threshold.idxmax()]
    cost_row = table.loc[table.cost_per_patient.idxmin()]  # idxmin -> first (lowest t)
    return sens_row, cost_row


def run_threshold_analysis(X_train, y_train, X_test, y_test, best, final):
    oof, rows, chosen = {}, [], {}
    for name in MODEL_NAMES:
        oof[name] = cross_val_predict(single_threaded(build_tuned_pipeline(name, best[name])),
                                      X_train, y_train, cv=cv_splitter(),
                                      method="predict_proba", n_jobs=-1)[:, 1]
        sens_row, cost_row = choose_thresholds(y_train, oof[name])
        rules = {
            "Default 0.5": 0.5,
            f"Sensitivity >= {TARGET_SENSITIVITY:.2f}": sens_row.threshold,
            f"Min cost (FN = {COST_FN:g} x FP)": cost_row.threshold,
        }
        chosen[name] = rules
        p_test = predict_scores(final[name], X_test)
        for rule, t in rules.items():
            oof_m = compute_metrics(y_train, oof[name], threshold=t)
            test_m = compute_metrics(y_test, p_test, threshold=t)
            row = {"model": name, "rule": rule, "threshold": t,
                   "oof_sensitivity": oof_m["recall"], "oof_specificity": oof_m["specificity"],
                   "oof_cost_per_patient": (COST_FN * oof_m["fn"] + COST_FP * oof_m["fp"])
                   / len(y_train)}
            row.update({f"test_{k}": v for k, v in test_m.items()})
            row["test_cost_per_patient"] = (COST_FN * test_m["fn"] + COST_FP * test_m["fp"]) / len(y_test)
            rows.append(row)
    return oof, chosen, pd.DataFrame(rows)


def plot_thresholds(y_train, oof, chosen):
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.5), sharex=True, sharey="row",
                             layout="constrained", gridspec_kw={"height_ratios": [1.6, 1]})
    grid = np.linspace(0, 1, 501)
    rule_styles = [(NEUTRAL, ":"), (TEXT_PRIMARY, "--"), (TEXT_PRIMARY, "-.")]
    for j, name in enumerate(MODEL_NAMES):
        curves = threshold_metrics(y_train, oof[name], grid)
        top, bottom = axes[0, j], axes[1, j]
        top.plot(grid, curves.sensitivity, color=CLASS_COLORS[1], lw=1.6)
        top.plot(grid, curves.specificity, color=CLASS_COLORS[0], lw=1.6)
        top.plot(grid, curves.precision, color=TEXT_SECONDARY, lw=1.4)
        bottom.plot(grid, curves.cost_per_patient, color=MODEL_COLORS[name], lw=1.6)
        cost_t = list(chosen[name].values())[2]
        min_cost = threshold_metrics(y_train, oof[name], [cost_t]).cost_per_patient.iloc[0]
        bottom.plot(cost_t, min_cost, "o", color=MODEL_COLORS[name], ms=4.5)
        bottom.annotate(f"min {min_cost:.2f}", (cost_t, min_cost), xytext=(3, -5),
                        textcoords="offset points", ha="left", va="top", fontsize=7,
                        color=TEXT_SECONDARY,
                        bbox={"fc": "white", "ec": "none", "pad": 0.6})  # keeps it legible over the guide lines
        for (color, style), t in zip(rule_styles, chosen[name].values()):
            for ax in (top, bottom):
                ax.axvline(t, color=color, ls=style, lw=1.1)
        top.set_title(name)
        bottom.set_xlabel("Decision threshold")
    axes[0, 0].set_ylabel("Rate (OOF, training)")
    axes[1, 0].set_ylabel("Expected cost\nper patient")
    # Costs explode for very high thresholds; zoom in on the useful region around the minimum.
    axes[1, 0].set_ylim(0.0, 1.2)
    names = list(chosen[MODEL_NAMES[0]])
    metric_handles = [
        Line2D([], [], color=CLASS_COLORS[1], label="Sensitivity"),
        Line2D([], [], color=CLASS_COLORS[0], label="Specificity"),
        Line2D([], [], color=TEXT_SECONDARY, label="Precision"),
    ]
    rule_handles = [Line2D([], [], color=c, ls=s, lw=1.1,
                           label=n.replace(">=", "≥").replace(" x ", " × "))
                    for (c, s), n in zip(rule_styles, names)]
    # The legend fills column by column: interleave so row 1 = metrics, row 2 = rules.
    handles = [h for pair in zip(metric_handles, rule_handles) for h in pair]
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    suptitle(fig, "Threshold analysis on out-of-fold training predictions")
    return savefig(fig, f"{PREFIX}threshold")


# --------------------------------------------------------------------- M5 calibration
def reliability_bins(y, p):
    """Equal-count (quantile) bins: mean prediction vs observed disease rate per bin."""
    bins = pd.qcut(p, N_CAL_BINS, labels=False, duplicates="drop")
    frame = pd.DataFrame({"y": np.asarray(y), "p": p, "bin": bins})
    return (frame.groupby("bin")
            .agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_fraction=("y", "mean"))
            .reset_index())


def calibration_stats(y, p, per_bin):
    """Brier score (with bootstrap CI), quantile-bin ECE and logistic calibration slope."""
    y = np.asarray(y)
    rng = np.random.default_rng(RANDOM_STATE)
    boot = [brier_score_loss(y[i], p[i])
            for i in (rng.integers(0, len(y), len(y)) for _ in range(N_BOOT))]
    gaps = (per_bin.observed_fraction - per_bin.mean_predicted).abs()
    ece = float((per_bin.n * gaps).sum() / len(y))
    # Calibration slope/intercept: unpenalised logistic regression of y on logit(p).
    # A slope < 1 means predictions are too extreme, > 1 means too timid.
    p_clip = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p_clip / (1 - p_clip))
    recal = LogisticRegression(C=np.inf).fit(logit.reshape(-1, 1), y)
    return {
        "brier": brier_score_loss(y, p),
        "brier_ci95_low": float(np.percentile(boot, 2.5)),
        "brier_ci95_high": float(np.percentile(boot, 97.5)),
        "ece_quantile_bins": ece,
        "max_bin_gap": float(gaps.max()),
        "calibration_slope": float(recal.coef_[0, 0]),
        "calibration_intercept": float(recal.intercept_[0]),
        "mean_predicted": float(p.mean()), "observed_rate": float(y.mean()),
    }


def run_calibration(final, X_test, y_test, y_train, oof):
    stats, curves, probs = {}, [], {}
    # Reference: predicting the training prevalence for everyone (no discrimination).
    prevalence = y_train.mean()
    brier_ref = brier_score_loss(y_test, np.full(len(y_test), prevalence))
    for name, model in final.items():
        p = predict_scores(model, X_test)
        probs[name] = p
        per_bin = reliability_bins(y_test, p)
        s = calibration_stats(y_test, p, per_bin)
        s["brier_skill_score"] = 1 - s["brier"] / brier_ref
        s["oof_train_brier"] = brier_score_loss(y_train, oof[name])
        stats[name] = s
        curves.append(per_bin.assign(model=name))
    stats["prevalence_only_reference"] = {"brier": brier_ref, "train_prevalence": prevalence}
    curves = pd.concat(curves, ignore_index=True)[
        ["model", "bin", "n", "mean_predicted", "observed_fraction"]]
    return stats, curves, probs


def plot_calibration(curves, stats, probs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.1), layout="constrained")
    ax1.plot([0, 1], [0, 1], color=NEUTRAL, ls="--", lw=1, label="Perfect calibration")
    for name in MODEL_NAMES:
        d = curves[curves.model == name]
        ax1.plot(d.mean_predicted, d.observed_fraction, marker="o", ms=3.5, lw=1.5,
                 color=MODEL_COLORS[name],
                 label=f"{SHORT_MODEL[name]} (Brier {stats[name]['brier']:.3f})")
    ax1.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability",
            ylabel="Observed fraction with disease")
    ax1.set_title(f"(a) Reliability diagram ({N_CAL_BINS} quantile bins)")
    ax1.legend(loc="upper left", fontsize=7.5)
    bins = np.linspace(0, 1, 21)
    for name in MODEL_NAMES:
        ax2.hist(probs[name], bins=bins, histtype="step", lw=1.5, color=MODEL_COLORS[name],
                 label=SHORT_MODEL[name])
    ax2.set(xlim=(0, 1), xlabel="Predicted probability of disease", ylabel="Test patients")
    ax2.set_title("(b) Distribution of predictions")
    ax2.legend(loc="upper center")
    suptitle(fig, "Calibration on the held-out test set")
    return savefig(fig, f"{PREFIX}calibration")


# ------------------------------------------------------------------- M6 error analysis
def build_error_frame(test_df, probs, threshold=0.5):
    """Per-patient test predictions and how many of the three models got them wrong."""
    frame = test_df[["site", *FEATURES, "target"]].copy()
    frame["n_missing_features"] = test_df[FEATURES].isna().sum(axis=1)
    for name in MODEL_NAMES:
        frame[f"prob_{SHORT_MODEL[name]}"] = probs[name]
        frame[f"wrong_{SHORT_MODEL[name]}"] = (
            (probs[name] >= threshold).astype(int) != frame.target).astype(int)
    frame["mean_prob"] = frame[[f"prob_{SHORT_MODEL[n]}" for n in MODEL_NAMES]].mean(axis=1)
    frame["n_models_wrong"] = frame[[f"wrong_{SHORT_MODEL[n]}" for n in MODEL_NAMES]].sum(axis=1)
    frame["group"] = np.select(
        [frame.n_models_wrong == 3, frame.n_models_wrong == 0],
        ["Wrong by all 3", "Wrong by none"], default="Wrong by 1-2")
    return frame


def summarise_group(g, n_total):
    out = {
        "n_patients": len(g),
        "pct_of_test_set": 100 * len(g) / n_total,
        "pct_true_disease": 100 * g.target.mean(),
        "mean_age": g.age.mean(),
        "pct_male": 100 * (g.sex == 1).mean(),
        "mean_missing_features": g.n_missing_features.mean(),
        "pct_with_3plus_missing": 100 * (g.n_missing_features >= 3).mean(),
        "mean_predicted_prob": g.mean_prob.mean(),
        "mean_abs_prob_minus_0.5": (g.mean_prob - 0.5).abs().mean(),
    }
    for site in SITES:
        out[f"pct_site_{site}"] = 100 * (g.site == site).mean()
    for code, label in CATEGORY_LABELS["cp"].items():
        out[f"pct_cp_{label}"] = 100 * (g.cp == code).mean()
    out["pct_exercise_angina"] = 100 * (g.exang == 1).mean()
    return out


def run_error_analysis(frame):
    groups = ["Wrong by all 3", "Wrong by 1-2", "Wrong by none"]
    n = len(frame)
    summary = {grp: summarise_group(frame[frame.group == grp], n) for grp in groups}
    summary["All test patients"] = summarise_group(frame, n)
    table = pd.DataFrame(summary)
    table.index.name = "characteristic"

    by_site = []
    for site in SITES:
        d = frame[frame.site == site]
        row = {"site": site, "n_test": len(d), "pct_disease": 100 * d.target.mean(),
               "pct_wrong_by_all_3": 100 * (d.n_models_wrong == 3).mean()}
        for name in MODEL_NAMES:
            row[f"error_rate_{SHORT_MODEL[name]}"] = 100 * d[f"wrong_{SHORT_MODEL[name]}"].mean()
        by_site.append(row)
    by_site = pd.DataFrame(by_site)

    all_wrong = frame[frame.group == "Wrong by all 3"]
    details = {
        "n_wrong_by_all_3": len(all_wrong),
        "all_3_false_negatives": int((all_wrong.target == 1).sum()),
        "all_3_false_positives": int((all_wrong.target == 0).sum()),
        "n_wrong_by_1_2": int((frame.group == "Wrong by 1-2").sum()),
        "n_wrong_by_none": int((frame.group == "Wrong by none").sum()),
        "errors_per_model": {name: int(frame[f"wrong_{SHORT_MODEL[name]}"].sum())
                             for name in MODEL_NAMES},
        "pairwise_error_overlap": {
            f"{SHORT_MODEL[a]}&{SHORT_MODEL[b]}": int(
                (frame[f"wrong_{SHORT_MODEL[a]}"] & frame[f"wrong_{SHORT_MODEL[b]}"]).sum())
            for i, a in enumerate(MODEL_NAMES) for b in MODEL_NAMES[i + 1:]
        },
    }
    return table, by_site, details


def plot_error_analysis(by_site, frame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.0), layout="constrained",
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    x = np.arange(len(SITES))
    width = 0.26
    rate_cols = [f"error_rate_{SHORT_MODEL[name]}" for name in MODEL_NAMES]
    for k, (name, col) in enumerate(zip(MODEL_NAMES, rate_cols)):
        ax1.bar(x + (k - 1) * width, by_site[col], width, color=MODEL_COLORS[name],
                label=SHORT_MODEL[name])
    # Two-line tick labels so the long site names do not collide.
    ticks = [f"{s.replace('Long Beach VA', 'Long Beach')}\n"
             f"{'VA, ' if s == 'Long Beach VA' else ''}n={n}"
             for s, n in zip(by_site.site, by_site.n_test)]
    ax1.set_xticks(x, ticks, fontsize=7.5)
    ax1.set_ylim(0, 1.3 * by_site[rate_cols].to_numpy().max())  # headroom for the legend
    ax1.set_ylabel("Misclassified (%)")
    ax1.set_title("(a) Error rate by site (threshold 0.5)")
    ax1.grid(axis="x", visible=False)
    ax1.legend(loc="upper left", ncol=3)

    # (b) Each patient's mean predicted probability, grouped by how many models get it
    # wrong: separates borderline cases from confident, shared mistakes.
    groups = ["Wrong by none", "Wrong by 1-2", "Wrong by all 3"]
    rng = np.random.default_rng(RANDOM_STATE)
    for pos, grp in enumerate(groups):
        d = frame[frame.group == grp]
        jitter = rng.uniform(-0.25, 0.25, len(d))
        for label in (0, 1):
            m = (d.target == label).to_numpy()
            ax2.scatter(pos + jitter[m], d.mean_prob.to_numpy()[m], s=9, alpha=0.75, lw=0,
                        color=CLASS_COLORS[label],
                        label=f"True: {CLASS_NAMES[label].lower()}" if pos == 0 else None)
    ax2.axhline(0.5, color=NEUTRAL, ls="--", lw=1)
    counts = frame.group.value_counts()
    ax2.set_xticks(range(len(groups)), [f"{short}\n(n={counts.get(g, 0)})" for short, g
                                        in zip(["None", "1-2", "All 3"], groups)])
    ax2.set(xlim=(-0.5, len(groups) - 0.5), ylim=(0, 1.18),
            xlabel="Models that misclassify the patient", ylabel="Mean predicted probability")
    ax2.set_yticks(np.linspace(0, 1, 6))
    ax2.set_title("(b) Confidence of the predictions")
    ax2.grid(axis="x", visible=False)
    ax2.legend(loc="upper center", ncol=2, fontsize=7.5, handletextpad=0.1, columnspacing=0.8)
    suptitle(fig, "Error analysis on the held-out test set")
    return savefig(fig, f"{PREFIX}error_analysis")


# ------------------------------------------------------------------------------- main
def main():
    start = time.time()
    df = load_dataset()
    train_df, test_df = split_data(df)
    X_train, y_train = get_xy(train_df)
    X_test, y_test = get_xy(test_df)
    best = get_best_params()
    final = fit_final_models(X_train, y_train, best)
    results = {"setup": {
        "n_train": len(y_train), "n_test": len(y_test),
        "curve_cv": f"{CV_FOLDS}-fold stratified x {CURVE_REPEATS} repeats",
        "best_params": best,
    }}

    # M1 ----------------------------------------------------------------------------
    lc = run_learning_curves(X_train, y_train, best)
    lc_summary = summarise_learning_curves(lc)
    save_csv(lc, "learning_curves")
    plot_learning_curves(lc, lc_summary)
    results["m1_learning_curves"] = lc_summary
    print(f"[M1] learning curves done ({time.time() - start:.0f}s)")
    for name, s in lc_summary.items():
        print(f"     {name:20s} train {s['train_auc_at_max']:.3f}  CV {s['cv_auc_at_max']:.3f}"
              f"  gap {s['train_cv_gap_at_max']:.3f}  gain 50->100% {s['cv_auc_gain_50_to_100pct']:+.3f}")

    # M2 ----------------------------------------------------------------------------
    vc = run_validation_curves(X_train, y_train, best)
    vc_summary = summarise_validation_curves(vc, best)
    save_csv(vc, "validation_curves")
    plot_validation_curves(vc, best)
    results["m2_validation_curves"] = vc_summary
    print(f"[M2] validation curves done ({time.time() - start:.0f}s)")
    for key, s in vc_summary.items():
        print(f"     {key:10s} CV-best {s['cv_best_value']} ({s['cv_best_auc']:.3f}); "
              f"tuned {s['tuned_value']} ({s['cv_auc_at_tuned']:.3f}); range {s['cv_auc_range']:.3f}")

    # M3 ----------------------------------------------------------------------------
    lr_table = lr_coefficient_table(final["Logistic Regression"], X_train)
    rf_table = rf_importance_table(final["Random Forest"])
    perm = run_permutation_importance(final, X_test, y_test)
    vec, rho, pairs = rank_agreement(lr_table, rf_table, perm)
    save_csv(lr_table, "lr_coefficients")
    save_csv(rf_table, "rf_importance")
    save_csv(perm, "permutation_importance")
    vec.index.name = "feature"
    save_csv(vec, "importance_by_feature", index=True)
    save_csv(rho, "rank_agreement_matrix", index=True)
    save_csv(pairs, "rank_agreement_pairs")
    plot_lr_coefficients(lr_table)
    plot_rf_importance(rf_table)
    plot_permutation_importance(perm)
    top_perm = {name: perm[perm.model == name].nlargest(5, "importance_mean")
                .feature.tolist() for name in MODEL_NAMES}
    results["m3_feature_importance"] = {
        "lr_top10_std_coefficients": lr_table.head(10)[
            ["label", "coefficient", "std_coefficient", "odds_ratio_per_unit", "odds_ratio_per_sd"]
        ].to_dict(orient="records"),
        "rf_top10_impurity": rf_table.head(10)[["label", "impurity_importance"]]
        .to_dict(orient="records"),
        "rf_impurity_by_feature": vec["RF impurity (sum per feature)"].sort_values(ascending=False)
        .to_dict(),
        "permutation_top5_features": top_perm,
        "permutation_importance": {
            name: perm[perm.model == name].set_index("feature")[["importance_mean", "importance_std"]]
            .sort_values("importance_mean", ascending=False).to_dict(orient="index")
            for name in MODEL_NAMES},
        "spearman_pairs": pairs.to_dict(orient="records"),
        # Context for the cholesterol result: a missing value marks Switzerland/VA records,
        # which have a much higher disease rate, so 'chol' carries site information.
        "train_chol_missingness": {
            "n_missing": int(train_df.chol.isna().sum()),
            "disease_rate_if_missing": train_df.loc[train_df.chol.isna(), "target"].mean(),
            "disease_rate_if_observed": train_df.loc[train_df.chol.notna(), "target"].mean(),
        },
    }
    print(f"[M3] feature importance done ({time.time() - start:.0f}s)")
    for name, feats in top_perm.items():
        print(f"     {name:20s} top-5 permutation: {', '.join(feats)}")
    print("     Spearman (permutation): " + "; ".join(
        f"{r.a.split()[-1]}-{r.b.split()[-1]} {r.spearman_rho:.2f}"
        for r in pairs.itertuples() if r.a.startswith("Perm") and r.b.startswith("Perm")))

    # M4 ----------------------------------------------------------------------------
    oof, chosen, thr = run_threshold_analysis(X_train, y_train, X_test, y_test, best, final)
    save_csv(thr, "threshold")
    plot_thresholds(y_train, oof, chosen)
    results["m4_threshold"] = {
        "cost_fn": COST_FN, "cost_fp": COST_FP, "target_sensitivity": TARGET_SENSITIVITY,
        "bayes_threshold_if_calibrated": COST_FP / (COST_FP + COST_FN),
        "rows": thr.to_dict(orient="records"),
    }
    print(f"[M4] threshold analysis done ({time.time() - start:.0f}s)")
    for r in thr.itertuples():
        print(f"     {r.model:20s} {r.rule:24s} t={r.threshold:.3f}  test sens {r.test_recall:.3f}"
              f"  spec {r.test_specificity:.3f}  FN {r.test_fn:2d} FP {r.test_fp:2d}"
              f"  cost {r.test_cost_per_patient:.3f}")

    # M5 ----------------------------------------------------------------------------
    cal_stats, cal_curves, probs = run_calibration(final, X_test, y_test, y_train, oof)
    save_csv(cal_curves, "calibration_curves")
    save_csv(pd.DataFrame({k: v for k, v in cal_stats.items() if k in MODEL_NAMES}).T
             .rename_axis("model"), "calibration_summary", index=True)
    plot_calibration(cal_curves, cal_stats, probs)
    results["m5_calibration"] = cal_stats
    print(f"[M5] calibration done ({time.time() - start:.0f}s)")
    for name in MODEL_NAMES:
        s = cal_stats[name]
        print(f"     {name:20s} Brier {s['brier']:.3f} [{s['brier_ci95_low']:.3f}, "
              f"{s['brier_ci95_high']:.3f}]  ECE {s['ece_quantile_bins']:.3f}  "
              f"slope {s['calibration_slope']:.2f}")

    # M6 ----------------------------------------------------------------------------
    frame = build_error_frame(test_df, probs)
    err_table, err_site, err_details = run_error_analysis(frame)
    save_csv(err_table, "error_analysis", index=True)
    save_csv(err_site, "error_by_site")
    save_csv(frame, "error_patients")
    plot_error_analysis(err_site, frame)
    results["m6_error_analysis"] = {
        **err_details,
        "group_summary": err_table.to_dict(),
        "by_site": err_site.set_index("site").to_dict(orient="index"),
    }
    print(f"[M6] error analysis done ({time.time() - start:.0f}s)")
    print(f"     wrong by all 3: {err_details['n_wrong_by_all_3']} "
          f"(FN {err_details['all_3_false_negatives']}, FP {err_details['all_3_false_positives']}), "
          f"by 1-2: {err_details['n_wrong_by_1_2']}, by none: {err_details['n_wrong_by_none']}")

    results["runtime_seconds"] = time.time() - start
    save_json(rounded(results), RESULTS_DIR / f"{PREFIX}results.json")
    print(f"Saved figures/{PREFIX}*.png and results/{PREFIX}*  ({time.time() - start:.0f}s total)")


if __name__ == "__main__":
    main()
