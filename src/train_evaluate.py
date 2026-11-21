"""Main experiment: tune, cross-validate, test and compare the three classifiers.

Pipeline of this script
  1. Hold-out split (80/20, stratified on site and target) and grid search of each
     model on the training set; the full grid results are saved for the report.
  2. Repeated stratified 5x10 cross-validation of the tuned pipelines (plus a
     majority-class baseline) on the training set, and a nested CV that also
     re-tunes inside every outer fold to give an unbiased performance estimate.
  3. One final fit on the whole training set and a single evaluation on the
     held-out test set, with 95% bootstrap confidence intervals and per-site metrics.
  4. Statistical comparison of the models (corrected resampled t-test on CV folds,
     exact McNemar and DeLong tests on the test set).
  5. Figures and a single JSON file with every number quoted in the report.

Run from the project root:  python src/train_evaluate.py
"""
import json
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.patches import Rectangle
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from config import CV_FOLDS, CV_REPEATS, RANDOM_STATE, RESULTS_DIR
from data import SITES, get_xy, load_dataset, split_data
from evaluation import bootstrap_ci, compute_metrics, predict_scores, save_json
from models import (
    BASELINE_NAME, MODEL_NAMES, build_tuned_pipeline, get_best_params,
    make_baseline, tune_model,
)
from plotting import (
    CLASS_NAMES, MODEL_COLORS, NEUTRAL, SEQUENTIAL_CMAP, TEXT_PRIMARY,
    TEXT_SECONDARY, savefig,
)

ALL_MODELS = MODEL_NAMES + [BASELINE_NAME]
CV_METRICS = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]
CI_METRICS = CV_METRICS  # bootstrap CIs for every bar shown in the test-metrics figure
N_BOOT = 2000
ALPHA = 0.05
SHORT_LABELS = {
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
    "specificity": "Specificity", "f1": "F1 score", "roc_auc": "ROC-AUC",
    "avg_precision": "Avg. precision",
}
SHORT_MODEL = {
    "Logistic Regression": "LR", "SVM (RBF)": "SVM", "Random Forest": "RF",
    BASELINE_NAME: "Baseline",
}


# --------------------------------------------------------------------------- helpers
def slug(name):
    """'SVM (RBF)' -> 'svm_rbf' (used in file names)."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def short_param(key):
    """'clf__estimator__C' -> 'C'."""
    return key.split("__")[-1]


def jsonable_params(params):
    """Round-trip through JSON so None/int/str compare the same way as the cache file."""
    return json.loads(json.dumps(params))


def rounded(df, ndigits=4):
    """Round every float column (keeps integer and text columns untouched)."""
    out = df.copy()
    float_cols = out.select_dtypes(include="float").columns
    out[float_cols] = out[float_cols].round(ndigits)
    return out


def fit_and_score(pipeline, X_train, y_train, X_test, y_test):
    """Fit a fresh copy of the pipeline and return metrics on the held-out part."""
    model = clone(pipeline).fit(X_train, y_train)
    return compute_metrics(y_test, predict_scores(model, X_test))


# ------------------------------------------------------------- 1. grid search tables
def run_grid_searches(X_train, y_train, cached_params):
    """Re-run the grid search of every model and save the full cv_results_ tables.

    The tuned parameters themselves are cached in results/best_params.json (shared by
    all experiment scripts); here we only check that this run reproduces them.
    """
    tables, matches = {}, {}
    for name in MODEL_NAMES:
        search = tune_model(name, X_train, y_train)
        res = search.cv_results_
        rows = []
        for i, params in enumerate(res["params"]):
            row = {short_param(k): ("None" if v is None else v) for k, v in params.items()}
            if name == "Logistic Regression":
                # l1_ratio is 0 or 1 here, i.e. a pure L2 or L1 penalty
                row["penalty"] = "L1" if params["clf__l1_ratio"] == 1 else "L2"
            row["mean_cv_auc"] = res["mean_test_score"][i]
            row["std_cv_auc"] = res["std_test_score"][i]
            row["rank"] = int(res["rank_test_score"][i])
            rows.append(row)
        table = pd.DataFrame(rows).sort_values("rank", kind="stable")
        rounded(table).to_csv(RESULTS_DIR / f"grid_search_{slug(name)}.csv", index=False)
        tables[name] = table

        found = jsonable_params(search.best_params_)
        matches[name] = found == cached_params[name]
        if not matches[name]:
            print(f"  WARNING: {name} best params {found} differ from cache {cached_params[name]}")
        print(f"  {name:<20} best CV AUC {search.best_score_:.4f}  {found}")
    return tables, matches


# ----------------------------------------------------- 2. repeated and nested CV
def repeated_cv(pipelines, X, y):
    """Repeated stratified k-fold CV of already-tuned pipelines on the training set.

    Every model is scored on exactly the same 50 train/validation splits, so fold
    scores are paired and can be compared with a corrected paired t-test.

    Note: the hyperparameters were chosen by CV on this same training set, so this
    estimate is mildly optimistic (selection bias); nested_cv() below removes that bias.
    """
    splitter = RepeatedStratifiedKFold(
        n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE
    )
    splits = list(splitter.split(X, y))
    jobs = [(name, i, tr, va) for name in pipelines for i, (tr, va) in enumerate(splits)]
    results = Parallel(n_jobs=-1)(
        delayed(fit_and_score)(pipelines[name], X.iloc[tr], y[tr], X.iloc[va], y[va])
        for name, i, tr, va in jobs
    )
    rows = [
        {"model": name, "repeat": i // CV_FOLDS + 1, "fold": i % CV_FOLDS + 1,
         **{m: res[m] for m in CV_METRICS}}
        for (name, i, _, _), res in zip(jobs, results)
    ]
    return pd.DataFrame(rows)


def summarise_cv(scores):
    """Mean and standard deviation of every metric per model (wide table)."""
    agg = scores.groupby("model", sort=False)[CV_METRICS].agg(["mean", "std"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    return agg.reset_index()


def nested_cv(X, y):
    """Nested CV: an outer 5-fold loop for evaluation, grid search in each outer fold.

    Because tuning only ever sees the outer-training part, the outer-fold scores are
    an unbiased estimate of how the whole "tune then fit" procedure generalises.
    """
    outer = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for fold, (tr, va) in enumerate(outer.split(X, y), start=1):
        for name in MODEL_NAMES:
            search = tune_model(name, X.iloc[tr], y[tr])
            met = compute_metrics(y[va], predict_scores(search, X.iloc[va]))
            params = {short_param(k): v for k, v in search.best_params_.items()}
            rows.append({
                "model": name, "outer_fold": fold, "inner_best_auc": search.best_score_,
                **{m: met[m] for m in CV_METRICS}, "best_params": json.dumps(params),
            })
        print(f"  outer fold {fold}/{CV_FOLDS} done")
    return pd.DataFrame(rows)


# --------------------------------------------------------------- 3. test evaluation
def evaluate_on_test(fitted, X_test, y_test):
    """Point estimates and 95% bootstrap CIs on the held-out test set."""
    scores = {name: predict_scores(model, X_test) for name, model in fitted.items()}
    metrics = {name: compute_metrics(y_test, s) for name, s in scores.items()}

    # bootstrap_ci recomputes all metrics for each resample, so the 24
    # (model, metric) jobs are run in parallel; each uses the same seed, so all
    # metrics of a model are computed on identical resamples.
    jobs = [(name, m) for name in fitted for m in CI_METRICS]
    cis = Parallel(n_jobs=-1)(
        delayed(bootstrap_ci)(y_test, scores[name], m, n_boot=N_BOOT, alpha=ALPHA)
        for name, m in jobs
    )
    ci = {name: {} for name in fitted}
    for (name, m), (lo, hi) in zip(jobs, cis):
        ci[name][m] = (lo, hi)
    return scores, metrics, ci


def metrics_by_site(test_df, y_test, scores):
    """Test metrics within each hospital (AUC/recall/specificity only where defined)."""
    rows = []
    for name, s in scores.items():
        for site in SITES:
            mask = (test_df["site"] == site).to_numpy()
            yt, ys = y_test[mask], s[mask]
            met = compute_metrics(yt, ys)
            n_pos = int(yt.sum())
            n_neg = int(len(yt) - n_pos)
            rows.append({
                "model": name, "site": site, "n": int(mask.sum()),
                "n_disease": n_pos, "n_no_disease": n_neg,
                "prevalence": yt.mean(),
                "accuracy": met["accuracy"],
                # compute_metrics returns 0 recall when there are no positives; NaN is honest
                "recall": met["recall"] if n_pos else np.nan,
                "specificity": met["specificity"] if n_neg else np.nan,
                "roc_auc": met["roc_auc"],
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------- 4. statistical comparison
def corrected_resampled_ttest(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for paired CV scores.

    The ordinary paired t-test treats the k*r fold scores as independent, but the
    training sets overlap heavily, which makes it far too liberal. The correction
    inflates the variance by the factor (1/J + n_test/n_train).
    """
    d = np.asarray(a) - np.asarray(b)
    J = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return float(d.mean()), float("nan"), 1.0
    t = d.mean() / np.sqrt((1 / J + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=J - 1)
    return float(d.mean()), float(t), float(p)


def mcnemar_exact(y_true, pred_a, pred_b):
    """Exact (binomial) McNemar test on the discordant test-set predictions."""
    correct_a, correct_b = pred_a == y_true, pred_b == y_true
    b = int(np.sum(correct_a & ~correct_b))  # only model A right
    c = int(np.sum(~correct_a & correct_b))  # only model B right
    p = 1.0 if b + c == 0 else stats.binomtest(min(b, c), b + c, 0.5).pvalue
    return b, c, float(p)


def _structural_components(y_true, score):
    """DeLong placements: V10 (per positive) and V01 (per negative)."""
    pos, neg = score[y_true == 1], score[y_true == 0]
    # psi = 1 if positive ranked above negative, 0.5 for ties
    psi = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    return psi.mean(axis=1), psi.mean(axis=0)


def delong_test(y_true, score_a, score_b):
    """DeLong et al. (1988) test for two correlated ROC-AUCs on the same test set."""
    v10_a, v01_a = _structural_components(y_true, score_a)
    v10_b, v01_b = _structural_components(y_true, score_b)
    auc_a, auc_b = v10_a.mean(), v10_b.mean()
    s10 = np.cov(np.vstack([v10_a, v10_b]))
    s01 = np.cov(np.vstack([v01_a, v01_b]))
    s = s10 / len(v10_a) + s01 / len(v01_a)
    var_diff = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    if var_diff <= 0:
        return float(auc_a - auc_b), float("nan"), 1.0
    z = (auc_a - auc_b) / np.sqrt(var_diff)
    return float(auc_a - auc_b), float(z), float(2 * stats.norm.sf(abs(z)))


def holm(pvals):
    """Holm step-down adjustment within one family of comparisons."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def compare_models(cv_scores, y_test, test_scores, n_train_fold, n_val_fold):
    """All pairwise tests between the three tuned models (one row per pair and test)."""
    pairs = [(a, b) for i, a in enumerate(MODEL_NAMES) for b in MODEL_NAMES[i + 1:]]
    rows = []
    for a, b in pairs:
        fa = cv_scores[cv_scores["model"] == a].sort_values(["repeat", "fold"])
        fb = cv_scores[cv_scores["model"] == b].sort_values(["repeat", "fold"])
        for metric in ["roc_auc", "accuracy"]:
            diff, t, p = corrected_resampled_ttest(
                fa[metric].to_numpy(), fb[metric].to_numpy(), n_train_fold, n_val_fold)
            rows.append({"model_a": a, "model_b": b,
                         "test": "Corrected resampled t-test (repeated CV)",
                         "metric": metric, "difference_a_minus_b": diff,
                         "statistic_name": "t (df = J - 1)", "statistic": t,
                         "p_value": p, "n_discordant_a_b": ""})

        pred_a = (test_scores[a] >= 0.5).astype(int)
        pred_b = (test_scores[b] >= 0.5).astype(int)
        n_ab, n_ba, p = mcnemar_exact(y_test, pred_a, pred_b)
        acc_diff = np.mean(pred_a == y_test) - np.mean(pred_b == y_test)
        rows.append({"model_a": a, "model_b": b, "test": "Exact McNemar (test set)",
                     "metric": "accuracy", "difference_a_minus_b": acc_diff,
                     "statistic_name": "min(b, c)", "statistic": float(min(n_ab, n_ba)),
                     "p_value": p,
                     "n_discordant_a_b": f"{n_ab}/{n_ba}"})

        diff, z, p = delong_test(y_test, test_scores[a], test_scores[b])
        rows.append({"model_a": a, "model_b": b, "test": "DeLong (test set)",
                     "metric": "roc_auc", "difference_a_minus_b": diff,
                     "statistic_name": "z", "statistic": z, "p_value": p,
                     "n_discordant_a_b": ""})

    table = pd.DataFrame(rows)
    # Holm correction across the three model pairs of each (test, metric) family
    table["p_holm"] = table.groupby(["test", "metric"])["p_value"].transform(holm)
    table["significant_0.05"] = table["p_holm"] < ALPHA
    table["interpretation"] = [interpret(r) for r in table.itertuples()]
    return table


def interpret(r):
    """One-line plain-English reading of a comparison row."""
    better = r.model_a if r.difference_a_minus_b > 0 else r.model_b
    where = "CV" if "CV" in r.test else "test"
    metric = {"accuracy": "accuracy", "roc_auc": "ROC-AUC"}[r.metric]
    if r.p_holm < ALPHA:
        return (f"{better} has significantly higher {where} {metric} "
                f"(diff {r.difference_a_minus_b:+.4f}, Holm p = {r.p_holm:.3g}).")
    return (f"No significant difference in {where} {metric} between {r.model_a} and "
            f"{r.model_b} (diff {r.difference_a_minus_b:+.4f}, Holm p = {r.p_holm:.3g}).")


# -------------------------------------------------------------------- 5. figures
def plot_cv_boxplots(cv_scores):
    """Fold-level ROC-AUC and F1 from repeated CV, one box per model."""
    order = ALL_MODELS[::-1]  # matplotlib draws the first box at the bottom
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7), sharey=True, layout="constrained")
    rng = np.random.default_rng(RANDOM_STATE)
    for ax, metric in zip(axes, ["roc_auc", "f1"]):
        data = [cv_scores.loc[cv_scores["model"] == m, metric].to_numpy() for m in order]
        bp = ax.boxplot(data, orientation="horizontal", widths=0.55, patch_artist=True,
                        showfliers=False, tick_labels=order,
                        medianprops={"color": TEXT_PRIMARY, "linewidth": 1.4},
                        whiskerprops={"color": TEXT_SECONDARY, "linewidth": 1},
                        capprops={"color": TEXT_SECONDARY, "linewidth": 1})
        for patch, m in zip(bp["boxes"], order):
            patch.set_facecolor(MODEL_COLORS[m])
            patch.set_alpha(0.35)
            patch.set_edgecolor(MODEL_COLORS[m])
        # individual folds on top so the spread (and the 50 folds) stay visible
        for i, (vals, m) in enumerate(zip(data, order), start=1):
            jitter = rng.uniform(-0.17, 0.17, len(vals))
            ax.scatter(vals, i + jitter, s=7, color=MODEL_COLORS[m], alpha=0.7,
                       linewidths=0, zorder=3)
        ax.set_xlabel(f"Fold {SHORT_LABELS[metric]}")
        ax.set_title(f"{SHORT_LABELS[metric]} per validation fold")
        ax.grid(axis="y", visible=False)
    axes[0].set_xlim(0.45, 1.0)
    axes[1].set_xlim(0.55, 1.0)
    fig.suptitle(f"Repeated stratified CV on the training set "
                 f"({CV_FOLDS} folds x {CV_REPEATS} repeats = {CV_FOLDS * CV_REPEATS} folds)",
                 fontsize=10.5, fontweight="bold")
    return savefig(fig, "results_cv_boxplot")


def plot_roc_curves(y_test, scores, metrics):
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    for name in MODEL_NAMES:
        fpr, tpr, _ = roc_curve(y_test, scores[name])
        ax.plot(fpr, tpr, color=MODEL_COLORS[name], lw=1.8,
                label=f"{name} ({metrics[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=NEUTRAL, label="Chance (0.500)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(f"ROC curves, test set (n = {len(y_test)})")
    # opaque legend box: otherwise the chance diagonal runs through the legend text
    ax.legend(title="Model (ROC-AUC)", title_fontsize=7.5, fontsize=7.5,
              loc="lower right", handlelength=1.5, frameon=True, facecolor="white",
              edgecolor="none", framealpha=0.95)
    fig.tight_layout()
    return savefig(fig, "results_roc_curves")


def plot_pr_curves(y_test, scores, metrics):
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    for name in MODEL_NAMES:
        prec, rec, _ = precision_recall_curve(y_test, scores[name])
        ax.plot(rec, prec, color=MODEL_COLORS[name], lw=1.8,
                label=f"{name} ({metrics[name]['avg_precision']:.3f})")
    prevalence = float(np.mean(y_test))
    ax.axhline(prevalence, ls="--", lw=1, color=NEUTRAL,
               label=f"No skill = prevalence ({prevalence:.3f})")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision-recall, test set")
    ax.legend(title="Model (average precision)", title_fontsize=7.5, fontsize=7.5,
              loc="lower left", handlelength=1.5)
    fig.tight_layout()
    return savefig(fig, "results_pr_curves")


def plot_confusion_matrices(metrics):
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5), sharey=True)
    labels = [CLASS_NAMES[0], CLASS_NAMES[1]]
    for ax, name in zip(axes, MODEL_NAMES):
        m = metrics[name]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        row_pct = cm / cm.sum(axis=1, keepdims=True) * 100
        im = ax.imshow(row_pct, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=100)
        for i in range(2):
            for j in range(2):
                color = "white" if row_pct[i, j] > 55 else TEXT_PRIMARY
                ax.text(j, i, f"{cm[i, j]}\n({row_pct[i, j]:.1f}%)", ha="center",
                        va="center", fontsize=8.5, color=color)
        ax.set_xticks([0, 1], labels)
        ax.set_yticks([0, 1], labels)
        ax.set_xlabel("Predicted")
        ax.set_title(f"{name}\naccuracy {m['accuracy']:.3f}", fontsize=9.5)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("Actual")
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("% of actual class", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.outline.set_visible(False)
    fig.suptitle("Test-set confusion matrices (threshold 0.5): counts and row %",
                 fontsize=10.5, fontweight="bold", x=0.47, y=1.04)
    return savefig(fig, "results_confusion_matrices")


def plot_test_metrics(metrics, ci):
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    x = np.arange(len(CV_METRICS))
    width = 0.2
    for k, name in enumerate(ALL_MODELS):
        vals = np.array([metrics[name][m] for m in CV_METRICS])
        lo = np.array([ci[name][m][0] for m in CV_METRICS])
        hi = np.array([ci[name][m][1] for m in CV_METRICS])
        pos = x + (k - 1.5) * width
        ax.bar(pos, vals, width, color=MODEL_COLORS[name], label=name)
        for xp, v in zip(pos, vals):
            if v == 0:  # e.g. baseline specificity: label it so the bar does not look missing
                ax.text(xp, 0.01, "0", ha="center", va="bottom", fontsize=7,
                        color=TEXT_SECONDARY)
        # the baseline's constant predictions give zero-width CIs (e.g. AUC 0.5);
        # drawing those would leave stray caps, so only real intervals get whiskers
        has_ci = hi > lo
        err = np.vstack([np.clip(vals - lo, 0, None), np.clip(hi - vals, 0, None)])
        ax.errorbar(pos[has_ci], vals[has_ci], yerr=err[:, has_ci], fmt="none",
                    ecolor=TEXT_PRIMARY, elinewidth=0.8, capsize=1.8, capthick=0.8)
    ax.set_xticks(x, [SHORT_LABELS[m] for m in CV_METRICS])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score on held-out test set")
    ax.grid(axis="x", visible=False)
    n_test = sum(metrics[MODEL_NAMES[0]][k] for k in ["tn", "fp", "fn", "tp"])
    ax.set_title(f"Test-set performance with 95% bootstrap CIs "
                 f"(n = {n_test}, {N_BOOT} resamples)", pad=24)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=8.5,
              handlelength=1.2, columnspacing=1.2)
    fig.tight_layout()
    return savefig(fig, "results_test_metrics")


def _heatmap(ax, values, row_labels, col_labels, vmin, vmax, best):
    """Annotated heatmap of mean CV AUC with the selected cell outlined."""
    im = ax.imshow(values, cmap=SEQUENTIAL_CMAP, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            frac = (values[i, j] - vmin) / (vmax - vmin)
            ax.text(j, i, f"{values[i, j]:.3f}".lstrip("0"), ha="center", va="center",
                    fontsize=6, color="white" if frac > 0.6 else TEXT_PRIMARY)
    ax.add_patch(Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1, fill=False,
                           edgecolor=TEXT_PRIMARY, lw=1.6))
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def plot_tuning(tables, best_params):
    """Grid-search landscapes: how mean 5-fold CV AUC depends on each hyperparameter."""
    fig = plt.figure(figsize=(6.5, 3.1), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 0.95])
    ax_lr, ax_svm, ax_rf = (fig.add_subplot(gs[0, i]) for i in range(3))

    # (a) logistic regression: AUC vs C for each penalty
    lr = tables["Logistic Regression"]
    styles = {"L2": {"color": MODEL_COLORS["Logistic Regression"], "ls": "-", "marker": "o"},
              "L1": {"color": TEXT_SECONDARY, "ls": "--", "marker": "s"}}
    for pen in ["L2", "L1"]:
        sub = lr[lr["penalty"] == pen].sort_values("C")
        ax_lr.plot(sub["C"], sub["mean_cv_auc"], ms=4, lw=1.5, label=pen, **styles[pen])
    lr_best = best_params["Logistic Regression"]
    best_row = lr[(lr["C"] == lr_best["clf__C"]) & (lr["l1_ratio"] == lr_best["clf__l1_ratio"])]
    ax_lr.scatter(best_row["C"], best_row["mean_cv_auc"], s=70, facecolors="none",
                  edgecolors=TEXT_PRIMARY, linewidths=1.3, zorder=4, label="Selected")
    ax_lr.set_xscale("log")
    ax_lr.set_xlabel("Inverse regularisation C")
    ax_lr.set_ylabel("Mean 5-fold CV ROC-AUC")
    ax_lr.set_title("Logistic Regression", fontsize=9.5)
    ax_lr.legend(loc="lower right", fontsize=7.5, title="Penalty", title_fontsize=7.5,
                 handlelength=1.6)

    # (b) SVM: C x gamma grid (gamma='scale' kept as its own column)
    svm = tables["SVM (RBF)"].copy()
    svm["gamma"] = svm["gamma"].astype(str)
    gammas = ["0.001", "0.01", "0.05", "0.1", "scale"]
    cs = sorted(svm["C"].unique())
    svm_grid = (svm.pivot(index="C", columns="gamma", values="mean_cv_auc")
                .loc[cs, gammas].to_numpy())

    # (c) RF: max_depth x min_samples_leaf for max_features='sqrt'
    rf = tables["Random Forest"].copy()
    rf = rf[rf["max_features"].astype(str) == "sqrt"]
    rf["max_depth"] = rf["max_depth"].astype(str)
    depths = ["4", "6", "10", "None"]
    leaves = sorted(rf["min_samples_leaf"].unique())
    rf_grid = (rf.pivot(index="max_depth", columns="min_samples_leaf", values="mean_cv_auc")
               .loc[depths, leaves].to_numpy())

    # one shared colour scale so the two heatmaps can be compared directly
    vmin = float(min(svm_grid.min(), rf_grid.min()))
    vmax = float(max(svm_grid.max(), rf_grid.max()))

    svm_best = best_params["SVM (RBF)"]
    best_svm = (cs.index(svm_best["clf__estimator__C"]),
                gammas.index(str(svm_best["clf__estimator__gamma"])))
    im = _heatmap(ax_svm, svm_grid, [f"{c:g}" for c in cs], gammas, vmin, vmax, best_svm)
    ax_svm.set_xlabel("Kernel width gamma")
    ax_svm.set_ylabel("C")
    ax_svm.set_title("SVM (RBF)", fontsize=9.5)
    ax_svm.tick_params(axis="x", labelsize=7, rotation=45)

    rf_best = best_params["Random Forest"]
    best_rf = (depths.index(str(rf_best["clf__max_depth"])),
               leaves.index(rf_best["clf__min_samples_leaf"]))
    _heatmap(ax_rf, rf_grid, depths, [str(v) for v in leaves], vmin, vmax, best_rf)
    ax_rf.set_xlabel("min_samples_leaf")
    ax_rf.set_ylabel("max_depth")
    ax_rf.set_title("Random Forest\n(max_features = sqrt)", fontsize=9.5)

    cbar = fig.colorbar(im, ax=[ax_svm, ax_rf], fraction=0.05, pad=0.03, aspect=25)
    cbar.set_label("Mean CV ROC-AUC", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_visible(False)
    fig.suptitle("Hyperparameter search (5-fold CV on the training set; selected = outlined)",
                 fontsize=10.5, fontweight="bold")
    return savefig(fig, "results_tuning")


# ---------------------------------------------------------------------------- main
def main():
    start = time.time()
    df = load_dataset()
    train_df, test_df = split_data(df)
    X_train, y_train = get_xy(train_df)
    X_test, y_test = get_xy(test_df)
    print(f"Train {len(train_df)} rows (prevalence {y_train.mean():.3f}), "
          f"test {len(test_df)} rows (prevalence {y_test.mean():.3f})")

    # 1. grid search -------------------------------------------------------------
    print("\n[1] Grid search (5-fold CV, ROC-AUC) on the training set")
    best_params = get_best_params()
    tables, params_match = run_grid_searches(X_train, y_train, best_params)

    pipelines = {name: build_tuned_pipeline(name, best_params[name]) for name in MODEL_NAMES}
    pipelines[BASELINE_NAME] = make_baseline()
    # The CV folds below are already fitted in parallel, so each forest is grown on a
    # single thread: this avoids oversubscribing the CPU with nested thread pools.
    # n_jobs does not change the fitted trees (random_state is fixed).
    pipelines["Random Forest"].set_params(clf__n_jobs=1)

    # 2. repeated and nested CV ----------------------------------------------------
    print(f"\n[2] Repeated stratified CV ({CV_FOLDS} folds x {CV_REPEATS} repeats)")
    cv_scores = repeated_cv(pipelines, X_train, y_train)
    rounded(cv_scores).to_csv(RESULTS_DIR / "cv_scores.csv", index=False)
    cv_summary = summarise_cv(cv_scores)
    rounded(cv_summary).to_csv(RESULTS_DIR / "cv_summary.csv", index=False)
    for r in cv_summary.itertuples():
        print(f"  {r.model:<20} AUC {r.roc_auc_mean:.3f} +/- {r.roc_auc_std:.3f}   "
              f"acc {r.accuracy_mean:.3f} +/- {r.accuracy_std:.3f}   "
              f"F1 {r.f1_mean:.3f} +/- {r.f1_std:.3f}")

    print("\n[2b] Nested CV (outer 5-fold, grid search inside each outer fold)")
    nested = nested_cv(X_train, y_train)
    rounded(nested).to_csv(RESULTS_DIR / "cv_nested.csv", index=False)
    nested_summary = nested.groupby("model", sort=False)[CV_METRICS].agg(["mean", "std"])
    for name in MODEL_NAMES:
        print(f"  {name:<20} AUC {nested_summary.loc[name, ('roc_auc', 'mean')]:.3f} +/- "
              f"{nested_summary.loc[name, ('roc_auc', 'std')]:.3f}   acc "
              f"{nested_summary.loc[name, ('accuracy', 'mean')]:.3f} +/- "
              f"{nested_summary.loc[name, ('accuracy', 'std')]:.3f}")

    # 3. final fit and test-set evaluation -------------------------------------------
    print("\n[3] Final fit on the full training set, evaluation on the test set")
    fitted = {name: clone(p).fit(X_train, y_train) for name, p in pipelines.items()}
    test_scores, test_metrics, test_ci = evaluate_on_test(fitted, X_test, y_test)

    metric_cols = CV_METRICS + ["avg_precision"]
    rows = []
    for name in ALL_MODELS:
        row = {"model": name}
        for m in metric_cols:
            row[m] = test_metrics[name][m]
            if m in test_ci[name]:
                row[f"{m}_ci_low"], row[f"{m}_ci_high"] = test_ci[name][m]
        row.update({k: test_metrics[name][k] for k in ["tn", "fp", "fn", "tp"]})
        rows.append(row)
    test_table = pd.DataFrame(rows)
    rounded(test_table).to_csv(RESULTS_DIR / "test_metrics.csv", index=False)
    test_json = {
        name: {
            "metrics": {m: round(float(test_metrics[name][m]), 4) for m in metric_cols},
            "ci95": {m: [round(v, 4) for v in test_ci[name][m]] for m in CI_METRICS},
            "confusion_matrix": {k: test_metrics[name][k] for k in ["tn", "fp", "fn", "tp"]},
        }
        for name in ALL_MODELS
    }
    save_json({"n_test": len(y_test), "threshold": 0.5, "n_bootstrap": N_BOOT,
               "models": test_json}, RESULTS_DIR / "test_metrics.json")
    for name in ALL_MODELS:
        m, c = test_metrics[name], test_ci[name]
        print(f"  {name:<20} acc {m['accuracy']:.3f} [{c['accuracy'][0]:.3f}, {c['accuracy'][1]:.3f}]  "
              f"F1 {m['f1']:.3f}  recall {m['recall']:.3f}  spec {m['specificity']:.3f}  "
              f"AUC {m['roc_auc']:.3f} [{c['roc_auc'][0]:.3f}, {c['roc_auc'][1]:.3f}]")

    by_site = metrics_by_site(test_df, y_test, test_scores)
    rounded(by_site).to_csv(RESULTS_DIR / "test_metrics_by_site.csv", index=False)

    # 4. statistical comparison -------------------------------------------------------
    print("\n[4] Pairwise model comparison")
    n_val_fold = len(y_train) // CV_FOLDS
    comparison = compare_models(cv_scores, y_test, test_scores,
                                n_train_fold=len(y_train) - n_val_fold, n_val_fold=n_val_fold)
    rounded(comparison).to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    for r in comparison.itertuples():
        print(f"  [{r.test.split(' (')[0]}, {r.metric}] {r.interpretation}")

    # 5. figures -------------------------------------------------------------------
    print("\n[5] Figures")
    paths = [
        plot_cv_boxplots(cv_scores),
        plot_roc_curves(y_test, test_scores, test_metrics),
        plot_pr_curves(y_test, test_scores, test_metrics),
        plot_confusion_matrices(test_metrics),
        plot_test_metrics(test_metrics, test_ci),
        plot_tuning(tables, best_params),
    ]
    for p in paths:
        print(f"  saved {p.name}")

    # 6. one JSON with every number the report quotes ------------------------------
    r4 = lambda v: None if pd.isna(v) else round(float(v), 4)  # noqa: E731
    main_results = {
        "data": {
            "n_total": len(df), "n_train": len(train_df), "n_test": len(test_df),
            "train_prevalence": r4(y_train.mean()), "test_prevalence": r4(y_test.mean()),
            "test_n_by_site": test_df["site"].value_counts().reindex(SITES).to_dict(),
        },
        "setup": {
            "cv_folds": CV_FOLDS, "cv_repeats": CV_REPEATS, "random_state": RANDOM_STATE,
            "tuning_scoring": "roc_auc", "decision_threshold": 0.5,
            "n_bootstrap": N_BOOT, "ci_level": 1 - ALPHA,
            "comparison_alpha": ALPHA, "multiple_testing": "Holm within each test/metric family",
        },
        "best_params": {name: {short_param(k): v for k, v in best_params[name].items()}
                        for name in MODEL_NAMES},
        "best_params_reproduced": params_match,
        "grid_search_best_cv_auc": {
            name: r4(tables[name]["mean_cv_auc"].max()) for name in MODEL_NAMES},
        "cv_repeated": {
            r.model: {m: {"mean": r4(getattr(r, f"{m}_mean")), "std": r4(getattr(r, f"{m}_std"))}
                      for m in CV_METRICS}
            for r in cv_summary.itertuples()
        },
        "cv_nested": {
            name: {
                **{m: {"mean": r4(nested_summary.loc[name, (m, "mean")]),
                       "std": r4(nested_summary.loc[name, (m, "std")])} for m in CV_METRICS},
                "params_per_outer_fold": nested.loc[nested["model"] == name, "best_params"]
                .map(json.loads).tolist(),
            }
            for name in MODEL_NAMES
        },
        "test": test_json,
        "test_by_site": {
            name: {r.site: {"n": r.n, "prevalence": r4(r.prevalence), "accuracy": r4(r.accuracy),
                            "recall": r4(r.recall), "specificity": r4(r.specificity),
                            "roc_auc": r4(r.roc_auc)}
                   for r in by_site[by_site["model"] == name].itertuples()}
            for name in ALL_MODELS
        },
        "comparisons": [
            {"model_a": r.model_a, "model_b": r.model_b, "test": r.test, "metric": r.metric,
             "difference_a_minus_b": r4(r.difference_a_minus_b),
             "statistic_name": r.statistic_name, "statistic": r4(r.statistic),
             "p_value": r4(r.p_value), "p_holm": r4(r.p_holm),
             "significant": bool(r.p_holm < ALPHA), "interpretation": r.interpretation}
            for r in comparison.itertuples()
        ],
        "figures": [p.name for p in paths],
    }
    save_json(main_results, RESULTS_DIR / "main_results.json")
    print(f"\nSaved results to {RESULTS_DIR} ({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
