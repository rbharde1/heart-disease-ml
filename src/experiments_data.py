"""Experiments on the data side of the project: preprocessing, missing values and
generalisation across the four hospitals.

E1  Preprocessing ablation: how much do the missing-data strategy and feature
    scaling matter for each tuned model? Includes a complete-case analysis.
E2  Leave-one-site-out (LOSO): train on three hospitals, test on the fourth, and
    compare with the ordinary random train/test split.
E3  Is missingness a hospital fingerprint? Can the pattern of missing values
    identify the hospital, how much disease signal does it carry on its own, and
    does telling the model the hospital help?
E4  Cleveland-only benchmark, for comparison with published Cleveland results.

Hyperparameters were tuned once on the pooled training split (cached in
results/best_params.json) and are reused unchanged here. Unless stated otherwise,
scores are stratified 5-fold CV repeated 3 times on the training split (736
patients); the hold-out test split is only used for the random-split reference
in E2.

Outputs: figures/exp_data_*.png, results/exp_data_*.csv, results/exp_data_results.json
"""
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_validate,
)
from sklearn.tree import DecisionTreeClassifier

from config import CV_FOLDS, RANDOM_STATE, RESULTS_DIR
from data import FEATURE_LABELS, FEATURES, SITES, get_xy, load_dataset, split_data
from evaluation import compute_metrics, predict_scores, save_json
from models import BASELINE_NAME, MODEL_NAMES, build_tuned_pipeline, get_best_params, make_baseline
from plotting import MODEL_COLORS, NEUTRAL, TEXT_PRIMARY, TEXT_SECONDARY, savefig
from preprocess import STRATEGIES, build_preprocessor

import matplotlib.pyplot as plt  # noqa: E402  (after plotting so the Agg backend/style are set)
from matplotlib.lines import Line2D  # noqa: E402

EXP_REPEATS = 3            # repeats of 5-fold CV on the training split
CLEVELAND_REPEATS = 10     # E4 uses more repeats because n=303 is small
N_BOOT = 2000              # bootstrap resamples for LOSO AUC confidence intervals
SCORING = {"roc_auc": "roc_auc", "accuracy": "accuracy"}

# key -> (strategy passed to build_preprocessor, scale numeric features?, plot label)
ABLATION_SETTINGS = {
    "indicator": ("indicator", True, "Indicators + scaling\n(default)"),
    "simple": ("simple", True, "Imputation only"),
    "drop_sparse": ("drop_sparse", True, "Drop ca, slope, thal"),
    "indicator_noscale": ("indicator", False, "Indicators, no scaling"),
}
SHORT_MODEL = {"Logistic Regression": "LR", "SVM (RBF)": "SVM", "Random Forest": "RF"}
MODEL_OFFSETS = dict(zip(MODEL_NAMES, [-0.22, 0.0, 0.22]))


# --------------------------------------------------------------------------- helpers
def repeated_cv(n_repeats=EXP_REPEATS):
    """Same seed everywhere, so different settings are scored on identical folds
    and their fold-wise differences are paired."""
    return RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=n_repeats, random_state=RANDOM_STATE)


def tuned_pipeline(name, best, preprocessor=None):
    """Tuned pipeline for one model. The forest runs single-threaded because the
    CV folds are already evaluated in parallel (avoids oversubscribing the CPU)."""
    pipe = build_tuned_pipeline(name, best[name], preprocessor)
    if name == "Random Forest":
        pipe.set_params(clf__n_jobs=1)
    return pipe


def cv_scores(estimator, X, y, cv, scoring=SCORING):
    """Fold-wise test scores, one array per metric."""
    res = cross_validate(estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {m: res[f"test_{m}"] for m in scoring}


def summarise(scores, digits=4):
    """mean and SD (across folds) of every metric."""
    out = {}
    for metric, values in scores.items():
        out[f"{metric}_mean"] = round(float(np.mean(values)), digits)
        out[f"{metric}_std"] = round(float(np.std(values, ddof=1)), digits)
    return out


def paired_delta(scores, reference, metric):
    """Mean fold-wise difference to a reference run on the same folds, and how
    many folds improved."""
    diff = np.asarray(scores[metric]) - np.asarray(reference[metric])
    return {
        f"{metric}_delta_mean": round(float(diff.mean()), 4),
        f"{metric}_delta_std": round(float(diff.std(ddof=1)), 4),
        f"{metric}_folds_better": int((diff > 0).sum()),
        "n_folds": int(diff.size),
    }


def auc_mann_whitney(y_true, y_score):
    """ROC-AUC via the Mann-Whitney U statistic (identical to roc_auc_score, ties included)."""
    ranks = rankdata(y_score)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def bootstrap_auc_ci(y_true, y_score, n_boot=N_BOOT, alpha=0.05, random_state=RANDOM_STATE):
    """Percentile bootstrap CI for ROC-AUC on a fixed test set.

    Same resampling scheme as evaluation.bootstrap_ci, but only the AUC is computed
    per resample (that helper evaluates every metric each time, which is ~100x
    slower and adds minutes for 24 site/model combinations).
    """
    rng = np.random.default_rng(random_state)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats.append(auc_mann_whitney(y_true[idx], y_score[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def missing_flags(X):
    """0/1 flag per feature (1 = value missing). The transformation is stateless,
    so building it before CV cannot leak information across folds."""
    return X[FEATURES].isna().astype(int).rename(columns=lambda c: f"{c}_missing")


def site_dummies(df):
    """One-hot hospital indicator with a fixed column order (stateless, known set of sites)."""
    return pd.get_dummies(pd.Categorical(df["site"], categories=SITES), prefix="site").astype(int).set_axis(df.index)


# --------------------------------------------------------------------------- E1
def e1_preprocessing(train_df, best):
    """Compare missing-data strategies / scaling for each tuned model with paired CV."""
    X, y = get_xy(train_df)
    cv = repeated_cv()
    rows, folds = [], {}
    for key, (strategy, scale, label) in ABLATION_SETTINGS.items():
        for name in MODEL_NAMES:
            prep = build_preprocessor(strategy, scale=scale)
            folds[(key, name)] = cv_scores(tuned_pipeline(name, best, prep), X, y, cv)
            rows.append({"setting": key, "label": label.replace("\n", " "), "model": name,
                         "n_rows": len(y), **summarise(folds[(key, name)])})
    for row in rows:
        ref = folds[("indicator", row["model"])]
        cur = folds[(row["setting"], row["model"])]
        for metric in SCORING:
            d = paired_delta(cur, ref, metric)
            row[f"{metric}_delta_vs_default"] = d[f"{metric}_delta_mean"]
    baseline = summarise(cv_scores(make_baseline(), X, y, cv))

    # Complete-case analysis: a different (smaller, almost all-Cleveland) population,
    # so these numbers are NOT directly comparable with the rows above.
    cc_df = train_df.dropna(subset=FEATURES)
    Xc, yc = get_xy(cc_df)
    cc_rows = []
    for name in MODEL_NAMES:
        cc_rows.append({"setting": "complete_case", "model": name, "n_rows": len(yc),
                        **summarise(cv_scores(tuned_pipeline(name, best), Xc, yc, cv))})
    complete_case = {
        "n_rows": len(cc_df),
        "n_train_rows": len(train_df),
        "share_of_train": round(len(cc_df) / len(train_df), 4),
        "rows_by_site": {s: int((cc_df["site"] == s).sum()) for s in SITES},
        "prevalence": round(float(yc.mean()), 4),
        "baseline": summarise(cv_scores(make_baseline(), Xc, yc, cv)),
        "results": cc_rows,
        "note": "Different, smaller population (complete rows are almost all from Cleveland); "
                "not comparable with the pooled-data rows.",
    }
    return {"strategies": {k: STRATEGIES[v[0]] + ("" if v[1] else "; numeric features NOT standardised")
                           for k, v in ABLATION_SETTINGS.items()},
            "results": rows, "majority_baseline": baseline,
            "complete_case": complete_case}, folds


# --------------------------------------------------------------------------- E2
def _site_metrics(y, score):
    m = compute_metrics(y, score)
    return {k: round(float(m[k]), 4) for k in
            ["roc_auc", "accuracy", "precision", "recall", "specificity", "f1"]} | {
        k: int(m[k]) for k in ["tn", "fp", "fn", "tp"]}


def e2_leave_one_site_out(df, train_df, test_df, best):
    """Train on three hospitals (all their rows), test on the fourth; contrast with
    the random split (fit on the training split, score the test rows of that site)."""
    X_tr, y_tr = get_xy(train_df)
    X_te, y_te = get_xy(test_df)
    names = MODEL_NAMES + [BASELINE_NAME]
    random_fits = {}
    for name in names:
        model = make_baseline() if name == BASELINE_NAME else tuned_pipeline(name, best)
        random_fits[name] = model.fit(X_tr, y_tr)

    rows = []
    for site in SITES:
        held = (df["site"] == site).to_numpy()
        X_a, y_a = get_xy(df[~held])
        X_b, y_b = get_xy(df[held])
        te_mask = (test_df["site"] == site).to_numpy()
        for name in names:
            loso = make_baseline() if name == BASELINE_NAME else tuned_pipeline(name, best)
            loso.fit(X_a, y_a)
            for scheme, model, Xs, ys, n_fit, fit_prev in [
                ("LOSO", loso, X_b, y_b, len(y_a), y_a.mean()),
                ("Random split", random_fits[name], X_te[te_mask], y_te[te_mask], len(y_tr), y_tr.mean()),
            ]:
                score = predict_scores(model, Xs)
                row = {"site": site, "model": name, "scheme": scheme,
                       "n_fit": n_fit, "fit_prevalence": round(float(fit_prev), 4),
                       "n_test": len(ys), "n_pos": int(ys.sum()), "n_neg": int((ys == 0).sum()),
                       "prevalence": round(float(ys.mean()), 4),
                       "mean_pred_prob": round(float(score.mean()), 4)}
                row["calibration_gap"] = round(row["mean_pred_prob"] - row["prevalence"], 4)
                row.update(_site_metrics(ys, score))
                if name != BASELINE_NAME:
                    lo, hi = bootstrap_auc_ci(ys, score)
                    row["roc_auc_ci_low"], row["roc_auc_ci_high"] = round(lo, 4), round(hi, 4)
                rows.append(row)

    table = pd.DataFrame(rows)
    summary = {}
    for name in MODEL_NAMES:
        sub = table[table["model"] == name]
        loso, rs = sub[sub["scheme"] == "LOSO"], sub[sub["scheme"] == "Random split"]
        pooled = compute_metrics(y_te, predict_scores(random_fits[name], X_te))
        summary[name] = {
            "loso_mean_auc_over_sites": round(float(loso["roc_auc"].mean()), 4),
            "loso_mean_accuracy_over_sites": round(float(loso["accuracy"].mean()), 4),
            "random_split_mean_auc_over_sites": round(float(rs["roc_auc"].mean()), 4),
            "random_split_mean_accuracy_over_sites": round(float(rs["accuracy"].mean()), 4),
            "random_split_pooled_test_auc": round(float(pooled["roc_auc"]), 4),
            "random_split_pooled_test_accuracy": round(float(pooled["accuracy"]), 4),
        }
    notes = [
        "Switzerland has only 8 patients without disease (2 in the random-split test rows), "
        "so its AUC and specificity are very unstable (see the bootstrap CI).",
        "LOSO trains on ALL rows of the other three sites (train and test split rows); "
        "hyperparameters are the ones tuned on the pooled training split.",
        "Random-split per-site numbers come from small test subsets "
        "(Cleveland 61, Hungary 58, Switzerland 25, Long Beach VA 40 patients).",
    ]
    return {"results": rows, "summary": summary, "notes": notes}, table


# --------------------------------------------------------------------------- E3
def e3_site_signal(train_df, best, e1_folds):
    """How much hospital / shortcut information do the missing-value flags carry?"""
    X, y = get_xy(train_df)
    flags = missing_flags(X)
    sites = train_df["site"].to_numpy()
    out = {"n_flags": flags.shape[1],
           "n_distinct_missing_patterns": int(flags.astype(str).agg("".join, axis=1).nunique())}

    # (a) predict the hospital from the 13 flags only
    clfs = {
        "Majority site": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision tree (depth 4)": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
    }
    site_scoring = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy"}
    site_rows = []
    for label, clf in clfs.items():
        s = cv_scores(clf, flags, sites, repeated_cv(), scoring=site_scoring)
        row = {"task": "site_from_flags", "model": label, **summarise(s)}
        if label != "Majority site":
            # per-site recall from one round of out-of-fold predictions
            pred = cross_val_predict(clf, flags, sites, cv=StratifiedKFold(
                CV_FOLDS, shuffle=True, random_state=RANDOM_STATE), n_jobs=-1)
            for site in SITES:
                row[f"recall_{site}"] = round(float((pred[sites == site] == site).mean()), 4)
        site_rows.append(row)
    tree = DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE).fit(flags, sites)
    imp = pd.Series(tree.feature_importances_, index=FEATURES).sort_values(ascending=False)
    out["site_from_flags"] = site_rows
    out["site_from_flags_tree_importance"] = {
        FEATURE_LABELS[f]: round(float(v), 4) for f, v in imp.items() if v > 0.01}
    out["majority_site_share"] = round(float(pd.Series(sites).value_counts(normalize=True).iloc[0]), 4)

    # (b) predict disease from the flags only (and compare with the hospital alone)
    dummies = site_dummies(train_df)

    def plain_lr():
        return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)

    inputs = {
        "Missing flags (LR)": (flags, plain_lr()),
        "Missing flags (tree)": (flags, DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)),
        "Hospital only (LR)": (dummies, plain_lr()),
        "Missing flags + hospital (LR)": (pd.concat([flags, dummies], axis=1), plain_lr()),
    }
    target_rows = []
    for label, (Xin, clf) in inputs.items():
        target_rows.append({"task": "target_from_flags", "model": label,
                            **summarise(cv_scores(clf, Xin, y, repeated_cv()))})
    target_rows.append({"task": "target_from_flags", "model": "All 13 features (tuned LR, reference)",
                        **summarise(e1_folds[("indicator", "Logistic Regression")])})
    out["target_from_flags"] = target_rows

    # (c) add the hospital (one-hot) as an explicit feature to each tuned model
    Xs = train_df[FEATURES + ["site"]]
    site_feature_rows = []
    for name in MODEL_NAMES:
        with_site = cv_scores(tuned_pipeline(name, best, build_preprocessor(include_site=True)),
                              Xs, y, repeated_cv())
        without = e1_folds[("indicator", name)]   # identical folds (same seed)
        row = {"model": name}
        for tag, s in [("without_site", without), ("with_site", with_site)]:
            row.update({f"{tag}_{k}": v for k, v in summarise(s).items()})
        for metric in SCORING:
            row.update({f"{k}": v for k, v in paired_delta(with_site, without, metric).items()})
        site_feature_rows.append(row)
    out["site_as_feature"] = site_feature_rows
    return out


# --------------------------------------------------------------------------- E4
def e4_cleveland(df, best):
    """Repeated 5-fold CV on the 303 Cleveland patients alone (no re-tuning)."""
    cle = df[df["site"] == "Cleveland"]
    X, y = get_xy(cle)
    scoring = {"roc_auc": "roc_auc", "accuracy": "accuracy", "f1": "f1"}
    cv = repeated_cv(CLEVELAND_REPEATS)
    rows = []
    for name in MODEL_NAMES + [BASELINE_NAME]:
        model = make_baseline() if name == BASELINE_NAME else tuned_pipeline(name, best)
        rows.append({"model": name, "n_rows": len(y), **summarise(cv_scores(model, X, y, cv, scoring))})
    return {"n_rows": len(cle), "prevalence": round(float(y.mean()), 4),
            "cv": f"RepeatedStratifiedKFold({CV_FOLDS} folds x {CLEVELAND_REPEATS} repeats)",
            "results": rows,
            "note": "Uses all 303 Cleveland rows (train and test split); hyperparameters are the "
                    "pooled-data ones, nothing re-tuned."}


# --------------------------------------------------------------------------- figures
def plot_preprocessing(e1):
    rows = pd.DataFrame(e1["results"])
    cc = pd.DataFrame(e1["complete_case"]["results"])
    cc_info = e1["complete_case"]
    keys = list(ABLATION_SETTINGS) + ["complete_case"]
    top_site, top_n = max(cc_info["rows_by_site"].items(), key=lambda kv: kv[1])
    labels = [v[2] for v in ABLATION_SETTINGS.values()] + [
        f"Complete cases only*\n(n={cc_info['n_rows']}, {top_n} {top_site})"]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.6), sharey=True)
    for ax, metric, title in zip(axes, SCORING, ["ROC-AUC", "Accuracy"]):
        for i, key in enumerate(keys):
            src = cc if key == "complete_case" else rows[rows["setting"] == key]
            for name in MODEL_NAMES:
                r = src[src["model"] == name].iloc[0]
                ax.errorbar(r[f"{metric}_mean"], i + MODEL_OFFSETS[name], xerr=r[f"{metric}_std"],
                            fmt="o", color=MODEL_COLORS[name], ms=5, capsize=2, elinewidth=1.2,
                            mfc="white" if key == "complete_case" else MODEL_COLORS[name])
        ax.axhline(len(ABLATION_SETTINGS) - 0.5, color=NEUTRAL, ls="--", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel(f"{title} (mean ± SD, 15 folds)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(keys)), labels)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Preprocessing")
    fig.suptitle("Effect of missing-data handling and scaling (training-set CV)", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    handles = [Line2D([], [], marker="o", ls="", color=MODEL_COLORS[n], label=n) for n in MODEL_NAMES]
    fig.legend(handles=handles, loc="center", ncol=3, bbox_to_anchor=(0.5, 0.06))
    fig.text(0.5, 0.0, "*Different, smaller population (hollow markers): not directly comparable "
             "with the rows above.", ha="center", fontsize=7.5, color=TEXT_SECONDARY)
    return savefig(fig, "exp_data_preprocessing")


def _site_ticklabels(table):
    lo = table[(table["scheme"] == "LOSO") & (table["model"] == MODEL_NAMES[0])].set_index("site")
    return [f"{s}\nn={lo.loc[s, 'n_test']}\n{lo.loc[s, 'n_neg']} no disease" for s in SITES]


def plot_loso(table):
    x = np.arange(len(SITES))
    # stacked panels give each hospital enough room for its label; legend on the right
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.5), sharex=True)
    for ax, metric, title in zip(axes, ["roc_auc", "accuracy"], ["ROC-AUC", "Accuracy"]):
        for name in MODEL_NAMES:
            sub = table[table["model"] == name]
            loso = sub[sub["scheme"] == "LOSO"].set_index("site").loc[SITES, metric].to_numpy()
            rs = sub[sub["scheme"] == "Random split"].set_index("site").loc[SITES, metric].to_numpy()
            xs = x + MODEL_OFFSETS[name]
            c = MODEL_COLORS[name]
            # thin connector shows the change from random split to unseen hospital
            ax.vlines(xs, np.minimum(loso, rs), np.maximum(loso, rs), color=c, lw=1, alpha=0.45)
            ax.plot(xs, rs, "D", ms=4.5, mfc="white", mec=c, mew=1.3)
            ax.plot(xs, loso, "o", ms=6, color=c)
        if metric == "roc_auc":
            ax.axhline(0.5, color=NEUTRAL, ls=":", lw=1)
            ax.text(x[-1] + 0.45, 0.51, "chance", ha="right", va="bottom", fontsize=7.5, color=TEXT_SECONDARY)
        else:
            base = table[(table["model"] == BASELINE_NAME) & (table["scheme"] == "LOSO")]
            base = base.set_index("site").loc[SITES, metric].to_numpy()
            ax.hlines(base, x - 0.36, x + 0.36, color=NEUTRAL, lw=2)
        ax.set_xlim(-0.5, len(SITES) - 0.5)
        ax.set_ylim(0.0 if metric == "accuracy" else 0.4, 1.02)
        ax.set_ylabel(title)
        ax.grid(axis="x", visible=False)
    axes[0].set_title("ROC-AUC on the held-out hospital", loc="left")
    axes[1].set_title("Accuracy on the held-out hospital (threshold 0.5)", loc="left")
    axes[1].set_xticks(x, _site_ticklabels(table), fontsize=7.8)
    axes[1].set_xlabel("Held-out hospital")
    sw = SITES.index("Switzerland")
    axes[0].text(sw, 0.99, "few negatives:\nAUC unstable", ha="center", va="top",
                 fontsize=7.2, color=TEXT_SECONDARY)
    handles = [Line2D([], [], marker="o", ls="", color=MODEL_COLORS[n], label=n) for n in MODEL_NAMES]
    handles += [
        Line2D([], [], marker="o", ls="", color=TEXT_PRIMARY, label="Unseen hospital\n(LOSO)"),
        Line2D([], [], marker="D", ls="", mfc="white", mec=TEXT_PRIMARY, label="Random split\n(same-site test rows)"),
        Line2D([], [], color=NEUTRAL, lw=2, label="Majority baseline\n(LOSO)"),
    ]
    fig.suptitle("Generalisation to an unseen hospital (leave-one-site-out)", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.74, 1))
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.75, 0.5), labelspacing=1.0)
    return savefig(fig, "exp_data_loso")


def plot_loso_calibration(table):
    """Calibration-in-the-large: mean predicted probability vs observed prevalence."""
    y = np.arange(len(SITES))
    loso = table[table["scheme"] == "LOSO"]
    fig, ax = plt.subplots(figsize=(3.2, 3.6))
    first = loso[loso["model"] == MODEL_NAMES[0]].set_index("site").loc[SITES]
    ax.vlines(first["prevalence"], y - 0.38, y + 0.38, color=TEXT_PRIMARY, lw=2.2)
    ax.vlines(first["fit_prevalence"], y - 0.38, y + 0.38, color=NEUTRAL, lw=1.4, ls="--")
    for name in MODEL_NAMES:
        vals = loso[loso["model"] == name].set_index("site").loc[SITES, "mean_pred_prob"]
        ax.plot(vals, y + MODEL_OFFSETS[name], "o", ms=5.5, color=MODEL_COLORS[name])
    ax.set_yticks(y, [s.replace("Long Beach VA", "Long Beach\nVA") for s in SITES])
    ax.set_xlim(0, 1)
    ax.set_ylim(len(SITES) - 0.5, -0.5)   # first hospital on top
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Proportion with disease")
    ax.set_ylabel("Held-out hospital")
    ax.set_title("Predicted vs observed\nprevalence (LOSO)")
    handles = [Line2D([], [], color=TEXT_PRIMARY, lw=2.2, label="Observed"),
               Line2D([], [], color=NEUTRAL, lw=1.4, ls="--", label="Training-site rate")]
    handles += [Line2D([], [], marker="o", ls="", color=MODEL_COLORS[n], label=f"{SHORT_MODEL[n]} mean prob.")
                for n in MODEL_NAMES]
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.08), ncol=2, fontsize=7.5,
               columnspacing=1.0)
    return savefig(fig, "exp_data_loso_calibration")


def plot_site_signal(e3):
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.2), gridspec_kw={"width_ratios": [1.0, 1.3, 0.9]})

    # (a) hospital from missingness
    ax = axes[0]
    rows = e3["site_from_flags"]
    names = ["Majority", "LR", "Tree"]
    colors = [NEUTRAL, MODEL_COLORS["Logistic Regression"], TEXT_SECONDARY]
    means = [r["accuracy_mean"] for r in rows]
    stds = [r["accuracy_std"] for r in rows]
    ax.bar(range(3), means, yerr=stds, color=colors, width=0.65, capsize=2, error_kw={"elinewidth": 1})
    for i, m in enumerate(means):
        ax.text(i, m + 0.04, f"{m:.2f}", ha="center", fontsize=7.8)
    ax.set_xticks(range(3), names, fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("CV accuracy")
    ax.set_xlabel("Classifier")
    ax.set_title("(a) Hospital from\nmissing flags", fontsize=9.5)
    ax.grid(axis="x", visible=False)

    # (b) disease from missingness (horizontal so the input names stay readable)
    ax = axes[1]
    rows = e3["target_from_flags"]
    names = ["Flags (LR)", "Flags (tree)", "Hospital only", "Flags + hospital", "All features (LR)"]
    for i, r in enumerate(rows):
        c = MODEL_COLORS["Logistic Regression"] if i == len(rows) - 1 else TEXT_SECONDARY
        ax.errorbar(r["roc_auc_mean"], i, xerr=r["roc_auc_std"], fmt="o", color=c, ms=5, capsize=2)
        ax.text(r["roc_auc_mean"], i - 0.22, f"{r['roc_auc_mean']:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.axvline(0.5, color=NEUTRAL, ls=":", lw=1)
    ax.text(0.51, len(rows) - 0.55, "chance", ha="left", va="bottom", fontsize=7.2, color=TEXT_SECONDARY)
    ax.set_yticks(range(len(rows)), names, fontsize=8)
    ax.set_ylim(len(rows) - 0.5, -0.7)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("CV ROC-AUC")
    ax.set_title("(b) Disease from\nmissing flags", fontsize=9.5)
    ax.grid(axis="y", visible=False)

    # (c) hospital as an extra feature
    ax = axes[2]
    rows = e3["site_as_feature"]
    for i, r in enumerate(rows):
        c = MODEL_COLORS[r["model"]]
        ax.errorbar(i - 0.13, r["without_site_roc_auc_mean"], yerr=r["without_site_roc_auc_std"],
                    fmt="o", mfc="white", mec=c, color=c, ms=5, capsize=2)
        ax.errorbar(i + 0.13, r["with_site_roc_auc_mean"], yerr=r["with_site_roc_auc_std"],
                    fmt="o", color=c, ms=5, capsize=2)
    ax.set_xticks(range(len(rows)), [SHORT_MODEL[r["model"]] for r in rows])
    ax.set_xlim(-0.5, len(rows) - 0.5)
    lo = min(min(r["without_site_roc_auc_mean"] - r["without_site_roc_auc_std"],
                 r["with_site_roc_auc_mean"] - r["with_site_roc_auc_std"]) for r in rows)
    hi = max(max(r["without_site_roc_auc_mean"] + r["without_site_roc_auc_std"],
                 r["with_site_roc_auc_mean"] + r["with_site_roc_auc_std"]) for r in rows)
    ax.set_ylim(np.floor(lo * 50) / 50, np.ceil(hi * 50) / 50 + 0.02)
    ax.set_ylabel("CV ROC-AUC")
    ax.set_xlabel("Model")
    ax.set_title("(c) Hospital as an\nextra feature", fontsize=9.5)
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=TEXT_PRIMARY, label="13 features"),
                       Line2D([], [], marker="o", ls="", color=TEXT_PRIMARY, label="+ hospital")],
              loc="upper right", fontsize=7.5, handletextpad=0.2, borderaxespad=0.2)
    fig.suptitle("How much hospital information do missing values carry?", fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    return savefig(fig, "exp_data_site_signal")


# --------------------------------------------------------------------------- main
def _fmt(r, metric):
    return f"{r[f'{metric}_mean']:.3f} +/- {r[f'{metric}_std']:.3f}"


def main():
    t0 = time.time()
    df = load_dataset()
    train_df, test_df = split_data(df)
    X_train, y_train = get_xy(train_df)
    best = get_best_params(X_train, y_train)

    print("E1 preprocessing ablation ...")
    e1, e1_folds = e1_preprocessing(train_df, best)
    print("E2 leave-one-site-out ...")
    e2, loso_table = e2_leave_one_site_out(df, train_df, test_df, best)
    print("E3 missingness as a hospital fingerprint ...")
    e3 = e3_site_signal(train_df, best, e1_folds)
    print("E4 Cleveland-only benchmark ...")
    e4 = e4_cleveland(df, best)

    figs = [plot_preprocessing(e1), plot_loso(loso_table), plot_loso_calibration(loso_table),
            plot_site_signal(e3)]

    # machine-readable outputs
    pd.DataFrame(e1["results"] + e1["complete_case"]["results"]).to_csv(
        RESULTS_DIR / "exp_data_preprocessing.csv", index=False)
    loso_table.to_csv(RESULTS_DIR / "exp_data_loso.csv", index=False)
    pd.DataFrame(e3["site_from_flags"] + e3["target_from_flags"]).to_csv(
        RESULTS_DIR / "exp_data_missingness.csv", index=False)
    pd.DataFrame(e3["site_as_feature"]).to_csv(RESULTS_DIR / "exp_data_site_feature.csv", index=False)
    pd.DataFrame(e4["results"]).to_csv(RESULTS_DIR / "exp_data_cleveland.csv", index=False)
    results = {
        "setup": {
            "n_total": len(df), "n_train": len(train_df), "n_test": len(test_df),
            "cv": f"RepeatedStratifiedKFold({CV_FOLDS} folds x {EXP_REPEATS} repeats) on the training split",
            "random_state": RANDOM_STATE, "best_params": best, "bootstrap_resamples": N_BOOT,
        },
        "E1_preprocessing": e1, "E2_leave_one_site_out": e2,
        "E3_missingness_site_signal": e3, "E4_cleveland_only": e4,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    save_json(results, RESULTS_DIR / "exp_data_results.json")

    # concise console summary
    print(f"\nE1 (CV on {len(train_df)} training rows) ROC-AUC / accuracy:")
    for r in e1["results"]:
        print(f"  {r['setting']:<18} {SHORT_MODEL[r['model']]:<4} {_fmt(r, 'roc_auc')}  {_fmt(r, 'accuracy')}")
    cc = e1["complete_case"]
    print(f"  complete cases: n={cc['n_rows']} {cc['rows_by_site']}")
    for r in cc["results"]:
        print(f"  complete_case      {SHORT_MODEL[r['model']]:<4} {_fmt(r, 'roc_auc')}  {_fmt(r, 'accuracy')}")
    print("\nE2 LOSO vs random split (AUC / accuracy):")
    for _, r in loso_table[loso_table["model"] != BASELINE_NAME].iterrows():
        print(f"  {r['site']:<14} {SHORT_MODEL[r['model']]:<4} {r['scheme']:<13} AUC {r['roc_auc']:.3f} "
              f"acc {r['accuracy']:.3f}  prev {r['prevalence']:.2f} pred {r['mean_pred_prob']:.2f}")
    print("\nE3 site from flags:", {r["model"]: r["accuracy_mean"] for r in e3["site_from_flags"]})
    print("   target from flags (AUC):", {r["model"]: r["roc_auc_mean"] for r in e3["target_from_flags"]})
    print("   + site feature (AUC delta):", {r["model"]: r["roc_auc_delta_mean"] for r in e3["site_as_feature"]})
    print("\nE4 Cleveland only:")
    for r in e4["results"]:
        print(f"  {r['model']:<20} AUC {_fmt(r, 'roc_auc')}  acc {_fmt(r, 'accuracy')}")
    print("\nSaved:", ", ".join(p.name for p in figs), f"| runtime {results['runtime_seconds']} s")


if __name__ == "__main__":
    main()
