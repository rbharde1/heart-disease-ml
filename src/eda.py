"""Exploratory data analysis and preprocessing evidence for the pooled UCI Heart Disease data.

Outputs
-------
figures/eda_*.png            figures for the Data section of the report
results/eda_*.csv, *.json    every number the report quotes
results/eda_tables.md        markdown tables ready to paste into the report
data/processed/*.csv         the cleaned data set and the fixed train/test split

Descriptive statistics and association tests use all 920 patients because they
describe the data set; no feature or model choice is made from them. The PCA,
which needs a fitted preprocessor, is fitted on the training split only so the
test set stays untouched until the final evaluation.
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import blended_transform_factory
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from config import DATA_PROCESSED, RANDOM_STATE, RESULTS_DIR
from data import (
    BINARY_FEATURES, CATEGORICAL_FEATURES, CATEGORY_LABELS, FEATURE_LABELS, FEATURES,
    NUMERIC_FEATURES, SITES, TARGET, clean, get_xy, load_raw, missing_summary, split_data,
)
from evaluation import save_json
from plotting import (
    CLASS_COLORS, CLASS_NAMES, DIVERGING_CMAP, NEUTRAL, SEQUENTIAL_CMAP, SITE_COLORS,
    TEXT_PRIMARY, TEXT_SECONDARY, savefig,
)
from preprocess import SPARSE_FEATURES, STRATEGIES, build_preprocessor

ALL_SITES = "All sites"

# Compact names for crowded axes (matrix labels, loadings); FEATURE_LABELS is used elsewhere.
SHORT_LABELS = {
    "age": "Age", "sex": "Sex (male)", "cp": "Chest pain", "trestbps": "Resting BP",
    "chol": "Cholesterol", "fbs": "Fasting blood sugar", "restecg": "Resting ECG",
    "thalach": "Max heart rate", "exang": "Exercise angina", "oldpeak": "ST depression",
    "slope": "ST slope", "ca": "Major vessels", "thal": "Thallium test", "target": "Heart disease",
}

# Wording follows the attribute documentation shipped with the data (heart-disease.names).
DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure on admission (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true)",
    "restecg": "Resting electrocardiogram result",
    "thalach": "Maximum heart rate achieved during exercise test (bpm)",
    "exang": "Exercise-induced angina (1 = yes)",
    "oldpeak": "ST depression induced by exercise relative to rest (mm)",
    "slope": "Slope of the peak-exercise ST segment",
    "ca": "Number of major vessels (0-3) coloured by fluoroscopy",
    "thal": "Thallium stress test result",
}

FEATURE_TYPES = {
    "age": "Numeric (continuous)", "trestbps": "Numeric (continuous)",
    "chol": "Numeric (continuous)", "thalach": "Numeric (continuous)",
    "oldpeak": "Numeric (continuous)", "ca": "Numeric (ordinal count)",
    "sex": "Binary", "fbs": "Binary", "exang": "Binary",
    "cp": "Categorical (nominal)", "restecg": "Categorical (nominal)",
    "slope": "Categorical (ordered levels)", "thal": "Categorical (nominal)",
}

# Features shown in the disease-rate-per-level figure, split over two balanced panels.
LEVEL_PANELS = [["sex", "cp", "exang", "ca"], ["fbs", "restecg", "slope", "thal"]]
LEVEL_FEATURES = LEVEL_PANELS[0] + LEVEL_PANELS[1]

# Variables with a meaningful order for the rank correlation matrix (nominal cp, restecg
# and thal are left out because their codes have no order). Target goes last so the
# bottom row of the lower triangle reads as "correlation with heart disease".
CORR_VARS = ["age", "sex", "trestbps", "chol", "fbs", "thalach", "exang",
             "oldpeak", "slope", "ca", TARGET]
DIST_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
SITE_BOX_FEATURES = ["age", "thalach", "chol", "oldpeak"]
UNITS = {"age": "Years", "thalach": "Beats per minute", "chol": "mg/dl", "oldpeak": "mm"}

# Shorter group headers for the level-rate figure (full FEATURE_LABELS do not fit there).
LEVEL_HEADERS = {
    "sex": "Sex", "cp": "Chest pain type", "exang": "Exercise-induced angina",
    "ca": "Major vessels coloured", "fbs": "Fasting blood sugar >120",
    "restecg": "Resting ECG", "slope": "Peak exercise ST slope", "thal": "Thallium stress test",
}

# savefig uses bbox='tight', which adds ~0.1 in of padding on each side, so figures drawn
# 6.3 in wide come out at ~6.5 in: the full text width of a single-column report.
FIG_W = 6.3


# --------------------------------------------------------------------------- helpers
def sig(x, digits=3):
    """Round to significant digits (for p-values that span many orders of magnitude)."""
    return float(f"{x:.{digits}g}")


def fmt_p(p):
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def magnitude(effect):
    """Conventional labels for |rank-biserial r| and Cramer's V with 1 degree of freedom."""
    a = abs(effect)
    if a < 0.1:
        return "negligible"
    if a < 0.3:
        return "small"
    if a < 0.5:
        return "medium"
    return "large"


def holm(pvalues):
    """Holm step-down adjustment for multiple testing."""
    p = np.asarray(pvalues, dtype=float)
    adjusted, running = np.empty_like(p), 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def records(df, orient="records"):
    """DataFrame -> dict/list for JSON, with NaN written as null (NaN is not valid JSON)."""
    return df.astype(object).where(df.notna(), None).to_dict(orient=orient)


def level_label(feature, code):
    if pd.isna(code):
        return "Missing"
    if feature == "ca":
        return f"{int(code)} vessel" + ("" if code == 1 else "s")
    return CATEGORY_LABELS[feature][int(code)]


def readable_column(name):
    """Human-readable name for a preprocessor output column (e.g. 'thal_-1.0')."""
    if name.startswith("missingindicator_"):
        return f"{SHORT_LABELS[name.removeprefix('missingindicator_')]}: missing"
    if name.startswith("site_"):
        return f"Site: {name.removeprefix('site_')}"
    if name in FEATURES:
        return SHORT_LABELS[name]
    feature, _, code = name.rpartition("_")
    return f"{SHORT_LABELS[feature]}: {level_label(feature, float(code)) if float(code) != -1 else 'missing'}"


def to_markdown(df):
    """Minimal GitHub-flavoured markdown table (numeric columns right-aligned)."""
    cols = list(df.columns)
    align = ["---:" if pd.api.types.is_numeric_dtype(df[c]) else "---" for c in cols]
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(align) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines)


def style_heatmap_axes(ax):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


# --------------------------------------------------------------------------- tables
def site_overview(df):
    """Patients, class counts and basic demographics per hospital and pooled."""
    groups = [(s, df[df["site"] == s]) for s in SITES] + [(ALL_SITES, df)]
    rows = []
    for name, g in groups:
        rows.append({
            "Site": name,
            "Patients": len(g),
            "Disease": int(g[TARGET].sum()),
            "No disease": int((g[TARGET] == 0).sum()),
            "Prevalence (%)": round(100 * g[TARGET].mean(), 1),
            "Share of data (%)": round(100 * len(g) / len(df), 1),
            "Median age": float(g["age"].median()),
            "Male (%)": round(100 * g["sex"].mean(), 1),
            "Complete cases (%)": round(100 * g[FEATURES].notna().all(axis=1).mean(), 1),
        })
    return pd.DataFrame(rows)


def num_distribution(df):
    """Counts of the original 0-4 diagnosis code per site."""
    table = pd.crosstab(df["site"], df["num"].astype(int)).reindex(SITES)
    table.loc[ALL_SITES] = table.sum()
    table.columns = [f"num={c}" for c in table.columns]
    return table.rename_axis("Site").reset_index()


def placeholder_zeros(raw):
    """chol == 0 / trestbps == 0 in the published files are missing-value placeholders."""
    target = (raw["num"] > 0).astype(int)
    out = {}
    for col in ("chol", "trestbps"):
        zero = raw[col] == 0
        by_site = raw[zero].groupby("site").size().reindex(SITES, fill_value=0)
        out[col] = {"total": int(by_site.sum()), "by_site": by_site.astype(int).to_dict(),
                    "disease_rate_among_zeros_pct": round(100 * target[zero].mean(), 1)}
    # Left as 0, the placeholder would read as "very low cholesterol" in the mostly sick
    # Swiss/VA patients and distort the cholesterol-disease association.
    for label, x in (("zeros_kept", raw["chol"]), ("zeros_as_missing", raw["chol"].mask(raw["chol"] == 0))):
        x1, x0 = x[target == 1].dropna(), x[target == 0].dropna()
        u = stats.mannwhitneyu(x1, x0, alternative="two-sided").statistic
        out["chol"][f"rank_biserial_{label}"] = round(float(2 * u / (len(x1) * len(x0)) - 1), 3)
    return out


def duplicate_report(raw, train_df, test_df):
    """Exact duplicate records and whether any pair straddles the train/test split."""
    cols = ["site"] + FEATURES + ["num"]
    dup_mask = raw.duplicated(subset=cols, keep=False)
    both = pd.concat([train_df.assign(split="train"), test_df.assign(split="test")])
    dup_split = both[both.duplicated(subset=cols, keep=False)]
    straddling = int(dup_split.groupby(cols, dropna=False)["split"].nunique().gt(1).sum())
    return {
        "exact_duplicate_rows": int(raw.duplicated(subset=cols).sum()),
        "rows_involved": int(dup_mask.sum()),
        "sites_of_duplicates": raw.loc[dup_mask, "site"].value_counts().to_dict(),
        "duplicate_pairs_split_across_train_test": straddling,
        "action": "kept: identical records cannot be told apart from two patients with "
                  "identical values, and 2 of 920 rows cannot change the conclusions",
    }


def feature_dictionary(df):
    """Feature name, type, observed values, missingness and pipeline treatment."""
    rows = []
    for f in FEATURES:
        x = df[f]
        if f in CATEGORY_LABELS:
            values = ", ".join(f"{k} = {v}" for k, v in CATEGORY_LABELS[f].items())
        elif f == "ca":
            values = "0-3"
        else:
            values = f"{x.min():g} to {x.max():g}"
        missing = 100 * x.isna().mean()
        if f in NUMERIC_FEATURES:
            treatment = ("Median imputation + missing indicator, standardised" if missing > 0
                         else "Standardised (no missing values)")
        elif f in BINARY_FEATURES:
            treatment = "Mode imputation, kept as 0/1" if missing > 0 else "Kept as 0/1 (no missing values)"
        else:
            treatment = "One-hot encoded; missing kept as its own level"
        if f in SPARSE_FEATURES:
            treatment += "; dropped in 'drop_sparse'"
        rows.append({
            "Feature": f, "Name": FEATURE_LABELS[f], "Description": DESCRIPTIONS[f],
            "Type": FEATURE_TYPES[f], "Values / range": values,
            "Missing (%)": round(missing, 1), "Treatment (default pipeline)": treatment,
        })
    return pd.DataFrame(rows)


def descriptive_by_target(df):
    """Mean / std / median / IQR per numeric and binary feature for each class."""
    rows = []
    for f in NUMERIC_FEATURES + BINARY_FEATURES:
        row = {"feature": f, "label": FEATURE_LABELS[f]}
        for cls, tag in ((0, "no_disease"), (1, "disease")):
            x = df.loc[df[TARGET] == cls, f].dropna()
            row.update({
                f"n_{tag}": len(x), f"mean_{tag}": round(x.mean(), 2), f"std_{tag}": round(x.std(), 2),
                f"median_{tag}": round(x.median(), 2), f"q1_{tag}": round(x.quantile(0.25), 2),
                f"q3_{tag}": round(x.quantile(0.75), 2),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def categorical_levels(df):
    """Disease rate for every level (including 'Missing') of the discrete features."""
    n_class = df[TARGET].value_counts()
    rows = []
    for f in LEVEL_FEATURES:
        codes = sorted(df[f].dropna().unique()) + ([np.nan] if df[f].isna().any() else [])
        for code in codes:
            mask = df[f].isna() if pd.isna(code) else df[f].eq(code)
            y = df.loc[mask, TARGET]
            rows.append({
                "feature": f, "level_code": None if pd.isna(code) else int(code),
                "level": level_label(f, code), "n": int(mask.sum()), "n_disease": int(y.sum()),
                "disease_rate_pct": round(100 * y.mean(), 1),
                "pct_of_no_disease": round(100 * (y == 0).sum() / n_class[0], 1),
                "pct_of_disease": round(100 * (y == 1).sum() / n_class[1], 1),
            })
    return pd.DataFrame(rows)


def univariate_tests(df):
    """Mann-Whitney U (numeric) and chi-square (binary/categorical) tests against the target.

    Effect sizes: rank-biserial r = 2*AUC - 1 (positive = larger values in the disease
    group) and Cramer's V. Missing values are excluded test by test; whether
    missingness itself is informative is checked separately in missingness_vs_target().
    """
    rows = []
    for f in NUMERIC_FEATURES:
        x1 = df.loc[df[TARGET] == 1, f].dropna()
        x0 = df.loc[df[TARGET] == 0, f].dropna()
        res = stats.mannwhitneyu(x1, x0, alternative="two-sided")
        auc = res.statistic / (len(x1) * len(x0))
        r = 2 * auc - 1

        def med_iqr(x):
            return f"{x.median():g} [{x.quantile(0.25):g}-{x.quantile(0.75):g}]"

        rows.append({
            "feature": f, "test": "Mann-Whitney U", "n": len(x1) + len(x0),
            "statistic": round(float(res.statistic), 1), "dof": None,
            "p_value": sig(res.pvalue), "effect_measure": "rank-biserial r",
            "effect_size": round(r, 3), "univariate_auc": round(auc, 3),
            "summary_measure": "median [IQR]",
            "no_disease": med_iqr(x0), "disease": med_iqr(x1),
            "direction": "higher with disease" if r > 0 else "lower with disease",
        })
    for f in BINARY_FEATURES + CATEGORICAL_FEATURES:
        sub = df[[f, TARGET]].dropna()
        table = pd.crosstab(sub[f], sub[TARGET])
        # Pearson chi-square without Yates' correction so the statistic and Cramer's V
        # are computed the same way for 2x2 and larger tables (n is large here).
        chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
        v = np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1)))
        # Summarise with the level whose disease rate is highest ("risk level").
        rates = table[1] / table.sum(axis=1)
        risk_code = rates.idxmax()
        share = 100 * table.loc[risk_code] / table.sum(axis=0)
        rows.append({
            "feature": f, "test": "Chi-square", "n": int(table.to_numpy().sum()),
            "statistic": round(float(chi2), 1), "dof": int(dof), "p_value": sig(p),
            "effect_measure": "Cramer's V", "effect_size": round(float(v), 3), "univariate_auc": None,
            "summary_measure": f"% {level_label(f, risk_code)}",
            "no_disease": f"{share[0]:.1f}%", "disease": f"{share[1]:.1f}%",
            "direction": f"'{level_label(f, risk_code)}' has highest disease rate ({100 * rates.max():.1f}%)",
        })
    out = pd.DataFrame(rows)
    out["p_holm"] = [sig(p) for p in holm(out["p_value"])]
    out["magnitude"] = out["effect_size"].map(magnitude)
    out["label"] = out["feature"].map(FEATURE_LABELS)
    out = out.reindex(out["effect_size"].abs().sort_values(ascending=False).index)
    return out.reset_index(drop=True)


def cochran_mantel_haenszel(flag, y, strata):
    """CMH chi-square (1 df, no continuity correction) for flag vs y, stratified by site.

    Tests whether a binary flag (e.g. 'value missing') is associated with the target
    *within* hospitals. Strata where the flag or the target is constant carry no
    information (zero variance) and drop out automatically.
    """
    dev, var = 0.0, 0.0
    for s in np.unique(strata):
        m = strata == s
        f, t = flag[m], y[m]
        n = m.sum()
        if n < 2:
            continue
        n_flag, n_pos = f.sum(), t.sum()
        a = (f & (t == 1)).sum()
        dev += a - n_flag * n_pos / n
        var += n_flag * (n - n_flag) * n_pos * (n - n_pos) / (n ** 2 * (n - 1))
    if var == 0:
        return float("nan"), float("nan")
    chi2 = dev ** 2 / var
    return float(chi2), float(stats.chi2.sf(chi2, df=1))


def missingness_vs_target(df):
    """Is a value being missing associated with disease, beyond which site it came from?

    'expected_rate_from_site_pct' is the disease rate we would see among the missing
    rows if missingness carried no information other than the hospital of origin
    (site prevalences weighted by where the missing values come from).

    Two tests are reported: the plain chi-square ('p_value') ignores the hospital, so it
    mostly re-detects site differences in prevalence; the Cochran-Mantel-Haenszel test
    ('cmh_p_value') compares missing vs observed rows *within* each hospital and is the
    one that answers "does missingness carry information beyond the site?".
    """
    site_prev = df.groupby("site")[TARGET].mean()
    y = df[TARGET].to_numpy()
    sites = df["site"].to_numpy()
    rows = []
    for f in FEATURES:
        miss = df[f].isna()
        if not miss.any():
            continue
        table = pd.crosstab(miss, df[TARGET])
        chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
        cmh_chi2, cmh_p = cochran_mantel_haenszel(miss.to_numpy(), y, sites)
        site_mix = df.loc[miss, "site"].value_counts()
        expected = (site_mix * site_prev.reindex(site_mix.index)).sum() / site_mix.sum()
        rows.append({
            "feature": f, "n_missing": int(miss.sum()), "missing_pct": round(100 * miss.mean(), 1),
            "disease_rate_if_missing_pct": round(100 * df.loc[miss, TARGET].mean(), 1),
            "disease_rate_if_observed_pct": round(100 * df.loc[~miss, TARGET].mean(), 1),
            "expected_rate_from_site_pct": round(100 * expected, 1),
            "chi2": round(float(chi2), 1), "p_value": sig(p),
            "cmh_chi2_within_site": None if np.isnan(cmh_chi2) else round(cmh_chi2, 1),
            "cmh_p_value": None if np.isnan(cmh_p) else sig(cmh_p),
            "share_of_missing_by_site_pct": {s: round(100 * site_mix.get(s, 0) / miss.sum(), 1) for s in SITES},
        })
    return pd.DataFrame(rows)


def site_heterogeneity(df):
    """Kruskal-Wallis across hospitals for the continuous features (+ prevalence chi-square)."""
    rows = []
    for f in DIST_FEATURES:
        groups = {s: df.loc[df["site"] == s, f].dropna() for s in SITES}
        groups = {s: g for s, g in groups.items() if len(g)}
        h, p = stats.kruskal(*groups.values())
        n = sum(len(g) for g in groups.values())
        row = {"feature": f, "test": "Kruskal-Wallis", "sites_with_data": len(groups),
               "statistic": round(float(h), 1), "p_value": sig(p),
               "epsilon_squared": round(float(h / (n - 1)), 3)}
        row.update({f"median_{s}": float(g.median()) for s, g in groups.items()})
        rows.append(row)
    table = pd.crosstab(df["site"], df[TARGET])
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    v = np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1)))
    rows.append({"feature": TARGET, "test": "Chi-square (site x target)", "sites_with_data": 4,
                 "statistic": round(float(chi2), 1), "p_value": sig(p),
                 "cramers_v": round(float(v), 3)})
    return pd.DataFrame(rows)


def complete_case_table(df):
    """Rows that survive listwise deletion, with all 13 features or without ca/slope/thal."""
    reduced = [f for f in FEATURES if f not in SPARSE_FEATURES]
    groups = [(s, df[df["site"] == s]) for s in SITES] + [(ALL_SITES, df)]
    rows = []
    for name, g in groups:
        full = g[FEATURES].notna().all(axis=1)
        red = g[reduced].notna().all(axis=1)
        rows.append({
            "Site": name, "Patients": len(g),
            "Complete (all 13 features)": int(full.sum()),
            "Complete, all 13 (%)": round(100 * full.mean(), 1),
            "Complete (without ca/slope/thal)": int(red.sum()),
            "Complete, without ca/slope/thal (%)": round(100 * red.mean(), 1),
        })
    return pd.DataFrame(rows)


def split_overview(train_df, test_df):
    rows = []
    for name, part in (("Train", train_df), ("Test", test_df)):
        row = {"Split": name, "Patients": len(part), "Disease": int(part[TARGET].sum()),
               "Prevalence (%)": round(100 * part[TARGET].mean(), 1)}
        row.update({s: int((part["site"] == s).sum()) for s in SITES})
        rows.append(row)
    return pd.DataFrame(rows)


def preprocessing_evidence(train_df):
    """Fit every preprocessing strategy on the training split and describe its output."""
    rows = []
    for strategy in STRATEGIES:
        for include_site in (False, True):
            cols = FEATURES + (["site"] if include_site else [])
            prep = build_preprocessor(strategy, include_site=include_site)
            z = prep.fit_transform(train_df[cols])
            names = list(prep.get_feature_names_out())
            indicators = [c for c in names if c.startswith("missingindicator_")]
            missing_levels = [c for c in names if c.endswith("_-1.0")]
            site_cols = [c for c in names if c.startswith("site_")]
            dummies = [c for c in names if c not in FEATURES and c not in indicators
                       and c not in missing_levels and c not in site_cols]
            rows.append({
                "strategy": strategy, "include_site": include_site,
                "description": STRATEGIES[strategy],
                "input_features": len([c for c in cols if c != "site"]),
                "n_rows": z.shape[0], "n_columns": z.shape[1],
                "n_missing_after": int(np.isnan(z).sum()),
                "missing_indicators": indicators, "missing_level_dummies": missing_levels,
                "one_hot_dummies": dummies, "site_dummies": site_cols, "columns": names,
            })
    return rows


# --------------------------------------------------------------------------- figures
def plot_class_balance(overview):
    sites = overview[overview["Site"] != ALL_SITES].set_index("Site")
    pooled = overview.set_index("Site").loc[ALL_SITES]
    fig, ax = plt.subplots(figsize=(FIG_W, 2.5), layout="constrained")
    ypos = np.arange(len(SITES))[::-1]
    for y, site in zip(ypos, SITES):
        neg, pos = int(sites.loc[site, "No disease"]), int(sites.loc[site, "Disease"])
        ax.barh(y, neg, height=0.62, color=CLASS_COLORS[0])
        ax.barh(y, pos, left=neg, height=0.62, color=CLASS_COLORS[1])
        note = f"{sites.loc[site, 'Prevalence (%)']:.1f}% disease"
        for left, width in ((0, neg), (neg, pos)):
            if width >= 20:
                ax.text(left + width / 2, y, str(width), ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold")
            else:  # segment too narrow for its count: report it next to the bar instead
                note += f" (only {width} without)"
        ax.text(neg + pos + 6, y, note, va="center", fontsize=8.5, color=TEXT_PRIMARY)
    ax.set_yticks(ypos, [f"{s} (n={int(sites.loc[s, 'Patients'])})" for s in SITES])
    ax.set_xlim(0, 430)
    ax.set_xlabel("Number of patients")
    ax.grid(axis="y", visible=False)
    ax.text(0.995, 0.03, f"All sites: {int(pooled['Disease'])}/{int(pooled['Patients'])}\n"
            f"= {pooled['Prevalence (%)']:.1f}% disease", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=8, fontweight="bold", color=TEXT_SECONDARY)
    handles = [Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c]) for c in (0, 1)]
    fig.legend(handles=handles, loc="outside upper right", ncols=2, fontsize=8)
    fig.suptitle("Disease prevalence by hospital", x=0.0, ha="left",
                 fontsize=10.5, fontweight="bold")
    return savefig(fig, "eda_class_balance")


def plot_missingness(miss, cc):
    order = miss[ALL_SITES].sort_values(ascending=False, kind="stable").index
    mat = miss.loc[order]
    fig = plt.figure(figsize=(FIG_W, 4.3), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1])
    ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    im = ax.imshow(mat.to_numpy(), cmap=SEQUENTIAL_CMAP, vmin=0, vmax=100, aspect="auto")
    for i, f in enumerate(order):
        for j, col in enumerate(mat.columns):
            v = mat.loc[f, col]
            ax.text(j, i, "0" if v == 0 else f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 55 else TEXT_PRIMARY)
    ax.set_xticks(range(mat.shape[1]), [c.replace("Long Beach VA", "Long Beach") for c in mat.columns],
                  rotation=30, ha="right", rotation_mode="anchor", fontsize=8)
    ax.set_yticks(range(len(order)), [FEATURE_LABELS[f] for f in order], fontsize=8)
    ax.axvline(len(SITES) - 0.5, color="white", lw=3)  # separate the pooled column
    style_heatmap_axes(ax)
    cbar = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.02, aspect=25)
    cbar.set_label("% missing", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.outline.set_visible(False)
    ax.set_title("Missing values (% of patients)", fontsize=9.5)

    labels = cc["Site"].str.replace("Long Beach VA", "Long Beach").tolist()
    y = np.arange(len(labels))[::-1]
    h = 0.38
    series = [("Complete, all 13 (%)", "All 13 features", SEQUENTIAL_CMAP(0.85)),
              ("Complete, without ca/slope/thal (%)", "Without ca, slope, thal", SEQUENTIAL_CMAP(0.45))]
    for k, (col, lab, color) in enumerate(series):
        yy = y + (h / 2 if k == 0 else -h / 2)
        ax2.barh(yy, cc[col], height=h, color=color, label=lab)
        for yi, v in zip(yy, cc[col]):
            ax2.text(v + 2, yi, f"{v:.0f}", va="center", fontsize=7.5, color=TEXT_SECONDARY)
    ax2.set_yticks(y, labels, fontsize=8)
    ax2.set_xlim(0, 118)
    ax2.set_xticks([0, 50, 100])
    ax2.tick_params(axis="x", labelsize=8)
    ax2.set_xlabel("Rows kept (%)", fontsize=8.5)
    ax2.grid(axis="y", visible=False)
    ax2.set_title("Complete cases", fontsize=9.5, loc="right")
    handles = [Patch(color=color, label=lab) for _, lab, color in series]
    fig.legend(handles=handles, loc="outside lower right", ncols=2, fontsize=7.5)
    fig.suptitle("Missingness is concentrated in a few features and hospitals",
                 x=0.0, ha="left", fontsize=10.5, fontweight="bold")
    return savefig(fig, "eda_missingness")


def plot_numeric_distributions(df, tests):
    effect = tests.set_index("feature")["effect_size"]
    fig, axes = plt.subplots(2, 3, figsize=(FIG_W, 4.3), layout="constrained")
    for ax, f in zip(axes.flat, DIST_FEATURES):
        x_all = df[f].dropna()
        bins = np.histogram_bin_edges(x_all, bins=18)
        grid = np.linspace(bins[0], bins[-1], 300)
        for cls in (0, 1):
            x = df.loc[df[TARGET] == cls, f].dropna()
            ax.hist(x, bins=bins, density=True, color=CLASS_COLORS[cls], alpha=0.28,
                    histtype="stepfilled", edgecolor="none")
            ax.plot(grid, stats.gaussian_kde(x)(grid), color=CLASS_COLORS[cls], lw=1.6)
        ax.set_title(f"{SHORT_LABELS[f]} (r = {effect[f]:+.2f})", fontsize=8.5)
        ax.set_xlabel(FEATURE_LABELS[f], fontsize=8)
        ax.tick_params(labelsize=7.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Density")

    # Major vessels is a 0-3 count, so bars of class shares read better than a density.
    ax = axes[1, 2]
    width = 0.38
    for k, cls in enumerate((0, 1)):
        x = df.loc[df[TARGET] == cls, "ca"].dropna()
        share = x.value_counts(normalize=True).reindex([0.0, 1.0, 2.0, 3.0], fill_value=0) * 100
        ax.bar(np.arange(4) + (k - 0.5) * width, share, width=width, color=CLASS_COLORS[cls])
    n_ca = int(df["ca"].notna().sum())
    ax.set_title(f"{SHORT_LABELS['ca']} (r = {effect['ca']:+.2f})", fontsize=8.5)
    ax.set_xticks(range(4), ["0", "1", "2", "3"])
    ax.set_xlabel(f"Vessels (n={n_ca} recorded)", fontsize=8)
    ax.set_ylabel("% of class", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.grid(axis="x", visible=False)

    handles = [Patch(color=CLASS_COLORS[c], alpha=0.6, label=CLASS_NAMES[c]) for c in (0, 1)]
    fig.legend(handles=handles, loc="outside lower center", ncols=2)
    fig.suptitle("Numeric features by diagnosis (r = rank-biserial effect size)",
                 x=0.0, ha="left", fontsize=10.5, fontweight="bold")
    return savefig(fig, "eda_numeric_distributions")


def plot_categorical_rates(levels, pooled_prev):
    n_rows = max(sum(1 + (levels["feature"] == f).sum() for f in panel) for panel in LEVEL_PANELS)
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 4.3), layout="constrained")
    for ax, panel in zip(axes, LEVEL_PANELS):
        y = n_rows - 1
        ticks, labels, is_header = [], [], []
        for f in panel:
            # Header row (feature name as a bold tick label), then one bar per level.
            ticks.append(y)
            labels.append(LEVEL_HEADERS[f])
            is_header.append(True)
            ax.text(118, y, "n", ha="right", va="center", fontsize=7, fontweight="bold",
                    color=TEXT_SECONDARY)
            y -= 1
            for row in levels[levels["feature"] == f].itertuples():
                color = NEUTRAL if row.level == "Missing" else CLASS_COLORS[1]
                ax.barh(y, row.disease_rate_pct, height=0.72, color=color)
                # Counts sit in a fixed column right of 100% so they never cross bars or the line.
                ax.text(118, y, str(row.n), ha="right", va="center", fontsize=7, color=TEXT_SECONDARY)
                ticks.append(y)
                labels.append(row.level)
                is_header.append(False)
                y -= 1
        ax.axvline(pooled_prev, color=TEXT_SECONDARY, lw=1, ls="--", zorder=3)
        ax.set_yticks(ticks, labels, fontsize=7.5)
        for tick, header in zip(ax.get_yticklabels(), is_header):
            if header:
                tick.set_fontweight("bold")
                tick.set_color(TEXT_PRIMARY)
        ax.set_ylim(-0.7, n_rows - 0.3)
        ax.set_xlim(0, 120)
        ax.set_xticks([0, 50, 100])
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel("Disease rate (%)", fontsize=8.5)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    handles = [Patch(color=CLASS_COLORS[1], label="Disease rate in level"),
               Patch(color=NEUTRAL, label="Value missing"),
               Line2D([], [], color=TEXT_SECONDARY, lw=1, ls="--",
                      label=f"Pooled prevalence ({pooled_prev:.1f}%)")]
    fig.legend(handles=handles, loc="outside lower center", ncols=3)
    fig.suptitle("Disease rate by level of each discrete feature", x=0.0, ha="left",
                 fontsize=10.5, fontweight="bold")
    return savefig(fig, "eda_categorical_rates")


def plot_correlation(rho):
    sub = rho.iloc[1:, :-1]
    # Lower triangle incl. diagonal of the shifted matrix = strictly lower triangle of rho.
    mask = np.triu(np.ones(sub.shape, dtype=bool), k=1)
    values = np.ma.masked_where(mask, sub.to_numpy())
    # Full -1..1 scale: a narrower range would paint moderate correlations (max |rho| is
    # below 0.5) in the darkest colour and make them look stronger than they are.
    lim = 1.0
    fig, ax = plt.subplots(figsize=(FIG_W, 4.3), layout="constrained")
    im = ax.imshow(values, cmap=DIVERGING_CMAP, vmin=-lim, vmax=lim)
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            if not mask[i, j]:
                v = sub.iat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.8,
                        color="white" if abs(v) > 0.4 * lim else TEXT_PRIMARY)
    ax.set_xticks(range(sub.shape[1]), [SHORT_LABELS[c] for c in sub.columns],
                  rotation=40, ha="right", rotation_mode="anchor", fontsize=8)
    ax.set_yticks(range(sub.shape[0]), [SHORT_LABELS[c] for c in sub.index], fontsize=8)
    style_heatmap_axes(ax)
    # Outline the target row: these are the univariate rank correlations with disease.
    ax.add_patch(Rectangle((-0.5, sub.shape[0] - 1.5), sub.shape[1], 1, fill=False,
                           edgecolor=TEXT_PRIMARY, lw=1.2))
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02, aspect=25)
    cbar.set_label("Spearman rank correlation", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.outline.set_visible(False)
    fig.suptitle("Spearman correlations (pairwise complete; outlined row = heart disease)",
                 x=0.0, ha="left", fontsize=10.5, fontweight="bold")
    return savefig(fig, "eda_correlation")


def plot_site_differences(df):
    fig, axes = plt.subplots(1, 4, figsize=(FIG_W, 2.6), sharey=True, layout="constrained")
    positions = {s: len(SITES) - i for i, s in enumerate(SITES)}  # Cleveland on top
    for ax, f in zip(axes, SITE_BOX_FEATURES):
        data, pos, colors = [], [], []
        for s in SITES:
            x = df.loc[df["site"] == s, f].dropna()
            if len(x):
                data.append(x.to_numpy())
                pos.append(positions[s])
                colors.append(SITE_COLORS[s])
            else:
                ax.text(0.5, positions[s], "not recorded", transform=blended_transform_factory(
                    ax.transAxes, ax.transData), ha="center", va="center", fontsize=7.5,
                    style="italic", color=TEXT_SECONDARY)
        bp = ax.boxplot(data, positions=pos, orientation="horizontal", widths=0.6,
                        patch_artist=True, showfliers=True,
                        medianprops={"color": TEXT_PRIMARY, "lw": 1.4},
                        flierprops={"marker": "o", "markersize": 2.2, "markeredgewidth": 0,
                                    "markerfacecolor": NEUTRAL, "alpha": 0.7},
                        whiskerprops={"color": TEXT_SECONDARY, "lw": 0.9},
                        capprops={"color": TEXT_SECONDARY, "lw": 0.9})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor(TEXT_SECONDARY)
        ax.set_title(SHORT_LABELS[f], fontsize=9.5)
        ax.set_xlabel(UNITS[f], fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(list(positions.values()), list(positions.keys()), fontsize=8)
    fig.suptitle("Patient characteristics differ between hospitals", x=0.0, ha="left",
                 fontsize=10.5, fontweight="bold")
    return savefig(fig, "eda_site_differences")


def plot_pca(coords, y, sites, var_ratio, no_exercise):
    rng = np.random.default_rng(RANDOM_STATE)
    order = rng.permutation(len(y))  # random draw order so no group hides another
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 3.1), sharex=True, sharey=True, layout="constrained")
    groupings = [
        ("Coloured by diagnosis", [CLASS_COLORS[v] for v in y],
         [(CLASS_NAMES[c], CLASS_COLORS[c]) for c in (0, 1)]),
        ("Coloured by hospital", [SITE_COLORS[s] for s in sites],
         [(s, SITE_COLORS[s]) for s in SITES]),
    ]
    for ax, (title, colors, legend) in zip(axes, groupings):
        ax.scatter(coords[order, 0], coords[order, 1], c=np.asarray(colors)[order], s=9,
                   alpha=0.65, linewidths=0)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel(f"PC1 ({100 * var_ratio[0]:.1f}% of variance)")
        handles = [Line2D([], [], marker="o", ls="", markersize=5, color=c, label=lab) for lab, c in legend]
        ax.legend(handles=handles, loc="upper right", fontsize=7.5, handletextpad=0.2, borderaxespad=0.3)
        block = coords[no_exercise]
        ax.annotate(f"No exercise-test\nvalues (n={int(no_exercise.sum())})",
                    xy=(block[:, 0].mean(), block[:, 1].min() - 0.2),
                    xytext=(block[:, 0].mean() - 1.5, block[:, 1].min() - 1.3), ha="center", va="top",
                    fontsize=7.5, color=TEXT_SECONDARY,
                    arrowprops={"arrowstyle": "-", "color": TEXT_SECONDARY, "lw": 0.8})
    axes[0].set_ylabel(f"PC2 ({100 * var_ratio[1]:.1f}% of variance)")
    fig.suptitle(f"PCA of the preprocessed training data (n={len(y)}, default preprocessing)",
                 x=0.0, ha="left", fontsize=10.5, fontweight="bold")
    return savefig(fig, "eda_pca")


# --------------------------------------------------------------------------- PCA
def pca_analysis(train_df):
    """2-D PCA of the training data after the default ('indicator') preprocessing."""
    X_train, y_train = get_xy(train_df)
    prep = build_preprocessor("indicator")
    z = prep.fit_transform(X_train)
    names = prep.get_feature_names_out()
    pca = PCA(random_state=RANDOM_STATE).fit(z)
    coords = pca.transform(z)[:, :2]
    cum = np.cumsum(pca.explained_variance_ratio_)

    loadings = pd.DataFrame(pca.components_[:2].T, index=names, columns=["PC1", "PC2"])
    loadings.insert(0, "readable", [readable_column(n) for n in names])
    top = {pc: [{"column": c, "readable": loadings.loc[c, "readable"],
                 "loading": round(float(loadings.loc[c, pc]), 3)}
                for c in loadings[pc].abs().sort_values(ascending=False).index[:6]]
           for pc in ("PC1", "PC2")}

    # Same projection without missing-value information, to show what drives the site split.
    z_simple = build_preprocessor("simple").fit_transform(X_train)
    var_simple = PCA(n_components=2, random_state=RANDOM_STATE).fit(z_simple).explained_variance_ratio_

    sites = train_df["site"].to_numpy()
    # Rows where blood pressure, max heart rate and ST depression are all missing form a
    # separate block on PC1 (their three missing indicators switch on together).
    no_exercise = train_df[["trestbps", "thalach", "oldpeak"]].isna().all(axis=1).to_numpy()
    summary = {
        "n_train_rows": int(z.shape[0]), "n_columns": int(z.shape[1]),
        "explained_variance_pct": [round(100 * v, 1) for v in pca.explained_variance_ratio_[:5]],
        "pc1_pc2_cumulative_pct": round(100 * cum[1], 1),
        "components_for_80pct": int(np.searchsorted(cum, 0.80) + 1),
        "components_for_90pct": int(np.searchsorted(cum, 0.90) + 1),
        "silhouette_2d_by_site": round(float(silhouette_score(coords, sites)), 3),
        "silhouette_2d_by_target": round(float(silhouette_score(coords, y_train)), 3),
        "top_loadings": top,
        "simple_strategy_explained_variance_pct": [round(100 * v, 1) for v in var_simple],
        "no_exercise_test_block": {
            "n": int(no_exercise.sum()),
            "sites": pd.Series(sites[no_exercise]).value_counts().to_dict(),
            "disease_rate_pct": round(100 * y_train[no_exercise].mean(), 1),
            "mean_pc1": round(float(coords[no_exercise, 0].mean()), 2),
            "mean_pc1_other_rows": round(float(coords[~no_exercise, 0].mean()), 2),
        },
    }
    return coords, y_train, sites, pca.explained_variance_ratio_, loadings.round(4), summary, no_exercise


# --------------------------------------------------------------------------- report tables
def write_markdown(path, overview, num_dist, miss, dictionary, tests, desc, cc, split, prep_rows):
    tests_md = pd.DataFrame({
        "Feature": tests["label"],
        "Test": tests["test"],
        "n": tests["n"],
        "Summary": tests["summary_measure"],
        "No disease": tests["no_disease"],
        "Disease": tests["disease"],
        "p (Holm)": tests["p_holm"].map(fmt_p),
        "Effect size": [f"{m} = {e:+.2f}" if m.startswith("rank") else f"V = {e:.2f}"
                        for m, e in zip(tests["effect_measure"], tests["effect_size"])],
        "Magnitude": tests["magnitude"],
    })
    desc_md = pd.DataFrame({
        "Feature": desc["label"],
        "No disease: mean (SD)": [f"{m:.2f} ({s:.2f})" for m, s in zip(desc["mean_no_disease"], desc["std_no_disease"])],
        "No disease: median": desc["median_no_disease"],
        "Disease: mean (SD)": [f"{m:.2f} ({s:.2f})" for m, s in zip(desc["mean_disease"], desc["std_disease"])],
        "Disease: median": desc["median_disease"],
    })
    miss_md = miss.loc[miss[ALL_SITES].sort_values(ascending=False, kind="stable").index]
    miss_md = miss_md.rename(index=FEATURE_LABELS).rename_axis("Feature").reset_index()
    prep_md = pd.DataFrame([{
        "Strategy": r["strategy"], "Site one-hot": "yes" if r["include_site"] else "no",
        "Output shape (train)": f"{r['n_rows']} x {r['n_columns']}",
        "Missing indicators": len(r["missing_indicators"]),
        "Missing-level dummies": len(r["missing_level_dummies"]),
        "One-hot dummies": len(r["one_hot_dummies"]),
    } for r in prep_rows])

    sections = [
        ("Table 1. Patients and disease prevalence per hospital", overview),
        ("Table 2. Original diagnosis code (num: 0 = no disease, 1-4 = increasing severity)", num_dist),
        ("Table 3. Missing values per feature (% of patients)", miss_md),
        ("Table 4. Feature dictionary", dictionary.drop(columns="Feature")),
        ("Table 5. Univariate association with heart disease (sorted by effect size; "
         "p-values Holm-adjusted over 13 tests)", tests_md),
        ("Table 6. Numeric and binary features by class (binary means are proportions)", desc_md),
        ("Table 7. Rows kept under complete-case deletion", cc),
        ("Table 8. Train/test split (stratified on site and target)", split),
        ("Table 9. Preprocessing strategies (fitted on the training split)", prep_md),
    ]
    text = "# Exploratory data analysis tables\n\n" + "\n\n".join(
        f"## {title}\n\n{to_markdown(table)}" for title, table in sections) + "\n"
    path.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- main
def main():
    start = time.perf_counter()
    raw = load_raw()
    df = clean(raw)
    train_df, test_df = split_data(df)

    # Save the cleaned data and the fixed split for transparency/reuse.
    df.to_csv(DATA_PROCESSED / "heart_disease_clean.csv", index=False)
    train_df.to_csv(DATA_PROCESSED / "heart_disease_train.csv", index=False)
    test_df.to_csv(DATA_PROCESSED / "heart_disease_test.csv", index=False)

    overview = site_overview(df)
    num_dist = num_distribution(df)
    zeros = placeholder_zeros(raw)
    dups = duplicate_report(raw, train_df, test_df)
    miss = missing_summary(df).rename(columns={"All sites": ALL_SITES})
    dictionary = feature_dictionary(df)
    desc = descriptive_by_target(df)
    levels = categorical_levels(df)
    tests = univariate_tests(df)
    miss_target = missingness_vs_target(df)
    site_tests = site_heterogeneity(df)
    cc = complete_case_table(df)
    split = split_overview(train_df, test_df)
    prep_rows = preprocessing_evidence(train_df)
    rho = df[CORR_VARS].corr(method="spearman")
    pair_n = df[CORR_VARS].notna().astype(int).T @ df[CORR_VARS].notna().astype(int)
    coords, y_train, train_sites, var_ratio, loadings, pca_summary, no_exercise = pca_analysis(train_df)

    # ---- CSV tables
    overview.to_csv(RESULTS_DIR / "eda_site_overview.csv", index=False)
    num_dist.to_csv(RESULTS_DIR / "eda_num_distribution.csv", index=False)
    miss.rename_axis("feature").to_csv(RESULTS_DIR / "eda_missingness.csv")
    dictionary.to_csv(RESULTS_DIR / "eda_feature_dictionary.csv", index=False)
    desc.to_csv(RESULTS_DIR / "eda_descriptive_by_target.csv", index=False)
    levels.to_csv(RESULTS_DIR / "eda_categorical_levels.csv", index=False)
    tests.to_csv(RESULTS_DIR / "eda_univariate_tests.csv", index=False)
    miss_target.drop(columns="share_of_missing_by_site_pct").to_csv(
        RESULTS_DIR / "eda_missingness_vs_target.csv", index=False)
    site_tests.to_csv(RESULTS_DIR / "eda_site_tests.csv", index=False)
    cc.to_csv(RESULTS_DIR / "eda_complete_case.csv", index=False)
    split.to_csv(RESULTS_DIR / "eda_split.csv", index=False)
    rho.round(3).to_csv(RESULTS_DIR / "eda_correlation_spearman.csv")
    loadings.to_csv(RESULTS_DIR / "eda_pca_loadings.csv")
    pd.DataFrame([{k: v for k, v in r.items() if k != "columns"} | {
        k: "; ".join(r[k]) for k in ("missing_indicators", "missing_level_dummies", "one_hot_dummies", "site_dummies")
    } for r in prep_rows]).to_csv(RESULTS_DIR / "eda_preprocessing_shapes.csv", index=False)
    write_markdown(RESULTS_DIR / "eda_tables.md", overview, num_dist, miss, dictionary, tests,
                   desc, cc, split, prep_rows)

    # ---- figures
    pooled_prev = float(overview.set_index("Site").loc[ALL_SITES, "Prevalence (%)"])
    figures = [
        plot_class_balance(overview),
        plot_missingness(miss, cc),
        plot_numeric_distributions(df, tests),
        plot_categorical_rates(levels, pooled_prev),
        plot_correlation(rho),
        plot_site_differences(df),
        plot_pca(coords, y_train, train_sites, var_ratio, no_exercise),
    ]

    # ---- JSON summary (every number the report quotes)
    ov = overview.set_index("Site")
    target_rho = rho[TARGET].drop(TARGET)
    summary = {
        "n_patients": len(df), "n_features": len(FEATURES),
        "feature_groups": {"numeric": NUMERIC_FEATURES, "binary": BINARY_FEATURES,
                           "categorical": CATEGORICAL_FEATURES},
        "pooled_prevalence_pct": pooled_prev,
        "class_counts": {"disease": int(df[TARGET].sum()), "no_disease": int((df[TARGET] == 0).sum())},
        "sites": {s: {"n": int(ov.loc[s, "Patients"]), "n_disease": int(ov.loc[s, "Disease"]),
                      "prevalence_pct": float(ov.loc[s, "Prevalence (%)"]),
                      "share_of_data_pct": float(ov.loc[s, "Share of data (%)"]),
                      "median_age": float(ov.loc[s, "Median age"]),
                      "male_pct": float(ov.loc[s, "Male (%)"])} for s in SITES},
        "num_distribution": num_dist.set_index("Site").to_dict(orient="index"),
        "placeholder_zeros_in_raw_files": zeros,
        "duplicates": dups,
        "missing_pct_by_feature_and_site": miss.to_dict(orient="index"),
        "missing_cells_pct_overall": round(100 * df[FEATURES].isna().to_numpy().mean(), 1),
        "features_without_missing": [f for f in FEATURES if df[f].notna().all()],
        "complete_case": cc.set_index("Site").to_dict(orient="index"),
        "missingness_vs_target": records(miss_target),
        "univariate_tests": records(tests),
        "strongest_associations": tests.head(5)[["feature", "test", "effect_measure", "effect_size",
                                                 "p_holm", "direction"]].to_dict(orient="records"),
        "not_significant_after_holm": tests.loc[tests["p_holm"] >= 0.05, "feature"].tolist(),
        "spearman_with_target": target_rho.round(3).sort_values(key=np.abs, ascending=False).to_dict(),
        "spearman_pairwise_n": {c: int(pair_n.loc[c, TARGET]) for c in CORR_VARS if c != TARGET},
        "site_heterogeneity": records(site_tests),
        "split": split.set_index("Split").to_dict(orient="index"),
        "preprocessing": [{k: v for k, v in r.items() if k != "columns"} | {"n_output_columns": r["n_columns"]}
                          for r in prep_rows],
        "pca": pca_summary,
        "notes": [
            "Descriptive statistics and tests use all 920 patients; no modelling decision is based on them.",
            "PCA uses the training split only (preprocessor fitted on train).",
            "Chi-square tests are Pearson (no continuity correction); missing values excluded per test.",
            "chol == 0 and trestbps == 0 in the raw files were recoded as missing before all analyses.",
        ],
        "figures": [p.name for p in figures],
    }
    save_json(summary, RESULTS_DIR / "eda_summary.json")

    # ---- console summary
    print(f"Patients: {len(df)} | pooled prevalence {pooled_prev:.1f}% | "
          f"train/test {len(train_df)}/{len(test_df)}")
    for s in SITES:
        print(f"  {s:<14} n={ov.loc[s, 'Patients']:>3}  prevalence {ov.loc[s, 'Prevalence (%)']:5.1f}%  "
              f"complete cases {ov.loc[s, 'Complete cases (%)']:5.1f}%")
    print(f"Placeholder zeros: chol={zeros['chol']['total']}, trestbps={zeros['trestbps']['total']}; "
          f"exact duplicates: {dups['exact_duplicate_rows']}")
    print("Pooled missing %: " + ", ".join(f"{f}={v:.1f}" for f, v in
                                           miss[ALL_SITES].sort_values(ascending=False).items() if v > 0))
    print("Strongest associations: " + ", ".join(
        f"{r.feature} ({r.effect_measure.split()[0]} {r.effect_size:+.2f})" for r in tests.head(5).itertuples()))
    print(f"PCA: PC1 {100 * var_ratio[0]:.1f}%, PC2 {100 * var_ratio[1]:.1f}% | silhouette by site "
          f"{pca_summary['silhouette_2d_by_site']:.2f} vs by target {pca_summary['silhouette_2d_by_target']:.2f}")
    print("Preprocessed shapes (train): " + ", ".join(
        f"{r['strategy']}{'+site' if r['include_site'] else ''}={r['n_rows']}x{r['n_columns']}" for r in prep_rows))
    print(f"Saved {len(figures)} figures and EDA tables in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
