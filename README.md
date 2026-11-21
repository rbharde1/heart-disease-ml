# Predicting Heart Disease Across Four Hospitals

Machine Learning course project: binary prediction of angiographically confirmed coronary
heart disease from 13 routine clinical attributes, using **all four cohorts** of the
[UCI Heart Disease data](https://doi.org/10.24432/C52P4X) (920 patients from Cleveland, Hungary,
Switzerland and Long Beach VA) instead of the usual 303-patient Cleveland subset.

**Report:** [`report/report.pdf`](report/report.pdf)

## Results at a glance

Held-out test set (184 patients, threshold 0.5, 95% bootstrap CIs for AUC):

| Model | Accuracy | Recall | Specificity | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (L2, C = 0.1) | 0.837 | 0.873 | 0.793 | 0.856 | **0.914** [0.866, 0.953] |
| SVM, RBF kernel (C = 5, γ = 0.01) | **0.848** | **0.912** | 0.768 | **0.869** | 0.913 [0.868, 0.953] |
| Random Forest (500 trees) | 0.842 | 0.892 | 0.780 | 0.863 | 0.900 [0.850, 0.943] |
| Majority baseline | 0.554 | 1.000 | 0.000 | 0.713 | 0.500 |

Key findings:

- The three models are statistically indistinguishable (corrected resampled *t*-test, McNemar,
  DeLong; all *p* > 0.18). Learning and validation curves show that performance is limited
  by the information in the features, not by model capacity.
- Missing values (16% of all cells) follow the hospital: the missing-value flags alone identify
  the hospital with 85% accuracy and predict disease with AUC 0.75, which is a shortcut that does
  not transfer to new hospitals.
- Leave-one-hospital-out validation lowers per-hospital AUC by 0.06-0.09 (excluding Switzerland,
  which has only 2 healthy test patients), and predicted probabilities tend to follow the
  training hospitals' prevalence.
- For screening, a threshold of about 0.41 (chosen on training data only) reaches 91–92%
  sensitivity on the test set.

## Project structure

```
data/raw/                 UCI processed files (cleveland, hungarian, switzerland, va) + documentation
data/processed/           cleaned data set and the train/test split (generated)
src/
  config.py               paths, random seed, split and CV settings
  data.py                 loading, cleaning (impossible zeros -> missing), stratified split
  preprocess.py           imputation + missing indicators, one-hot encoding, scaling (sklearn Pipeline)
  models.py               model definitions, hyperparameter grids, tuning
  evaluation.py           metrics, bootstrap confidence intervals
  plotting.py             shared figure style and colours
  eda.py                  exploratory data analysis and statistics
  train_evaluate.py       tuning, repeated and nested CV, test evaluation, statistical tests
  experiments_data.py     preprocessing ablation, leave-one-site-out, missingness, Cleveland benchmark
  experiments_model.py    learning/validation curves, importance, thresholds, calibration, errors
figures/                  all figures (PNG, 300 dpi)
results/                  every reported number (CSV / JSON)
report/                   report source (HTML/CSS), build script and PDF
run_all.py                reproduces every figure and result
```

## How to run

Requires Python 3.10+.

```bash
pip install -r requirements.txt
python run_all.py                 # about 4 minutes; regenerates figures/ and results/
python report/build_report.py     # optional: re-render report/report.pdf (needs Chrome or Edge and internet for MathJax)
```

`results/best_params.json` caches the tuned hyperparameters; delete it to re-run the grid search
from scratch. Every random step uses seed 42, so results are reproducible.

## Data

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X (CC BY 4.0).
The principal investigators requested acknowledgement: Andras Janosi, M.D. (Hungarian Institute of
Cardiology, Budapest); William Steinbrunn, M.D. (University Hospital, Zurich); Matthias Pfisterer, M.D.
(University Hospital, Basel); Robert Detrano, M.D., Ph.D. (V.A. Medical Center, Long Beach and Cleveland
Clinic Foundation).
