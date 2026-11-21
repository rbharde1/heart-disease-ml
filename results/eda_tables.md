# Exploratory data analysis tables

## Table 1. Patients and disease prevalence per hospital

| Site | Patients | Disease | No disease | Prevalence (%) | Share of data (%) | Median age | Male (%) | Complete cases (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cleveland | 303 | 139 | 164 | 45.9 | 32.9 | 56.0 | 68.0 | 98.0 |
| Hungary | 294 | 106 | 188 | 36.1 | 32.0 | 49.0 | 72.4 | 0.3 |
| Switzerland | 123 | 115 | 8 | 93.5 | 13.4 | 56.0 | 91.9 | 0.0 |
| Long Beach VA | 200 | 149 | 51 | 74.5 | 21.7 | 60.0 | 97.0 | 0.5 |
| All sites | 920 | 509 | 411 | 55.3 | 100.0 | 54.0 | 78.9 | 32.5 |

## Table 2. Original diagnosis code (num: 0 = no disease, 1-4 = increasing severity)

| Site | num=0 | num=1 | num=2 | num=3 | num=4 |
|---|---:|---:|---:|---:|---:|
| Cleveland | 164 | 55 | 36 | 35 | 13 |
| Hungary | 188 | 106 | 0 | 0 | 0 |
| Switzerland | 8 | 48 | 32 | 30 | 5 |
| Long Beach VA | 51 | 56 | 41 | 42 | 10 |
| All sites | 411 | 265 | 109 | 107 | 28 |

## Table 3. Missing values per feature (% of patients)

| Feature | Cleveland | Hungary | Switzerland | Long Beach VA | All sites |
|---|---:|---:|---:|---:|---:|
| Major vessels coloured (0-3) | 1.3 | 99.0 | 95.9 | 99.0 | 66.4 |
| Thallium stress test | 0.7 | 90.5 | 42.3 | 83.0 | 52.8 |
| Slope of peak exercise ST | 0.0 | 64.6 | 13.8 | 51.0 | 33.6 |
| Serum cholesterol (mg/dl) | 0.0 | 7.8 | 100.0 | 28.0 | 22.0 |
| Fasting blood sugar > 120 mg/dl | 0.0 | 2.7 | 61.0 | 3.5 | 9.8 |
| ST depression (oldpeak) | 0.0 | 0.0 | 4.9 | 28.0 | 6.7 |
| Resting blood pressure (mm Hg) | 0.0 | 0.3 | 1.6 | 28.5 | 6.5 |
| Max heart rate achieved (bpm) | 0.0 | 0.3 | 0.8 | 26.5 | 6.0 |
| Exercise-induced angina | 0.0 | 0.3 | 0.8 | 26.5 | 6.0 |
| Resting ECG | 0.0 | 0.3 | 0.8 | 0.0 | 0.2 |
| Age (years) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Sex | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Chest pain type | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Table 4. Feature dictionary

| Name | Description | Type | Values / range | Missing (%) | Treatment (default pipeline) |
|---|---|---|---|---:|---|
| Age (years) | Age in years | Numeric (continuous) | 28 to 77 | 0.0 | Standardised (no missing values) |
| Resting blood pressure (mm Hg) | Resting blood pressure on admission (mm Hg) | Numeric (continuous) | 80 to 200 | 6.5 | Median imputation + missing indicator, standardised |
| Serum cholesterol (mg/dl) | Serum cholesterol (mg/dl) | Numeric (continuous) | 85 to 603 | 22.0 | Median imputation + missing indicator, standardised |
| Max heart rate achieved (bpm) | Maximum heart rate achieved during exercise test (bpm) | Numeric (continuous) | 60 to 202 | 6.0 | Median imputation + missing indicator, standardised |
| ST depression (oldpeak) | ST depression induced by exercise relative to rest (mm) | Numeric (continuous) | -2.6 to 6.2 | 6.7 | Median imputation + missing indicator, standardised |
| Major vessels coloured (0-3) | Number of major vessels (0-3) coloured by fluoroscopy | Numeric (ordinal count) | 0-3 | 66.4 | Median imputation + missing indicator, standardised; dropped in 'drop_sparse' |
| Sex | Sex (1 = male, 0 = female) | Binary | 0 = Female, 1 = Male | 0.0 | Kept as 0/1 (no missing values) |
| Fasting blood sugar > 120 mg/dl | Fasting blood sugar > 120 mg/dl (1 = true) | Binary | 0 = No, 1 = Yes | 9.8 | Mode imputation, kept as 0/1 |
| Exercise-induced angina | Exercise-induced angina (1 = yes) | Binary | 0 = No, 1 = Yes | 6.0 | Mode imputation, kept as 0/1 |
| Chest pain type | Chest pain type | Categorical (nominal) | 1 = Typical angina, 2 = Atypical angina, 3 = Non-anginal pain, 4 = Asymptomatic | 0.0 | One-hot encoded; missing kept as its own level |
| Resting ECG | Resting electrocardiogram result | Categorical (nominal) | 0 = Normal, 1 = ST-T abnormality, 2 = LV hypertrophy | 0.2 | One-hot encoded; missing kept as its own level |
| Slope of peak exercise ST | Slope of the peak-exercise ST segment | Categorical (ordered levels) | 1 = Upsloping, 2 = Flat, 3 = Downsloping | 33.6 | One-hot encoded; missing kept as its own level; dropped in 'drop_sparse' |
| Thallium stress test | Thallium stress test result | Categorical (nominal) | 3 = Normal, 6 = Fixed defect, 7 = Reversible defect | 52.8 | One-hot encoded; missing kept as its own level; dropped in 'drop_sparse' |

## Table 5. Univariate association with heart disease (sorted by effect size; p-values Holm-adjusted over 13 tests)

| Feature | Test | n | Summary | No disease | Disease | p (Holm) | Effect size | Magnitude |
|---|---|---:|---|---|---|---|---|---|
| Chest pain type | Chi-square | 920 | % Asymptomatic | 25.3% | 77.0% | 9.1e-57 | V = 0.54 | large |
| Thallium stress test | Chi-square | 434 | % Reversible defect | 20.3% | 62.3% | 1.9e-23 | V = 0.50 | large |
| Major vessels coloured (0-3) | Mann-Whitney U | 309 | median [IQR] | 0 [0-0] | 1 [0-2] | 9.4e-17 | rank-biserial r = +0.50 | medium |
| Exercise-induced angina | Chi-square | 865 | % Yes | 14.1% | 59.5% | 2.9e-41 | V = 0.46 | medium |
| Max heart rate achieved (bpm) | Mann-Whitney U | 865 | median [IQR] | 151 [134.5-167.5] | 128 [112-145] | 9.4e-31 | rank-biserial r = -0.46 | medium |
| ST depression (oldpeak) | Mann-Whitney U | 858 | median [IQR] | 0 [0-0.8] | 1.05 [0-2] | 9.4e-31 | rank-biserial r = +0.45 | medium |
| Slope of peak exercise ST | Chi-square | 611 | % Downsloping | 6.4% | 12.5% | 3.6e-19 | V = 0.38 | medium |
| Age (years) | Mann-Whitney U | 920 | median [IQR] | 51 [43-57] | 57 [51-62] | 7.7e-18 | rank-biserial r = +0.34 | medium |
| Sex | Chi-square | 920 | % Male | 65.0% | 90.2% | 9.3e-20 | V = 0.31 | medium |
| Serum cholesterol (mg/dl) | Mann-Whitney U | 718 | median [IQR] | 233 [204-270.25] | 248 [216-284.75] | 0.002 | rank-biserial r = +0.15 | small |
| Fasting blood sugar > 120 mg/dl | Chi-square | 830 | % Yes | 11.1% | 21.7% | 1.6e-04 | V = 0.14 | small |
| Resting blood pressure (mm Hg) | Mann-Whitney U | 860 | median [IQR] | 130 [120-140] | 130 [120-145] | 0.003 | rank-biserial r = +0.13 | small |
| Resting ECG | Chi-square | 918 | % ST-T abnormality | 14.8% | 23.3% | 0.003 | V = 0.11 | small |

## Table 6. Numeric and binary features by class (binary means are proportions)

| Feature | No disease: mean (SD) | No disease: median | Disease: mean (SD) | Disease: median |
|---|---|---:|---|---:|
| Age (years) | 50.55 (9.43) | 51.0 | 55.90 (8.72) | 57.0 |
| Resting blood pressure (mm Hg) | 129.91 (16.87) | 130.0 | 134.26 (19.62) | 130.0 |
| Serum cholesterol (mg/dl) | 240.16 (55.77) | 233.0 | 254.01 (60.62) | 248.0 |
| Max heart rate achieved (bpm) | 148.80 (23.61) | 151.0 | 128.26 (24.02) | 128.0 |
| ST depression (oldpeak) | 0.42 (0.72) | 0.0 | 1.26 (1.20) | 1.05 |
| Major vessels coloured (0-3) | 0.28 (0.64) | 0.0 | 1.13 (1.01) | 1.0 |
| Sex | 0.65 (0.48) | 1.0 | 0.90 (0.30) | 1.0 |
| Fasting blood sugar > 120 mg/dl | 0.11 (0.31) | 0.0 | 0.22 (0.41) | 0.0 |
| Exercise-induced angina | 0.14 (0.35) | 0.0 | 0.59 (0.49) | 1.0 |

## Table 7. Rows kept under complete-case deletion

| Site | Patients | Complete (all 13 features) | Complete, all 13 (%) | Complete (without ca/slope/thal) | Complete, without ca/slope/thal (%) |
|---|---:|---:|---:|---:|---:|
| Cleveland | 303 | 297 | 98.0 | 303 | 100.0 |
| Hungary | 294 | 1 | 0.3 | 261 | 88.8 |
| Switzerland | 123 | 0 | 0.0 | 0 | 0.0 |
| Long Beach VA | 200 | 1 | 0.5 | 97 | 48.5 |
| All sites | 920 | 299 | 32.5 | 661 | 71.8 |

## Table 8. Train/test split (stratified on site and target)

| Split | Patients | Disease | Prevalence (%) | Cleveland | Hungary | Switzerland | Long Beach VA |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 736 | 407 | 55.3 | 242 | 236 | 98 | 160 |
| Test | 184 | 102 | 55.4 | 61 | 58 | 25 | 40 |

## Table 9. Preprocessing strategies (fitted on the training split)

| Strategy | Site one-hot | Output shape (train) | Missing indicators | Missing-level dummies | One-hot dummies |
|---|---|---|---:|---:|---:|
| indicator | no | 736 x 30 | 5 | 3 | 13 |
| indicator | yes | 736 x 34 | 5 | 3 | 13 |
| simple | no | 736 x 22 | 0 | 0 | 13 |
| simple | yes | 736 x 26 | 0 | 0 | 13 |
| drop_sparse | no | 736 x 15 | 0 | 0 | 7 |
| drop_sparse | yes | 736 x 19 | 0 | 0 | 7 |
