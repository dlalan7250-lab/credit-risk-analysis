# Credit default prediction

Probability-of-default model for consumer loans using the public [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset). The original notebook treated this as a Colab experiment with accuracy as the headline metric. This rewrite is a local, leak-free scoring pipeline.

## Why this rewrite

The previous version had several issues that inflate reported performance:

- Duplicates were never actually dropped (`drop_duplicates()` result was discarded).
- Missing values, encoding, and scaling were fit on the **full** dataset before the train/test split.
- The KNN neighborhood size was chosen by scanning the **test** set.
- The target was described as fraud; it is loan default (`loan_status`).
- Accuracy (~93%) looks strong because about 78% of borrowers do not default. This project selects models by **ROC-AUC** and **PR-AUC**, then sets a recall-oriented (F2) threshold.

## Dataset

| Column | Meaning |
| --- | --- |
| `person_age` | Applicant age |
| `person_income` | Annual income |
| `person_home_ownership` | RENT / OWN / MORTGAGE / OTHER |
| `person_emp_length` | Years employed |
| `loan_intent` | Loan purpose |
| `loan_grade` | Assigned grade A–G |
| `loan_amnt` | Requested amount |
| `loan_int_rate` | Interest rate |
| `loan_percent_income` | Loan amount / income |
| `cb_person_default_on_file` | Prior default (Y/N) |
| `cb_person_cred_hist_length` | Credit history length |
| `loan_status` | **Target:** 1 = default, 0 = non-default |

Impossible rows are removed (age outside 18–100, employment longer than `age - 14`, income above $2M). Remaining missing `person_emp_length` and `loan_int_rate` values are imputed **inside** the sklearn pipeline on training folds only.

## Setup

Use the Anaconda Python that already has scikit-learn, or any 3.10+ environment:

```powershell
cd c:\Projects\credit-analysis
python -m pip install -r requirements.txt
```

On this machine the interpreter with scikit-learn is `C:\ProgramData\anaconda3\python.exe`. The Microsoft Store `python` on PATH does not have the ML stack.

## Train

```powershell
python -m src.train
```

This stratified 80/20 split, 5-fold CV comparison, F2 threshold, and held-out test evaluation writes:

- `models/best_model.joblib` — preprocessing + classifier + threshold
- `reports/model_comparison.csv`
- `reports/feature_importance.csv`
- `reports/summary.json`

## Score an applicant

```powershell
python -m src.predict --age 29 --income 62000 --home RENT --emp-length 4 --intent EDUCATION --grade B --amount 10000 --rate 11.5 --prior-default N --cred-hist 5
```

Interactive UI:

```powershell
streamlit run app.py
```

## Analysis notebook

Walkthrough of EDA and methodology: `notebooks/credit_risk_analysis.ipynb`.

## Results

Histogram gradient boosting wins on 5-fold train ROC-AUC. Held-out test (6,482 loans, 21.9% default rate):

| Metric | Value | Meaning |
| --- | ---: | --- |
| ROC-AUC | 0.949 | Ranking quality |
| PR-AUC | 0.906 | Default-class ranking |
| KS | 0.761 | Score separation |
| Recall (default) | 0.866 | Share of defaults caught at the F2 threshold 0.35 |
| Precision (default) | 0.636 | Share of flagged loans that actually default |
| Accuracy | 0.862 | Not used for selection; the old notebook's ~93% figure was inflated by class imbalance and leakage |

Cross-validation ROC-AUC (train folds, threshold 0.5):

| Model | ROC-AUC | PR-AUC |
| --- | ---: | ---: |
| Histogram gradient boosting | 0.942 | 0.897 |
| Random forest | 0.933 | 0.883 |
| Decision tree | 0.911 | 0.863 |
| k-nearest neighbors | 0.892 | 0.812 |
| Logistic regression | 0.871 | 0.707 |

Permutation importance on the test set is dominated by `loan_percent_income`, `loan_grade`, `person_income`, and `person_home_ownership`.

## Models compared

Logistic regression, k-nearest neighbors, decision tree, random forest, and histogram gradient boosting. Trees and logistic regression use `class_weight="balanced"` instead of SMOTE — oversampling mixed numeric/categorical credit records after encoding creates synthetic borrowers that do not exist.

## Project layout

```
data/credit_risk_dataset.csv
src/                 training, scoring, metrics
notebooks/           EDA and methodology
app.py               Streamlit scorer
reports/             metrics after training
tests/               data-cleaning checks
```

## Contributors

- Sanket Prakash

- Lalan Kumar Das
