# ML Experiment Tracking Platform

A full-stack MLOps platform for uploading datasets, training and comparing multiple ML models, tracking experiment history, and deploying the best model behind a live prediction API — built to demonstrate ML engineering skills beyond "train one model in a notebook."

## What it does

1. **Upload** a CSV dataset and specify the target column
2. **Train** up to 6 classification algorithms simultaneously, with configurable outlier handling and class-imbalance correction
3. **Compare** every experiment ever run, filterable by dataset, sortable by any metric
4. **Promote** the best model (auto-selected by metric, or chosen manually) to production
5. **Predict** via a live REST API, accepting raw human-readable input (no manual encoding required)

## Why this project

Most ML portfolio projects stop at "I trained a model and got X% accuracy." This project is about the layer most tutorials skip: **experiment reproducibility, model comparison, and deployment infrastructure** — the actual day-to-day work of an ML engineer, not just a data scientist.

## Tech stack

- **Backend:** FastAPI, SQLite, scikit-learn, pandas, joblib
- **Frontend:** Vanilla HTML/CSS/JS (served directly by FastAPI — single deployable process)
- **Models:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, Gaussian Naive Bayes, KNN

## Folder structure

```
ml-experiment-tracker/
├── core/
│   ├── __init__.py
│   ├── db.py               # SQLite schema: datasets, experiments, production_model
│   ├── dataset_manager.py  # Upload, SHA-256 versioning, validation
│   ├── trainer.py          # Preprocessing, training, preprocessing pipeline system
│   └── registry.py         # Best-model selection and promotion logic
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI endpoints
│   └── static/
│       └── index.html       # Frontend (HTML/CSS/JS)
├── storage/
│   ├── datasets/             # Versioned uploaded CSVs
│   ├── models/                # Serialized model + pipeline bundles (.joblib)
│   └── tracker.db             # SQLite database
├── requirements.txt
├── .gitignore
└── README.md
```

## Key design decisions

**Dataset versioning by content hash, not filename.** Re-uploading identical data doesn't create duplicates — the file is hashed (SHA-256) and matched against existing records.

**Preprocessing is intentionally scoped, not "smart."** Missing-value imputation (median/mode), categorical encoding, and *optional* outlier capping (IQR, off by default) — deliberately not a full data-cleaning framework. Every decision is logged in a `preprocessing_report` returned to the caller, so nothing happens silently.

**Class imbalance handling is model-aware, not one-size-fits-all.** scikit-learn doesn't support `class_weight` uniformly — `GradientBoostingClassifier` has no such parameter at all. The registry tracks each model's supported strategy (`class_weight` vs `sample_weight` vs `none`) and applies the correct mechanism per model.

**Promotion is a separate step from training.** Finishing a training run does not deploy it. A model only goes live when explicitly promoted — either by auto-selecting the best experiment on a chosen metric, or by hand-picking a specific one. Both are scoped to a single dataset, so a fraud-detection model can never accidentally get promoted while working on a churn dataset.

**The full preprocessing pipeline is saved with the model, not just the model.** This was the most important fix in the project. Initially, `/predict` failed on raw input like `"Female"` because the model was trained on `LabelEncoder`-transformed integers and had no memory of that transformation. The fix: every training run now serializes a `pipeline` object (fitted encoders, fill values, outlier bounds, feature order) alongside the model, and `/predict` replays those exact transformations on new data — including graceful fallback for categories never seen during training.

## Real bugs found and fixed

This project was built with genuine iterative debugging, not a clean first pass. A few worth highlighting:

- **GradientBoosting scored ROC-AUC 0.35 (worse than random) on the Credit Card Fraud dataset** (0.17% positive rate). Root cause: `GradientBoostingClassifier` has no `class_weight` parameter, unlike `RandomForestClassifier`/`LogisticRegression`. Fixed with manually computed `sample_weight='balanced'` passed to `.fit()` — AUC recovered to ~0.98.
- **A continuous numeric column was wrongly dropped as an "ID column"** because it happened to be 100% unique by chance. Fixed by restricting ID-detection to non-float columns.
- **`TotalCharges` in the Telco Churn dataset was silently label-encoded as categorical** because blank-string entries broke pandas' automatic numeric dtype inference. Fixed with a threshold-based `pd.to_numeric(errors='coerce')` coercion step.
- **Auto-promotion picked the wrong model entirely** — it searched across every dataset ever trained on, not just the current one, once promoting a fraud-detection model while the user was working on churn data. Fixed by threading a `dataset_id` filter through the full stack.

## Results — tested on two structurally different real datasets

**Credit Card Fraud Detection** (284,807 rows, 0.17% fraud rate, fully numeric, extreme imbalance)

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.810 | 0.694 | 0.747 | 0.949 |
| Decision Tree | 0.890 | 0.745 | 0.811 | 0.810 |
| **Random Forest** | 0.906 | 0.786 | **0.842** | 0.957 |
| Gradient Boosting | 0.179 | 0.908 | 0.299 | **0.977** |
| Gaussian Naive Bayes | 0.115 | 0.765 | 0.200 | 0.968 |
| KNN | 1.000 | 0.031 | 0.059 | 0.636 |

**Telco Customer Churn** (7,043 rows, 27% churn rate, mixed categorical/numeric, real missing values)

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.498 | 0.789 | 0.610 | 0.837 |
| Decision Tree | 0.485 | 0.719 | 0.579 | 0.756 |
| Random Forest | 0.565 | 0.642 | 0.601 | 0.820 |
| **Gradient Boosting** | 0.523 | 0.802 | **0.633** | **0.844** |
| Gaussian Naive Bayes | 0.481 | 0.791 | 0.598 | 0.817 |
| KNN | 0.575 | 0.439 | 0.498 | 0.753 |

**No single model wins universally** — Random Forest was best for extreme-imbalance fraud detection, Gradient Boosting was best for moderate-imbalance churn prediction. This is the core thesis the platform demonstrates: you measure, you don't assume.

## Live inference validation

The promoted Random Forest model was stress-tested against a batch of 17 confirmed real fraud transactions and correctly caught 15 of them (~88%), consistent with its measured 78.6% recall on the held-out test set.

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/upload` | Upload a CSV, get back dataset_id + validation info |
| GET | `/models/available` | List available model types |
| POST | `/train` | Train one or more models on an uploaded dataset |
| GET | `/experiments` | List all logged experiments |
| POST | `/promote` | Promote an experiment (auto-pick or manual) to production |
| GET | `/production` | View the currently deployed model |
| POST | `/predict` | Run inference using the production model |

## Running locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/ml-experiment-tracker.git
cd ml-experiment-tracker

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Start the server (serves both API and frontend)
uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000` in your browser.

## What I'd add next

- Persist SQLite + model artifacts to cloud object storage for a true multi-instance deployment
- Multi-class classification support (currently binary only)
- Configurable decision threshold at inference time, instead of the fixed 0.5 default
- Automated hyperparameter tuning per model, logged as additional experiment variants
