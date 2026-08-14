"""
trainer.py
Preprocesses a dataset, trains multiple models, evaluates them, and logs
each run as an "experiment" via db.py.
"""

import joblib
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from core.db import insert_experiment
from core.dataset_manager import load_dataset

MODEL_DIR = Path(__file__).resolve().parent.parent / "storage" / "models"

MODEL_REGISTRY = {
    "LogisticRegression": {
        "class": LogisticRegression,
        "default_params": {"max_iter": 1000, "random_state": 42},
        "imbalance_strategy": "class_weight",
    },
    "DecisionTree": {
        "class": DecisionTreeClassifier,
        "default_params": {"random_state": 42, "max_depth": 10},
        "imbalance_strategy": "class_weight",
    },
    "RandomForest": {
        "class": RandomForestClassifier,
        "default_params": {"n_estimators": 100, "random_state": 42},
        "imbalance_strategy": "class_weight",
    },
    "GradientBoosting": {
        "class": GradientBoostingClassifier,
        "default_params": {"n_estimators": 100, "random_state": 42},
        "imbalance_strategy": "sample_weight",
    },
    "GaussianNB": {
        "class": GaussianNB,
        "default_params": {},
        "imbalance_strategy": "sample_weight",
    },
    "KNN": {
        "class": KNeighborsClassifier,
        "default_params": {"n_neighbors": 5},
        "imbalance_strategy": "none",
    },
}


def cap_outliers_iqr(df: pd.DataFrame, numeric_columns) -> pd.DataFrame:
    df = df.copy()
    for col in numeric_columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def preprocess(df: pd.DataFrame, target_column: str, handle_outliers: bool = False):
    """
    Returns (X, y, preprocessing_report, pipeline).

    pipeline captures every transformation applied (which columns were
    dropped, coerced, imputed, encoded, and with what fitted values/encoders)
    so the exact same steps can be replayed on new data at prediction time.
    Without this, /predict has no way to know "Female" needs to become 0
    before reaching the model — discovered when raw categorical strings sent
    to /predict caused "could not convert string to float" errors, since the
    model was trained on LabelEncoder-transformed integers, not raw text.
    """
    df = df.copy()
    report = {"missing_filled": {}, "categorical_encoded": [], "outliers_capped": False}
    pipeline = {
        "target_column": target_column,
        "dropped_id_like_columns": [],
        "coerced_to_numeric_columns": [],
        "numeric_fill_values": {},
        "categorical_fill_value": "Unknown",
        "categorical_encoders": {},   # col -> fitted LabelEncoder
        "target_encoder": None,       # fitted LabelEncoder for y, or None if y was already numeric
        "outlier_bounds": {},         # col -> (lower, upper)
        "feature_order": [],          # exact column order the model expects
    }

    y = df[target_column]
    X = df.drop(columns=[target_column])

    id_like_cols = [
        col for col in X.columns
        if X[col].nunique() == len(X) and X[col].dtype != np.float64
    ]
    if id_like_cols:
        X = X.drop(columns=id_like_cols)
        report["dropped_id_like_columns"] = id_like_cols
        pipeline["dropped_id_like_columns"] = id_like_cols

    coerced_cols = []
    for col in X.select_dtypes(exclude=[np.number]).columns:
        non_null = X[col].notna().sum()
        if non_null == 0:
            continue
        converted = pd.to_numeric(X[col], errors="coerce")
        newly_nan = converted.isnull().sum() - X[col].isnull().sum()
        if non_null > 0 and (newly_nan / non_null) <= 0.3:
            X[col] = converted
            coerced_cols.append(col)
    if coerced_cols:
        report["coerced_to_numeric"] = coerced_cols
        pipeline["coerced_to_numeric_columns"] = coerced_cols

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    for col in numeric_cols:
        median_val = X[col].median()
        pipeline["numeric_fill_values"][col] = float(median_val) if pd.notna(median_val) else 0.0
        n_missing = int(X[col].isnull().sum())
        if n_missing > 0:
            X[col] = X[col].fillna(median_val)
            report["missing_filled"][col] = n_missing

    for col in categorical_cols:
        n_missing = int(X[col].isnull().sum())
        if n_missing > 0:
            X[col] = X[col].fillna("Unknown")
            report["missing_filled"][col] = n_missing

    if handle_outliers and numeric_cols:
        for col in numeric_cols:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            pipeline["outlier_bounds"][col] = (float(lower), float(upper))
        X = cap_outliers_iqr(X, numeric_cols)
        report["outliers_capped"] = True
        report["outlier_capped_columns"] = numeric_cols

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        report["categorical_encoded"].append(col)
        pipeline["categorical_encoders"][col] = le

    if y.dtype == object:
        target_le = LabelEncoder()
        y = target_le.fit_transform(y.astype(str))
        pipeline["target_encoder"] = target_le

    pipeline["feature_order"] = X.columns.tolist()

    return X, y, report, pipeline


def apply_pipeline(records: list, pipeline: dict) -> pd.DataFrame:
    """
    Applies a saved preprocessing pipeline to new incoming records (e.g. from
    /predict) so they match exactly what the model was trained on.
    """
    X = pd.DataFrame(records)

    for col in pipeline["dropped_id_like_columns"]:
        if col in X.columns:
            X = X.drop(columns=[col])

    for col in pipeline["coerced_to_numeric_columns"]:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    for col, fill_val in pipeline["numeric_fill_values"].items():
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(fill_val)
        else:
            X[col] = fill_val  # column missing entirely from the request — use training-time median

    for col in pipeline["categorical_encoders"]:
        if col not in X.columns:
            X[col] = pipeline["categorical_fill_value"]

    for col, le in pipeline["categorical_encoders"].items():
        X[col] = X[col].fillna(pipeline["categorical_fill_value"]).astype(str)
        known = set(le.classes_)
        # Map any category never seen during training to the first known
        # class rather than crashing — protects against real-world requests
        # containing values the training data never had.
        X[col] = X[col].apply(lambda v: v if v in known else le.classes_[0])
        X[col] = le.transform(X[col])

    for col, (lower, upper) in pipeline["outlier_bounds"].items():
        if col in X.columns:
            X[col] = X[col].clip(lower=lower, upper=upper)

    # Enforce the exact column order/set the model was trained on
    X = X.reindex(columns=pipeline["feature_order"], fill_value=0)

    return X


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = round(roc_auc_score(y_true, y_proba), 4)
        except ValueError:
            pass
    return metrics


def train_single_model(model_name: str, X_train, X_test, y_train, y_test,
                        custom_params: dict = None, handle_class_imbalance: bool = False):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )

    entry = MODEL_REGISTRY[model_name]
    params = {**entry["default_params"], **(custom_params or {})}
    strategy = entry["imbalance_strategy"]

    sample_weight = None
    if handle_class_imbalance:
        if strategy == "class_weight":
            params["class_weight"] = "balanced"
        elif strategy == "sample_weight":
            sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    model = entry["class"](**params)
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)
    return model, metrics, params


def run_experiment(dataset_id: int, dataset_filepath: str, target_column: str,
                    model_names: list, handle_outliers: bool = False,
                    handle_class_imbalance: bool = False,
                    custom_params: dict = None):
    df = load_dataset(dataset_filepath)
    X, y, preprocessing_report, pipeline = preprocess(df, target_column, handle_outliers)

    if not np.issubdtype(np.array(y).dtype, np.number):
        y = LabelEncoder().fit_transform(np.array(y).astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for model_name in model_names:
        model_params = (custom_params or {}).get(model_name, {})
        model, metrics, used_params = train_single_model(
            model_name, X_train, X_test, y_train, y_test, model_params,
            handle_class_imbalance=handle_class_imbalance
        )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        model_filename = f"{model_name}_{timestamp}.joblib"
        model_path = MODEL_DIR / model_filename
        # Bundle the fitted model together with its preprocessing pipeline —
        # not just the raw model — so /predict can correctly transform new,
        # raw-format input (e.g. "Female") the same way training data was
        # transformed, instead of requiring pre-encoded numeric input.
        joblib.dump({"model": model, "pipeline": pipeline}, model_path)

        experiment_id = insert_experiment(
            dataset_id=dataset_id,
            model_type=model_name,
            hyperparams=used_params,
            metrics=metrics,
            model_path=str(model_path),
            problem_type="classification",
        )

        results.append({
            "experiment_id": experiment_id,
            "model_type": model_name,
            "metrics": metrics,
            "hyperparams": used_params,
            "model_path": str(model_path),
        })

    return {
        "results": results,
        "preprocessing_report": preprocessing_report,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }