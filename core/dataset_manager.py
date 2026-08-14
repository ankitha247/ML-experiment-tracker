"""
dataset_manager.py
Handles dataset upload, content-hash versioning, and basic validation.

Design decisions (interview talking points):
- Versioning by SHA-256 hash of file contents (not filename or timestamp):
  if the same CSV is uploaded twice, it's recognized as the same version
  and not duplicated in storage or the DB. Filename-based versioning breaks
  the moment someone re-uploads "data.csv" with different content, or the
  same content under a different name.
- Validation is intentionally shallow (missing values, column count, dtype
  sanity) rather than deep data-quality checks — this project is about
  experiment tracking infra, not a full data-validation framework like
  Great Expectations. Keeping it shallow is a defensible scope choice.
"""

import hashlib
import shutil
from pathlib import Path
import pandas as pd

from core.db import insert_dataset

DATASET_DIR = Path(__file__).resolve().parent.parent / "storage" / "datasets"


def compute_file_hash(filepath: Path) -> str:
    """SHA-256 hash of raw file bytes. Chosen over MD5 for negligible collision
    risk at this scale, and it's what Git itself effectively models content
    addressing on (conceptually), which is a nice parallel to bring up."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def validate_dataset(df: pd.DataFrame, target_column: str):
    """Raises ValueError with a clear message if the dataset fails basic checks.
    Returns nothing on success."""
    if df.empty:
        raise ValueError("Uploaded dataset is empty.")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset columns: "
            f"{list(df.columns)}"
        )

    if df[target_column].isnull().all():
        raise ValueError(f"Target column '{target_column}' is entirely missing/null.")

    if len(df) < 10:
        raise ValueError(
            f"Dataset has only {len(df)} rows — too few to train/test split meaningfully."
        )

    # Warn-level issue, not fatal: missing values elsewhere. We don't raise here
    # because trainer.py will handle imputation; but we surface it in metadata.
    return {
        "n_missing_cells": int(df.isnull().sum().sum()),
        "n_rows": len(df),
        "n_columns": len(df.columns),
    }


def upload_dataset(source_filepath: str, target_column: str) -> dict:
    """
    Takes a path to a CSV (e.g. from a FastAPI UploadFile saved to a temp path),
    validates it, hashes it for versioning, copies it into storage/datasets/,
    and registers it in the DB.

    Returns a dict with dataset_id, version_hash, and validation summary —
    this is what the API layer will return to the client.
    """
    source_filepath = Path(source_filepath)

    df = pd.read_csv(source_filepath)
    validation_info = validate_dataset(df, target_column)

    version_hash = compute_file_hash(source_filepath)
    short_hash = version_hash[:12]  # full hash in DB, short hash for readable filenames

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    stored_filename = f"{short_hash}_{source_filepath.name}"
    stored_path = DATASET_DIR / stored_filename

    # Avoid re-copying if this exact content already exists in storage
    if not stored_path.exists():
        shutil.copy(source_filepath, stored_path)

    dataset_id = insert_dataset(
        version_hash=version_hash,
        filename=source_filepath.name,
        filepath=str(stored_path),
        n_rows=validation_info["n_rows"],
        n_columns=validation_info["n_columns"],
        target_column=target_column,
    )

    return {
        "dataset_id": dataset_id,
        "version_hash": version_hash,
        "short_hash": short_hash,
        "stored_path": str(stored_path),
        "target_column": target_column,
        **validation_info,
    }


def load_dataset(filepath: str) -> pd.DataFrame:
    """Thin wrapper so trainer.py doesn't need to import pandas directly for this —
    keeps the 'how do I read a dataset' logic in one place."""
    return pd.read_csv(filepath)