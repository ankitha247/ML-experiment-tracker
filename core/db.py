"""
db.py
SQLite database layer for the ML Experiment Tracking Platform.

Design decisions (interview talking points):
- SQLite chosen over Postgres: single-user local tool, zero setup overhead,
  file-based so the whole DB can be versioned/copied like an artifact.
- Three tables: datasets, experiments, production_model.
  - datasets: one row per unique dataset version (hash-based dedup).
  - experiments: one row per training run, storing hyperparams/metrics as
    JSON strings (SQLite has no native JSON type, so we serialize).
  - production_model: single-row table acting as a pointer to whichever
    experiment is "live" for the /predict endpoint. This means deploying
    a new model is just an UPDATE, not a code change or redeploy.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager

DB_PATH = Path(__file__).resolve().parent.parent / "storage" / "tracker.db"


@contextmanager
def get_connection():
    """Context manager so every caller gets a connection that's always closed,
    even if an exception is raised mid-query."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets us access columns by name, not just index
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Creates all tables if they don't exist yet. Safe to call on every app startup."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_hash TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                n_rows INTEGER,
                n_columns INTEGER,
                target_column TEXT,
                uploaded_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER NOT NULL,
                model_type TEXT NOT NULL,
                hyperparams TEXT NOT NULL,       -- JSON string
                metrics TEXT NOT NULL,           -- JSON string
                model_path TEXT NOT NULL,
                problem_type TEXT NOT NULL,      -- 'classification' or 'regression'
                trained_at TEXT NOT NULL,
                FOREIGN KEY (dataset_id) REFERENCES datasets (id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS production_model (
                id INTEGER PRIMARY KEY CHECK (id = 1),  -- enforces single row
                experiment_id INTEGER NOT NULL,
                promoted_at TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
        """)


def insert_dataset(version_hash, filename, filepath, n_rows, n_columns, target_column):
    """Returns the dataset's id. If the hash already exists (same file uploaded
    twice), returns the existing row's id instead of creating a duplicate."""
    with get_connection() as conn:
        existing = conn.execute(
            "SELECT id FROM datasets WHERE version_hash = ?", (version_hash,)
        ).fetchone()
        if existing:
            return existing["id"]

        cursor = conn.execute(
            """INSERT INTO datasets
               (version_hash, filename, filepath, n_rows, n_columns, target_column, uploaded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (version_hash, filename, filepath, n_rows, n_columns, target_column,
             datetime.now(timezone.utc).isoformat())
        )
        return cursor.lastrowid


def insert_experiment(dataset_id, model_type, hyperparams: dict, metrics: dict,
                       model_path, problem_type):
    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO experiments
               (dataset_id, model_type, hyperparams, metrics, model_path, problem_type, trained_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (dataset_id, model_type, json.dumps(hyperparams), json.dumps(metrics),
             model_path, problem_type, datetime.now(timezone.utc).isoformat())
        )
        return cursor.lastrowid


def get_all_experiments():
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT e.*, d.filename as dataset_filename, d.version_hash,
                   d.target_column as dataset_target_column, d.n_rows as dataset_n_rows
            FROM experiments e
            JOIN datasets d ON e.dataset_id = d.id
            ORDER BY e.trained_at DESC
        """).fetchall()
        return [dict(row) for row in rows]

def get_best_experiment(metric_name: str, higher_is_better: bool = True, dataset_id: int = None):
    """Picks the best experiment by a given metric key inside the metrics JSON,
    optionally restricted to a single dataset. Without dataset_id, this
    searches across every dataset ever trained on — which caused a real bug:
    promoting a fraud-detection model when the person actually wanted their
    churn-detection model, simply because the fraud model's F1 score was
    higher in absolute terms across a completely different problem."""
    experiments = get_all_experiments()
    if dataset_id is not None:
        experiments = [e for e in experiments if e["dataset_id"] == dataset_id]

    scored = []
    for exp in experiments:
        metrics = json.loads(exp["metrics"])
        if metric_name in metrics:
            scored.append((metrics[metric_name], exp))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0], reverse=higher_is_better)
    return scored[0][1]


def set_production_model(experiment_id: int):
    """Promotes an experiment to production. Overwrites the single existing row."""
    with get_connection() as conn:
        conn.execute("DELETE FROM production_model WHERE id = 1")
        conn.execute(
            "INSERT INTO production_model (id, experiment_id, promoted_at) VALUES (1, ?, ?)",
            (experiment_id, datetime.now(timezone.utc).isoformat())
        )


def get_production_model():
    with get_connection() as conn:
        row = conn.execute("""
            SELECT e.*, d.filename as dataset_filename, d.target_column as dataset_target_column
            FROM production_model p
            JOIN experiments e ON p.experiment_id = e.id
            JOIN datasets d ON e.dataset_id = d.id
            WHERE p.id = 1
        """).fetchone()
        return dict(row) if row else None