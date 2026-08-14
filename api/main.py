"""
main.py
FastAPI layer exposing the ML Experiment Tracking Platform:
  POST /upload        -> upload a CSV, get back dataset_id + validation info
  POST /train          -> train one or more models on an uploaded dataset
  GET  /experiments    -> list all logged experiments
  POST /promote         -> promote an experiment (or auto-pick the best) to production
  GET  /production      -> see what's currently deployed
  POST /predict          -> run inference using the production model

Design decisions (interview talking points):
- Upload saves the file to a temp path first, then hands off to
  dataset_manager.upload_dataset() for hashing/validation/storage — keeps
  "receiving an HTTP upload" separate from "what a dataset upload means",
  so dataset_manager stays testable without spinning up FastAPI at all
  (which is exactly how we tested it during development).
- /predict loads the production model fresh on each request rather than
  caching it in memory. For a portfolio-scale project this is simpler and
  correct-by-construction (no stale-model-in-memory bugs after a promotion);
  the tradeoff (reload cost per request) would matter at production scale,
  which is a good "how would you scale this" talking point.
- Errors from the core layer (ValueError for bad input) are caught and
  turned into proper HTTP 400s instead of leaking as 500s — the API layer's
  job is translating domain errors into HTTP semantics.
"""

import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from core.dataset_manager import upload_dataset as _upload_dataset
from core.trainer import run_experiment, MODEL_REGISTRY, apply_pipeline
from core.registry import (
    list_experiments, promote_to_production, get_current_production_model,
)
from core.db import init_db

app = FastAPI(title="ML Experiment Tracking Platform")



@app.on_event("startup")
def startup():
    init_db()


# ---------- Request/response schemas ----------

class TrainRequest(BaseModel):
    dataset_id: int
    dataset_filepath: str
    target_column: str
    model_names: List[str]
    handle_outliers: bool = False
    handle_class_imbalance: bool = False


class PromoteRequest(BaseModel):
    experiment_id: Optional[int] = None
    metric_name: str = "f1_score"
    higher_is_better: bool = True
    dataset_id: Optional[int] = None


class PredictRequest(BaseModel):
    records: List[dict]  # list of {"feature_name": value, ...}


# ---------- Endpoints ----------

@app.post("/upload")
async def upload_endpoint(file: UploadFile = File(...), target_column: str = Form(...)):
    """Accepts a CSV upload, saves it temporarily, then hands off to
    dataset_manager for hashing/validation/permanent storage."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = _upload_dataset(tmp_path, target_column)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/models/available")
def available_models():
    """Lists model types the trainer can build, so a frontend can render
    checkboxes without hardcoding model names."""
    return {"models": list(MODEL_REGISTRY.keys())}


@app.post("/train")
def train_endpoint(req: TrainRequest):
    """Trains the requested models on an already-uploaded dataset and logs
    each as an experiment."""
    unknown = [m for m in req.model_names if m not in MODEL_REGISTRY]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model(s): {unknown}. Available: {list(MODEL_REGISTRY.keys())}"
        )

    try:
        output = run_experiment(
            dataset_id=req.dataset_id,
            dataset_filepath=req.dataset_filepath,
            target_column=req.target_column,
            model_names=req.model_names,
            handle_outliers=req.handle_outliers,
            handle_class_imbalance=req.handle_class_imbalance,
        )
        return output
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/experiments")
def experiments_endpoint():
    """Returns all logged experiments, most recent first — this is what a
    comparison dashboard would render as a table."""
    return {"experiments": list_experiments()}


@app.post("/promote")
def promote_endpoint(req: PromoteRequest):
    try:
        promoted = promote_to_production(
            experiment_id=req.experiment_id,
            metric_name=req.metric_name,
            higher_is_better=req.higher_is_better,
            dataset_id=req.dataset_id,
        )
        return {"promoted_experiment": promoted}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/production")
def production_endpoint():
    """Shows what's currently deployed."""
    prod = get_current_production_model()
    if prod is None:
        raise HTTPException(status_code=404, detail="No model has been promoted to production yet.")
    return prod


@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    """
    Runs inference using whatever model is currently in production.
    Accepts raw-format records (e.g. "Female", "Yes") — the saved
    preprocessing pipeline handles encoding them the same way training
    data was encoded, so callers don't need to pre-encode anything.
    """
    prod = get_current_production_model()
    if prod is None:
        raise HTTPException(status_code=404, detail="No model has been promoted to production yet.")

    bundle = joblib.load(prod["model_path"])

    # Backward compatibility: older saved models are just the raw model,
    # not a {"model": ..., "pipeline": ...} bundle. Retrain to get the fix.
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise HTTPException(
            status_code=400,
            detail="This model was trained before pipeline support was added. Please retrain it to enable predictions."
        )

    model = bundle["model"]
    pipeline = bundle["pipeline"]

    try:
        df = apply_pipeline(req.records, pipeline)
        predictions = model.predict(df)
        result = {"predictions": predictions.tolist()}
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(df)[:, 1]
            result["probabilities"] = probabilities.tolist()
        return result
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {e}"
        )

app.mount("/", StaticFiles(directory="api/static", html=True), name="static")