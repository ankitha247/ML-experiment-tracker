"""
registry.py
Model registry: selecting the "best" experiment by a chosen metric and
promoting it to production.

Design decisions (interview talking points):
- "Best model" selection is a simple MAX/MIN over a chosen metric — no
  weighted scoring across multiple metrics. This is a deliberate scope
  choice: a weighted composite score (e.g. 0.5*f1 + 0.3*precision + ...)
  is easy to add later but hard to defend the weights for without a real
  business cost-of-error tradeoff. Keeping selection to "pick the metric
  that matters for this problem" is simpler and more honest.
- Promotion is decoupled from training: training logs an experiment,
  promotion is a separate explicit action. This mirrors real MLOps practice
  — a model finishing training doesn't mean it should auto-deploy; someone
  (or some validation gate) decides that separately.
- The registry doesn't retrain or re-validate a model before promoting it —
  it just points the "production" pointer at an existing experiment's saved
  artifact. Loading that artifact for inference is handled by whoever calls
  get_production_model() (the API layer).
"""

from core.db import get_all_experiments, get_best_experiment, set_production_model, get_production_model

# Which metric to optimize for by default, and its direction.
# Classification: F1 is a reasonable default because it balances precision/recall,
# which matters most on imbalanced data (like fraud detection) where accuracy
# alone is misleading — a model predicting "not fraud" every time still gets
# ~99% accuracy but is useless.
DEFAULT_METRIC = "f1_score"
DEFAULT_HIGHER_IS_BETTER = True


def list_experiments():
    """Returns all logged experiments, most recent first."""
    return get_all_experiments()


def find_best_model(metric_name: str = DEFAULT_METRIC, higher_is_better: bool = DEFAULT_HIGHER_IS_BETTER,
                     dataset_id: int = None):
    best = get_best_experiment(metric_name, higher_is_better, dataset_id=dataset_id)
    if best is None:
        raise ValueError(
            f"No experiments found with metric '{metric_name}'"
            f"{f' for dataset_id {dataset_id}' if dataset_id else ''}. "
            f"Run at least one experiment first."
        )
    return best


def promote_to_production(experiment_id: int = None, metric_name: str = DEFAULT_METRIC,
                           higher_is_better: bool = DEFAULT_HIGHER_IS_BETTER, dataset_id: int = None):
    if experiment_id is None:
        best = find_best_model(metric_name, higher_is_better, dataset_id=dataset_id)
        experiment_id = best["id"]
    else:
        all_exps = {e["id"]: e for e in list_experiments()}
        if experiment_id not in all_exps:
            raise ValueError(f"Experiment id {experiment_id} not found.")
        best = all_exps[experiment_id]

    set_production_model(experiment_id)
    return best


def get_current_production_model():
    """Returns the currently deployed experiment's metadata, or None if
    nothing has been promoted yet."""
    return get_production_model()