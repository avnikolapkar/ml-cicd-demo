"""Train the model, evaluate it, and save it to the model/ folder.

Run:  python -m src.train
"""
import json
from pathlib import Path

import joblib
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
MIN_R2 = 0.9  # quality gate: CI fails if the model is worse than this


def train() -> dict:
    # 1. Load real data (bundled with scikit-learn -> no internet needed in CI)
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    # 2. Split so we can measure the model on data it has never seen
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Train
    model = Ridge(alpha=0.1)
    model.fit(X_train, y_train)

    # 4. Evaluate
    r2 = r2_score(y_test, model.predict(X_test))
    metrics = {"r2": round(float(r2), 4), "n_features": X.shape[1]}

    # 5. Save model + metrics (these are the "artifacts")
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    result = train()
    print(f"Training done: {result}")
    if result["r2"] < MIN_R2:
        raise SystemExit(f"Quality gate FAILED: r2 {result['r2']} < {MIN_R2}")
    print("Quality gate PASSED")
