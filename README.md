# ml-cicd-demo
ML API (FastAPI + scikit-learn) with a full CI/CD pipeline on GitHub Actions.

## Run locally
    python -m venv .venv
    .venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
    pip install -r requirements-dev.txt
    python -m src.train
    pytest -v
    uvicorn src.app:app --reload    # open http://127.0.0.1:8000/docs

## Pipeline
push / PR -> lint -> train + quality gate -> tests -> docker build -> smoke test -> (main only) push image to GHCR -> optional deploy
