from src.train import MIN_R2, MODEL_PATH, train


def test_training_creates_model_file():
    train()
    assert MODEL_PATH.exists()


def test_model_meets_quality_gate():
    metrics = train()
    assert metrics["r2"] >= MIN_R2
