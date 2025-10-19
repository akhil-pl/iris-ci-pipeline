import joblib
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

MODEL_PATH = "models/model.joblib"
SAMPLE_DATA = "data/iris_v1.csv"  # can use iris_v1.csv for evaluation

@pytest.fixture
def model():
    """Load the trained model."""
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        pytest.fail(f"Model not found at {MODEL_PATH}")
    except Exception as e:
        pytest.fail(f"Error loading model: {e}")

@pytest.fixture
def sample_data():
    """Load a small sample of data."""
    df = pd.read_csv(SAMPLE_DATA)
    X = df.drop("species", axis=1)
    y = df["species"]
    return X, y

def test_model_prediction(model, sample_data):
    """Check that the model can make predictions without errors."""
    X, y = sample_data
    preds = model.predict(X)
    assert len(preds) == len(y), "Prediction length mismatch"

def test_model_accuracy(model, sample_data):
    """Check that model has reasonable accuracy (>=70%)."""
    X, y = sample_data
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    assert acc >= 0.7, f"Model accuracy too low: {acc}"
