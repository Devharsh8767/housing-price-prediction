from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = str(BASE_DIR / "model.pkl")
PIPELINE_FILE = str(BASE_DIR / "pipeline.pkl")
DATA_FILE = str(BASE_DIR / "housing.csv")
INPUT_FILE = str(BASE_DIR / "input.csv")
PREDICTIONS_FILE = str(BASE_DIR / "predictions.csv")
METRICS_FILE = str(BASE_DIR / "metrics.json")
