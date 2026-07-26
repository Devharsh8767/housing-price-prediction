import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import main


def test_prepare_metrics_returns_dictionary():
    metrics = main.prepare_metrics(10.0, 2.0, 0.5)

    assert isinstance(metrics, dict)
    assert metrics["rmse"] == 10.0
    assert metrics["mae"] == 2.0
    assert metrics["r2"] == 0.5


def test_save_predictions_creates_csv(tmp_path):
    df = pd.DataFrame({"feature": [1, 2], "predicted_median_house_value": [100.0, 200.0]})
    output_path = tmp_path / "predictions.csv"

    main.save_predictions(df, output_path)

    assert output_path.exists()
    assert pd.read_csv(output_path).shape[0] == 2
