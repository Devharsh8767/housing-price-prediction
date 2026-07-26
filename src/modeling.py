import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import DATA_FILE, INPUT_FILE, METRICS_FILE, MODEL_FILE, PIPELINE_FILE, PREDICTIONS_FILE
from src.pipeline import build_pipeline, create_stratified_split, prepare_income_category
from src.visualizations import save_visualizations


def prepare_metrics(*args):
    if len(args) == 3:
        rmse, mae, r2 = args
        return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}

    if len(args) != 2:
        raise TypeError("prepare_metrics expects either (y_true, y_pred) or (rmse, mae, r2)")

    y_true, y_pred = args
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def save_metrics(metrics, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def save_predictions(predictions_df, output_path):
    predictions_df.to_csv(output_path, index=False)


def train_and_evaluate():
    housing = pd.read_csv(DATA_FILE)
    housing = prepare_income_category(housing)
    train_set, test_set = create_stratified_split(housing, INPUT_FILE)

    X_train = train_set.drop(columns=["median_house_value"])
    y_train = train_set["median_house_value"].copy()
    X_test = test_set.drop(columns=["median_house_value"])
    y_test = test_set["median_house_value"].copy()

    num_attributes = X_train.drop(columns=["ocean_proximity"]).columns.tolist()
    cat_attributes = ["ocean_proximity"]

    pipeline = build_pipeline(num_attributes, cat_attributes)
    X_train_prepared = pipeline.fit_transform(X_train)

    model = RandomForestRegressor(random_state=42, n_estimators=200)
    model.fit(X_train_prepared, y_train)

    X_test_prepared = pipeline.transform(X_test)
    predictions = model.predict(X_test_prepared)

    metrics = prepare_metrics(y_test, predictions)
    save_metrics(metrics, METRICS_FILE)

    predictions_df = X_test.copy()
    predictions_df["actual_median_house_value"] = y_test
    predictions_df["predicted_median_house_value"] = predictions
    save_predictions(predictions_df, PREDICTIONS_FILE)

    save_visualizations(housing, output_dir=".")

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("Model trained and saved.")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"R²: {metrics['r2']:.3f}")


def run_inference():
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv(INPUT_FILE)
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)

    input_data["predicted_median_house_value"] = predictions
    save_predictions(input_data, PREDICTIONS_FILE)

    print("Inference complete and predictions saved to predictions.csv.")


def main():
    if not os.path.exists(MODEL_FILE):
        train_and_evaluate()
    else:
        run_inference()
