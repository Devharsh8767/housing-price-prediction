# Gurgaon Housing Price Prediction

## Overview

Gurgaon’s real estate market is growing rapidly, and property pricing can be difficult to estimate because it depends on many factors such as location, size, number of rooms, and nearby amenities. This project builds a machine learning model to predict house prices using historical housing data and helps buyers, sellers, and real estate professionals make more informed decisions.

## Why This Project Matters

Real estate pricing is influenced by a wide range of variables, and predicting prices accurately can support better planning and decision-making. This project focuses on building a practical and interpretable regression model for housing price prediction.

## Project Goal

The goal of this project is to:
- build a predictive model for house prices,
- use relevant housing features to estimate property values,
- evaluate model performance using standard regression metrics,
- create a reusable workflow that can later be adapted to real Gurgaon housing data.

## Approach

A clean Gurgaon-specific housing dataset is not readily available, so this project uses the California Housing dataset as a substitute for development and evaluation. The model is trained using features such as:
- median income,
- location-related information,
- number of rooms,
- population-related signals.

This provides a strong baseline for housing price prediction and can later be adapted to real Gurgaon property data when it becomes available.

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- joblib
- Jupyter / Python scripts

## Project Workflow

1. Load the housing dataset.
2. Prepare the data for modeling.
3. Create a training and test split.
4. Build a preprocessing pipeline.
5. Train a regression model.
6. Evaluate the model using metrics such as RMSE, MAE, and R².
7. Save the trained model and generate predictions.

## Model Performance

The current model is evaluated using regression metrics such as:
- RMSE
- MAE
- R²

These results are saved in the project output files after running the script.

## Visualizations

The project generates several useful plots to understand the data better:

![House Price Distribution](house_price_distribution.png)

![Median Income Distribution](median_income_distribution.png)

![Correlation Heatmap](correlation_heatmap.png)

![Income vs Price Scatter Plot](income_vs_price_scatter.png)

## Files in This Project

- main.py - training and prediction workflow
- app.py - interactive Streamlit app for predictions
- housing.csv - input housing dataset
- input.csv - test input data used for inference
- predictions.csv - generated predictions
- model.pkl - saved trained model
- pipeline.pkl - saved preprocessing pipeline
- metrics.json - evaluation metrics
- house_price_distribution.png - visualization of house price spread
- median_income_distribution.png - visualization of income distribution
- correlation_heatmap.png - correlation heatmap of numeric features
- income_vs_price_scatter.png - scatter plot of income vs price

## How to Run

1. Create and activate a virtual environment.
2. Install the required dependencies.
3. Run the main script:

```bash
python main.py
```

## Future Improvements

Possible next steps for this project:
- compare multiple regression models,
- add visualizations for actual vs predicted values,
- improve feature engineering,
- deploy the model as a web app,
- replace the sample dataset with real Gurgaon housing data.

## Summary

This project demonstrates an end-to-end machine learning workflow for housing price prediction and serves as a strong foundation for building a more realistic real-estate pricing solution for Gurgaon.
