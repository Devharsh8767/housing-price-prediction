import os

import joblib
import pandas as pd
import streamlit as st

from src.config import MODEL_FILE, PIPELINE_FILE
from src.modeling import train_and_evaluate


@st.cache_resource
def load_model_and_pipeline():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(PIPELINE_FILE):
        train_and_evaluate()

    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    return model, pipeline


def build_input_dataframe(
    longitude,
    latitude,
    housing_median_age,
    total_rooms,
    total_bedrooms,
    population,
    households,
    median_income,
    ocean_proximity,
):
    input_values = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity,
    }
    input_df = pd.DataFrame([input_values])
    expected_columns = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
        "ocean_proximity",
    ]
    return input_df[expected_columns]


def main():
    st.set_page_config(page_title="Housing Price Predictor", page_icon="🏠", layout="wide")
    st.title("🏠 Housing Price Predictor")
    st.write("Estimate a home's value using a trained machine learning model.")
    st.caption("These values match the features used during training, so realistic inputs give more meaningful predictions.")

    model, pipeline = load_model_and_pipeline()

    with st.sidebar:
        st.header("Input Controls")
        st.write("Adjust the housing features below to generate a new prediction.")
        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=-122.23,
            step=0.01,
            help="The east-west location of the property.",
        )
        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=37.88,
            step=0.01,
            help="The north-south location of the property.",
        )
        housing_median_age = st.number_input(
            "Housing Median Age",
            min_value=0.0,
            value=20.0,
            step=1.0,
            help="Typical age of homes in the area.",
        )
        total_rooms = st.number_input(
            "Total Rooms",
            min_value=0.0,
            value=2000.0,
            step=10.0,
            help="Total number of rooms in the house.",
        )
        total_bedrooms = st.number_input(
            "Total Bedrooms",
            min_value=0.0,
            value=500.0,
            step=10.0,
            help="Total bedrooms in the house.",
        )
        population = st.number_input(
            "Population",
            min_value=0.0,
            value=1200.0,
            step=10.0,
            help="Population of the area.",
        )
        households = st.number_input(
            "Households",
            min_value=0.0,
            value=400.0,
            step=10.0,
            help="Number of households in the area.",
        )
        median_income = st.number_input(
            "Median Income",
            min_value=0.0,
            value=3.0,
            step=0.1,
            help="Typical income level of residents.",
        )
        ocean_proximity = st.selectbox(
            "Ocean Proximity",
            ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"],
            help="The home's proximity to the ocean.",
        )

        st.divider()
        if st.button("Predict Price", type="primary"):
            input_df = build_input_dataframe(
                longitude=longitude,
                latitude=latitude,
                housing_median_age=housing_median_age,
                total_rooms=total_rooms,
                total_bedrooms=total_bedrooms,
                population=population,
                households=households,
                median_income=median_income,
                ocean_proximity=ocean_proximity,
            )
            transformed = pipeline.transform(input_df)
            prediction = model.predict(transformed)[0]
            st.session_state.prediction = prediction

    st.subheader("Prediction Result")
    if "prediction" in st.session_state:
        prediction = st.session_state.prediction
        st.metric("Estimated House Price", f"${prediction:,.2f}")
        st.info("This is an estimated value based on the model's learned patterns from historical housing data.")
    else:
        st.info("Use the sidebar to enter property details and click Predict Price.")


if __name__ == "__main__":
    main()
