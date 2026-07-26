from pathlib import Path

import pandas as pd

from src.visualizations import save_visualizations


def test_save_visualizations_creates_image_files(tmp_path):
    housing = pd.DataFrame(
        {
            "median_house_value": [100000, 200000, 300000, 400000],
            "median_income": [2.0, 3.0, 4.0, 5.0],
            "housing_median_age": [10, 20, 30, 40],
            "latitude": [37.0, 37.5, 38.0, 38.5],
            "longitude": [-122.0, -122.5, -123.0, -123.5],
        }
    )

    output_paths = save_visualizations(housing, output_dir=tmp_path)

    assert all(Path(path).exists() for path in output_paths.values())
