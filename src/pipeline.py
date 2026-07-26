import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(num_attributes, cat_attributes):
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    return ColumnTransformer(
        [
            ("nums", num_pipeline, num_attributes),
            ("cat", cat_pipeline, cat_attributes),
        ]
    )


def prepare_income_category(housing):
    housing = housing.copy()
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0, 1.5, 3, 4.5, 6, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    return housing


def create_stratified_split(housing, input_path, test_size=0.2, random_state=42):
    split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    for train_index, test_index in split.split(housing, housing["income_cat"]):
        train_set = housing.iloc[train_index].drop(columns=["income_cat"]).copy()
        test_set = housing.iloc[test_index].drop(columns=["income_cat"]).copy()

    test_set.to_csv(input_path, index=False)
    return train_set, test_set
