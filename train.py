import random

import dill as pickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import (
    TransformedTargetRegressor,
    make_column_selector,
    make_column_transformer,
)
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

random.seed(42)
np.random.seed(42)




class DataFramePreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.X_train = self.format_columns(X.copy())
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_ = self.format_columns(X_)

        na_cols = X_.columns[X_.isna().any()].tolist()
        X_[na_cols] = X_[na_cols].fillna(self.X_train[na_cols].median())

        X_["engine"] = pd.to_numeric(X_["engine"], downcast="integer")
        X_["seats"] = pd.to_numeric(X_["seats"], downcast="integer")

        X_ = X_.drop("name", axis=1)

        X_["seats"] = X_["seats"].astype("object")

        return X_

    def convert_torque(self, text):
        text = text.strip()

        if text.count("kgm") == 1:
            text = text.replace("kgm", "")
            text = float(text) * 9.80665
        elif text.count("nm") == 1:
            text = text.replace("nm", "")
            text = float(text)

        return int(float(text) + 0.5)

    def convert_max_torque(self, text):
        text = text.strip()
        text = text.replace("rpm", "")

        if "+/-" in text:
            max_torque, _ = text.split("+/-")
            text = float(max_torque)

        elif "~" in text or "-" in text:
            text = text.replace("-", " ")
            text = text.replace("~", " ")
            max_torque_low, max_torque_high = text.split()
            text = (float(max_torque_low) + float(max_torque_high)) / 2

        return int(float(text) + 0.5)

    def convert(self, text):
        if pd.isna(text):
            return [text, text]

        text = text.lower()
        text = text.replace("at", "@")
        text = text.replace(",", "")
        text = text.replace(")", "")

        if text.count("(") == 1 and text.count("@") == 2:
            parts = text.split("(")
            first_part, second_part = parts[0].split("@"), parts[1].split("@")
            torque = first_part[0] + second_part[0]
            max_torque = first_part[1] + second_part[1]
            return [self.convert_torque(torque), self.convert_max_torque(max_torque)]

        elif text.count("(") == 0 and text.count("@") == 1:
            torque, max_torque = text.split("@")
            return [self.convert_torque(torque), self.convert_max_torque(max_torque)]

        elif "/" in text:
            torque, max_torque = text.split("/")
            return [self.convert_torque(torque), self.convert_max_torque(max_torque)]

        elif text.count("(") == 1 and text.count("@") == 1:
            text = text[text.find("(") + 1 :]
            torque, max_torque = text.split("@")
            return [self.convert_torque(torque), self.convert_max_torque(max_torque)]

        else:
            return [self.convert_torque(text), np.nan]

    def remove_extra_units(self, text):
        if pd.isna(text):
            return text

        text = text.replace("kmpl", "").replace("km/kg", "")
        text = text.replace("CC", "")
        text = text.replace("bhp", "")

        text = text.strip()

        if text:
            return text
        else:
            return np.nan

    def format_columns(self, df):
        df[["torque", "max_torque_rpm"]] = df["torque"].apply(
            lambda x: pd.Series(self.convert(x))
        )
        df["max_torque_rpm"] = pd.to_numeric(df["max_torque_rpm"], downcast="float")
        df["torque"] = pd.to_numeric(df["torque"], downcast="float")
        df["mileage"] = pd.to_numeric(
            df["mileage"].apply(self.remove_extra_units), downcast="float"
        )
        df["engine"] = pd.to_numeric(
            df["engine"].apply(self.remove_extra_units), downcast="float"
        )
        df["max_power"] = pd.to_numeric(
            df["max_power"].apply(self.remove_extra_units), downcast="float"
        )
        return df

    @staticmethod
    def apply_log(x):
        return np.log(x + 1)

    @staticmethod
    def inverse_log(x):
        return np.exp(x) - 1


if __name__ == "__main__":
    df_train = pd.read_csv(
        "https://raw.githubusercontent.com/hse-mlds/ml/main/hometasks/HT1/cars_train.csv"
    )
    df_test = pd.read_csv(
        "https://raw.githubusercontent.com/hse-mlds/ml/main/hometasks/HT1/cars_test.csv"
    )

    na_sum_train = df_train.isna().sum()
    na_sum_test = df_test.isna().sum()
    na_cols_train = na_sum_train[na_sum_train > 0].index.to_list()
    na_cols_test = na_sum_test[na_sum_test > 0].index.to_list()

    duplicates = df_train.drop("selling_price", axis=1).duplicated()

    df_train = df_train[~duplicates]

    assert df_train.shape == (5840, 13)

    df_train.reset_index(drop=True, inplace=True)

    df_train_wo_duplicates = df_train.copy()
    df_test_wo_duplicates = df_test.copy()

    pipe = make_pipeline(
        DataFramePreprocessor(),
        make_column_transformer(
            (
                OneHotEncoder(
                    drop="first", sparse_output=False, handle_unknown="ignore"
                ),
                make_column_selector(dtype_include=["category", "object"]),
            ),
            remainder="passthrough",
        ),
        StandardScaler(),
        TransformedTargetRegressor(
            ElasticNet(),
            func=DataFramePreprocessor.apply_log,
            inverse_func=DataFramePreprocessor.inverse_log,
        ),
    )

    param_grid = {
        "transformedtargetregressor__regressor__alpha": [
            0.001,
            0.01,
            0.1,
            1.0,
            10.0,
            100.0,
        ],
        "transformedtargetregressor__regressor__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1],
    }

    grid_search = GridSearchCV(
        pipe, param_grid, cv=10, scoring="r2", n_jobs=-1, verbose=1
    )
    X = df_train_wo_duplicates.drop("selling_price", axis=1)
    y = df_train_wo_duplicates["selling_price"]
    grid_search.fit(X, y)

    X_t = df_test_wo_duplicates.drop("selling_price", axis=1)
    y_t = df_test_wo_duplicates["selling_price"]

    pred = grid_search.predict(X_t)
    print("R2:", r2_score(y_t, pred))
    print("MSE:", MSE(y_t, pred))

    with open("model.pkl", "wb") as f:
        pickle.dump(grid_search, f)
