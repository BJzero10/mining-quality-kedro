from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


def train_xgb(train_df: pd.DataFrame, target_col: str, date_col: str, xgb_params: Dict) -> XGBRegressor:
    if target_col not in train_df.columns:
        raise ValueError(f"target_col='{target_col}' no existe. Columnas: {list(train_df.columns)}")

    X = train_df.drop(columns=[target_col])
    if date_col in X.columns:
        X = X.drop(columns=[date_col])
    y = train_df[target_col].astype(float)

    model = XGBRegressor(**xgb_params)
    model.fit(X, y)
    return model


def predict(model: XGBRegressor, test_df: pd.DataFrame, target_col: str, date_col: str) -> pd.DataFrame:
    """Devuelve dataframe con date, y_true, y_pred y residual."""
    if target_col not in test_df.columns:
        raise ValueError(f"target_col='{target_col}' no existe en test_df.")

    X = test_df.drop(columns=[target_col])
    if date_col in X.columns:
        X = X.drop(columns=[date_col])

    y_true = test_df[target_col].astype(float).to_numpy()
    y_pred = model.predict(X)

    out = pd.DataFrame(
        {
            date_col: pd.to_datetime(test_df[date_col], errors="coerce").to_numpy(),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    out["residual"] = out["y_true"] - out["y_pred"]
    out = out.sort_values(date_col).reset_index(drop=True)
    return out


def evaluate(predictions: pd.DataFrame) -> Dict:
    y_true = predictions["y_true"].to_numpy()
    y_pred = predictions["y_pred"].to_numpy()

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "n_test": int(len(predictions)),
    }
