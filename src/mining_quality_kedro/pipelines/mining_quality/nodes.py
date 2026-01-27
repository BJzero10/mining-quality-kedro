from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import shap
import matplotlib.pyplot as plt


def _coerce_decimal_comma_to_float(df: pd.DataFrame, exclude_cols: Optional[list[str]] = None) -> pd.DataFrame:
    """Convert columns with values like '55,2' -> 55.2 to float safely."""
    exclude_cols = exclude_cols or []
    out = df.copy()

    for col in out.columns:
        if col in exclude_cols:
            continue

        # If column already numeric, keep it
        if pd.api.types.is_numeric_dtype(out[col]):
            continue

        # Convert strings with decimal comma to dot
        s = out[col].astype(str).str.replace(",", ".", regex=False)

        # to_numeric will produce NaN for non-numeric tokens
        out[col] = pd.to_numeric(s, errors="coerce")

    return out


def clean_data(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    - Parse datetime
    - Convert all numeric columns that may come as strings with decimal comma
    - Sort by date
    - Drop duplicates & NaNs
    """
    out = df.copy()

    if date_col not in out.columns:
        raise ValueError(f"date_col='{date_col}' no existe en el dataframe. Columnas: {list(out.columns)}")

    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])

    # Convert everything except date
    out = _coerce_decimal_comma_to_float(out, exclude_cols=[date_col])

    # Remove duplicates in timestamp (keep last)
    out = out.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")

    # Drop rows with any NaN after conversion
    out = out.dropna(axis=0, how="any").reset_index(drop=True)

    return out


def resample_time(df: pd.DataFrame, date_col: str, rule: str) -> pd.DataFrame:
    """
    Resample by time (e.g., '1H') using mean aggregation.
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    out = out.set_index(pd.DatetimeIndex(out[date_col]))
    out = out.drop(columns=[date_col])

    out = out.resample(rule).mean()

    out = out.dropna(axis=0, how="any")
    out = out.reset_index().rename(columns={"index": date_col})

    return out


def time_split(df: pd.DataFrame, date_col: str, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split preserving time order (no shuffle).
    test_size: fraction (0-1)
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size debe estar entre 0 y 1")

    out = df.sort_values(date_col).reset_index(drop=True)
    n = len(out)
    cut = int(np.floor(n * (1.0 - test_size)))
    cut = max(1, min(cut, n - 1))

    train_df = out.iloc[:cut].copy()
    test_df = out.iloc[cut:].copy()
    return train_df, test_df


def train_xgb(train_df: pd.DataFrame, target_col: str, date_col: str, xgb_params: Dict) -> XGBRegressor:
    """
    Train XGBRegressor.
    """
    if target_col not in train_df.columns:
        raise ValueError(f"target_col='{target_col}' no existe en train_df")

    X = train_df.drop(columns=[target_col])
    if date_col in X.columns:
        X = X.drop(columns=[date_col])
    y = train_df[target_col].astype(float)

    model = XGBRegressor(**xgb_params)
    model.fit(X, y)
    return model


def evaluate(model: XGBRegressor, test_df: pd.DataFrame, target_col: str, date_col: str) -> Dict:
    """
    Compute metrics on test set.
    """
    X = test_df.drop(columns=[target_col])
    if date_col in X.columns:
        X = X.drop(columns=[date_col])
    y_true = test_df[target_col].astype(float)

    y_pred = model.predict(X)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "n_test": int(len(test_df)),
    }


def shap_explainability(
    model: XGBRegressor,
    train_df: pd.DataFrame,
    target_col: str,
    date_col: str,
    max_samples: int = 5000,
):
    """
    Create SHAP summary plot figure (bar + beeswarm-like default summary).
    Returns a matplotlib Figure for MatplotlibWriter.
    """
    X = train_df.drop(columns=[target_col])
    if date_col in X.columns:
        X = X.drop(columns=[date_col])

    # Sample for speed
    if len(X) > max_samples:
        Xs = X.sample(n=max_samples, random_state=42)
    else:
        Xs = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xs)

    fig = plt.figure()
    shap.summary_plot(shap_values, Xs, show=False)
    plt.tight_layout()
    return fig
