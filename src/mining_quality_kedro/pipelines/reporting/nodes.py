from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBRegressor


def plot_target_timeseries(hourly_df: pd.DataFrame, date_col: str, target_col: str):
    """Plot del target resampleado (serie de tiempo)."""
    df = hourly_df.sort_values(date_col)
    fig = plt.figure()
    plt.plot(df[date_col], df[target_col])
    plt.title(f"Target over time (resampled): {target_col}")
    plt.xlabel(date_col)
    plt.ylabel(target_col)
    plt.tight_layout()
    return fig


def plot_pred_vs_true(predictions: pd.DataFrame, date_col: str):
    """Plot y_true vs y_pred en el tiempo."""
    df = predictions.sort_values(date_col)
    fig = plt.figure()
    plt.plot(df[date_col], df["y_true"], label="y_true")
    plt.plot(df[date_col], df["y_pred"], label="y_pred")
    plt.title("Prediction vs True (time)")
    plt.xlabel(date_col)
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    return fig


def plot_residuals(predictions: pd.DataFrame, date_col: str):
    """Plot residuals en el tiempo."""
    df = predictions.sort_values(date_col)
    fig = plt.figure()
    plt.plot(df[date_col], df["residual"])
    plt.title("Residuals over time (y_true - y_pred)")
    plt.xlabel(date_col)
    plt.ylabel("residual")
    plt.tight_layout()
    return fig


def shap_summary_plot(
    model: XGBRegressor,
    train_df: pd.DataFrame,
    target_col: str,
    date_col: str,
    max_samples: int = 5000,
):
    """SHAP summary plot (beeswarm)."""
    X = train_df.drop(columns=[target_col])
    if date_col in X.columns:
        X = X.drop(columns=[date_col])

    Xs = X.sample(n=min(len(X), max_samples), random_state=42) if len(X) > max_samples else X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xs)

    fig = plt.figure()
    shap.summary_plot(shap_values, Xs, show=False)
    plt.tight_layout()
    return fig
