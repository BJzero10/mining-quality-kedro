from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd


def validate_and_sort(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Assumes catalog already parsed dates and decimals (Kaggle-style load_args).
    - Validates required column exists
    - Ensures datetime
    - Sorts by date
    - Drops duplicate timestamps (keep last)
    """
    if date_col not in df.columns:
        raise ValueError(f"'{date_col}' no existe. Columnas: {list(df.columns)}")

    out = df.copy()

    # Ensure datetime (catalog should do it, but keep it robust)
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])

    out = out.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")
    out = out.reset_index(drop=True)

    return out


def resample_time(df: pd.DataFrame, date_col: str, rule: str) -> pd.DataFrame:
    """Resample by time (e.g., '1h') using mean aggregation."""
    if df.empty:
        return df.copy()

    # compatibility guard
    rule = rule.replace("H", "h")

    out = df.copy()
    out = out.set_index(pd.DatetimeIndex(out[date_col]))
    out = out.drop(columns=[date_col])

    out = out.resample(rule).mean()
    out = out.dropna(axis=0, how="any")

    out = out.reset_index().rename(columns={"index": date_col})
    return out


def time_split(df: pd.DataFrame, date_col: str, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Time-ordered split (no shuffle)."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size debe estar entre 0 y 1")

    out = df.sort_values(date_col).reset_index(drop=True)
    n = len(out)
    cut = int(np.floor(n * (1.0 - test_size)))
    cut = max(1, min(cut, n - 1))

    train_df = out.iloc[:cut].copy()
    test_df = out.iloc[cut:].copy()
    return train_df, test_df
