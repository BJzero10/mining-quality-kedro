from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    plot_target_timeseries,
    plot_pred_vs_true,
    plot_residuals,
    shap_summary_plot,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=plot_target_timeseries,
                inputs=dict(
                    hourly_df="flotation_hourly",
                    date_col="params:date_col",
                    target_col="params:target_col",
                ),
                outputs="target_timeseries_plot",
                name="rp_target_timeseries",
            ),
            node(
                func=plot_pred_vs_true,
                inputs=dict(
                    predictions="predictions",
                    date_col="params:date_col",
                ),
                outputs="pred_vs_true_plot",
                name="rp_pred_vs_true",
            ),
            node(
                func=plot_residuals,
                inputs=dict(
                    predictions="predictions",
                    date_col="params:date_col",
                ),
                outputs="residuals_plot",
                name="rp_residuals",
            ),
            node(
                func=shap_summary_plot,
                inputs=dict(
                    model="xgb_model",
                    train_df="train_set",
                    target_col="params:target_col",
                    date_col="params:date_col",
                    max_samples="params:shap_max_samples",
                ),
                outputs="shap_summary_plot",
                name="rp_shap_summary",
            ),
        ],
        tags=["reporting"],
    )
