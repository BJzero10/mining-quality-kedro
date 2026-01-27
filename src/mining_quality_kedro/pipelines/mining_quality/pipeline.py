from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    clean_data,
    resample_time,
    time_split,
    train_xgb,
    evaluate,
    shap_explainability,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_data,
                inputs=dict(df="flotation_raw", date_col="params:date_col"),
                outputs="flotation_clean",
                name="clean_data_node",
            ),
            node(
                func=resample_time,
                inputs=dict(
                    df="flotation_clean",
                    date_col="params:date_col",
                    rule="params:resample_rule",
                ),
                outputs="flotation_hourly",
                name="resample_time_node",
            ),
            node(
                func=time_split,
                inputs=dict(
                    df="flotation_hourly",
                    date_col="params:date_col",
                    test_size="params:test_size",
                ),
                outputs=["train_set", "test_set"],
                name="time_split_node",
            ),
            node(
                func=train_xgb,
                inputs=dict(
                    train_df="train_set",
                    target_col="params:target_col",
                    date_col="params:date_col",
                    xgb_params="params:xgb_params",
                ),
                outputs="xgb_model",
                name="train_xgb_node",
            ),
            node(
                func=evaluate,
                inputs=dict(
                    model="xgb_model",
                    test_df="test_set",
                    target_col="params:target_col",
                    date_col="params:date_col",
                ),
                outputs="metrics",
                name="evaluate_node",
            ),
            node(
                func=shap_explainability,
                inputs=dict(
                    model="xgb_model",
                    train_df="train_set",
                    target_col="params:target_col",
                    date_col="params:date_col",
                ),
                outputs="shap_summary_plot",
                name="shap_explainability_node",
            ),
        ],
        tags=["mining_quality"],
    )
