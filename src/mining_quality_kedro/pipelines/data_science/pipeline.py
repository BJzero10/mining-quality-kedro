from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_xgb, predict, evaluate


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_xgb,
                inputs=dict(
                    train_df="train_set",
                    target_col="params:target_col",
                    date_col="params:date_col",
                    xgb_params="params:xgb_params",
                ),
                outputs="xgb_model",
                name="ds_train_xgb",
            ),
            node(
                func=predict,
                inputs=dict(
                    model="xgb_model",
                    test_df="test_set",
                    target_col="params:target_col",
                    date_col="params:date_col",
                ),
                outputs="predictions",
                name="ds_predict",
            ),
            node(
                func=evaluate,
                inputs="predictions",
                outputs="metrics",
                name="ds_evaluate",
            ),
        ],
        tags=["data_science"],
    )
