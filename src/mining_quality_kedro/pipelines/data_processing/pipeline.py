from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import validate_and_sort, resample_time, time_split


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=validate_and_sort,
                inputs=dict(df="flotation_raw", date_col="params:date_col"),
                outputs="mq_flotation_validated",
                name="dp_validate_and_sort",
            ),
            node(
                func=resample_time,
                inputs=dict(
                    df="mq_flotation_validated",
                    date_col="params:date_col",
                    rule="params:resample_rule",
                ),
                outputs="mq_flotation_hourly",
                name="dp_resample_time",
            ),
            node(
                func=time_split,
                inputs=dict(
                    df="mq_flotation_hourly",
                    date_col="params:date_col",
                    test_size="params:test_size",
                ),
                outputs=["mq_train_set", "mq_test_set"],
                name="dp_time_split",
            ),
        ],
        tags=["data_processing"],
    )
