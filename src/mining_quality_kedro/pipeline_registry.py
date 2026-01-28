from __future__ import annotations

from typing import Dict
from kedro.pipeline import Pipeline

from mining_quality_kedro.pipelines import data_processing as dp
from mining_quality_kedro.pipelines import data_science as ds
from mining_quality_kedro.pipelines import reporting as rp


def register_pipelines() -> Dict[str, Pipeline]:
    data_processing = dp.create_pipeline()
    data_science = ds.create_pipeline()
    reporting = rp.create_pipeline()

    mining_quality = data_processing + data_science + reporting

    return {
        "data_processing": data_processing,
        "data_science": data_science,
        "reporting": reporting,
        "mining_quality": mining_quality,
        "__default__": mining_quality,
    }
