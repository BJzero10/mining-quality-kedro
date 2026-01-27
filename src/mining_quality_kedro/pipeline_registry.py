"""Project pipelines."""
from __future__ import annotations

from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from mining_quality_kedro.pipelines.mining_quality import pipeline as mining_quality_pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    mining_quality = mining_quality_pipeline.create_pipeline()

    return {
        "mining_quality": mining_quality,
        "__default__": mining_quality,
    }