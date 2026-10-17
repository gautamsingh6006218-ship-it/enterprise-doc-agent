"""Pipeline module: run ingest → preprocess → chunk → embed and record the result."""

from agent.pipeline.models import PipelineResult, StageResult
from agent.pipeline.orchestrator import DocumentPipeline
from agent.pipeline.service import PipelineService

__all__ = [
    "PipelineResult",
    "StageResult",
    "DocumentPipeline",
    "PipelineService",
]
