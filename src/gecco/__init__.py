from __future__ import annotations

from .config import GeCCoConfig

__all__ = ["GeCCoConfig", "GeCCoPipeline", "PipelineResult", "post_process"]


def __getattr__(name: str):
    if name == "GeCCoPipeline":
        from .pipeline import GeCCoPipeline
        return GeCCoPipeline
    if name == "PipelineResult":
        from .pipeline import PipelineResult
        return PipelineResult
    if name == "post_process":
        from .tree import post_process
        return post_process
    raise AttributeError(f"module 'gecco' has no attribute {name!r}")
