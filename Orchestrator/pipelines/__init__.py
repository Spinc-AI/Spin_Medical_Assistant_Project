"""The pipeline registry.

Each pipeline is a plain Python module — no JSON instruction files, no
generic step-loop. To add one: write pipelines/<name>.py exposing ID, NAME,
DESCRIPTION, and run(), then list its module below.
"""
from . import greeting

_MODULES = [greeting]

PIPELINES = {module.ID: module for module in _MODULES}


def list_pipelines() -> list[dict]:
    return [{"id": m.ID, "name": m.NAME, "description": m.DESCRIPTION} for m in _MODULES]
