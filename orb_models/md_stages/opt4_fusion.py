"""ORBv3 Opt4 fusion entry point.

No ORB fusion candidate is currently active. The full-processor AOT attempts
were rejected because compiler reassociation in the forward pass changed the
force VJP beyond the frozen numerical contract on both validation systems.
The failed implementations remain available in Git history only.
"""
from __future__ import annotations

from md_benchmark.opt4_registry import FusionSetupError

def refresh(model, options) -> None:
    """There is no compiled ORB boundary to refresh."""

    del model, options


def install(model, passes, report, options):
    del model, report, options
    if passes:
        raise FusionSetupError("ORBv3 has no active Opt4 fusion candidate")
