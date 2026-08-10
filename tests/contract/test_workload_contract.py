"""Collect workload contract scenarios that pin decisions and evidence boundaries."""

from __future__ import annotations

from pathlib import Path

import metrifid
from metrifid import _npz, _timegrid, _workload
from metrifid.version import __version__ as CURRENT_VERSION

ROOT = Path(__file__).resolve().parents[2]


def test_workload_loaders_stay_internal_while_writers_are_public() -> None:
    """Expose artifact writers without exposing internal admission machinery."""
    assert metrifid.__version__ == CURRENT_VERSION
    assert {"write_state_artifact", "write_actions_artifact"} <= set(metrifid.__all__)
    assert not any(
        name in metrifid.__all__
        for name in (
            "ArtifactAdmissionRefusal",
            "LoadedStateArtifact",
            "LoadedActionsArtifact",
            "WorkloadArtifacts",
            "TimeGrid",
            "load_state_artifact",
            "load_actions_artifact",
            "build_time_grid",
        )
    )
    assert _npz.__name__ == "metrifid._npz"
    assert _workload.__name__ == "metrifid._workload"
    assert _timegrid.__name__ == "metrifid._timegrid"
