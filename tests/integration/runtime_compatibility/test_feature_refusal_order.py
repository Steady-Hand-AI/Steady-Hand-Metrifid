"""Integration coverage for claim-specific MuJoCo feature refusal ordering."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import mujoco  # type: ignore[import-untyped]
import pytest

from metrifid._model_admission import (
    MujocoClaimSurface,
    admit_compiled_model,
    admit_external_implementation_free_model,
    require_supported_runtime,
)
from metrifid._model_closure import ModelAdmissionRefusal, ModelClosureSnapshot
from metrifid._mujoco_runtime import admit_model_feature_coverage, measure_model_feature_facts
from metrifid.compare import _model_pair
from metrifid.operational import OperationalReasonCode

_MIMO_XML = """<mujoco model="multi_input_fixture">
  <worldbody>
    <site name="reference" pos="0 0 0"/>
    <body name="payload">
      <freejoint/>
      <geom name="shape" type="sphere" size="0.1" mass="1"/>
      <site name="end_effector" pos="0 0 0"/>
    </body>
  </worldbody>
  <actuator>
    <orientation name="attitude" site="end_effector" refsite="reference"
                 kp="1" dampratio="1"/>
  </actuator>
</mujoco>
"""

_HISTORY_XML = """<mujoco model="history_fixture">
  <worldbody>
    <body name="payload">
      <joint name="joint" type="hinge"/>
      <geom name="shape" type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
  <sensor>
    <jointpos name="delayed_position" joint="joint" nsample="2" delay="1"/>
  </sensor>
</mujoco>
"""


def _multi_input_model() -> tuple[object, bool]:
    """Return a native modern fixture when available, or its exact measured signature double."""
    try:
        return mujoco.MjModel.from_xml_string(_MIMO_XML), True
    except ValueError:
        return (
            SimpleNamespace(
                nactuator=1,
                nu=3,
                nout=3,
                actuator_ctrlnum=[3],
                actuator_outnum=[3],
                actuator_ctrlspec=[1],
            ),
            False,
        )


def _history_model() -> tuple[object, bool]:
    """Return a native history fixture when available, or a closed admission-compatible double."""
    try:
        return mujoco.MjModel.from_xml_string(_HISTORY_XML), True
    except ValueError:
        sensor_kind = int(mujoco.mjtSensor.mjSENS_JOINTPOS)
        return (
            SimpleNamespace(
                nu=0,
                nplugin=0,
                npluginstate=0,
                body_plugin=[],
                geom_plugin=[],
                actuator_plugin=[],
                sensor_plugin=[],
                actuator_dyntype=[],
                actuator_gaintype=[],
                actuator_biastype=[],
                nsensor=1,
                sensor_type=[sensor_kind],
                nhistory=1,
                actuator_historyadr=[],
                sensor_historyadr=[0],
                nmocap=0,
                body_mocapid=[-1],
            ),
            False,
        )


@pytest.mark.parametrize(
    "surface",
    [
        pytest.param(MujocoClaimSurface.STATIC_MODEL_REVIEW, id="static_claim"),
        pytest.param(MujocoClaimSurface.DYNAMIC_REPLAY, id="dynamic_claim"),
    ],
)
def test_multi_input_actuator_refuses_each_one_control_claim_surface(
    surface: MujocoClaimSurface,
) -> None:
    """Name the exact native actuator signature that the current action contract cannot encode."""
    model, _native_fixture = _multi_input_model()
    admission = require_supported_runtime(surface)
    with pytest.raises(ModelAdmissionRefusal) as caught:
        admit_model_feature_coverage(model, admission, "candidate")
    assert caught.value.reason is OperationalReasonCode.MUJOCO_FEATURE_COVERAGE_INCOMPLETE
    unsupported = caught.value.evidence["unsupported_features"]["actuator_signatures"]
    assert unsupported == (
        {
            "actuator_index": 0,
            "control_inputs": 3,
            "force_outputs": 3,
            "control_spec": 1,
        },
    )
    facts = measure_model_feature_facts(model)
    assert facts.control_width == 3
    assert facts.output_width == 3


def test_multi_input_refusal_precedes_descriptor_construction_and_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Stop the dynamic path before descriptor indexing or any product artifact is published."""
    model, _native_fixture = _multi_input_model()

    def compiled_model(_snapshot: ModelClosureSnapshot, _role: str) -> object:
        """Supply the already measured feature fixture at the normal post-compile seam."""
        return model

    def unchanged(_snapshot: ModelClosureSnapshot, _role: str) -> None:
        """Keep this ordering test focused on the post-compilation feature gate."""

    def descriptor_must_not_run(*_arguments: object, **_keywords: object) -> None:
        """Fail if descriptor construction is reached after an unsupported signature."""
        raise AssertionError("descriptor construction ran before feature admission")

    monkeypatch.setattr(_model_pair, "compile_snapshot_model", compiled_model)
    monkeypatch.setattr(_model_pair, "verify_model_closure_unchanged", unchanged)
    monkeypatch.setattr(_model_pair, "compile_model_identity", descriptor_must_not_run)
    admission = require_supported_runtime(MujocoClaimSurface.DYNAMIC_REPLAY)
    synthetic_snapshot = cast(ModelClosureSnapshot, object())
    with pytest.raises(ModelAdmissionRefusal) as caught:
        _model_pair._admitted_model(synthetic_snapshot, "candidate", admission)
    assert caught.value.reason is OperationalReasonCode.MUJOCO_FEATURE_COVERAGE_INCOMPLETE
    assert list(tmp_path.iterdir()) == []


def test_history_backed_execution_refuses_while_compile_only_admission_remains_available() -> None:
    """Preserve certification of complete bytes while refusing replay without full history state."""
    model, _native_fixture = _history_model()
    compiled_admission = require_supported_runtime(MujocoClaimSurface.COMPILED_ARTIFACT)
    admit_external_implementation_free_model(model, "baseline", compiled_admission)

    dynamic_admission = require_supported_runtime(MujocoClaimSurface.DYNAMIC_REPLAY)
    with pytest.raises(ModelAdmissionRefusal) as caught:
        admit_compiled_model(model, "baseline", dynamic_admission)
    assert caught.value.reason is OperationalReasonCode.UNSUPPORTED_HISTORY_STATE
    assert int(caught.value.evidence["nhistory"]) > 0
    assert caught.value.evidence["active_history_references"] == {"sensor_historyadr": (0,)}
