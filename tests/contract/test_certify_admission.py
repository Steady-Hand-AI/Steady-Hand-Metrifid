"""The shared compile-only admission seam and everything Certify must not inherit."""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import mujoco
import numpy as np
import pytest

from metrifid import _model_admission as admission
from metrifid import _model_compile as model_compile
from metrifid._atomic_output import PairedOutputDirectory
from metrifid._model_closure import (
    ModelAdmissionRefusal,
    ModelClosureSnapshot,
    ModelRole,
)
from metrifid._owned_artifacts import RetainedArtifactPair
from metrifid.operational import OperationalReasonCode

_SIMPLE = """
<mujoco model="admission">
  <worldbody>
    <body name="b" pos="0 0 1">
      <geom name="g" type="sphere" size="0.1"/>
      <joint name="j" type="hinge" axis="0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

_MOCAP = """
<mujoco model="mocap">
  <worldbody>
    <body name="target" mocap="true" pos="0 0 1"><geom type="sphere" size="0.05"/></body>
    <body name="b" pos="0 0 1">
      <geom name="g" type="sphere" size="0.1"/>
      <joint name="j" type="hinge" axis="0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _compile_test_model(xml: str = _SIMPLE) -> mujoco.MjModel:
    """Compile an in-memory MJCF fixture for admission-boundary checks."""
    return mujoco.MjModel.from_xml_string(xml)


def test_the_seam_admits_an_ordinary_model() -> None:
    """Protect the certification admission assurance boundary from behavioral drift.

    This scenario exercises the seam admits an ordinary model; the assertions pin the user-
    visible result and the evidence needed to explain that result.
    """
    admission.admit_external_implementation_free_model(_compile_test_model(), "baseline")


def test_the_seam_requires_a_baseline_or_candidate_role() -> None:
    """Protect the certification admission assurance boundary from behavioral drift.

    This scenario exercises the seam requires a baseline or candidate role; the assertions pin
    the user-visible result and the evidence needed to explain that result.
    """
    with pytest.raises(ValueError):
        admission.admit_external_implementation_free_model(_compile_test_model(), "comparison")


def test_the_seam_refuses_active_plugin_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Protect the certification admission assurance boundary from behavioral drift.

    This scenario exercises the seam refuses active plugin state; the assertions pin the user-
    visible result and the evidence needed to explain that result.
    """
    monkeypatch.setattr(
        model_compile, "_active_plugin_arrays", lambda _compile_test_model: {"body_plugin": [0]}
    )
    with pytest.raises(ModelAdmissionRefusal) as caught:
        admission.admit_external_implementation_free_model(_compile_test_model(), "baseline")
    assert caught.value.reason is OperationalReasonCode.UNSUPPORTED_PLUGIN_STATE


def test_the_seam_refuses_user_actuators_and_user_sensors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Protect the certification admission assurance boundary from behavioral drift.

    This scenario exercises the seam refuses user actuators and user sensors; the assertions pin
    the user-visible result and the evidence needed to explain that result.
    """
    monkeypatch.setattr(model_compile, "_user_actuator_indices", lambda _compile_test_model: [3])
    with pytest.raises(ModelAdmissionRefusal) as caught:
        admission.admit_external_implementation_free_model(_compile_test_model(), "candidate")
    assert caught.value.reason is OperationalReasonCode.UNSUPPORTED_USER_CALLBACK
    assert caught.value.evidence["user_actuator_indices"] == (3,)


def test_the_seam_refuses_plugin_sensors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Protect the certification admission assurance boundary from behavioral drift.

    This scenario exercises the seam refuses plugin sensors; the assertions pin the user-visible
    result and the evidence needed to explain that result.
    """
    real = model_compile._sensor_indices
    monkeypatch.setattr(
        model_compile,
        "_sensor_indices",
        lambda model, kind: (
            [7] if kind == int(mujoco.mjtSensor.mjSENS_PLUGIN) else real(model, kind)
        ),
    )
    with pytest.raises(ModelAdmissionRefusal) as caught:
        admission.admit_external_implementation_free_model(_compile_test_model(), "baseline")
    assert caught.value.reason is OperationalReasonCode.UNSUPPORTED_PLUGIN_STATE


def test_the_seam_does_not_inherit_the_mocap_refusal() -> None:
    """A mocap body is a replay concern. It never blocks a statement about compiled bytes."""
    model = _compile_test_model(_MOCAP)
    assert int(model.nmocap) == 1
    admission.admit_external_implementation_free_model(model, "baseline")
    with pytest.raises(ModelAdmissionRefusal) as caught:
        admission.admit_compiled_model(model, "baseline")
    assert caught.value.reason is OperationalReasonCode.UNSUPPORTED_MOCAP_STATE


def test_the_seam_does_not_inherit_the_history_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Protect the certification admission assurance boundary from behavioral drift.

    This scenario exercises the seam does not inherit the history refusal; the assertions pin
    the user-visible result and the evidence needed to explain that result.
    """
    model = _compile_test_model()
    monkeypatch.setattr(
        model_compile, "_active_history_arrays", lambda _compile_test_model: {"body_history": [0]}
    )
    admission.admit_external_implementation_free_model(model, "baseline")
    with pytest.raises(ModelAdmissionRefusal) as caught:
        admission.admit_compiled_model(model, "baseline")
    assert caught.value.reason is OperationalReasonCode.UNSUPPORTED_HISTORY_STATE


def test_certify_certifies_a_mocap_model_that_replay_would_refuse(tmp_path: Path) -> None:
    """Protect the certification admission assurance boundary from behavioral drift.

    This scenario exercises certify certifies a mocap model that replay would refuse; the
    assertions pin the user-visible result and the evidence needed to explain that result.
    """
    from metrifid.certify import certify_models

    root = tmp_path / "tree"
    root.mkdir()
    (root / "model.xml").write_text(_MOCAP, encoding="utf-8")
    result = certify_models(str(root / "model.xml"), str(root / "model.xml"), str(tmp_path / "out"))
    assert result.status.value == "CERTIFIED_COMPILED_EQUIVALENCE"


def test_the_compile_guard_still_refuses_an_active_global_callback(tmp_path: Path) -> None:
    """Protect the certification admission assurance boundary from behavioral drift.

    This scenario exercises the compile guard still refuses an active global callback; the
    assertions pin the user-visible result and the evidence needed to explain that result.
    """
    from metrifid.certify import certify_models
    from metrifid.compare._failure import ComparisonOperationError

    root = tmp_path / "tree"
    root.mkdir()
    (root / "model.xml").write_text(_SIMPLE, encoding="utf-8")
    mujoco.set_mjcb_control(lambda _compile_test_model, _data: None)
    try:
        with pytest.raises(ComparisonOperationError) as caught:
            certify_models(str(root / "model.xml"), str(root / "model.xml"), str(tmp_path / "out"))
    finally:
        mujoco.set_mjcb_control(None)
    assert caught.value.failure.reason.code.value == "UNSUPPORTED_USER_CALLBACK"
    assert caught.value.failure.operation == "certify"


def test_certify_refuses_a_source_tree_mutated_during_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Protect the certification admission assurance boundary from behavioral drift.

    This scenario exercises certify refuses a source tree mutated during the run; the assertions
    pin the user-visible result and the evidence needed to explain that result.
    """
    from metrifid.certify import _run as run_module
    from metrifid.certify import certify_models
    from metrifid.compare._failure import ComparisonOperationError

    root = tmp_path / "tree"
    root.mkdir()
    entrypoint = root / "model.xml"
    entrypoint.write_text(_SIMPLE, encoding="utf-8")
    real = run_module.serialize_complete_artifact

    def mutate_then_serialize(model: object, role: str, scratch: Path) -> object:
        """Apply the targeted mutate then serialize mutation to otherwise valid evidence.

        Resealing the result isolates the semantic contradiction exercised by certify refuses a
        source tree mutated during the run.
        """
        entrypoint.write_text(_SIMPLE.replace("0.1", "0.2"), encoding="utf-8")
        return real(model, role, scratch)  # type: ignore[arg-type]

    monkeypatch.setattr(run_module, "serialize_complete_artifact", mutate_then_serialize)
    with pytest.raises(ComparisonOperationError) as caught:
        certify_models(str(entrypoint), str(entrypoint), str(tmp_path / "out"))
    assert caught.value.failure.reason.code.value == "MODEL_CLOSURE_MUTATED"
    assert not (tmp_path / "out" / "certification.json").exists()


def test_certify_reverifies_both_sources_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep both admitted source bindings live through the publication decision."""
    from metrifid import _runtime_identity as runtime_identity
    from metrifid.certify import _run as run_module
    from metrifid.certify import certify_models

    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    baseline_root.mkdir()
    candidate_root.mkdir()
    baseline = baseline_root / "model.xml"
    candidate = candidate_root / "model.xml"
    baseline.write_text(_SIMPLE, encoding="utf-8")
    candidate.write_text(_SIMPLE, encoding="utf-8")
    events: list[str] = []
    real_verify = run_module.verify_model_closure_unchanged
    real_publish = run_module.publish_paired_results

    def observe_verification(snapshot: ModelClosureSnapshot, role: ModelRole) -> None:
        """Record each real closure verification at the orchestrator boundary."""
        real_verify(snapshot, role)
        events.append(f"verify:{role}")

    def observe_publication(
        output: PairedOutputDirectory, *, json_bytes: bytes, markdown_text: str
    ) -> RetainedArtifactPair:
        """Require both retained roles to be reverified before committing outputs."""
        assert events[-2:] == ["verify:baseline", "verify:candidate"]
        events.append("publish")
        return real_publish(output, json_bytes=json_bytes, markdown_text=markdown_text)

    monkeypatch.setattr(run_module, "verify_model_closure_unchanged", observe_verification)
    monkeypatch.setattr(run_module, "publish_paired_results", observe_publication)
    monkeypatch.setattr(run_module, "installed_distribution_sha256", lambda: "1" * 64)
    monkeypatch.setattr(runtime_identity, "installed_distribution_sha256", lambda: "1" * 64)
    certify_models(str(baseline), str(candidate), str(tmp_path / "out"))

    assert events == [
        "verify:baseline",
        "verify:candidate",
        "publish",
        "verify:baseline",
        "verify:candidate",
    ]


def test_certify_refuses_output_inside_model_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse a canonical output below either model root before creating it."""
    from metrifid.certify import _run as run_module
    from metrifid.certify import certify_models
    from metrifid.compare._failure import ComparisonOperationError

    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    baseline_root.mkdir()
    candidate_root.mkdir()
    baseline = baseline_root / "model.xml"
    candidate = candidate_root / "model.xml"
    baseline.write_text(_SIMPLE, encoding="utf-8")
    candidate.write_text(_SIMPLE, encoding="utf-8")
    output_alias = tmp_path / "baseline-alias"
    output_alias.symlink_to(baseline_root, target_is_directory=True)
    output = output_alias / "results"
    monkeypatch.setattr(run_module, "installed_distribution_sha256", lambda: "1" * 64)

    with pytest.raises(ComparisonOperationError) as caught:
        certify_models(str(baseline), str(candidate), str(output))

    assert caught.value.failure.reason.code is OperationalReasonCode.OUTPUT_PATH_INVALID
    assert caught.value.failure.reason.evidence["issue"] == "output_inside_model_root"
    assert not (baseline_root / "results").exists()


def test_the_source_tree_is_never_modified(tmp_path: Path) -> None:
    """Protect the certification admission assurance boundary from behavioral drift.

    This scenario exercises the source tree is never modified; the assertions pin the user-
    visible result and the evidence needed to explain that result.
    """
    import hashlib

    from metrifid.certify import certify_models

    root = tmp_path / "tree"
    root.mkdir()
    entrypoint = root / "model.xml"
    entrypoint.write_text(_SIMPLE, encoding="utf-8")
    before = hashlib.sha256(entrypoint.read_bytes()).hexdigest()
    before_entries = sorted(item.name for item in root.iterdir())
    certify_models(str(entrypoint), str(entrypoint), str(tmp_path / "out"))
    assert hashlib.sha256(entrypoint.read_bytes()).hexdigest() == before
    assert sorted(item.name for item in root.iterdir()) == before_entries


def test_no_private_artifact_survives_a_completed_run(tmp_path: Path) -> None:
    """Protect the certification admission assurance boundary from behavioral drift.

    This scenario exercises no private artifact survives a completed run; the assertions pin the
    user-visible result and the evidence needed to explain that result.
    """
    from metrifid.certify import certify_models

    root = tmp_path / "tree"
    root.mkdir()
    (root / "model.xml").write_text(_SIMPLE, encoding="utf-8")
    certify_models(str(root / "model.xml"), str(root / "model.xml"), str(tmp_path / "out"))
    assert not list(Path(tmp_path).rglob("*.mjb"))
    assert not list(Path("/tmp").glob("metrifid-certify-*"))


def _admit(
    monkeypatch: pytest.MonkeyPatch,
    *,
    system: str = "Linux",
    mujoco_package: str = "3.10.0",
    mujoco_native: int = 3010000,
) -> None:
    """Drive the one shared runtime gate directly against a declared environment."""
    monkeypatch.setattr(model_compile.os, "name", "posix")
    monkeypatch.setattr(model_compile.platform, "system", lambda: system)
    monkeypatch.setattr(mujoco, "__version__", mujoco_package)
    monkeypatch.setattr(mujoco, "mj_version", lambda: mujoco_native)
    admission.require_supported_runtime()


def test_certify_no_longer_declares_a_second_envelope() -> None:
    """Remove the Certify-only compatibility envelope and every constant that supported it."""
    from metrifid.certify import _run as run_module

    for removed in (
        "_require_certify_envelope",
        "_CERTIFY_PLATFORMS",
        "_CERTIFY_PLATFORM_NAMES",
        "_CERTIFY_PYTHON",
    ):
        assert not hasattr(run_module, removed), removed


def test_certify_calls_the_shared_runtime_gate_exactly_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Admit Certify through the shared gate once, with no second narrower gate behind it."""
    from metrifid.certify import _run as run_module

    calls: list[int] = []

    def _record() -> None:
        """Count one shared-gate admission for this Certify invocation."""
        calls.append(1)

    monkeypatch.setattr(run_module, "require_supported_runtime", _record)
    root = tmp_path / "src"
    root.mkdir()
    (root / "model.xml").write_text(_SIMPLE, encoding="utf-8")
    run_module.certify_models(
        str(root / "model.xml"), str(root / "model.xml"), str(tmp_path / "out")
    )
    assert calls == [1]


@pytest.mark.parametrize("system", ["Darwin", "Linux"])
@pytest.mark.parametrize("machine", ["x86_64", "arm64", "aarch64"])
def test_certify_is_admitted_on_both_systems_and_any_architecture(
    monkeypatch: pytest.MonkeyPatch, system: str, machine: str
) -> None:
    """Admit Certify on Linux and Darwin without consulting the machine architecture."""
    monkeypatch.setattr(model_compile.platform, "machine", lambda: machine)
    _admit(monkeypatch, system=system)


@pytest.mark.parametrize("python", [(3, 11, 9), (3, 13, 1), (3, 14, 0)])
def test_certify_is_no_longer_restricted_to_one_python_minor(
    monkeypatch: pytest.MonkeyPatch, python: tuple[int, int, int]
) -> None:
    """Admit Certify on any supported minor: the 3.12-only Certify restriction is gone."""
    version = namedtuple("version", "major minor micro releaselevel serial")
    monkeypatch.setattr(model_compile.sys, "version_info", version(*python, "final", 0))
    _admit(monkeypatch)


def test_certify_refuses_windows_through_the_shared_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse native Windows through the shared gate rather than a Certify-only envelope."""
    monkeypatch.setattr(model_compile.os, "name", "nt")
    monkeypatch.setattr(model_compile.platform, "system", lambda: "Windows")
    with pytest.raises(ModelAdmissionRefusal) as caught:
        admission.require_supported_runtime()
    assert caught.value.reason is OperationalReasonCode.UNSUPPORTED_PLATFORM


def test_certify_accepts_a_binding_only_post_release(monkeypatch: pytest.MonkeyPatch) -> None:
    """Admit a binding-only 3.10.0.postN package that targets the same native engine."""
    _admit(monkeypatch, mujoco_package="3.10.0.post1")


def test_certify_refuses_a_mujoco_package_native_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse a package outside the engine family and a native engine that disagrees with it."""
    with pytest.raises(ModelAdmissionRefusal) as caught:
        _admit(monkeypatch, mujoco_package="3.11.0")
    assert caught.value.reason is OperationalReasonCode.UNSUPPORTED_MUJOCO_VERSION
    with pytest.raises(ModelAdmissionRefusal) as caught:
        _admit(monkeypatch, mujoco_native=3011000)
    assert caught.value.reason is OperationalReasonCode.MUJOCO_PYTHON_NATIVE_VERSION_MISMATCH
    assert caught.value.evidence["native_version_integer"] == 3011000


def test_the_runtime_identity_stays_compact() -> None:
    """Protect the certification admission assurance boundary from behavioral drift.

    This scenario exercises the runtime identity stays compact; the assertions pin the user-
    visible result and the evidence needed to explain that result.
    """
    from metrifid._runtime_identity import build_certify_runtime_identity

    identity = build_certify_runtime_identity((54321, 8, 92, 3010000, 471)).to_primitive()
    assert len(identity) < 25
    for value in identity.values():
        assert not isinstance(value, dict), "no nested manifest belongs in the runtime identity"
    assert identity["execution_mode"] == "NO_MJDATA_EXECUTION"
    assert identity["mjb_header_words"] == [54321, 8, 92, 3010000, 471]
    assert np.isscalar(identity["mujoco_version_integer"])
