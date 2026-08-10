"""Root and entrypoint resolution for the frozen Certify command contract."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from metrifid._model_closure import ModelAdmissionRefusal
from metrifid.certify._entrypoint import resolve_entrypoint
from metrifid.operational import OperationalReasonCode

_XML = "<mujoco><worldbody><body><geom size='1'/></body></worldbody></mujoco>"


def _tree(root: Path, relative: str = "model.xml") -> Path:
    """Construct the tree fixture used by certification entrypoint scenarios.

    Deterministic setup isolates certification entrypoint without bypassing the contract
    boundary under assertion.
    """
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_XML, encoding="utf-8")
    return path


def test_an_omitted_root_uses_the_real_parent_directory_and_basename(tmp_path: Path) -> None:
    """Protect the certification entrypoint assurance boundary from behavioral drift.

    This scenario exercises an omitted root uses the real parent directory and basename; the
    assertions pin the user-visible result and the evidence needed to explain that result.
    """
    entrypoint = _tree(tmp_path / "tree")
    resolved = resolve_entrypoint(str(entrypoint), None, "baseline")
    assert resolved.model_root == (tmp_path / "tree").resolve()
    assert resolved.entrypoint == "model.xml"


def test_an_omitted_root_resolves_symlinked_parent_directories(tmp_path: Path) -> None:
    """Protect the certification entrypoint assurance boundary from behavioral drift.

    This scenario exercises an omitted root resolves symlinked parent directories; the
    assertions pin the user-visible result and the evidence needed to explain that result.
    """
    _tree(tmp_path / "real")
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "real")
    resolved = resolve_entrypoint(str(link / "model.xml"), None, "baseline")
    assert resolved.model_root == (tmp_path / "real").resolve()
    assert resolved.entrypoint == "model.xml"


def test_a_supplied_root_yields_a_normal_relative_posix_entrypoint(tmp_path: Path) -> None:
    """Protect the certification entrypoint assurance boundary from behavioral drift.

    This scenario exercises a supplied root yields a normal relative posix entrypoint; the
    assertions pin the user-visible result and the evidence needed to explain that result.
    """
    root = tmp_path / "tree"
    _tree(root, "models/arm/model.xml")
    resolved = resolve_entrypoint(
        str(root / "models" / "arm" / "model.xml"), str(root), "candidate"
    )
    assert resolved.model_root == root.resolve()
    assert resolved.entrypoint == "models/arm/model.xml"
    assert not Path(resolved.entrypoint).is_absolute()


def test_a_supplied_root_equal_to_the_parent_directory_is_accepted(tmp_path: Path) -> None:
    """Protect the certification entrypoint assurance boundary from behavioral drift.

    This scenario exercises a supplied root equal to the parent directory is accepted; the
    assertions pin the user-visible result and the evidence needed to explain that result.
    """
    root = tmp_path / "tree"
    entrypoint = _tree(root)
    resolved = resolve_entrypoint(str(entrypoint), str(root), "baseline")
    assert resolved.entrypoint == "model.xml"


def test_a_relative_command_line_path_is_resolved_against_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Protect the certification entrypoint assurance boundary from behavioral drift.

    This scenario exercises a relative command line path is resolved against the working
    directory; the assertions pin the user-visible result and the evidence needed to explain
    that result.
    """
    root = tmp_path / "tree"
    _tree(root)
    monkeypatch.chdir(root)
    resolved = resolve_entrypoint("model.xml", None, "baseline")
    assert resolved.model_root == root.resolve()
    assert resolved.entrypoint == "model.xml"


def test_a_dot_segment_path_still_resolves_to_a_normal_entrypoint(tmp_path: Path) -> None:
    """Protect the certification entrypoint assurance boundary from behavioral drift.

    This scenario exercises a dot segment path still resolves to a normal entrypoint; the
    assertions pin the user-visible result and the evidence needed to explain that result.
    """
    root = tmp_path / "tree"
    _tree(root, "models/model.xml")
    resolved = resolve_entrypoint(str(root / "models" / "." / "model.xml"), str(root), "baseline")
    assert resolved.entrypoint == "models/model.xml"


def test_a_parent_segment_path_cannot_escape_the_supplied_root(tmp_path: Path) -> None:
    """Protect the certification entrypoint assurance boundary from behavioral drift.

    This scenario exercises a parent segment path cannot escape the supplied root; the
    assertions pin the user-visible result and the evidence needed to explain that result.
    """
    root = tmp_path / "tree"
    _tree(root, "models/model.xml")
    outside = _tree(tmp_path / "outside")
    with pytest.raises(ModelAdmissionRefusal) as caught:
        resolve_entrypoint(
            str(root / "models" / ".." / ".." / "outside" / "model.xml"), str(root), "baseline"
        )
    assert caught.value.reason is OperationalReasonCode.MODEL_CLOSURE_PATH_ESCAPE
    assert outside.is_file()


def test_an_entrypoint_beside_the_supplied_root_refuses(tmp_path: Path) -> None:
    """Protect the certification entrypoint assurance boundary from behavioral drift.

    This scenario exercises an entrypoint beside the supplied root refuses; the assertions pin
    the user-visible result and the evidence needed to explain that result.
    """
    root = tmp_path / "tree"
    root.mkdir()
    entrypoint = _tree(tmp_path / "other")
    with pytest.raises(ModelAdmissionRefusal) as caught:
        resolve_entrypoint(str(entrypoint), str(root), "candidate")
    assert caught.value.reason is OperationalReasonCode.MODEL_CLOSURE_PATH_ESCAPE


def test_a_symlinked_entrypoint_refuses(tmp_path: Path) -> None:
    """Protect the certification entrypoint assurance boundary from behavioral drift.

    This scenario exercises a symlinked entrypoint refuses; the assertions pin the user-visible
    result and the evidence needed to explain that result.
    """
    entrypoint = _tree(tmp_path / "tree")
    link = tmp_path / "tree" / "alias.xml"
    link.symlink_to(entrypoint)
    with pytest.raises(ModelAdmissionRefusal) as caught:
        resolve_entrypoint(str(link), None, "baseline")
    assert caught.value.reason is OperationalReasonCode.MODEL_CLOSURE_SYMLINK_REFUSED


def test_a_directory_given_as_an_entrypoint_refuses(tmp_path: Path) -> None:
    """Protect the certification entrypoint assurance boundary from behavioral drift.

    This scenario exercises a directory given as an entrypoint refuses; the assertions pin the
    user-visible result and the evidence needed to explain that result.
    """
    root = tmp_path / "tree"
    root.mkdir()
    with pytest.raises(ModelAdmissionRefusal) as caught:
        resolve_entrypoint(str(root), None, "baseline")
    assert caught.value.reason is OperationalReasonCode.MODEL_ENTRYPOINT_INVALID
    assert caught.value.evidence["issue"] == "entrypoint_not_a_regular_file"


def test_a_fifo_given_as_an_entrypoint_refuses(tmp_path: Path) -> None:
    """Protect the certification entrypoint assurance boundary from behavioral drift.

    This scenario exercises a fifo given as an entrypoint refuses; the assertions pin the user-
    visible result and the evidence needed to explain that result.
    """
    fifo = tmp_path / "pipe.xml"
    os.mkfifo(fifo)
    with pytest.raises(ModelAdmissionRefusal) as caught:
        resolve_entrypoint(str(fifo), None, "baseline")
    assert caught.value.reason is OperationalReasonCode.MODEL_ENTRYPOINT_INVALID


def test_an_absent_entrypoint_refuses(tmp_path: Path) -> None:
    """Protect the certification entrypoint assurance boundary from behavioral drift.

    This scenario exercises an absent entrypoint refuses; the assertions pin the user-visible
    result and the evidence needed to explain that result.
    """
    with pytest.raises(ModelAdmissionRefusal) as caught:
        resolve_entrypoint(str(tmp_path / "absent.xml"), None, "candidate")
    assert caught.value.reason is OperationalReasonCode.MODEL_ENTRYPOINT_INVALID
    assert caught.value.evidence["issue"] == "entrypoint_unavailable"


def test_an_absent_root_refuses(tmp_path: Path) -> None:
    """Protect the certification entrypoint assurance boundary from behavioral drift.

    This scenario exercises an absent root refuses; the assertions pin the user-visible result
    and the evidence needed to explain that result.
    """
    entrypoint = _tree(tmp_path / "tree")
    with pytest.raises(ModelAdmissionRefusal) as caught:
        resolve_entrypoint(str(entrypoint), str(tmp_path / "absent"), "baseline")
    assert caught.value.reason is OperationalReasonCode.MODEL_ROOT_INVALID


def test_a_file_given_as_a_root_refuses(tmp_path: Path) -> None:
    """Protect the certification entrypoint assurance boundary from behavioral drift.

    This scenario exercises a file given as a root refuses; the assertions pin the user-visible
    result and the evidence needed to explain that result.
    """
    entrypoint = _tree(tmp_path / "tree")
    with pytest.raises(ModelAdmissionRefusal) as caught:
        resolve_entrypoint(str(entrypoint), str(entrypoint), "baseline")
    assert caught.value.reason is OperationalReasonCode.MODEL_ROOT_INVALID


def test_a_symlinked_root_resolves_to_its_real_directory(tmp_path: Path) -> None:
    """Protect the certification entrypoint assurance boundary from behavioral drift.

    This scenario exercises a symlinked root resolves to its real directory; the assertions pin
    the user-visible result and the evidence needed to explain that result.
    """
    real = tmp_path / "real"
    _tree(real, "models/model.xml")
    link = tmp_path / "link"
    link.symlink_to(real)
    resolved = resolve_entrypoint(str(link / "models" / "model.xml"), str(link), "baseline")
    assert resolved.model_root == real.resolve()
    assert resolved.entrypoint == "models/model.xml"


def test_the_same_path_may_be_used_for_both_roles(tmp_path: Path) -> None:
    """Protect the certification entrypoint assurance boundary from behavioral drift.

    This scenario exercises the same path may be used for both roles; the assertions pin the
    user-visible result and the evidence needed to explain that result.
    """
    entrypoint = _tree(tmp_path / "tree")
    baseline = resolve_entrypoint(str(entrypoint), None, "baseline")
    candidate = resolve_entrypoint(str(entrypoint), None, "candidate")
    assert baseline == candidate
