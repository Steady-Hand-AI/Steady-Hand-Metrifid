"""Collect MuJoCo callback surface scenarios that pin decisions and evidence boundaries."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, cast

import mujoco  # type: ignore[import-untyped]
import pytest

from metrifid import _model_compile as model_compile

_FROZEN_CALLBACK_NAMES = (
    "mjcb_control",
    "mjcb_sensor",
    "mjcb_passive",
    "mjcb_act_dyn",
    "mjcb_act_gain",
    "mjcb_act_bias",
    "mjcb_contactfilter",
    "mjcb_time",
    "mju_user_warning",
    "mju_user_malloc",
    "mju_user_free",
)

_PROBE_SCRIPT = r"""
import ctypes
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco

from metrifid import _model_admission as admission
from metrifid import _model_closure as closure
from metrifid import _model_compile as model_compile
from metrifid import _model_identity as identity
from metrifid.json_values import thaw_canonical

mode = sys.argv[1]
baseline = Path(sys.argv[2])
candidate = Path(sys.argv[3])
libc = ctypes.CDLL(None)
libc.posix_memalign.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_size_t,
    ctypes.c_size_t,
]
libc.posix_memalign.restype = ctypes.c_int
libc.free.argtypes = [ctypes.c_void_p]
libc.free.restype = None
malloc_type = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t)
free_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
calls = {"malloc": 0, "free": 0}
blocked_calls = []

@malloc_type
def allocate(size, _calls=calls, _libc=libc):
    _calls["malloc"] += 1
    pointer = ctypes.c_void_p()
    result = _libc.posix_memalign(ctypes.byref(pointer), 64, max(1, size))
    return pointer.value if result == 0 else 0

@free_type
def release(pointer, _calls=calls, _libc=libc):
    _calls["free"] += 1
    _libc.free(pointer)

def address(value):
    return None if value is None else ctypes.cast(value, ctypes.c_void_p).value

def forbidden(*args, **kwargs):
    del args, kwargs
    blocked_calls.append("called")
    raise RuntimeError("callback refusal did not precede a forbidden operation")

previous_malloc = mujoco.get_mju_user_malloc()
previous_free = mujoco.get_mju_user_free()
result = {}
try:
    if mode in {"malloc", "both"}:
        mujoco.set_mju_user_malloc(allocate)
    if mode in {"free", "both"}:
        mujoco.set_mju_user_free(release)
    if mode == "both":
        model_compile.discover_snapshot_dependencies = forbidden
        model_compile._require_mjcf_root = forbidden
        model_compile.mujoco.MjModel = SimpleNamespace(from_xml_path=forbidden)
    before = [address(mujoco.get_mju_user_malloc()), address(mujoco.get_mju_user_free())]
    try:
        identity.build_model_pair_identity(
            baseline,
            "model.xml",
            candidate,
            "model.xml",
        )
    except closure.ModelAdmissionRefusal as exc:
        result = {
            "outcome": "refused",
            "reason": exc.reason.value,
            "role": exc.role,
            "evidence": thaw_canonical(exc.evidence),
        }
    except Exception as exc:
        result = {
            "outcome": "unexpected_exception",
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
    else:
        result = {"outcome": "accepted"}
    result.update(
        {
            "before": before,
            "after": [
                address(mujoco.get_mju_user_malloc()),
                address(mujoco.get_mju_user_free()),
            ],
            "calls": calls,
            "blocked_calls": blocked_calls,
        }
    )
finally:
    mujoco.set_mju_user_free(previous_free)
    mujoco.set_mju_user_malloc(previous_malloc)
    result["restored"] = [
        address(mujoco.get_mju_user_malloc()),
        address(mujoco.get_mju_user_free()),
    ]
    result["previous"] = [address(previous_malloc), address(previous_free)]

sys.stdout.write(json.dumps(result, sort_keys=True))
sys.stdout.flush()
os._exit(0)
"""


def _write_model_pair(root: Path) -> tuple[Path, Path]:
    """Write write model pair data into the isolated test workspace.

    The MuJoCo callback surface scenario observes real bytes and filesystem effects for MuJoCo
    callback surface.
    """
    xml = (
        '<mujoco><worldbody><body><joint name="joint"/>'
        '<geom size="0.1" mass="1"/></body></worldbody></mujoco>'
    )
    baseline = root / "baseline"
    candidate = root / "candidate"
    for model_root in (baseline, candidate):
        model_root.mkdir()
        (model_root / "model.xml").write_text(xml, encoding="utf-8")
    return baseline.resolve(), candidate.resolve()


def _run_allocator_probe(tmp_path: Path, mode: str) -> dict[str, Any]:
    """Construct the run allocator probe fixture used by MuJoCo callback surface scenarios.

    Deterministic setup isolates MuJoCo callback surface without bypassing the contract boundary
    under assertion.
    """
    baseline, candidate = _write_model_pair(tmp_path)
    script = tmp_path / "allocator_probe.py"
    script.write_text(textwrap.dedent(_PROBE_SCRIPT), encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, str(script), mode, str(baseline), str(candidate)],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not result.stderr
    value = json.loads(result.stdout)
    assert type(value) is dict
    return cast(dict[str, Any], value)


def test_callback_registry_matches_complete_mujoco_310_surface() -> None:
    """Prevent unmeasured model state from supporting an equivalence claim.

    This scenario exercises callback registry matches complete MuJoCo 310 surface; the
    assertions bind admission to exact model bytes, resource boundaries, or an explicit refusal
    reason.
    """
    registry_names = tuple(name for name, _getter in model_compile._CALLBACK_ACCESSORS)
    assert registry_names == _FROZEN_CALLBACK_NAMES

    discovered = {name.removeprefix("get_") for name in dir(mujoco) if name.startswith("get_mjcb_")}
    discovered.update({"mju_user_warning", "mju_user_malloc", "mju_user_free"})
    assert discovered == set(_FROZEN_CALLBACK_NAMES)


@pytest.mark.parametrize(
    ("mode", "expected_active"),
    [
        ("malloc", ["mju_user_malloc"]),
        ("free", ["mju_user_free"]),
    ],
)
def test_each_allocator_callback_alone_refuses_without_invocation(
    tmp_path: Path,
    mode: str,
    expected_active: list[str],
) -> None:
    """Prevent unmeasured model state from supporting an equivalence claim.

    This scenario exercises each allocator callback alone refuses without invocation; the
    assertions bind admission to exact model bytes, resource boundaries, or an explicit refusal
    reason.
    """
    result = _run_allocator_probe(tmp_path, mode)
    assert result["outcome"] == "refused"
    assert result["reason"] == "UNSUPPORTED_USER_CALLBACK"
    assert result["role"] == "baseline"
    assert result["evidence"] == {"active_callbacks": expected_active}
    assert result["calls"] == {"malloc": 0, "free": 0}
    assert result["before"] == result["after"]
    assert result["restored"] == result["previous"]


def test_both_allocator_callbacks_refuse_before_dependencies_root_or_compile(
    tmp_path: Path,
) -> None:
    """Prevent unmeasured model state from supporting an equivalence claim.

    This scenario exercises both allocator callbacks refuse before dependencies root or compile;
    the assertions bind admission to exact model bytes, resource boundaries, or an explicit
    refusal reason.
    """
    result = _run_allocator_probe(tmp_path, "both")
    assert result["outcome"] == "refused"
    assert result["reason"] == "UNSUPPORTED_USER_CALLBACK"
    assert result["role"] == "baseline"
    assert result["evidence"] == {"active_callbacks": ["mju_user_malloc", "mju_user_free"]}
    assert result["calls"] == {"malloc": 0, "free": 0}
    assert result["blocked_calls"] == []
    assert result["before"] == result["after"]
    assert result["restored"] == result["previous"]
