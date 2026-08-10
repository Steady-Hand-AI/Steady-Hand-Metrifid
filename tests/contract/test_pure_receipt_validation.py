"""Certification receipt validation must work without any native dependency.

A reader who receives a certificate should be able to revalidate it with nothing installed but
Metrifid itself. These tests hold that line two ways: by inspecting the transitive import graph of
the receipt module, and by driving the public raw loader inside a fresh interpreter where importing
MuJoCo or NumPy raises ``ModuleNotFoundError``.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_BLOCK_NATIVE = textwrap.dedent(
    """
    import sys

    class _BlockNative:
        def find_spec(self, name, path=None, target=None):
            root = name.split(".", 1)[0]
            if root in {"mujoco", "numpy"}:
                raise ModuleNotFoundError(f"{name} is unavailable in this environment")
            return None

    sys.meta_path.insert(0, _BlockNative())
    """
)


_MODEL_XML = """
<mujoco model="pure-receipt">
  <worldbody>
    <body name="b" pos="0 0 1">
      <geom name="g" type="sphere" size="0.1"/>
      <joint name="j" type="hinge" axis="0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def certification_receipt_bytes(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    """Produce one real serialized certification receipt for pure revalidation."""
    from metrifid.certify import certify_models

    root: Path = tmp_path_factory.mktemp("pure-receipt").resolve()
    # The output directory must sit outside both model roots, so each model gets its own root.
    baseline_root = root / "baseline"
    candidate_root = root / "candidate"
    baseline_root.mkdir()
    candidate_root.mkdir()
    baseline = baseline_root / "model.xml"
    candidate = candidate_root / "model.xml"
    baseline.write_text(_MODEL_XML, encoding="utf-8")
    candidate.write_text(_MODEL_XML, encoding="utf-8")
    output = root / "out"
    certify_models(str(baseline), str(candidate), str(output))
    return (output / "certification.json").read_bytes()


def _run_pure(body: str) -> subprocess.CompletedProcess[str]:
    """Run one script in a fresh interpreter where MuJoCo and NumPy cannot be imported."""
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_NATIVE + textwrap.dedent(body)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_receipt_import_graph_excludes_native_and_producer_modules() -> None:
    """Keep MuJoCo, NumPy, the native field producer, and live runtime measurement out."""
    completed = _run_pure(
        """
        import sys
        import importlib

        importlib.import_module("metrifid.certify._receipt")
        loaded = set(sys.modules)
        assert not any(m == "mujoco" or m.startswith("mujoco.") for m in loaded), "mujoco imported"
        assert not any(m == "numpy" or m.startswith("numpy.") for m in loaded), "numpy imported"
        assert "metrifid.certify._fields" not in loaded, "native field producer imported"
        assert "metrifid._runtime_identity" not in loaded, "runtime measurement imported"
        print("PURE")
        """
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "PURE" in completed.stdout


def test_public_certify_names_import_without_native_dependencies() -> None:
    """Import the certify package and both receipt entry points with natives blocked."""
    completed = _run_pure(
        """
        import metrifid.certify
        from metrifid.certify import load_and_validate_certification_receipt, validate_receipt

        assert callable(validate_receipt)
        assert callable(load_and_validate_certification_receipt)
        print("IMPORTED")
        """
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "IMPORTED" in completed.stdout


def test_raw_loader_validates_and_rejects_without_native_dependencies(
    certification_receipt_bytes: bytes,
) -> None:
    """Revalidate a real receipt and reject a duplicate-name mutation in a pure interpreter."""
    payload = certification_receipt_bytes.decode("utf-8")
    completed = _run_pure(
        f"""
        from metrifid.certify import load_and_validate_certification_receipt

        raw = {payload!r}
        receipt = load_and_validate_certification_receipt(raw)
        assert isinstance(receipt, dict)
        assert receipt["schema"] == "metrifid.compiled_equivalence_receipt"

        marker = '"status"'
        index = raw.index(marker)
        mutated = raw[:index] + '"status": "INJECTED", ' + raw[index:]
        try:
            load_and_validate_certification_receipt(mutated)
        except Exception:
            print("VALIDATED_AND_REJECTED")
        else:
            raise AssertionError("duplicate member name was accepted")
        """
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "VALIDATED_AND_REJECTED" in completed.stdout


def test_raw_loader_requires_a_root_object(certification_receipt_bytes: bytes) -> None:
    """Refuse a well-formed document whose root is not an object."""
    from metrifid._json_admission import JsonAdmissionError
    from metrifid.certify import load_and_validate_certification_receipt

    with pytest.raises(JsonAdmissionError):
        load_and_validate_certification_receipt(b"[1, 2, 3]")


def test_raw_loader_rejects_noncanonical_numeric_tokens() -> None:
    """Refuse raw float tokens before any semantic validation runs."""
    from metrifid._json_admission import JsonAdmissionError
    from metrifid.certify import load_and_validate_certification_receipt

    with pytest.raises(JsonAdmissionError):
        load_and_validate_certification_receipt(b'{"schema": "x", "value": 1.5}')
