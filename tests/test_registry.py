"""Registry: manifest parse, path derivation, hashing, offline resolve.
No network — asserts only on the manifest and on already-present files."""

import hashlib

import pytest

from segformer.model import registry as reg


def test_manifest_loads_and_has_default():
    models = reg.manifest()
    assert models, "models.toml has no [models.*]"
    d = reg.default_model()
    assert d in models
    spec = models[d]
    assert spec.arch in ("b0", "b2")
    if spec.source == "hf":
        assert spec.repo and spec.revision and spec.sha256


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        reg.get("does-not-exist")


def test_paths_are_under_models_root(tmp_path, monkeypatch):
    monkeypatch.setenv("SEGFORMER_MODELS_DIR", str(tmp_path))
    spec = reg.get(None)
    assert reg.models_root() == tmp_path
    assert reg.artifact_path(spec) == tmp_path / spec.name / "model.mlx.safetensors"
    assert reg.source_path(spec).parent == tmp_path / spec.name


def test_sha256_file(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"segformer")
    assert reg.sha256_file(p) == hashlib.sha256(b"segformer").hexdigest()


def test_resolve_reuses_existing_artifact():
    """If weights were already converted (CI/dev machine), resolve is offline."""
    spec = reg.get(None)
    if not reg.artifact_path(spec).is_file():
        pytest.skip("artifact not built; run: segformer models convert")
    path, got = reg.resolve()
    assert path == reg.artifact_path(spec)
    assert got.name == spec.name
