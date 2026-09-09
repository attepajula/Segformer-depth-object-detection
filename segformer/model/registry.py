"""Model registry: resolve a model name -> a local MLX-ready weights file.

Manifest: ``segformer/models.toml``.
Cache dir: ``$SEGFORMER_MODELS_DIR`` else ``<repo>/weights/`` (gitignored).
Per model: ``weights/<name>/{source.*, model.mlx.safetensors, model.json}``.

No huggingface_hub. HF downloads are one pinned HTTPS GET, sha256-verified.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import tomllib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent          # segformer/
_REPO = _PKG.parent                                    # repo root
MANIFEST = _PKG / "models.toml"

HF_RESOLVE = "https://huggingface.co/{repo}/resolve/{revision}/{file}"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    arch: str
    input: int
    classes: str
    source: str                # "hf" | "local"
    file: str
    description: str = ""
    repo: str | None = None
    revision: str | None = None
    sha256: str | None = None
    size: int | None = None
    parent: str | None = None


def _load() -> tuple[dict[str, ModelSpec], dict]:
    data = tomllib.loads(MANIFEST.read_text())
    models = {
        name: ModelSpec(name=name, **entry)
        for name, entry in data.get("models", {}).items()
    }
    return models, data.get("defaults", {})


def manifest() -> dict[str, ModelSpec]:
    return _load()[0]


def default_model() -> str:
    return _load()[1].get("model", next(iter(manifest())))


def get(name: str | None) -> ModelSpec:
    name = name or default_model()
    models = manifest()
    if name not in models:
        raise KeyError(f"unknown model {name!r}; have: {', '.join(models)}")
    return models[name]


# -- paths ----------------------------------------------------------------
def models_root() -> Path:
    env = os.environ.get("SEGFORMER_MODELS_DIR")
    return Path(env).expanduser() if env else _REPO / "weights"


def model_dir(spec: ModelSpec) -> Path:
    return models_root() / spec.name


def source_path(spec: ModelSpec) -> Path:
    return model_dir(spec) / spec.file


def artifact_path(spec: ModelSpec) -> Path:
    return model_dir(spec) / "model.mlx.safetensors"


def sidecar_path(spec: ModelSpec) -> Path:
    return model_dir(spec) / "model.json"


# -- hashing ------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hf_cache_hit(spec: ModelSpec) -> Path | None:
    """Reuse an existing huggingface_hub download if one is already on disk."""
    if spec.source != "hf" or not spec.repo or not spec.revision:
        return None
    mangled = "models--" + spec.repo.replace("/", "--")
    p = Path.home() / ".cache/huggingface/hub" / mangled / "snapshots" / spec.revision / spec.file
    return p if p.is_file() else None


# -- ensure source / artifact -----------------------------------------
def ensure_source(spec: ModelSpec, *, quiet: bool = False) -> Path:
    dst = source_path(spec)
    if dst.is_file() and (not spec.sha256 or sha256_file(dst) == spec.sha256):
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)

    if spec.source == "local":
        raise FileNotFoundError(
            f"local model {spec.name!r}: put {spec.file} at {dst}"
        )

    hit = _hf_cache_hit(spec)
    if hit:
        if not quiet:
            print(f"  reusing HF cache: {hit}")
        shutil.copyfile(hit, dst)
    else:
        url = HF_RESOLVE.format(repo=spec.repo, revision=spec.revision, file=spec.file)
        if not quiet:
            print(f"  GET {url}")
        tmp = dst.with_suffix(dst.suffix + ".part")
        with urllib.request.urlopen(url) as r, tmp.open("wb") as fh:  # noqa: S310
            shutil.copyfileobj(r, fh)
        tmp.replace(dst)

    if spec.sha256:
        got = sha256_file(dst)
        if got != spec.sha256:
            dst.unlink()
            raise ValueError(f"{spec.name}: sha256 mismatch\n  want {spec.sha256}\n  got  {got}")
    return dst


def ensure_artifact(spec: ModelSpec, *, quiet: bool = False) -> Path:
    art = artifact_path(spec)
    if art.is_file():
        return art
    from segformer.model.convert import convert

    src = ensure_source(spec, quiet=quiet)
    if not quiet:
        print(f"  converting {src.name} -> {art.name}")
    params = convert(src, spec.arch)
    import mlx.core as mx

    mx.save_safetensors(str(art), params)
    sidecar_path(spec).write_text(json.dumps({
        "name": spec.name,
        "arch": spec.arch,
        "classes": spec.classes,
        "input": spec.input,
        "source": spec.source,
        "repo": spec.repo,
        "revision": spec.revision,
        "source_sha256": spec.sha256,
        "converted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2))
    return art


def resolve(name: str | None = None, *, quiet: bool = True) -> tuple[Path, ModelSpec]:
    spec = get(name)
    return ensure_artifact(spec, quiet=quiet), spec
