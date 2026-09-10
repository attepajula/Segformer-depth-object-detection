# System changes — segformer

Same rule as the `xtion` repo: no permanent system settings touched. This
records anything outside the project directory.

## Outside the project

### Model weights
- Live in the project: `weights/<name>/` (gitignored). The registry
  (`segformer/models.toml`) downloads a pinned, sha256-checked safetensors with
  a plain HTTPS GET and converts it to an MLX artifact — no `huggingface_hub`.
- `~/.cache/huggingface/hub/` — populated when the **torch backend** or
  `tools/compare_hf.py` runs (`.[torch]`); `transformers` caches weights there.
  The MLX registry also *reuses* that cache if present, to avoid a re-download.
  Remove with `rm -rf ~/.cache/huggingface/hub/models--nvidia--segformer-*`.

### Nothing else persistent
- `.venv/` is project-local and gitignored. `.[mlx]` adds `mlx` (~small);
  `.[torch]` adds `torch` + `torchvision` + `transformers` (~2.5 GB) — all
  inside `.venv/`, no system install (not brew).
- Uses the system Homebrew `python3.14` (already installed for other projects).
  No new interpreter, unlike `xtion` (which needs x86_64 CPython for libOpenNI2).
- No `brew install`, no `defaults`, no launch agents, no PATH/rc changes.
