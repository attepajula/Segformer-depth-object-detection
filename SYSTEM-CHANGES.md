# System changes — segformer

Same rule as the `xtion` repo: no permanent system settings touched. This
records anything outside the project directory.

## Outside the project

### Model weights
- Live in the project: `weights/<name>/` (gitignored). The registry
  (`segformer/models.toml`) downloads a pinned, sha256-checked safetensors with
  a plain HTTPS GET and converts it to an MLX artifact — no `huggingface_hub`.
- `~/.cache/huggingface/hub/` — only if `.[reference]` (torch + transformers,
  for `tools/compare_hf.py`) was installed and run; `transformers` caches
  weights there. The registry also *reuses* that cache if it exists, to avoid a
  re-download. Remove with `rm -rf ~/.cache/huggingface/hub/models--nvidia--segformer-*`.

### Nothing else persistent
- `.venv/` is project-local and gitignored. It carries `mlx`, `opencv`, and
  (for `tools/compare_hf.py`) `torch` + `torchvision` + `transformers` — CPU
  wheels, ~2.5 GB, all inside `.venv/`.
- Uses the system Homebrew `python3.14` (already installed for other projects).
  No new interpreter, unlike `xtion` (which needs x86_64 CPython for libOpenNI2).
- No `brew install`, no `defaults`, no launch agents, no PATH/rc changes.
