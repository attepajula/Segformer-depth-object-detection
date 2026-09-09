# System changes — segformer

Same rule as the `xtion` repo: no permanent system settings touched. This
records anything outside the project directory.

## Outside the project

### Hugging Face model cache
- `~/.cache/huggingface/hub/` — SegFormer weights downloaded on first use:
  `nvidia/segformer-b0-finetuned-ade-512-512` (~15 MB safetensors + config).
  B2 (~110 MB) if used.
- Additive cache, managed by `huggingface_hub`. Remove with
  `huggingface-cli delete-cache` or `rm -rf ~/.cache/huggingface/hub/models--nvidia--segformer-*`.

### Nothing else persistent
- `.venv/` is project-local and gitignored. It carries `mlx`, `opencv`, and
  (for `tools/compare_hf.py`) `torch` + `torchvision` + `transformers` — CPU
  wheels, ~2.5 GB, all inside `.venv/`.
- Uses the system Homebrew `python3.14` (already installed for other projects).
  No new interpreter, unlike `xtion` (which needs x86_64 CPython for libOpenNI2).
- No `brew install`, no `defaults`, no launch agents, no PATH/rc changes.
