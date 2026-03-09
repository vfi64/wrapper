# Install and Sync (Private -> Local Wrapper Clone)

This document defines the local handover routine from this source repository to the local clone of `vfi64/wrapper`.

## Goal

- Use a compatible Python version automatically.
- Install all required packages deterministically.
- Sync tracked wrapper files safely into local `vfi64/wrapper`.
- Keep `vfi64/wrapper` remote untouched until explicit release decision.

## 1) Environment bootstrap

```bash
bash scripts/setup_venv.sh
```

What it does:
- Reads minimum Python requirement from `pyproject.toml` (`requires-python`).
- Selects the first compatible interpreter from common candidates.
- Creates/uses `.venc`.
- Installs project package with extras (`local-dev` by default).
- Runs `pip check`.

Useful flags:

```bash
bash scripts/setup_venv.sh --dry-run
bash scripts/setup_venv.sh --python python3.12
bash scripts/setup_venv.sh --extras none
```

## 2) Local sync to wrapper clone

Dry-run first:

```bash
bash scripts/sync_to_wrapper_local.sh --target /path/to/wrapper
```

Apply sync:

```bash
bash scripts/sync_to_wrapper_local.sh --target /path/to/wrapper --apply --validate
```

Safety controls:
- Verifies target is a git repo with origin containing `vfi64/wrapper`.
- Blocks dirty target repos unless `--allow-dirty-target` is set.
- Copies only tracked files from source repo.
- Skips local secrets/caches/log artifacts.

## 3) Release gating (later step)

Do not push `vfi64/wrapper` remote in this phase.
First complete verification in source repo, then copy locally, then adapt naming/links for public context, then release.
