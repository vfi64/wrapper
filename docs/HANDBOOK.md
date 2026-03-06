# Comm-SCI-Control-App Handbook (EN)

This handbook complements the README for two audiences:

- Beginner users (students, non-programmers, domain experts)
- Advanced users (developers, technical operators, CI-oriented teams)

## 1) DOI policy (stable references)

For durable references in README, handbook, and website, use concept DOIs:

- Runtime app concept DOI: [10.5281/zenodo.18445672](https://doi.org/10.5281/zenodo.18445672)
- Ruleset concept DOI: [10.5281/zenodo.17928357](https://doi.org/10.5281/zenodo.17928357)

Why: release/version DOIs change per release. Concept DOIs stay stable and expose the full version history.

## 2) Installation for beginners

### Prerequisites

- OS with Python 3 available (project minimum in `pyproject.toml`: `>=3.10`)
- terminal access

### Recommended steps

```bash
cd /path/to/repo
bash scripts/setup_venv.sh
source .venc/bin/activate
python Comm-SCI-Control-App.py
```

What the script does:

- creates/reuses `.venc`
- updates `pip/setuptools/wheel`
- installs project + dependencies via `pip install -e ".[local-dev]"`

### Typical failure modes

- `python3.14/python3 not found`: install Python or provide an explicit interpreter.
- `No module named ...`: activate venv and rerun setup.
- provider does not answer: verify API keys (see section 4).

## 3) Installation for advanced users

### Goal

- reproducible local setup
- clear test and diagnostics path

### Procedure

```bash
cd /path/to/repo
PYTHON_BIN=python3.14 VENV_DIR=.venc bash scripts/setup_venv.sh
source .venc/bin/activate
python -m pytest -q tests
python Comm-SCI-Control-App.py
```

Notes:

- project minimum remains `>=3.10`; local checks currently prefer Python 3.14.
- for CI-like reproducibility, pin Python minor version and dependency source.

## 4) API keys: onboarding, cost, security

### 4.1 Provider portals (create keys)

- Gemini / Google AI Studio:
  - API key: [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
  - pricing: [ai.google.dev/pricing](https://ai.google.dev/pricing)
- OpenRouter:
  - API keys: [openrouter.ai/keys](https://openrouter.ai/keys)
  - models/pricing: [openrouter.ai/models](https://openrouter.ai/models)
- Hugging Face:
  - access tokens: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
  - pricing: [huggingface.co/pricing](https://huggingface.co/pricing)

### 4.2 Important notice for non-professional users

- API calls can generate costs (token-based/rate-limited by provider/model).
- verify budgets, limits, and billing model before production use.
- never share keys in screenshots, chats, public repos, or issue trackers.
- if a key leaks, revoke and rotate it immediately in the provider portal.

### 4.3 Where keys are stored

- standard local key file (gitignored): `Config/Comm-SCI-API-Keys.json`
- template file (no real secrets): `Config/Comm-SCI-API-Keys.example.json`
- production/team recommendation: prefer environment variables over plaintext files

### 4.4 Runtime key precedence

Current runtime behavior prefers environment variables first, then local key files.
This is safer than static plaintext secrets inside the repository.

### 4.5 Encryption support: current status

- already supported for Gemini: `api_key_enc` + `api_key_salt` decryption (Fernet/PBKDF2)
- passphrase is read from ENV: `COMM_SCI_KEY_PASSPHRASE`
- OpenRouter/Hugging Face currently rely mainly on ENV/plaintext fields

Pragmatic recommendation:

- short term: use environment variables for all providers.
- mid term: add uniform encryption/decryption support for all providers and expose it in UI.

## 5) API key dialog (set/change/delete)

This is feasible and a good professional enhancement.

Current status:

- backend helper `set_api_key_for_provider(...)` exists.
- panel does not yet provide a fully wired dedicated key dialog.

Suggested implementation:

1. panel modal: provider selector + `set/change/delete` controls.
2. backend routes: `panel_action` actions `set_api_key` and `delete_api_key`.
3. optional secure mode: write encrypted payload (`api_key_enc` + `api_key_salt`).
4. hard rule: never expose keys in logs/audits/UI echo.

## 6) License and maintainer

- License: Apache-2.0 (see `LICENSE`)
- Maintainer: Volker Fickert
