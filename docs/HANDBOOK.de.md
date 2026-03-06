# Comm-SCI-Control-App Handbuch (DE)

Dieses Handbuch ergaenzt die README fuer zwei Zielgruppen:

- Einsteiger (Laien, Schueler, Studierende, fachfremde Wissenschaftler)
- Fortgeschrittene (Entwickler, technisch versierte Nutzer, CI-orientierte Teams)

## 1) DOI-Policy (stabile Referenzen)

Fuer dauerhaft gueltige Verweise in README, Handbuch und Website sollten nur Concept-DOIs genutzt werden:

- Runtime-App Concept DOI: [10.5281/zenodo.18445672](https://doi.org/10.5281/zenodo.18445672)
- Regelwerk Concept DOI: [10.5281/zenodo.17928357](https://doi.org/10.5281/zenodo.17928357)

Warum: Ein Release-DOI aendert sich pro Release. Concept-DOIs bleiben stabil und fuehren auf die Versionsuebersicht.

## 2) Installation fuer Einsteiger

### Voraussetzungen

- Betriebssystem mit Python 3 (Projektminimum laut `pyproject.toml`: `>=3.10`)
- Terminalzugang

### Empfohlene Schrittfolge

```bash
cd /pfad/zum/repo
bash scripts/setup_venv.sh
source .venc/bin/activate
python Comm-SCI-Control-App.py
```

Was das Skript macht:

- erstellt ein virtuelles Environment `.venc` (oder nutzt ein vorhandenes)
- aktualisiert `pip/setuptools/wheel`
- installiert Projekt + Abhaengigkeiten (`pip install -e ".[local-dev]"`)

### Typische Fehlerbilder

- `python3.14/python3 nicht gefunden`: Python installieren oder den Interpreter explizit setzen.
- `No module named ...`: venv aktivieren (`source .venc/bin/activate`) und Setup erneut ausfuehren.
- Provider antwortet nicht: API-Key pruefen (siehe Abschnitt 4).

## 3) Installation fuer Fortgeschrittene

### Ziel

- reproduzierbarer lokaler Build
- klare Test- und Diagnosepfade

### Vorgehen

```bash
cd /pfad/zum/repo
PYTHON_BIN=python3.14 VENV_DIR=.venc bash scripts/setup_venv.sh
source .venc/bin/activate
python -m pytest -q tests
python Comm-SCI-Control-App.py
```

Hinweise:

- Projektminimum bleibt `>=3.10`; lokal wird aktuell Python 3.14 bevorzugt.
- Fuer CI-nahe Reproduzierbarkeit: gleiche Python-Minor-Version und gleiche Dependency-Quelle verwenden.

## 4) API-Keys: Beschaffung, Kosten, Sicherheit

### 4.1 Provider-Portale (Key erzeugen)

- Gemini / Google AI Studio:
  - API-Key: [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
  - Pricing/Uebersicht: [ai.google.dev/pricing](https://ai.google.dev/pricing)
- OpenRouter:
  - API-Keys: [openrouter.ai/keys](https://openrouter.ai/keys)
  - Modelle/Preise: [openrouter.ai/models](https://openrouter.ai/models)
- Hugging Face:
  - Access Tokens: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
  - Pricing: [huggingface.co/pricing](https://huggingface.co/pricing)

### 4.2 Wichtige Hinweise fuer nicht professionelle Nutzer

- API-Aufrufe koennen Kosten verursachen (tokenbasiert/ratelimitiert je Provider/Modell).
- Vor produktiver Nutzung Budget, Limits und Abrechnungsmodell pruefen.
- API-Keys niemals in Screenshots, Chats, oeffentlichen Repos oder Issues posten.
- Bei Verdacht auf Leak: Key sofort im Provider-Portal widerrufen und neu erzeugen.

### 4.3 Wo Keys im Wrapper liegen

- Standarddatei (lokal, gitignored): `Config/Comm-SCI-API-Keys.json`
- Vorlage ohne echte Secrets: `Config/Comm-SCI-API-Keys.example.json`
- Empfohlen in Produktion/Team: ENV-Variablen statt Klartextdatei

### 4.4 Schluessel prioritaet (Runtime)

Gemaess aktuellem Code werden Keys bevorzugt aus ENV gelesen, danach aus lokalen Key-Dateien.
Das ist sicherer als statische Klartextablage im Repo.

### 4.5 Verschluesselung: aktueller Stand

- Bereits unterstuetzt (Runtime): entschluesseln von `api_key_enc` + `api_key_salt` (Fernet/PBKDF2) fuer Gemini.
- Passphrase kommt aus ENV: `COMM_SCI_KEY_PASSPHRASE`.
- OpenRouter/Hugging Face sind aktuell primar auf ENV/Klartextfelder ausgelegt.

Pragmatische Empfehlung:

- Kurzfristig: ENV-Variablen als Standard fuer alle Provider.
- Mittelfristig: einheitliche Encrypt/Decrypt-Pipeline fuer alle Provider plus UI-Dialog.

## 5) API-Keys per Dialog eingeben/aendern/loeschen

Ist technisch sinnvoll und professionell umsetzbar.

Aktueller Stand:

- Backend-Helfer `set_api_key_for_provider(...)` ist vorhanden.
- Ein dedizierter Key-Dialog im Panel ist noch nicht durchgaengig verdrahtet.

Empfohlene Nachruestung:

1. Panel-UI: Modal mit Providerauswahl + Felder `Setzen/Aendern/Loeschen`.
2. Backend-Route: `panel_action`-Action `set_api_key`/`delete_api_key`.
3. Optional: Modus `encrypted` fuer Dateiablage (`api_key_enc` + `api_key_salt`).
4. Security: niemals Keys in Logs/Audits/UI-Echo ausgeben.

## 6) Lizenz und Maintainer

- Lizenz: Apache-2.0 (siehe `LICENSE`)
- Maintainer: Volker Fickert
