# Comm-SCI-Wrapper v136

**Deterministischer Single-File-Python-Wrapper für die auditierbare Nutzung von LLMs unter der normativen Governance-Spezifikation von Comm-SCI-Control (v19.6.8).**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)
[![Comm-SCI-Control](https://img.shields.io/badge/Comm--SCI--Control-v19.6.8-orange.svg)](https://github.com/vfi64/Comm-SCI-Control)
[![DOI](https://zenodo.org/badge/1137466025.svg)](https://doi.org/10.5281/zenodo.18445673)
[![Zenodo (ruleset)](https://zenodo.org/badge/DOI/10.5281/zenodo.18108395.svg)](https://doi.org/10.5281/zenodo.18108395)
[![tests](https://github.com/vfi64/wrapper/actions/workflows/tests.yml/badge.svg)](https://github.com/vfi64/wrapper/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

> Kernidee:  
> **Comm-SCI-Control** = Governance-Regelwerk (JSON, *keine Runtime, keine Code-Ausführung*).  
> **Wrapper-136.py** = technische Durchsetzung und Session-Schicht (Zustandsautomat, UI, Export, Audit, Provider-Wechsel).

Referenz-Regelwerk: https://github.com/vfi64/Comm-SCI-Control
Wrapper DOI: https://doi.org/10.5281/zenodo.18445673

Empfohlene Zitierung (Regelwerk): https://doi.org/10.5281/zenodo.18108395

---

## Schnellstart

1. Abhängigkeiten installieren:
   ```bash
   pip install pywebview requests markdown pytest bleach cryptography google-genai
   ```
2. Wrapper starten:
   ```bash
   python3 Wrapper-136.py
   ```
3. Testsuite ausführen:
   ```bash
   python3 -m pytest -vv -s --tb=long Test-136.py
   ```
4. Im Chat-Fenster testen:
   - `Comm Help`
   - `Comm State`
   - `QC Override` (öffnet die Slider-UI für temporäre QC-Anpassungen, falls aktiviert)

---

## Konzept: Governance vs. Runtime

**Comm-SCI-Control** ist eine *reine Spezifikation* (JSON). Sie definiert *was passieren soll*: Profile, Befehle, strukturierte Workflows (SCI), QC-Metriken mit Abweichungsberichten, eine Uncertainty-Taxonomie sowie Post-Answer-Self-Audits.

Dieser Wrapper ist die *Exekutive*: Er hält einen externen Zustandsautomaten vor und erzeugt strukturierte Logs, sodass Governance **sichtbar, testbar und auditierbar** wird (statt „Prompt-Magie“).

---

## Kernfunktionen (v136)

Testsuite: **67 Tests** (offline, deterministisch; keine GUI, keine echten Provider-Calls).

- **Audit-v2-Export** (enthält `trace_id`, Provider-/Modell-Kontext, Ruleset-Hash, Wrapper-File-Hash und einen Event-Stream)
- **QC-Override-UI** (6 Slider) und deterministische QC-Delta-Behandlung
- **Chat-Log-Replay** sowie optional **Load & Fork** für neue Sessions
- **Multi-Provider-Support** (Gemini / OpenRouter / Hugging-Face-Katalog), konfigurationsgetrieben
- **Guardrails** (z. B. „keine Netzwerk-Calls bei UI-only Actions“) + Rate Limiting
- **HTML-Sanitization** via `bleach` (verhindert unsichere HTML-Ausgabe im Rendering)
- **Optionale verschlüsselte Key-Ablage** (Fernet / `cryptography`)
- **Offline-Regression-Suite** in `Test-136.py` (keine GUI, keine echten Provider-Calls)

---

## Was dieses Repository enthält

- [`Wrapper-136.py`](https://github.com/vfi64/wrapper/blob/main/Wrapper-136.py) — die Runtime (UI, Session-State, Guardrails, Exporte, Audit)
- [`Test-136.py`](https://github.com/vfi64/wrapper/blob/main/Test-136.py) — das Offline-Regression-Gate (**keine GUI, keine echten Provider-Calls**)
- `JSON/Comm-SCI-v19.6.8.json` — das Regelwerk (**Source of Truth**)

---

## Verzeichnisstruktur

Erwartetes Layout:

```text
.
├── Wrapper-136.py
├── Test-136.py
├── JSON/
│   ├── Comm-SCI-v19.6.8.json
│   └── Comm-SCI-API-Keys.json        # nur lokal, NICHT committen
├── Config/
│   └── Comm-SCI-Config.json
└── Logs/
    ├── Audit/
    └── Chats/
```

---

## Abhängigkeiten (Kurzliste)

- `pywebview` — UI/WebView
- `requests` — HTTP (providerabhängig)
- `markdown` — Rendering (Fallback möglich)
- `bleach` — HTML-Sanitization
- `pytest` — Tests
- `google-genai` — Gemini-Client (falls genutzt)
- `cryptography` — optionale verschlüsselte Key-Ablage

> Hinweis: `pywebview` benötigt OS-spezifische WebView-Backends (macOS: WebKit/Cocoa; Windows: WebView2; Linux: GTK/QT – je nach Installation).

---

## Nutzung (kurz)

Häufige Befehle (aus Comm-SCI-Control):

- `Comm Start` / `Comm Stop`
- `Comm Help`
- `Comm State`
- `Profile Expert`
- `SCI on` / `SCI menu`
- `Strict on` / `Strict off`
- `Color on` (falls im Regelwerk aktiviert)

Wrapper-/UI-Aktionen:

- `QC Override` — öffnet eine Slider-UI für temporäre QC-Anpassungen (falls in Config/UI aktiviert)

Zustandsänderungen werden über **Status + Audit-Events** sichtbar gemacht; Chat-/Audit-Logs können exportiert werden.

---

## Bekannte Einschränkungen (ehrlich)

- **Provider-APIs ändern sich:** Der Wrapper braucht ggf. kleine Updates bei Endpoint-/SDK-Änderungen (wo möglich wird OpenAI-kompatible Semantik genutzt).
- **Kein Cloud-Log-Sharing:** Exporte sind bewusst lokal.
- **Governance verbessert Disziplin, nicht Wahrheit:** Transparenz und Reproduzierbarkeit steigen, faktische Fehler lassen sich damit nicht garantieren.

---

## Mitmachen (Contribution)

Issues und PRs sind willkommen, insbesondere für:
- Verbesserungen an Provider-Adaptern (bei Beibehaltung: „no network on UI-only actions“)
- Tests und Fixtures (pytest muss grün bleiben)
- Dokumentation und Minimalbeispiele

---

## Lizenz / Zitierung

- **Wrapper (Zenodo):** https://doi.org/10.5281/zenodo.18445673

- **Wrapper-Code:** Apache License 2.0 (siehe `LICENSE`).
- **Comm-SCI-Control (Regelwerk):** bitte eine archivierte Zenodo-Version zitieren (siehe DOI oben). Lizenz und Attribution für das Regelwerk werden vom Upstream-Projekt Comm-SCI-Control geregelt.