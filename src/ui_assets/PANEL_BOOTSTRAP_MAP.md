# Panel Bootstrap Map (S7)

Ziel: `panel.html` sicher auslagern, ohne die pywebview-Bridge oder das dynamische Rebuild zu brechen.

## Sichtbare Symptomatik beim fehlgeschlagenen S7-Panel-Schritt

- Statische obere Panel-Elemente sichtbar (`Provider`, `Model`, `Answer language`, `Logs`, `Manual Test`)
- Dynamische Sektionen fehlen:
  - `Comm Core`
  - `Profiles`
  - `SCI Workflow`
  - `Modes & Overlays`
  - `Tools` (inkl. `Dynamic one-shot on`)
- Buttons/Controls sichtbar, aber funktional wirkungslos
- Status blieb bei `Panel boot…` / offline

Das spricht fuer einen Bootstrap-/Bridge-Ausfall, nicht fuer einzelne Action-Handler.

## Bootstrap-Reihenfolge (panel.js in HTML_PANEL)

1. `DOMContentLoaded` -> `initPanelFailOpen()`
2. `buildUIFromData(_fallbackData())`
   - rendert nur statischen Offline-Zustand
   - `comm/profiles/sci/overlays/tools` sind in Fallback absichtlich leer
3. `_tryBringOnline()`
   - `_api()` muss Bridge liefern
   - `ping`
   - `buildUI()` -> `get_ui`
4. `buildUIFromData(data_from_python)`
   - rendert dynamische Sektionen aus `data.comm`, `data.profiles`, `data.sci`, `data.overlays`, `data.tools`

## Kritische JS-Funktionen (Panel)

- `_api()`
- `_apiCall(name, args, timeoutMs)`
- `buildUIFromData(raw)`
- `buildUI()`
- `initPanelFailOpen()`
- `_startRetryLoop()`
- `_tryBringOnline()`
- `run(cmd)` (Comm/Profile/SCI/Modes/Tools Buttons)
- `changeProvider()`
- `changeModel()`
- `changeAnswerLanguage()`
- `refreshModels()`
- `refreshLogList()`
- `loadSelectedLog()`
- `clearChat()`
- `startManualTestRunner()`
- `stopManualTestRunner()`

## Python-Bridge-Abhaengigkeiten

### `get_ui`
Liefert Daten fuer `buildUIFromData(...)`, insbesondere:
- `logs`
- `providers`
- `current_provider`
- `available_models`
- `current_model`
- `answer_language`
- `chat_logs`
- `chat_log_selected`
- `comm`
- `profiles`
- `sci`
- `overlays`
- `tools`

Wenn `get_ui` nicht kommt oder JS es nicht aufruft, bleiben die dynamischen Button-Sektionen leer.

### `panel_action`
Wird genutzt fuer:
- `cmd`
- `set_provider`
- `refresh_models`
- `set_model`
- `set_answer_language`
- `list_chat_logs`
- `load_chat_log`
- `clear_chat`
- `manual_test_monitor_*`
- `save_manual_test_report`
- `qc_override_apply` / `qc_override_clear` (teils direkt, teils Fallback)

### Weitere Bridge-Methoden
- `ping`
- `qc_get_state`
- `export` (optional; Manual-Test kann Warnung ausgeben, wenn nicht vorhanden)
- `ask` (optional; Manual-Test nutzt Fallback auf `panel_action('ask', ...)`)

## Historisch gewachsene Funktionsblöcke (plausible Reihenfolge)

1. Comm/Profile/SCI/Modes/Color
2. Provider-/LLM-Verwaltung
3. QC-Override / Dynamic one-shot
4. Manual Test Runner + Monitor

Konsequenz:
- Panel-HTML/JS und Python-Handler sind im Monolithen an mehreren Stellen erweitert worden
- Auslagerung erfordert strikt gleiche Bootstrap-/Bridge-Semantik

## S7-Naechster sicherer Ansatz fuer `panel.html` (umgesetzt)

1. Externe Datei statisch pruefen (Marker-/Struktur-Selbsttest)
2. Externes `panel.html` als aktive Quelle laden (nur bei bestandenem Static-Selftest)
3. Runtime-Selbsttest vor erstem sichtbaren `show()`:
   - `ping` erfolgreich
   - `buildUI()` erfolgreich
   - DOM-Kernmarker vorhanden
   - `data.comm/profiles/...` nicht leer (falls ruleset geladen)
4. Bei Fehler/Timeout:
   - automatische Rueckfallebene auf eingebettetes `HTML_PANEL` **vor dem Anzeigen**
   - Log-Hinweis in Session-Events (`panel_bootstrap`)

## Aktueller S7-Stand

- `qc_override.html` extern aktiv (mit Fallback)
- `manual_test_monitor.html` extern aktiv (mit Fallback)
- `chat_template.html` extern aktiv (mit Fallback)
- `panel.html` extern aktiv, aber nur nach Static-Selftest + Runtime-Selbsttest; bei Fehler/Timeout automatischer Fallback auf eingebettetes `HTML_PANEL`
- Verbleibendes S7-Risiko: manuelle GUI-Validierung auf pywebview-Backend (Mac) fuer echtes Fensterverhalten/Timing
