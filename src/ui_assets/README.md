# UI Assets (S7)

Dieses Verzeichnis ist fuer die schrittweise Auslagerung der eingebetteten UI-Assets aus
`src/Comm-SCI-Control-App.py` vorgesehen.

S7-Regel:
- Externe Dateien sind optional.
- Wenn eine Datei fehlt oder nicht lesbar ist, faellt der Wrapper deterministisch auf den
  eingebetteten HTML/CSS/JS-String zurueck (fail-open, keine Verhaltensaenderung).

Geplante Dateien (schrittweise):
- `panel.html`
- `panel_manual_test_runner.js`
- `qc_override.html`
- spaeter ggf. `chat.html`

Hinweis:
Der Wrapper ersetzt weiterhin dynamische Platzhalter (z. B. Wrapper-Name) nach dem Laden,
so dass eingebettete und externe Assets gleich behandelt werden.
