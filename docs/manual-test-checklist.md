# Comm-SCI-Control-App: Manuelle Test-Checkliste

Repo (manuelle Tests):
`/pfad/zum/repo`

## Ziel
Reproduzierbarer GUI-Testlauf fuer Release-/Aenderungspruefungen mit Fokus auf:
- Providerwechsel (Gemini / Hugging Face / OpenRouter)
- Antwortformatierung (Color, SCI, Self-Debunking)
- QC-Matrix / Footer / Export / Audit

## Verbindliche Entwicklungsregel (ACTUAL-TEST)
- Nach jedem relevanten Code-Patch muss der in-App `ACTUAL-TEST`-Flow im UI-Szenario mitgepflegt werden.
- Neue oder geaenderte Logikpfade muessen als konkrete Checks im ACTUAL-TEST landen (nicht nur als externer Einmal-Report).
- Bei Aenderungen am Panel-Szenario immer beide Quellen synchron halten:
  - `src/ui_assets/panel.html`
  - Embedded-Mirror in `src/Comm-SCI-Control-App.py`

## Vorbereitung

```bash
cd /pfad/zum/repo
./scripts/run_local_tests.sh
python Comm-SCI-Control-App.py
```

Pruefen vor Teststart:
- Fenster startet ohne Crash
- Titel ist `Comm-SCI-Control-App`
- Ruleset `Comm-SCI-v20.0.3.json` wurde geladen
- Panel reagiert (Buttons/Dropdowns klickbar)

## Test A: Basisantwort (ohne SCI)

Einstellungen:
- Profil: `Standard`
- SCI: `off`
- Color: `on`
- QC: `on`

Frage:
- `Was ist Zeit?`

Pruefen:
- Antwort kommt ohne Fehlermeldung
- Kein doppelter `Self-Debunking`-Block
- `QC-Matrix` im Footer vorhanden und vollstaendig (6 Werte)
- Keine abgeschnittene QC-Zeile
- Bei Color on: Marker/Farbicons nur, wenn das Modell sie liefert

## Test B: Providerwechsel (Pfadtest)

Reihenfolge:
1. Gemini -> 1 Frage
2. Hugging Face -> 2-3 Fragen
3. OpenRouter -> 1 Frage
4. Hugging Face -> 1 Frage
5. Gemini -> 1 Frage

Pruefen pro Wechsel:
- Modellliste aktualisiert sich passend zum Provider
- Antwort kommt vom gewaehlten Provider
- Keine falschen Provider-Fehlertexte (z. B. HF-Fehler als "OpenRouter")
- Keine unerwarteten Session-/Reconnect-Fehler

## Test C: SCI-Modi (Formatierung)

Mindestens testen:
- SCI A
- SCI B

Optional (Release-Kandidaten):
- SCI C-H je 1 Kurzfrage

Frage (gleich halten):
- `Was ist Zeit?`

Pruefen:
- SCI-Trace als separater Block (nicht mitten im Fliesstext)
- Schritte sauber formatiert (Nummerierung/Labels)
- Self-Debunking korrekt formatiert und nummeriert
- `QC-Matrix` bleibt im Footer vollstaendig
- Keine Markdown-Leaks wie `##`, `###`, `####` in der Anzeige

## Test D: QC-Override

Einstellungen:
- Profil: `Expert`
- SCI: `B`
- QC-Override setzen (z. B. `Brevity=0`, Rest hoch)

Frage:
- `Was ist Zeit?`

Pruefen:
- Antwort wird nicht ungueltig kurz (modellabhaengig, nicht absolut)
- Override-Werte erscheinen korrekt in der `QC-Matrix`
- Footer bleibt vollstaendig
- Self-Debunking bleibt sauber (keine Doppelnummerierung)

## Test E: Fehlerfall-Check (wenn reproduzierbar)

Ziel:
- Ein echter Providerfehler (z. B. Credits/Rate-Limit) soll korrekt benannt werden

Pruefen:
- HF-Fehler zeigt `Hugging Face` (nicht faelschlich `OpenRouter`)
- OpenRouter-Fehler zeigt `OpenRouter`
- Meldung ist verstaendlich und nicht als roher JSON-Fehler sichtbar

## Test F: Export / Logs / Audit

Aktion:
- Export ausloesen

Pruefen:
- Chat-Export erstellt unter `Logs/Chats/`
- Audit-Export erstellt unter `Logs/Audit/`
- `AuditStream_YYYYMMDD.jsonl` wird geschrieben/erweitert
- Keine Crashs beim Export

## Akzeptanzkriterien (Release-tauglich)

Muss erfuellt sein:
- Keine GUI-Abstuerze
- Providerwechsel stabil
- QC-Matrix immer vollstaendig bei Inhaltsantworten
- Kein doppeltes Self-Debunking
- SCI-Formatierung lesbar und stabil
- Fehlertexte mit korrektem Providernamen

## Testprotokoll (zum Ausfuellen)

- Datum/Uhrzeit:
- Getestete Version/Commit:
- Providerfolge getestet:
- SCI-Modi getestet:
- QC-Override getestet (ja/nein):
- Auffaelligkeiten:
- Reproduzierbar (ja/nein):
- Verwendete Logs:
  - Chat:
  - Audit:
  - AuditStream:
