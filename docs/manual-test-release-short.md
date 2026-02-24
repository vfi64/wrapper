# Comm-SCI-Control-App: Manueller Release-Kurztest

Repo:
`/Users/hof/Dropbox/Privat/GitHub/Comm-SCI-Control-private`

## Schnellablauf (5-10 Minuten)

1. Start
- App startet ohne Crash
- Ruleset `Comm-SCI-v20.0.3.json` geladen

2. Gemini (1 Frage)
- `Was ist Zeit?`
- QC-Matrix vollstaendig

3. Hugging Face (2 Fragen)
- Providerwechsel korrekt
- Keine falsche "OpenRouter"-Fehlermeldung
- Self-Debunking sauber

4. OpenRouter (1 Frage)
- Antwort oder sauberer Fehlertext
- Providername im Fehlertext korrekt

5. SCI B (1 Frage, ideal mit `Expert`)
- SCI-Trace sauber formatiert
- Self-Debunking korrekt
- QC-Matrix vollstaendig

6. QC-Override (Brevity=0) + 1 Frage
- QC-Matrix zeigt Override-Werte korrekt
- Antwort nicht offensichtlich entgleist (zu kurz/kaputt formatiert)

7. Export
- Chat- und Audit-Export funktionieren

## Abbruchkriterien (Release stoppen)

- GUI-Crash / Freeze
- Providerwechsel liefert falschen Providerpfad
- QC-Matrix fehlt wiederholt
- Doppelte Self-Debunking-Bloecke
- SCI-Trace-Formatierung bricht sichtbar
