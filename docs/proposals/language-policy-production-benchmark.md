# Proposal: Language Policy Mode (`production` / `benchmark`)

- Status: proposed
- Owner: Wrapper Maintainer
- Issue: TBD
- Letzte Aktualisierung: 2026-03-06

## Kontext

Im `Comm State` wird derzeit u. a. diese Zeile gezeigt:

`Language policy: production`

Das ist bereits in Runtime und Config verdrahtet, aber fuer Nutzer noch nicht als klarer, erklaerter Bedieneintrag im Panel sichtbar.

## a) Zweck

Der Zweck der Language-Policy ist die Trennung von zwei Betriebsmodi fuer Sprachkonformitaet:

- `production`:
  Sprachregeln werden strikt erzwungen. Sprachverstosse gelten als harte Vertragsverletzungen.
- `benchmark`:
  Sprachverstosse werden nur protokolliert/warnend behandelt (soft), um Modellverhalten ohne harte Eingriffe zu beobachten.

Damit wird ein klassischer Zielkonflikt aufgeloest:

- maximale Konsistenz in der Produktivnutzung
- maximale Messbarkeit in Evaluation/Benchmarking

## b) States und ihre Bedeutung

### State 1: `production`

- Bedeutung:
  Der Wrapper behandelt Sprachverletzungen als hard violations.
- Wirkung fuer Nutzer:
  Hohe Konsistenz und niedrige Toleranz fuer Sprachdrift.
- Geeignet fuer:
  Alltag, Lehre, reproduzierbare Outputs, dokumentationspflichtige Nutzung.

### State 2: `benchmark`

- Bedeutung:
  Sprachverletzungen werden als soft violations bewertet.
- Wirkung fuer Nutzer:
  Mehr Rohsignal aus dem Modell, weniger harte Korrektureingriffe.
- Geeignet fuer:
  Modellvergleich, Prompt-Evaluation, Diagnose von Sprachproblemen.

### Fallback/Normierung

- Ungueltige Werte werden auf `production` normiert (fail-safe).

## c) Moegliche Vorteile einer aktiven Aenderung per nachgeruestetem Button im Wrapper

Ein sichtbarer Schalter (z. B. Dropdown im Panel) bringt unmittelbaren Mehrwert:

1. Transparenz
- Nutzer sehen den aktiven Modus explizit und muessen ihn nicht aus Config-Dateien ableiten.

2. Schnellere Fehlersuche
- Bei Sprachproblemen kann direkt auf `benchmark` umgeschaltet werden, um zu sehen, ob die Policy oder das Modell die Ursache ist.

3. Bessere Vergleichbarkeit
- A/B-Tests zwischen Providern/Modellen werden sauberer, wenn der Sprachmodus kontrolliert variiert werden kann.

4. Geringere Bedienkosten
- Kein manuelles Editieren von `Config/Comm-SCI-Config.json` noetig.

5. Sauberes Governance-Signal
- Der Modus wird im State/Audit klar sichtbar und ist damit besser nachvollziehbar.

## d) Moegliche Realisierung

## D1. UX-Entwurf

- Position:
  Panel, im Provider/Model-Bereich oder im Runtime-&-Governance-Bereich.
- Widget:
  Dropdown `Language policy` mit Optionen:
  - `production (strict)`
  - `benchmark (log-only)`
- Verhalten:
  - On change: sofort persistieren
  - danach `buildUI()` refresh
  - Systemhinweis in Main-Chat: `Language policy mode set to: <mode>.`

## D2. Technische Anbindung (ist groesstenteils schon vorhanden)

Bereits vorhanden:

- `set_language_policy_mode(...)` im App-Code
- `get_language_policy_mode(...)` in Config/Runtime
- `panel_action('set_language_policy_mode', {mode: ...})` in `ui_controller.py`
- Darstellung im `Comm State`

Fehlend fuer Endnutzer:

- sichtbares Panel-Control mit Handler

## D3. Minimaler Implementierungsplan

1. Panel-HTML erweitern
- Neues Select-Feld `id="langPolicyMode"` plus Label.

2. UI-Bindung in `buildUIFromData(...)`
- `data.language_policy_mode` einlesen und Select setzen.

3. Event-Handler
- `changeLanguagePolicyMode()` implementieren:
  - call: `panel_action('set_language_policy_mode', {mode})`
  - danach `buildUI()`

4. Tests
- Panel-Template-Tests fuer neues Feld/Handler
- `panel_action` Contract-Test fuer `set_language_policy_mode`
- Zustandstest: `get_ui()` liefert den gesetzten Modus

5. Optional
- Tooltips mit kurzer Erklaerung (`strict` vs `log-only`)
- kleine Warnung bei Wechsel auf `benchmark`

## D4. Risiken und Gegenmassnahmen

- Risiko: Nutzer schalten versehentlich auf `benchmark` und wundern sich ueber weniger strikte Sprachkontrolle.
  - Gegenmassnahme: klarer Tooltip + sichtbarer State in `Comm State`.

- Risiko: Inkonsistenz zwischen Runtime-State und persistierter Config.
  - Gegenmassnahme: immer beide setzen und nach dem Setzen `buildUI()` erzwingen.

## Professionelle Ablage solcher Erweiterungsplaene auf GitHub

Typischer, professioneller Ablauf:

1. GitHub Issue anlegen
- Problem, Ziel, Akzeptanzkriterien, offene Fragen.
- Labels, z. B. `proposal`, `governance`, `ui`, `needs-decision`.

2. Proposal im Repo pflegen
- Datei in `docs/proposals/` mit Status (`draft/proposed/accepted/...`).
- Issue-Link im Kopf der Datei.

3. Architekturentscheidung trennen
- Nach finaler Entscheidung kompakten ADR-Eintrag in `docs/ADR-light.md`.

4. Umsetzung separat steuern
- Nur bei `accepted` in Milestone/Project Board aufnehmen.

Damit bleiben auch nicht direkt umgesetzte Ideen auffindbar, begruendet und teamfaehig.

## Empfehlung fuer dieses Repo

- Diese Notiz als Referenz belassen.
- Bei Bedarf ein kleines Folge-PR nur fuer den Panel-Schalter erstellen.
- Danach ADR-light-Eintrag mit finaler Entscheidung ergaenzen.
