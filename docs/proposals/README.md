# Proposals (Backlog-Designs)

Dieser Ordner ist fuer geplante oder diskutierte Erweiterungen gedacht, die noch nicht zwingend implementiert werden.

## Ziel

- Ideen nachvollziehbar festhalten
- Entscheidungen vorbereiten, ohne Code sofort zu aendern
- Kontext fuer spaetere Umsetzung/Review erhalten

## Empfohlene Praxis (GitHub-nah)

1. Lege fuer jede groessere Idee ein Proposal-Dokument in diesem Ordner an.
2. Verlinke ein passendes GitHub-Issue im Proposal.
3. Nutze im Titel und im Dokument klaren Status:
   - `draft` (Idee in Arbeit)
   - `proposed` (reviewbereit)
   - `accepted` (zur Umsetzung freigegeben)
   - `deferred` (bewusst verschoben)
   - `rejected` (verworfen, mit Begruendung)
4. Wenn eine Entscheidung final ist, ueberfuehre die Kernaussage zusaetzlich in `docs/ADR-light.md`.
5. Halte Proposal-Dateien klein, testbar und mit klaren Trade-offs.

## Mini-Template

```md
# Proposal: <Titel>

- Status: draft|proposed|accepted|deferred|rejected
- Owner: <Name/Team>
- Issue: <Link oder TBD>
- Letzte Aktualisierung: YYYY-MM-DD

## Problem
...

## Ziel
...

## Optionen
1. ...
2. ...

## Entscheidungsvorschlag
...

## Auswirkungen/Risiken
...

## Umsetzungsentwurf
...
```
