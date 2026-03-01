# ADR-light

## 2026-03-01 — S15.1.1: Komplextest-Sicherung + Monitor-Abbruch

- Entscheidung:
  - `komplexttest` schreibt deterministische Export-Checkpoints vor `clear_chat`-Phasen und zum Laufende.
  - Das Manual-Test-Monitorfenster erhält einen aktiven Stop-Button, der den laufenden Test über `manual_test_stop` sauber abbricht.
  - Bei `STOPPED`/`ERROR` wird zusätzlich ein partieller Export erzeugt, bevor der Report gespeichert wird.
- Begründung:
  - Bei langen Matrixläufen gingen visuelle/verarbeitete Zustände durch Chat-Reset verloren.
  - Ein sicherer Abbruchpfad ist für lange Tests erforderlich, ohne Zwischenstände zu verlieren.
  - Die Lösung hält den Ablauf deterministisch und erhöht Diagnosequalität ohne Eingriff in Kern-Governance-Logik.
- Verworfene Alternativen:
  - Nur ManualTest-JSON ohne Export-Snapshot: unzureichend für visuelle/Rendering-Fehleranalyse.
  - Abbruch nur über Panel-Stop im Hauptfenster: im Monitor-Workflow nicht ergonomisch und fehleranfällig.
  - Harte Thread-/Window-Termination: erhöht Risiko für inkonsistente Runner-/UI-Zustände.
- Commitbezug:
  - Wird nach Commit ergänzt (`TBD` vor Commit).
