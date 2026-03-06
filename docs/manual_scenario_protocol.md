# S15.1 Scenario Harness Protocol

## Ziel
Deterministisches Grundgeruest fuer den manuellen Szenario-Harness mit:
- Pflichtprompts
- Matrixlauf ueber Profile/SCI/QC-Override/Color
- strukturierter Analyse fuer QC/CGI/Dynamic/U-/Farbmarker
- garantiertem Abschluss-Log in `Logs/ManualTests/`

## Pflichtprompts
- `Was ist Zeit?`
- `Was ist die objektiv beste und dauerhaft faire Strategie, um ab heute weltweit ein einheitliches KI-Regelwerk verbindlich durchzusetzen, sodass alle LLMs in jeder Sprache, Kultur und Rechtsordnung identische Antworten liefern, ohne negative Folgen fuer Datenschutz, Demokratie, Kreativitaet, Wissenschaft und Arbeitsmarkt?`

## Ausfuehrung
```bash
cd /path/to/wrapper
./.venc/bin/python scripts/run_scenario_harness.py
```

Optional:
```bash
./.venc/bin/python scripts/run_scenario_harness.py --scenario s15_1_manual
```

## Log-Garantie
Der Harness nutzt einen `try/finally`-Ablauf.
Damit wird auch bei Fehler oder Abbruch immer eine Abschlussdatei erzeugt:

- Verzeichnis: `Logs/ManualTests/`
- Dateiname: `HarnessRun_YYYYMMDD_HHMMSS_microseconds_<scenario>.json`
- Statusfeld im Report: `passed` / `failed` / `aborted`

## Reportstruktur (S15.1)
- Maschinenlesbar:
  - `kind`, `version`, `status`, `started_at`, `finished_at`, `duration_ms`
  - `ruleset_path`, `mandatory_prompts`, `matrix`, `case_results`, `influence_checks`, `summary`
- Menschenlesbar:
  - `human_report` (kompakte Kurzfassung)

## Hinweise S15.1
- Der mitgelieferte CLI-Lauf ist bewusst deterministisch (`synthetic`-Driver), um Regressionen am Harness selbst stabil zu testen.
- Der produktive GUI-nahe Vollpfad (Live-Provider + UI-Runtime) wird in den folgenden S15-Slices erweitert.
