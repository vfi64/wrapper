# Comm-SCI-Control Wrapper Handbuch (DE)

Versionsbasis: Wrapper `1.0.0`  
Default-Regelwerk zur Laufzeit: `JSON/Comm-SCI-v20.2.0.json`

## 1. Zweck

Dieses Repository enthaelt die Runtime-Implementierung der Comm-SCI-Governance.
Das oeffentliche Regelwerk definiert die Normen, der Wrapper setzt diese operativ durch:

- deterministisches Command-Routing
- Runtime-State-Transitions
- Output-Contract-Checks mit Repair
- Audit/Export-Funktionen
- UI/Panel-Integration fuer den Betrieb

Der Wrapper ist bewusst strikt bei Command-Tokens und fail-soft bei optionalen Modulen.

## 2. Beziehung der Repositories

- Oeffentliches Regelwerk und Konzepte: [vfi64/Comm-SCI-Control](https://github.com/vfi64/Comm-SCI-Control)
- Runtime-Implementierung: [vfi64/wrapper](https://github.com/vfi64/wrapper)
- Dieses Repository: Staging- und Integrationslinie fuer Wrapper-Weiterentwicklung

Praxisregel:
JSON ist die Governance-Quelle, Wrapper ist die Ausfuehrungsquelle.

## 3. Architektur-Ueberblick

Zentrale Implementierungsdateien:

- `src/Comm-SCI-Control-App.py`: monolithische Runtime-Orchestrierung
- `src/app_bootstrap.py`: Start-/Bootstrap- und Window-Lifecycle-Guards
- `src/provider_service.py`: Provider- und Modellsteuerung
- `src/governance_service.py`: State-/Governance-Helfer
- `src/ui_assets/`: Panel-/Chat-HTML-Assets
- `src/Module/`: optionale Module (`auditstream`, `rendering_utils`, `compliance_scan`)

Testfokus:

- `tests/test_app.py`: Verhaltensvertraege und Regressionen
- `tests/test_app_bootstrap.py`: Bootstrap-Lifecycle und Panel-Readiness

## 4. Runtime-Ausfuehrungsmodell

Der operative Ablauf folgt deterministischen Stufen:

1. Input parsen und in Command vs Content klassifizieren.
2. Command-Intent routen und Governance-State aktualisieren.
3. Governed Prompt bauen (bei Content-Route).
4. Provider-Call ausfuehren.
5. Contracts durchsetzen (SCI Trace, Self-Debunking, QC-Footer, Verification Gates).
6. Deterministisches HTML und Session-Metadaten rendern.

Wichtig:
UI-Kommandos wie `Comm Help`, `Comm State`, `Comm Config`, `Comm Anchor` und `Comm Audit` sollen lokal deterministisch gerendert werden und keine Model-Calls benoetigen.

## 5. Command-Disziplin

Command-Parsing ist absichtlich standalone-only.
Erwartetes Verhalten:

- exakte Command-Token-Zeile
- kein Mix aus Command + Inhaltsfrage in einer Zeile
- SCI-Buchstabe `A-H` gilt nur waehrend SCI-Pending als Variantenwahl

Damit werden unbeabsichtigte Moduswechsel reduziert und Automation reproduzierbar.

## 6. SCI-Governance

SCI-Modus aktiviert trace-erzwungene Antwortstruktur.

- `SCI on` aktiviert die Variantenwahl.
- Buchstabe `A-H` waehlt das Trace-Template.
- `SCI recurse` erweitert genau einen Turn bei aktivem SCI.
- `SCI off` schaltet zur nicht-tracebasierten Ausgabe zurueck.

Variantenmetadaten (Name/Fokus/Mapping/Steps) kommen aus dem geladenen JSON und werden in Help/UI angezeigt.

## 7. QC, CGI und Control Layer

### QC

QC-Footer-Werte werden normalisiert und gegen Profilziele geprueft.
Overrides sind erlaubt, muessen aber sichtbar bleiben (`Comm State` / Footer-Metadaten).

### CGI

CGI verarbeitet Nutzer-Feedback-Signale und kann Anpassungslogik beeinflussen.
CGI-Inputs sollen lokal abgefangen werden, wo es vertraglich vorgesehen ist.

### Control Layer

Der Control Layer erzwingt:

- Output-Struktur und Reihenfolge
- erforderliche/verbotene Bloecke je Route
- deterministisches Fallback-Verhalten

## 8. Language Policy: `production` vs `benchmark`

Die Runtime fuehrt `language_policy_mode` als expliziten Zustand.

- `production`: Nutzerbetrieb, konservative und stabile Defaults
- `benchmark`: Evaluationsmodus fuer kontrollierte Vergleiche und Regressionen

Nutzen:

- klare Trennung zwischen Tagesbetrieb und Messbetrieb
- explizite Dokumentation des Experimentkontexts
- weniger Mehrdeutigkeit bei Zeitvergleichen ueber Modelle und Versionen

## 9. API-Key-Modi und Passphrase-Flow

Der Wrapper unterstuetzt je Provider zwei lokale Speicherarten in `Config/Comm-SCI-API-Keys.json`:

- `api_key_plain`: unverschluesselt gespeichert, kein Passphrase-Dialog noetig
- `api_key_enc`: verschluesselt gespeichert, Entsperrung per Passphrase-Dialog

Laufzeitverhalten:

- Beim Programmstart fragt der Wrapper eine Passphrase ab, wenn der initiale Provider einen `api_key_enc` nutzt.
- Beim Provider-Wechsel erscheint der Passphrase-Dialog nur dann, wenn der Ziel-Provider verschluesselt gespeichert ist.
- Verschiedene Provider koennen gemischt betrieben werden (z. B. Gemini verschluesselt, OpenRouter unverschluesselt).
- API-Keys koennen im Panel unter `Provider & LLM` ueber den Button `API-Key` eingegeben, aktualisiert oder geloescht werden.

## 10. Operativer Ablauf

Empfohlene Basissequenz:

1. Regelwerk laden und mit `Comm State` pruefen.
2. `Comm Start`
3. Profil waehlen (`Profile Expert`, `Profile Standard`, ...).
4. Optional: `SCI on` + Variantenbuchstabe.
5. Inhaltsfrage stellen.
6. Bei langen Sitzungen regelmaessig `Comm Anchor` und `Comm Audit`.

Bei Instabilitaet:

- zuerst Command-Format und State pruefen
- danach Provider/Modell/API-Key im Panel validieren
- vor Weiterarbeit re-anchoren und auditieren
- Eingabe-History wird beim Beenden automatisch nach `Logs/History/InputLineHistory.json` gespeichert und beim Start geladen.

## 11. Dokumentations- und Webstruktur

Projektnahe Dokumentation liegt unter `docs/`:

- `docs/index.html` / `docs/index.de.html`: zweisprachige Landingpages
- `docs/glossary.html` / `docs/glossar.de.html`: didaktische Fachbegriffserklaerungen inkl. SCI A-H Beispiele
- `docs/install-beginner.html` / `docs/install-beginner.de.html`: gefuehrte Installation fuer nicht technische Nutzer
- `docs/install-pro.html` / `docs/install-pro.de.html`: vertiefte Installation fuer reproduzierbaren Profi-Betrieb
- `docs/HANDBOOK.md` / `docs/HANDBOOK.de.md`: operatives Handbuch
- `docs/manual-*.md`: Release-/Manual-Test-Checklisten
- `docs/proposals/`: Design-Backlog und Erweiterungsideen

## 12. Proposal- und Planungs-Praxis

Professionelle GitHub-Praxis fuer Ideen, die noch nicht umgesetzt sind:

1. Leanes Proposal in `docs/proposals/` anlegen.
2. Zugehoeriges Issue verlinken.
3. Status klar fuehren (`draft`, `proposed`, `accepted`, `deferred`, `rejected`).
4. Bei Annahme die dauerhafte Architekturentscheidung in `docs/ADR-light.md` uebernehmen.

So bleiben Implementierung und Strategie sauber getrennt.

## 13. DOI und Zitation

Aktuelle DOI-Referenzen:

- Wrapper Concept DOI: [10.5281/zenodo.18445672](https://doi.org/10.5281/zenodo.18445672)
- Wrapper Release DOI: [10.5281/zenodo.18759479](https://doi.org/10.5281/zenodo.18759479)
- Regelwerk Concept DOI: [10.5281/zenodo.17928357](https://doi.org/10.5281/zenodo.17928357)
- Regelwerk Release DOI: [10.5281/zenodo.18154098](https://doi.org/10.5281/zenodo.18154098)

Primadaten fuer Zitationen liegen in `CITATION.cff`.
