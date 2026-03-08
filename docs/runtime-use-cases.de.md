# Runtime Use Cases

## Wann JSON-only oft reicht

JSON-only-Nutzung im Chat reicht oft fuer:

- erste Konzepttests,
- leichte Explorations-Workflows,
- didaktische Demonstration der Kommando-Logik.

## Wann die Wrapper-Runtime klar im Vorteil ist

Der Wrapper ist klar vorzuziehen bei:

- reproduzierbaren Laeufen,
- strengeren Command-/State-Contracts,
- Diagnostik und Audit-Trails,
- Modellvergleich unter stabiler Ausfuehrungslogik.

## Typische Szenarien

1. Wiederholte Benchmark-Prompts ueber mehrere Modelle.
2. Lange Sitzungen mit sichtbar zu haltendem Drift-Risiko.
3. QA-Pruefungen vor Releases.
4. Lehr-/Demo-Sessions mit transparenter Runtime-Kontrolle.

## Kernunterscheidung

- Regelwerk: wie Antworten sein sollen.
- Wrapper-Runtime: wie dieses Verhalten erzwungen und beobachtet wird.
