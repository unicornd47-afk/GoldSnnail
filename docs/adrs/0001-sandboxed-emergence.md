# ADR-0001: Sandboxed Emergence

**Status:** Accepted  
**Date:** 2026-08-09  
**Deciders:** Lead Architecture Team

## Context

Klassische SNN-Engines und sicherheitskritische Systeme operieren nach dem _Zero-Trust_-Paradigma: Jede Abweichung von erwarteten numerischen Bereichen wird als Fehler gewertet, der zu Panics, Exceptions oder harten Abbruchbedingungen führt. Dieses Modell ist für deterministische, eng begrenzte Aufgaben geeignet, aber es ist fundamental ungeeignet für Systeme, die _Emergenz_ und _kritisches Verhalten_ erforschen sollen.

Ein neuronales Netzwerk, das am "Edge of Chaos" operiert, muss notwendigerweise mit:

- **Numerischen Ausreißern** umgehen (Membranpotentiale, die spontan oszillieren)
- **Singularitäten** in geometrischen Transformationen tolerieren (Poincaré-Ball-Grenzen)
- **Stochastischem Rauschen** als strukturellem Feature (nicht als Bug) akzeptieren

Das Werfen von Errors bei diesen Bedingungen zerstört die kontinuierliche Dynamik, die für Emergenz erforderlich ist. Ein System, das bei jeder kritischen Situation crasht, kann keine kritischen Zustände erreichen.

## Decision

Wir ersetzen _Zero Trust_ durch **Sandboxed Emergence**:

> **"Bending, not Breaking"**

Das System umschließt alle numerischen Operationen mit elastischen Begrenzungsfunktionen. Statt Panics oder Errors zu werfen, werden Werte über asymptotische Funktionen (z. B. `tanh`, `atanh`, Softmax mit Temperatur) weich in den gültigen Bereich skaliert. Die Validierung erfolgt nicht mehr als _Gatekeeping_ an Layer-Grenzen, sondern als _passive Telemetrie_ im Hintergrund.

Konkrete Regeln:

1. **Keine `unwrap()` in Kernpfaden.** Alle Operationen auf Membranpotentialen, Gewichten und geometrischen Zuständen nutzen `soft_clamp`- oder `soft_normalize`-Funktionen.
2. **Keine `Result`-Propagierung für mathematische Ausreißer.** Statt `Err(OutOfBounds)` wird der Wert asymptotisch begrenzt und ein Telemetrie-Event geloggt.
3. **Passive Observer statt aktiver Guards.** Die Telemetrie-Schicht zeichnet Verteilungen (Power-Law, Avalanches) auf, ohne den Datenfluss zu blockieren.
4. **Rausch-Injektion als Strukturelement.** QLIF-Implementationen injizieren kontrolliertes Rauschen (`noise_std`) in jedes Zeitstep, um das Netzwerk aus dem Gleichgewichtszustand zu halten.

## Consequences

### Positive

- **Robustheit:** Das System überlebt numerische Extremzustände ohne Crash.
- **Emergenz-Fähigkeit:** Kritische Zustände und Avalanchen können sich frei entfalten.
- **CUDA-Kompatibilität:** Flache Arrays ohne verschachtelte Fehlerbehandlung lassen sich direkt in GPU-Kernels übertragen.
- **Wartbarkeit:** Keine verschachtelten `match`-Bäume für Fehlerbehandlung in mathematischen Kernroutinen.

### Negative

- **Schwierigere Debugging:** Fehler sind nicht mehr sofort als Panics sichtbar. Sie müssen über Telemetrie-Dashboards erkannt werden.
- **Laufzeit-Overhead:** Soft-Clamp-Operationen addieren minimale Latenz (vernachlässigbar im Vergleich zu GPU-Transfer).
- **Risiko der "Silent Corruption":** Wenn Telemetrie deaktiviert ist, können numerische Driften unbemerkt bleiben. Dies wird durch automatische Telemetrie-Pflicht in allen `step()`-Aufrufen adressiert.

### Neutral

- Das Architektur-Team muss sich auf elastische Grenzwerte einigen (z. B. `PoincaréSoftLimit`, `MembraneSoftLimit`). Diese sind in `src/geometry/poincare.rs` zentral definiert.

## References

- [Poincaré Ball Model - Elastic Boundaries](../math/poincare.md)
- [Telemetry Observer Pattern](../architecture/telemetry.md)
- [Substrate Memory Layout](../architecture/substrate.md)

