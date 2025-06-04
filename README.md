# Projektübersicht

Dieses Repository enthält den vollständigen Quellcode zur datenbasierten Analyse der Hilfsfristen des Rettungsdienstes der Stadt Hagen. 
Die enthaltenen Skripte decken sämtliche Schritte von der Datenvorverarbeitung bis zur Visualisierung und Modellierung ab. 
Ziel des Projekts ist es, auf Grundlage historischer Einsatz-, Wetter- und Geodaten Prognosemodelle für Hilfsfristen zu erstellen und räumliche Analysen mithilfe von Isochronen durchzuführen. 
Der Code bietet eine modulare Struktur, sodass einzelne Komponenten (bspw. Dateneinleseprozesse, Explorative Datenanalyse oder Modelltraining) unabhängig voneinander nachvollzogen und erweitert werden können.

## Datenquellen und Vorbereitung

Die Hauptdatengrundlage bilden zwei Exportdateien aus der Leitstellensoftware des Rettungsdienstes für die Jahre 2018–2022 und für 2023. 
Diese Dateien enthalten Zeitstempel für Meldungseingang, Alarmierung, Statusmeldungen von Rettungsfahrzeugen sowie Koordinaten der Einsatzorte. 
Die Daten sind nicht enthalten.
Wetterdaten werden automatisch aus ERA5-Land im GRIB-Format geladen und auf stündliche Werte transformiert, wobei nur Variablen genutzt werden, die einen plausiblen Einfluss auf Fahrzeiten haben. 
Die Zuordnung von Feiertags- und Ferieninformationen nutzt öffentlich verfügbare Daten für Nordrhein-Westfalen.
Geodaten der Stadt Hagen werden automatisch heruntergeladen.

## Installation und Abhängigkeiten

Für alle Analyseschritte wird Python 3.10 oder höher empfohlen. Die benötigten Python-Bibliotheken sind in requirements.txt aufgelistet. Eine typische Installation verläuft über

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Hinweise zum Datenschutz und zu Ergebnissen

Die Analysen dieses Projekts basieren auf anonymisierten Einsatzdaten und öffentlich zugänglichen Wetter- und Geodaten. 
Im Zuge der Veröffentlichung auf GitHub werden keine numerischen Ergebnisse, statistischen Kennzahlen oder Schlussfolgerungen veröffentlicht, die Rückschlüsse auf einzelne Stadtteile oder Versorgungssituationen zulassen. 
Ziel des Repositories ist allein die Bereitstellung der methodischen Ansätze und des Analyse- und Modellierungscodes. Bitte beachten Sie, dass alle Dateien mit sensiblen Daten (z. B. Originalauszüge aus dem Leitstellensystem) stets lokal bleiben und nicht in dieses Repository hochgeladen werden dürfen.

## Lizenz

Dieses Projekt steht unter der MIT-Lizenz. Details finden sich in der Datei LICENSE. Nutzerinnen und Nutzer sind eingeladen, den Code zum eigenen Gebrauch zu klonen oder abzuwandeln, solange die Lizenzbedingungen eingehalten werden.
