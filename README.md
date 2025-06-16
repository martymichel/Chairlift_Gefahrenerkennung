# YOLO Dual Model Video Annotator - Sturzerkennung Skilift

Eine Python-Anwendung zur Demonstration der Tauglichkeit von KI für die Sturzerkennung an Skiliften mittels Video-Annotation mit YOLO-Modellen und Field of Interest (FOI) Überwachung.

## Features

- **Dual-Model-Architektur**: Kombination aus YOLO Detection und YOLO Pose Estimation
- **Mehrklassen-Pose-Detection**: Möglichkeit, mehrere Klassen für Pose-Detection zu definieren
- **Field of Interest (FOI)**: Interaktives Viereck zur Überwachung spezifischer Bereiche
- **Intelligente Skilift-Steuerung**: Automatische Verlangsamung/Stopp bei Gefahrensituationen
- **Echtzeit-Personenzählung**: Zählung von Personen im definierten FOI-Bereich
- **Endlos-Video-Wiedergabe**: Automatische Schleife durch mehrere Videos
- **Echtzeit-Alarmsystem**: Visuelle Warnung bei erkannten Gefahrensituationen
- **Benutzerfreundliche Konfiguration**: Grafischer Dialog für alle Einstellungen
- **Modulare Architektur**: Aufgeteilter Code für bessere Wartbarkeit

## Field of Interest (FOI) System

### Funktionsweise
Das FOI-System überwacht einen vom Benutzer definierten viereckigen Bereich im Video:

1. **Personenzählung**: Kontinuierliche Zählung einer konfigurierbaren Objektklasse im FOI
2. **Alert-System**: Überwachung einer zweiten Objektklasse für Notfall-Reaktionen
3. **Skilift-Steuerung**: Automatische Statusmeldungen für Liftbetrieb

### Skilift-Status-Management
- **Normalbetrieb**: Keine besonderen Ereignisse
- **Lift verlangsamt**: Alert-Objekt im FOI erkannt
- **Lift wieder auf Normalgeschwindigkeit**: Alert-Objekt innerhalb der Timeout-Zeit verschwunden
- **Lift wird gestoppt. Personal informiert**: Alert-Objekt länger als Timeout-Zeit im FOI

### FOI-Interaktion
- **Anpassung**: Ziehen Sie die Eckpunkte des Vierecks zur Größen-/Positionsänderung
- **Echtzeit-Updates**: Änderungen werden sofort übernommen und gespeichert
- **Visuelle Rückmeldung**: Hover-Effekte und Drag-Indikatoren

## Installation

1. **Repository klonen oder Dateien herunterladen**

2. **Abhängigkeiten installieren:**
```bash
pip install -r requirements.txt
```

3. **YOLO-Modelle bereitstellen:**
   - Detection-Modell (z.B. `yolov8n.pt`, `yolov8s.pt`)
   - Pose-Modell (z.B. `yolov8n-pose.pt`, `yolov8s-pose.pt`)

## Projektstruktur

```
project/
├── main.py                 # Hauptanwendung
├── requirements.txt        # Abhängigkeiten
├── README.md              # Diese Datei
├── config/
│   ├── __init__.py
│   ├── config_manager.py   # Konfigurationsverwaltung
│   └── constants.py        # Konstanten und Standardwerte
├── core/
│   ├── __init__.py
│   ├── detection_worker.py # YOLO Detection Worker
│   ├── frame_renderer.py   # Frame-Rendering
│   └── foi_manager.py      # Field of Interest Management
└── ui/
    ├── __init__.py
    ├── video_player.py      # Hauptfenster
    └── settings_dialog.py   # Einstellungsdialog
```

## Verwendung

1. **Anwendung starten:**
```bash
python main.py
```

2. **Modelle und Videos konfigurieren:**
   - Auf "⚙ Einstellungen" klicken
   - YOLO Detection-Modell auswählen
   - YOLO Pose-Modell auswählen (optional)
   - Videos hinzufügen

3. **FOI konfigurieren:**
   - FOI aktivieren
   - **Zählklasse**: Objektklasse für Personenzählung auswählen
   - **Alert-Klasse**: Objektklasse für Lift-Verlangsamung auswählen
   - **Alert Timeout**: Zeit in Sekunden bis Lift-Stopp

4. **Pose-Detection konfigurieren:**
   - In der Klassentabelle die gewünschten Klassen für Pose-Detection aktivieren
   - Mehrere Klassen können gleichzeitig ausgewählt werden
   - Konfidenz-Schwellenwerte anpassen

5. **Wiedergabe starten:**
   - "▶ Abspielen" klicken
   - FOI durch Ziehen der Eckpunkte anpassen
   - Videos laufen in Endlosschleife

## Neue Features (Version 2.1)

### Field of Interest (FOI) System
- **Interaktives Viereck**: Anpassbare Überwachungszone durch Ziehen der Eckpunkte
- **Dual-Objektklassen**: Separate Konfiguration für Zählung und Alert-System
- **Skilift-Simulation**: Realistische Nachbildung von Liftsteuerung
- **Echtzeit-Feedback**: Sofortige visuelle und Status-Updates

### Erweiterte Pose-Detection
- **Mehrpersonen-Support**: Eine Bounding Box kann mehrere Personen mit Pose enthalten
- **Verbesserte Genauigkeit**: Optimierte Erkennung in überfüllten Szenen

### Benutzerfreundlichkeit
- **Status-Bar**: Kontinuierliche Anzeige des Lift-Status
- **Farbkodierte Meldungen**: Visuelle Unterscheidung verschiedener Zustände
- **Mouse-Interaktion**: Intuitive FOI-Manipulation

## Konfiguration

Die Anwendung speichert alle Einstellungen automatisch in `config.json`:

```json
{
  "detection_model_path": "path/to/detection/model.pt",
  "pose_model_path": "path/to/pose/model.pt", 
  "class_config": {
    "0": {
      "name": "Person",
      "color": [0, 255, 0],
      "conf": 0.5,
      "iou": 0.4
    }
  },
  "pose_config": {
    "pose_detect_classes": ["0", "1"],
    "min_confidence": 0.3,
    "line_thickness": 2,
    "keypoint_radius": 3,
    "show_keypoints": true,
    "show_skeleton": true
  },
  "display_config": {
    "box_thickness": 2,
    "font_scale": 5,
    "text_thickness": 1,
    "alarm_class": "0"
  },
  "foi_config": {
    "enabled": true,
    "points": [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]],
    "count_class": "0",
    "alert_class": "1",
    "alert_timeout": 10.0,
    "foi_color": [0, 255, 255],
    "foi_thickness": 3
  },
  "video_files": ["path/to/video1.mp4", "path/to/video2.mp4"]
}
```

## Anwendungsszenarien Skilift

### Typische Konfiguration
- **Zählklasse**: "Person" - für normale Skifahrer/Snowboarder
- **Alert-Klasse**: "Gestürzter" oder "Hilfsbedürftiger" - für Notfälle
- **FOI-Position**: Liftbereich, kritische Stellen, Ausstiegsbereiche

### Workflow
1. **Normale Fahrt**: Personen werden gezählt, Status "Normalbetrieb"
2. **Sturz erkannt**: Alert-Klasse im FOI → "Lift verlangsamt"
3. **Schnelle Hilfe**: Person verlässt FOI < 10s → "Normalgeschwindigkeit"
4. **Ernstfall**: Person bleibt > 10s → "Lift gestoppt, Personal informiert"

## Technische Details

### YOLO-Modelle
- **Detection**: Erkennt Objekte (Personen, Fahrzeuge, etc.)
- **Pose**: Schätzt Körperhaltung bei erkannten Objekten
- **Kompatibilität**: YOLOv8 und neuere Versionen
- **Mehrpersonen-Pose**: Unterstützt mehrere Personen pro Bounding Box

### FOI-Technologie
- **Polygon-Erkennung**: Präzise Punkt-in-Polygon-Tests
- **Relative Koordinaten**: Auflösungsunabhängige Speicherung
- **Echtzeit-Updates**: Sofortige Anpassung während Wiedergabe

### Performance-Optimierung
- **Multi-Threading**: Parallele Verarbeitung von Frames
- **Frame-Pufferung**: Vermeidung von Blockaden
- **Optimiertes Rendering**: Effiziente OpenCV-Integration
- **Smart-FOI**: Nur bei Bedarf aktive Berechnungen

## Troubleshooting

### Häufige Probleme

1. **FOI reagiert nicht:**
   - Stellen Sie sicher, dass FOI in den Einstellungen aktiviert ist
   - Überprüfen Sie, ob Videos abgespielt werden
   - Klicken Sie direkt auf die Eckpunkte (rote Kreise)

2. **Keine Personenzählung:**
   - Überprüfen Sie die Zählklasse-Konfiguration
   - Stellen Sie sicher, dass das Detection-Modell die gewählte Klasse erkennt
   - Testen Sie mit niedrigerer Konfidenz-Schwelle

3. **Alert-System funktioniert nicht:**
   - Überprüfen Sie die Alert-Klasse-Konfiguration
   - Stellen Sie sicher, dass sich die Alert-Klasse von der Zählklasse unterscheidet
   - Prüfen Sie das Timeout (Standard: 10 Sekunden)

4. **Modell lädt nicht:**
   - Überprüfen Sie den Dateipfad
   - Stellen Sie sicher, dass es sich um ein gültiges YOLO-Modell handelt

5. **Video spielt nicht ab:**
   - Überprüfen Sie das Videoformat (MP4, AVI, MOV, MKV)
   - Stellen Sie sicher, dass OpenCV das Codec unterstützt

6. **Langsame Performance:**
   - Verwenden Sie kleinere YOLO-Modelle (nano/small statt large)
   - Reduzieren Sie die Videoauflösung
   - Deaktivieren Sie FOI wenn nicht benötigt

## Entwicklung

### Code-Beiträge
Das Projekt ist modular aufgebaut und ermöglicht einfache Erweiterungen:

- **FOI-Erweitungen**: Bearbeiten Sie `foi_manager.py`
- **Neue Modelle**: Erweitern Sie `detection_worker.py`
- **UI-Verbesserungen**: Bearbeiten Sie die `ui/`-Module
- **Konfiguration**: Erweitern Sie `config/constants.py`

### Testing
```bash
# Grundlegende Funktionalität testen
python main.py

# FOI-System testen
# 1. FOI aktivieren in Einstellungen
# 2. Videos abspielen
# 3. Eckpunkte ziehen und Position testen
# 4. Objektklassen-Erkennung überprüfen
```

## Lizenz

Dieses Projekt dient der Demonstration von KI-Anwendungen für Sicherheitssysteme in Skigebieten.

## Support

Bei Fragen oder Problemen:
1. Überprüfen Sie die FOI-Konfiguration in den Einstellungen
2. Testen Sie die Objektklassen-Erkennung
3. Prüfen Sie die Konsole auf Fehlermeldungen
4. Stellen Sie sicher, dass alle Abhängigkeiten installiert sind
5. Überprüfen Sie die Mouse-Interaktion mit FOI-Eckpunkten