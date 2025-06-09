import os
import cv2
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QFrame, QProgressBar, QMessageBox, QStatusBar
)
from PyQt6.QtGui import QPixmap, QImage, QMouseEvent
from PyQt6.QtCore import QTimer, Qt, QThreadPool
from ultralytics import YOLO

from config.config_manager import ConfigManager
from config.constants import DEFAULT_CONFIG
from core.detection_worker import DualDetectionWorker
from core.frame_renderer import FrameRenderer
from core.foi_manager import FOIManager
from ui.settings_dialog import SettingsDialog

class VideoPlayer(QWidget):
    """Hauptklasse für den Video Player mit YOLO Dual Model Annotation"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Dual Model Video Annotator - Sturzerkennung Skilift")
        
        # Vollbild-Modus sauber auf Windows 11
        # Vollbild mit Fenster mit Rahmen, ohne die Taskleiste zu verdecken
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowCloseButtonHint)
        # Bildschirmgröße abrufen
        screen_geometry = self.screen().availableGeometry()
        # Grösse einstellen (y - Höhe der Taskleiste)
        self.resize(screen_geometry.width(), screen_geometry.height() - 40)

        # Thread pool for parallel processing
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(4)
        
        # Configuration Manager
        self.config_manager = ConfigManager()
        
        # Initialize default values
        self._init_default_values()
        
        # Setup UI
        self.setup_ui()
        self.connect_signals()
        
        # Video playback state
        self._init_video_state()
        
        # Frame renderer und FOI Manager
        self.frame_renderer = FrameRenderer(
            self.class_config, self.pose_config, self.display_config
        )
        self.foi_manager = FOIManager(self.foi_config)
        
        # Mouse interaction state
        self.mouse_pressed = False
        self.last_mouse_pos = None
        
        # Status and alarm system
        self._init_alarm_system()
        
        # Setup timers
        self._init_timers()
        
        # Load default config
        self.load_default_config()
    
    def _init_default_values(self):
        """Initialisiert die Standard-Konfigurationswerte"""
        self.detection_model = None
        self.pose_model = None
        self.detection_model_path = ""
        self.pose_model_path = ""
        self.class_config = {}
        self.pose_config = DEFAULT_CONFIG['pose_config'].copy()
        self.display_config = DEFAULT_CONFIG['display_config'].copy()
        self.foi_config = DEFAULT_CONFIG['foi_config'].copy()
        self.video_files = []
    
    def _init_video_state(self):
        """Initialisiert den Video-Wiedergabe-Status"""
        self.cap = None
        self.current_video_idx = 0
        self.current_frame = None
        self.processing_frame = False
        self.last_detections = []
        self.last_poses = []
    
    def _init_alarm_system(self):
        """Initialisiert das Alarmsystem"""
        self.alarm_active = False
        self.pulse_value = 0
        self.pulse_direction = 1
    
    def _init_timers(self):
        """Initialisiert die Timer"""
        # Frame timer - läuft mit fester 30 FPS
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.timer.setInterval(33)  # ~30 FPS fest eingestellt
        
        # Alarm animation timer
        self.alarm_timer = QTimer()
        self.alarm_timer.timeout.connect(self.pulse_alarm)
    
    def setup_ui(self):
        """Erstellt die Benutzeroberfläche"""
        main_layout = QVBoxLayout(self)
        
        # Toolbar
        toolbar_layout = self._create_toolbar()
        main_layout.addLayout(toolbar_layout)
        
        # Video display
        self.video_container = self._create_video_container()
        main_layout.addWidget(self.video_container, 1)
        
        # Status bar für Lift-Status
        self.status_bar = QStatusBar()
        self.status_bar.showMessage("Lift Normalbetrieb")
        self.status_bar.setStyleSheet("QStatusBar { background-color: #f0f0f0; border-top: 1px solid #ccc; }")
        main_layout.addWidget(self.status_bar)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(3)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        main_layout.addWidget(self.progress)
    
    def _create_toolbar(self):
        """Erstellt die Toolbar"""
        toolbar_layout = QHBoxLayout()
        
        self.btn_settings = QPushButton("⚙ Einstellungen")
        self.btn_settings.setStyleSheet("padding: 8px; font-size: 14px;")
        
        self.btn_play_pause = QPushButton("▶ Abspielen")
        self.btn_play_pause.setStyleSheet("padding: 8px; font-size: 14px;")
        self.btn_play_pause.setEnabled(False)
        
        self.lbl_status = QLabel("Bereit - Konfiguration für Sturzerkennung an Skiliften")
        self.lbl_status.setStyleSheet("color: #666; font-size: 12px;")
        
        toolbar_layout.addWidget(self.btn_settings)
        toolbar_layout.addWidget(self.btn_play_pause)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.lbl_status)
        
        return toolbar_layout
    
    def _create_video_container(self):
        """Erstellt den Video-Container"""
        video_container = QFrame()
        video_container.setStyleSheet("background-color: #222; border-radius: 5px;")
        video_layout = QVBoxLayout(video_container)
        
        self.label = QLabel("Bitte wählen Sie Videos und YOLO-Modelle in den Einstellungen aus")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("color: #aaa; font-size: 16px;")
        
        # Mouse-Events für FOI-Interaktion aktivieren
        self.label.setMouseTracking(True)
        self.label.mousePressEvent = self.mousePressEvent
        self.label.mouseMoveEvent = self.mouseMoveEvent
        self.label.mouseReleaseEvent = self.mouseReleaseEvent
        
        video_layout.addWidget(self.label)
        
        return video_container
    
    def connect_signals(self):
        """Verbindet die Signale mit ihren Slots"""
        self.btn_settings.clicked.connect(self.open_settings)
        self.btn_play_pause.clicked.connect(self.toggle_playback)
    
    def open_settings(self):
        """Öffnet den Einstellungsdialog"""
        dialog = SettingsDialog(
            self.detection_model_path,
            self.pose_model_path,
            self.class_config,
            self.pose_config,
            self.display_config,
            self.foi_config,
            self.video_files,
            self
        )
        
        if dialog.exec():
            settings = dialog.get_settings()
            self._apply_settings(settings)
            self._update_models()
            self._update_status()
            self.save_config()
    
    def _apply_settings(self, settings):
        """Wendet die Einstellungen aus dem Dialog an"""
        self.detection_model_path = settings['detection_model_path']
        self.pose_model_path = settings['pose_model_path']
        self.class_config = settings['class_config']
        self.pose_config = settings['pose_config']
        self.display_config = settings['display_config']
        self.foi_config = settings['foi_config']
        self.video_files = settings['video_files']
        
        # Frame renderer und FOI Manager aktualisieren
        self.frame_renderer.update_config(
            self.class_config, self.pose_config, self.display_config
        )
        self.foi_manager.update_config(self.foi_config)
    
    def _update_models(self):
        """Lädt die Modelle neu falls sie geändert wurden"""
        # Detection Modell neu laden
        if self.detection_model_path:
            try:
                self.detection_model = YOLO(self.detection_model_path)
                detection_model_name = os.path.basename(self.detection_model_path)
                print(f"Detection Modell geladen: {detection_model_name}")
            except Exception as e:
                QMessageBox.critical(self, "Fehler", f"Detection Modell konnte nicht geladen werden: {str(e)}")
                self.detection_model = None
        else:
            self.detection_model = None
        
        # Pose Modell neu laden
        if self.pose_model_path:
            try:
                self.pose_model = YOLO(self.pose_model_path)
                pose_model_name = os.path.basename(self.pose_model_path)
                print(f"Pose Modell geladen: {pose_model_name}")
            except Exception as e:
                QMessageBox.critical(self, "Fehler", f"Pose Modell konnte nicht geladen werden: {str(e)}")
                self.pose_model = None
        else:
            self.pose_model = None
    
    def _update_status(self):
        """Aktualisiert die Statusanzeige"""
        status_parts = []
        if self.detection_model:
            status_parts.append(f"Detection: {os.path.basename(self.detection_model_path)}")
        if self.pose_model:
            status_parts.append(f"Pose: {os.path.basename(self.pose_model_path)}")
        
        if status_parts:
            self.lbl_status.setText(" | ".join(status_parts))
        else:
            self.lbl_status.setText("Keine Modelle geladen")
        
        # Videos vorbereiten
        if self.video_files and (self.detection_model or self.pose_model):
            self.current_video_idx = 0
            self.btn_play_pause.setEnabled(True)
            self.label.setText("Bereit zum Abspielen")
            video_count = len(self.video_files)
            current_status = self.lbl_status.text()
            self.lbl_status.setText(f"{current_status} | {video_count} Video(s) bereit")
        else:
            self.btn_play_pause.setEnabled(False)
            if not self.video_files:
                self.label.setText("Keine Videos ausgewählt")
            elif not (self.detection_model or self.pose_model):
                self.label.setText("Keine Modelle ausgewählt")
    
    def start_video(self):
        """Startet die Video-Wiedergabe"""
        if not (self.detection_model or self.pose_model):
            QMessageBox.warning(self, "Warnung", "Bitte wählen Sie mindestens ein YOLO-Modell aus.")
            return
            
        if not self.video_files:
            QMessageBox.warning(self, "Warnung", "Bitte wählen Sie Videos aus.")
            return
        
        if self.cap:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(self.video_files[self.current_video_idx])
        self.timer.start()
        
        self.btn_play_pause.setText("⏸ Pausieren")
        
        # Status aktualisieren
        video_name = os.path.basename(self.video_files[self.current_video_idx])
        current_num = self.current_video_idx + 1
        total_num = len(self.video_files)
        self.setWindowTitle(f"YOLO Dual Model Video Annotator - {video_name} ({current_num}/{total_num})")
        
        status_parts = []
        if self.detection_model:
            status_parts.append(f"Detection: {os.path.basename(self.detection_model_path)}")
        if self.pose_model:
            status_parts.append(f"Pose: {os.path.basename(self.pose_model_path)}")
        status_parts.append(f"Video: {video_name} ({current_num}/{total_num})")
        
        self.lbl_status.setText(" | ".join(status_parts))
    
    def toggle_playback(self):
        """Wechselt zwischen Abspielen und Pausieren"""
        if not self.cap:
            self.start_video()
            return
            
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play_pause.setText("▶ Abspielen")
            current_status = self.lbl_status.text()
            if "| Video:" in current_status:
                self.lbl_status.setText(current_status.replace("| Video:", "| Pausiert | Video:"))
            else:
                self.lbl_status.setText("Pausiert")
        else:
            self.timer.start()
            self.btn_play_pause.setText("⏸ Pausieren")
            current_status = self.lbl_status.text()
            if "| Pausiert |" in current_status:
                self.lbl_status.setText(current_status.replace("| Pausiert |", "|"))
    
    def pulse_alarm(self):
        """Animation für den Alarmzustand"""
        if not self.alarm_active:
            self.alarm_timer.stop()
            self.video_container.setStyleSheet("background-color: #222; border-radius: 5px;")
            return
            
        self.pulse_value += (5 * self.pulse_direction)
        if self.pulse_value >= 100:
            self.pulse_value = 100
            self.pulse_direction = -1
        elif self.pulse_value <= 0:
            self.pulse_value = 0
            self.pulse_direction = 1
            
        intensity = 100 + int(self.pulse_value * 1.55)
        self.video_container.setStyleSheet(f"""
            background-color: rgb({intensity}, 0, 0);
            border-radius: 5px;
        """)
    
    def handle_detection_result(self, result):
        """Verarbeitet die Erkennungsergebnisse vom Worker-Thread"""
        if not result:
            return
            
        frame, detections, poses = result
        self.current_frame = frame
        self.last_detections = detections
        self.last_poses = poses
        
        # Frame-Dimensionen für FOI Manager setzen
        if frame is not None:
            h, w = frame.shape[:2]
            self.foi_manager.set_frame_dimensions(w, h)
        
        # FOI-Analysen durchführen
        if self.foi_config.get('enabled', False):
            self.foi_manager.count_objects_in_foi(detections)
            self.foi_manager.check_alert_objects_in_foi(detections)
            
            # Status-Bar aktualisieren
            lift_status = self.foi_manager.get_lift_status()
            self.status_bar.showMessage(lift_status)
            
            # Status-Bar-Farbe je nach Status ändern
            if "verlangsamt" in lift_status:
                self.status_bar.setStyleSheet("QStatusBar { background-color: #fff3cd; border-top: 1px solid #ccc; color: #856404; }")
            elif "gestoppt" in lift_status:
                self.status_bar.setStyleSheet("QStatusBar { background-color: #f8d7da; border-top: 1px solid #ccc; color: #721c24; }")
            elif "Normalgeschwindigkeit" in lift_status:
                self.status_bar.setStyleSheet("QStatusBar { background-color: #d1edff; border-top: 1px solid #ccc; color: #004085; }")
            else:
                self.status_bar.setStyleSheet("QStatusBar { background-color: #d4edda; border-top: 1px solid #ccc; color: #155724; }")
        
        # Prüfung auf Standard-Alarmzustand
        alarm_class_id = self.display_config.get('alarm_class')
        if alarm_class_id:
            alarm_triggered = any(str(d['class_id']) == alarm_class_id for d in detections)
            
            if alarm_triggered and not self.alarm_active:
                self.alarm_active = True
                self.alarm_timer.start(50)
            elif not alarm_triggered:
                self.alarm_active = False
                
        # Frame mit Erkennungen und Posen rendern
        self.render_frame()
        
        # Processing-Lock freigeben
        self.processing_frame = False
    
    def render_frame(self):
        """Zeichnet Erkennungen und Posen auf den Frame und zeigt ihn an"""
        if self.current_frame is None:
            return
            
        # Frame mit Erkennungen und Posen rendern
        rendered_frame = self.frame_renderer.render_frame(
            self.current_frame, self.last_detections, self.last_poses
        )
        
        # FOI auf Frame zeichnen
        if self.foi_config.get('enabled', False):
            rendered_frame = self.foi_manager.draw_foi_on_frame(rendered_frame)
            rendered_frame = self.foi_manager.draw_count_display(rendered_frame)
        
        # In Qt-Format für Anzeige konvertieren
        rgb_frame = cv2.cvtColor(rendered_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.AspectRatioMode.KeepAspectRatio))
    
    def next_frame(self):
        """Holt den nächsten Video-Frame und verarbeitet ihn"""
        if not self.cap or self.processing_frame:
            return
            
        # Frame holen
        ret, frame = self.cap.read()
        if not ret:
            # Ende des aktuellen Videos, zum nächsten in Endlosschleife
            self.current_video_idx = (self.current_video_idx + 1) % len(self.video_files)
            
            # Status für nächstes Video aktualisieren
            video_name = os.path.basename(self.video_files[self.current_video_idx])
            current_num = self.current_video_idx + 1
            total_num = len(self.video_files)
            self.setWindowTitle(f"YOLO Dual Model Video Annotator - {video_name} ({current_num}/{total_num})")
            
            status_parts = []
            if self.detection_model:
                status_parts.append(f"Detection: {os.path.basename(self.detection_model_path)}")
            if self.pose_model:
                status_parts.append(f"Pose: {os.path.basename(self.pose_model_path)}")
            status_parts.append(f"Video: {video_name} ({current_num}/{total_num})")
            
            self.lbl_status.setText(" | ".join(status_parts))
            
            # Nächstes Video starten
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.video_files[self.current_video_idx])
            return
        
        # Processing-Lock setzen
        self.processing_frame = True
        
        # An Thread-Pool übergeben
        worker = DualDetectionWorker(frame, self.detection_model, self.pose_model, 
                                   self.class_config, self.pose_config)
        worker.signals.result.connect(self.handle_detection_result)
        worker.signals.error.connect(lambda err: print(f"Error: {err}"))
        self.threadpool.start(worker)
    
    def load_default_config(self):
        """Lädt die Standard-Konfiguration falls config.json existiert"""
        try:
            config = self.config_manager.load_config()
            
            # Konfiguration anwenden
            self.detection_model_path = config.get('detection_model_path', '')
            self.pose_model_path = config.get('pose_model_path', '')
            self.class_config = config.get('class_config', {})
            self.pose_config = config.get('pose_config', DEFAULT_CONFIG['pose_config'].copy())
            self.display_config = config.get('display_config', DEFAULT_CONFIG['display_config'].copy())
            self.foi_config = config.get('foi_config', DEFAULT_CONFIG['foi_config'].copy())
            self.video_files = config.get('video_files', [])
            
            # Frame renderer und FOI Manager aktualisieren
            self.frame_renderer.update_config(
                self.class_config, self.pose_config, self.display_config
            )
            self.foi_manager.update_config(self.foi_config)
            
            # Detection Modell laden falls Pfad vorhanden
            if self.detection_model_path and os.path.exists(self.detection_model_path):
                try:
                    self.detection_model = YOLO(self.detection_model_path)
                    detection_model_name = os.path.basename(self.detection_model_path)
                    print(f"Detection Modell geladen: {detection_model_name}")
                except Exception as e:
                    print(f"Fehler beim Laden des Detection Modells: {e}")
                    self.detection_model = None
            
            # Pose Modell laden falls Pfad vorhanden
            if self.pose_model_path and os.path.exists(self.pose_model_path):
                try:
                    self.pose_model = YOLO(self.pose_model_path)
                    pose_model_name = os.path.basename(self.pose_model_path)
                    print(f"Pose Modell geladen: {pose_model_name}")
                except Exception as e:
                    print(f"Fehler beim Laden des Pose Modells: {e}")
                    self.pose_model = None
            
            # Status aktualisieren
            self._update_initial_status()
                        
        except Exception as e:
            print(f"Fehler beim Laden der Konfiguration: {e}")
            # Verwende Standard-Konfiguration bei Fehlern
            self.class_config = {
                "0": {"name": "Person", "color": (0, 255, 0), "conf": 0.5, "iou": 0.4}
            }
    
    def _update_initial_status(self):
        """Aktualisiert den initialen Status nach dem Laden der Konfiguration"""
        status_parts = []
        if self.detection_model:
           status_parts.append(f"Detection: {os.path.basename(self.detection_model_path)}")
        if self.pose_model:
            status_parts.append(f"Pose: {os.path.basename(self.pose_model_path)}")
        
        if status_parts and self.video_files:
            self.btn_play_pause.setEnabled(True)
            video_count = len(self.video_files)
            status_parts.append(f"{video_count} Video(s) bereit")
            self.lbl_status.setText(" | ".join(status_parts))
            self.label.setText("Bereit zum Abspielen")
        elif status_parts:
            self.lbl_status.setText(" | ".join(status_parts))
        else:
            self.lbl_status.setText("Keine Modelle geladen - Konfiguration für Sturzerkennung an Skiliften")
    
    def save_config(self):
        """Speichert die aktuelle Konfiguration"""
        config = {
            'detection_model_path': self.detection_model_path,
            'pose_model_path': self.pose_model_path,
            'class_config': self.class_config,
            'pose_config': self.pose_config,
            'display_config': self.display_config,
            'foi_config': self.foi_config,
            'video_files': self.video_files
        }
        
        success = self.config_manager.save_config(config)
        if not success:
            print("Fehler beim Speichern der Konfiguration")
    
    def closeEvent(self, event):
        """Wird beim Schließen der Anwendung aufgerufen"""
        # Ressourcen aufräumen
        self.timer.stop()
        self.alarm_timer.stop()
        if self.cap:
            self.cap.release()
            
        # Konfiguration beim Beenden speichern
        self.save_config()
        event.accept()
    
    # Mouse-Event-Handler für FOI-Interaktion, passend zu FOIManager
    def mousePressEvent(self, event):
        """Behandelt Mausklicks für FOI-Interaktion"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.mouse_pressed = True
            self.last_mouse_pos = (event.x(), event.y())
            corner_idx = self.foi_manager.get_corner_at_position(event.x(), event.y())
            if corner_idx >= 0:
                self.foi_manager.set_active_corner(corner_idx)
        elif event.button() == Qt.MouseButton.RightButton:
            # Rechtsklick zum Zurücksetzen des FOI
            self.foi_manager.reset_foi()
            self.render_frame()
            self.status_bar.showMessage("FOI zurückgesetzt")

    def mouseMoveEvent(self, event):
        """Behandelt Mausbewegungen für FOI-Interaktion"""
        if self.mouse_pressed and self.last_mouse_pos is not None:
            self.foi_manager.update_foi_selection(event.position())
            self.render_frame()

    def mouseReleaseEvent(self, event):
        """Behandelt das Loslassen der Maustaste für FOI-Interaktion"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.mouse_pressed = False
            self.last_mouse_pos = None
            self.foi_manager.finalize_foi_selection(event.position())
            self.render_frame()
