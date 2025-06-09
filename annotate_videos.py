import sys
import os
import cv2
import numpy as np
import json
import threading
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QSlider, QGroupBox, QFormLayout, QSpinBox, QComboBox,
    QSplitter, QFrame, QTabWidget, QLineEdit, QColorDialog, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QProgressBar,
    QDialog, QGridLayout, QListWidget, QListWidgetItem
)
from PyQt6.QtGui import QPixmap, QImage, QColor, QIcon
from PyQt6.QtCore import QTimer, Qt, QPropertyAnimation, QThread, pyqtSignal, QRunnable, QThreadPool, QObject
from ultralytics import YOLO

# Erweiterte Farbpalette für mehr Klassen (in BGR format für OpenCV)
COLORS = {
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
    "Yellow": (0, 255, 255),
    "Cyan": (255, 255, 0),
    "Magenta": (255, 0, 255),
    "Orange": (0, 165, 255),
    "Purple": (128, 0, 128),
    "Brown": (42, 42, 165),
    "Pink": (203, 192, 255),
    "Lime": (0, 255, 128),
    "Teal": (128, 128, 0),
    "Navy": (128, 0, 0),
    "Maroon": (0, 0, 128),
    "Olive": (0, 128, 128),
    "Silver": (192, 192, 192),
    "Gold": (0, 215, 255),
    "Coral": (80, 127, 255),
    "Turquoise": (208, 224, 64),
    "Violet": (238, 130, 238),
}

class WorkerSignals(QObject):
    """Defines the signals available from the worker thread."""
    result = pyqtSignal(object)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

class DetectionWorker(QRunnable):
    """Worker thread for processing video frames in parallel."""
    def __init__(self, frame, model, class_config):
        super().__init__()
        self.frame = frame.copy()
        self.model = model
        self.class_config = class_config
        self.signals = WorkerSignals()
        
    def run(self):
        try:
            # Process frame with YOLO
            results = self.model.predict(self.frame, verbose=False)[0]
            
            # Format the results for rendering
            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                cls_id = str(cls)
                cfg = self.class_config.get(cls_id)
                if not cfg or conf < float(cfg.get("conf", 0.5)):
                    continue
                
                detections.append({
                    'box': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                    'conf': conf,
                    'class_id': cls,
                    'class_name': cfg.get('name', f"Class {cls}")
                })
            
            # Emit the result
            self.signals.result.emit((self.frame, detections))
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()

class SettingsDialog(QDialog):
    """Großer übersichtlicher Dialog für alle Einstellungen."""
    def __init__(self, model_path, class_config, display_config, video_files, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Einstellungen")
        self.resize(1000, 700)
        
        self.model_path = model_path
        self.class_config = class_config.copy()
        self.display_config = display_config.copy()
        self.video_files = video_files.copy()
        
        self.setup_ui()
        self.load_settings()
    
    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        
        # Linke Seite: Modell und Videos
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Modell-Auswahl
        model_group = QGroupBox("YOLO Modell")
        model_layout = QVBoxLayout(model_group)
        
        model_select_layout = QHBoxLayout()
        self.txt_model_path = QLineEdit(self.model_path)
        self.txt_model_path.setReadOnly(True)
        self.btn_select_model = QPushButton("Modell auswählen")
        model_select_layout.addWidget(self.txt_model_path, 3)
        model_select_layout.addWidget(self.btn_select_model, 1)
        model_layout.addLayout(model_select_layout)
        
        self.lbl_model_info = QLabel("Kein Modell geladen")
        model_layout.addWidget(self.lbl_model_info)
        
        left_layout.addWidget(model_group)
        
        # Video-Auswahl
        video_group = QGroupBox("Videos (Endlosschleife)")
        video_layout = QVBoxLayout(video_group)
        
        video_buttons_layout = QHBoxLayout()
        self.btn_add_videos = QPushButton("Videos hinzufügen")
        self.btn_remove_video = QPushButton("Video entfernen")
        self.btn_clear_videos = QPushButton("Alle entfernen")
        video_buttons_layout.addWidget(self.btn_add_videos)
        video_buttons_layout.addWidget(self.btn_remove_video)
        video_buttons_layout.addWidget(self.btn_clear_videos)
        video_layout.addLayout(video_buttons_layout)
        
        self.video_list = QListWidget()
        video_layout.addWidget(self.video_list)
        
        left_layout.addWidget(video_group)
        
        # Darstellungs-Einstellungen
        display_group = QGroupBox("Darstellung")
        display_form = QFormLayout(display_group)
        
        self.box_thickness = QSpinBox()
        self.box_thickness.setRange(1, 10)
        self.box_thickness.setValue(self.display_config.get('box_thickness', 2))
        display_form.addRow("Rahmendicke:", self.box_thickness)
        
        self.font_scale = QSpinBox()
        self.font_scale.setRange(1, 20)
        self.font_scale.setValue(self.display_config.get('font_scale', 5))
        display_form.addRow("Textgrösse (x10):", self.font_scale)
        
        self.text_thickness = QSpinBox()
        self.text_thickness.setRange(1, 5)
        self.text_thickness.setValue(self.display_config.get('text_thickness', 1))
        display_form.addRow("Textdicke:", self.text_thickness)
        
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(60)
        self.speed_slider.setValue(self.display_config.get('playback_speed', 30))
        self.speed_slider.setTickInterval(5)
        self.speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.speed_label = QLabel(str(self.speed_slider.value()))
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(self.speed_slider)
        speed_layout.addWidget(self.speed_label)
        display_form.addRow("FPS:", speed_layout)
        
        self.alarm_class_dropdown = QComboBox()
        display_form.addRow("Alarmklasse:", self.alarm_class_dropdown)
        
        left_layout.addWidget(display_group)
        left_layout.addStretch()
        
        # Rechte Seite: Klasseneinstellungen
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        classes_group = QGroupBox("Klasseneinstellungen")
        classes_layout = QVBoxLayout(classes_group)
        
        # Tabelle für Klasseneinstellungen
        self.class_table = QTableWidget()
        self.class_table.setColumnCount(5)  # Name, Farbe, Konfidenz, IoU, Aktiv
        self.class_table.setHorizontalHeaderLabels(['Klassen-ID', 'Name', 'Farbe', 'Konfidenz', 'IoU'])
        self.class_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        classes_layout.addWidget(self.class_table)
        
        note_label = QLabel("Hinweis: Klassen werden automatisch aus dem YOLO-Modell geladen.")
        note_label.setWordWrap(True)
        note_label.setStyleSheet("color: #666; font-style: italic;")
        classes_layout.addWidget(note_label)
        
        right_layout.addWidget(classes_group)
        
        # Layout zusammenfügen
        main_layout.addWidget(left_widget, 1)
        main_layout.addWidget(right_widget, 2)
        
        # Buttons unten
        button_layout = QHBoxLayout()
        self.btn_save = QPushButton("Speichern")
        self.btn_cancel = QPushButton("Abbrechen")
        self.btn_load_config = QPushButton("Konfiguration laden")
        self.btn_save_config = QPushButton("Konfiguration speichern")
        
        button_layout.addWidget(self.btn_load_config)
        button_layout.addWidget(self.btn_save_config)
        button_layout.addStretch()
        button_layout.addWidget(self.btn_cancel)
        button_layout.addWidget(self.btn_save)
        
        # Hauptlayout erweitern
        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        
        dialog_layout = QVBoxLayout(self)
        dialog_layout.addWidget(main_widget)
        dialog_layout.addLayout(button_layout)
        
        # Signal-Verbindungen
        self.connect_signals()
    
    def connect_signals(self):
        self.btn_select_model.clicked.connect(self.select_model)
        self.btn_add_videos.clicked.connect(self.add_videos)
        self.btn_remove_video.clicked.connect(self.remove_video)
        self.btn_clear_videos.clicked.connect(self.clear_videos)
        self.speed_slider.valueChanged.connect(lambda v: self.speed_label.setText(str(v)))
        
        self.btn_save.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_load_config.clicked.connect(self.load_config)
        self.btn_save_config.clicked.connect(self.save_config)
    
    def select_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "YOLO-Modell auswählen", "", "YOLO Models (*.pt)"
        )
        if file_path:
            try:
                model = YOLO(file_path)
                self.model_path = file_path
                self.txt_model_path.setText(file_path)
                
                model_name = os.path.basename(file_path)
                self.lbl_model_info.setText(f"Modell: {model_name}")
                
                # Klassen aus Modell extrahieren
                self.extract_model_classes(model)
                
            except Exception as e:
                QMessageBox.critical(self, "Fehler", f"Modell konnte nicht geladen werden: {str(e)}")
    
    def extract_model_classes(self, model):
        """Extract class information from the loaded YOLO model"""
        class_names = model.names
        
        # Bestehende Konfiguration beibehalten, neue Klassen hinzufügen
        color_list = list(COLORS.values())
        
        for cls_id, name in class_names.items():
            cls_id_str = str(cls_id)
            
            if cls_id_str not in self.class_config:
                color_idx = cls_id % len(color_list)
                color = color_list[color_idx]
                
                self.class_config[cls_id_str] = {
                    'name': name,
                    'color': color,
                    'conf': 0.5,
                    'iou': 0.5
                }
            else:
                # Namen aktualisieren
                self.class_config[cls_id_str]['name'] = name
        
        self.load_class_table()
        self.update_alarm_classes()
    
    def load_class_table(self):
        self.class_table.setRowCount(0)
        for cls_id, cfg in sorted(self.class_config.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999):
            row = self.class_table.rowCount()
            self.class_table.insertRow(row)
            
            # Klassen-ID (nicht editierbar)
            id_item = QTableWidgetItem(cls_id)
            id_item.setFlags(id_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.class_table.setItem(row, 0, id_item)
            
            # Name
            self.class_table.setItem(row, 1, QTableWidgetItem(cfg.get('name', f"Class {cls_id}")))
            
            # Farbe (Dropdown)
            color_combo = QComboBox()
            for color_name in COLORS:
                color_combo.addItem(color_name)
            
            current_color = cfg.get('color', (0, 0, 255))
            closest_color = self.find_closest_color(current_color)
            color_combo.setCurrentText(closest_color)
            self.class_table.setCellWidget(row, 2, color_combo)
            
            # Konfidenz
            conf_spin = QDoubleSpinBox()
            conf_spin.setRange(0.1, 1.0)
            conf_spin.setSingleStep(0.05)
            conf_spin.setValue(float(cfg.get('conf', 0.5)))
            self.class_table.setCellWidget(row, 3, conf_spin)
            
            # IoU
            iou_spin = QDoubleSpinBox()
            iou_spin.setRange(0.1, 1.0)
            iou_spin.setSingleStep(0.05)
            iou_spin.setValue(float(cfg.get('iou', 0.5)))
            self.class_table.setCellWidget(row, 4, iou_spin)
    
    def find_closest_color(self, bgr_color):
        """Find the closest predefined color name for a BGR value"""
        if not bgr_color:
            return "Red"
            
        min_distance = float('inf')
        closest_name = "Red"
        
        for name, color in COLORS.items():
            distance = sum((a - b) ** 2 for a, b in zip(bgr_color, color))
            if distance < min_distance:
                min_distance = distance
                closest_name = name
                
        return closest_name
    
    def add_videos(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Videos auswählen", "", "Videos (*.mp4 *.avi *.mov *.mkv)"
        )
        for file in files:
            if file not in self.video_files:
                self.video_files.append(file)
                item = QListWidgetItem(os.path.basename(file))
                item.setToolTip(file)
                self.video_list.addItem(item)
    
    def remove_video(self):
        current_row = self.video_list.currentRow()
        if current_row >= 0:
            self.video_files.pop(current_row)
            self.video_list.takeItem(current_row)
    
    def clear_videos(self):
        self.video_files.clear()
        self.video_list.clear()
    
    def update_alarm_classes(self):
        current_data = self.alarm_class_dropdown.currentData()
        self.alarm_class_dropdown.clear()
        
        for cls_id, cfg in self.class_config.items():
            self.alarm_class_dropdown.addItem(f"{cls_id}: {cfg['name']}", userData=cls_id)
        
        if current_data:
            index = self.alarm_class_dropdown.findData(current_data)
            if index >= 0:
                self.alarm_class_dropdown.setCurrentIndex(index)
    
    def load_settings(self):
        # Videos laden
        for video_file in self.video_files:
            item = QListWidgetItem(os.path.basename(video_file))
            item.setToolTip(video_file)
            self.video_list.addItem(item)
        
        # Klassen laden
        self.load_class_table()
        self.update_alarm_classes()
        
        # Modell-Info aktualisieren
        if self.model_path:
            model_name = os.path.basename(self.model_path)
            self.lbl_model_info.setText(f"Modell: {model_name}")
    
    def get_settings(self):
        # Klasseneinstellungen aus Tabelle sammeln
        for row in range(self.class_table.rowCount()):
            cls_id = self.class_table.item(row, 0).text()
            name = self.class_table.item(row, 1).text()
            
            color_combo = self.class_table.cellWidget(row, 2)
            color_name = color_combo.currentText()
            color = COLORS[color_name]
            
            conf = self.class_table.cellWidget(row, 3).value()
            iou = self.class_table.cellWidget(row, 4).value()
            
            self.class_config[cls_id] = {
                'name': name,
                'color': color,
                'conf': conf,
                'iou': iou
            }
        
        # Display-Einstellungen sammeln
        self.display_config = {
            'box_thickness': self.box_thickness.value(),
            'font_scale': self.font_scale.value(),
            'text_thickness': self.text_thickness.value(),
            'playback_speed': self.speed_slider.value(),
            'alarm_class': self.alarm_class_dropdown.currentData()
        }
        
        return {
            'model_path': self.model_path,
            'class_config': self.class_config,
            'display_config': self.display_config,
            'video_files': self.video_files
        }
    
    def save_config(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Konfiguration speichern", "", "JSON (*.json)"
        )
        if file_path:
            try:
                config = self.get_settings()
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)
                QMessageBox.information(self, "Erfolg", "Konfiguration wurde gespeichert.")
            except Exception as e:
                QMessageBox.critical(self, "Fehler", f"Fehler beim Speichern: {str(e)}")
    
    def load_config(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Konfiguration laden", "", "JSON (*.json)"
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    config = json.load(f)
                
                # Konfiguration laden
                self.model_path = config.get('model_path', '')
                self.class_config = config.get('class_config', {})
                self.display_config = config.get('display_config', {})
                self.video_files = config.get('video_files', [])
                
                # Farben konvertieren
                for cls_id, cfg in self.class_config.items():
                    if 'color' in cfg and isinstance(cfg['color'], list):
                        cfg['color'] = tuple(cfg['color'])
                
                # UI aktualisieren
                self.txt_model_path.setText(self.model_path)
                if self.model_path:
                    model_name = os.path.basename(self.model_path)
                    self.lbl_model_info.setText(f"Modell: {model_name}")
                
                self.box_thickness.setValue(self.display_config.get('box_thickness', 2))
                self.font_scale.setValue(self.display_config.get('font_scale', 5))
                self.text_thickness.setValue(self.display_config.get('text_thickness', 1))
                self.speed_slider.setValue(self.display_config.get('playback_speed', 30))
                
                # Videos neu laden
                self.video_list.clear()
                for video_file in self.video_files:
                    item = QListWidgetItem(os.path.basename(video_file))
                    item.setToolTip(video_file)
                    self.video_list.addItem(item)
                
                self.load_class_table()
                self.update_alarm_classes()
                
                QMessageBox.information(self, "Erfolg", "Konfiguration wurde geladen.")
            except Exception as e:
                QMessageBox.critical(self, "Fehler", f"Fehler beim Laden: {str(e)}")

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Video Annotator")
        
        # Vollbild-Modus aktivieren
        self.showMaximized()
        
        # Thread pool for parallel processing
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(4)
        
        # Initial default values
        self.model = None
        self.model_path = ""
        self.class_config = {}
        self.display_config = {
            'box_thickness': 2,
            'font_scale': 5,
            'text_thickness': 1,
            'playback_speed': 30,
            'alarm_class': None
        }
        self.video_files = []
        
        # Setup UI
        self.setup_ui()
        self.connect_signals()
        
        # Video playback
        self.cap = None
        self.current_video_idx = 0
        self.current_frame = None
        self.processing_frame = False
        self.last_detections = []
        
        # Status
        self.alarm_active = False
        self.pulse_value = 0
        self.pulse_direction = 1
        
        # Setup timers
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        
        self.alarm_timer = QTimer()
        self.alarm_timer.timeout.connect(self.pulse_alarm)
        
        # Load default config
        self.load_default_config()
    
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        # Toolbar
        toolbar_layout = QHBoxLayout()
        
        self.btn_settings = QPushButton("⚙ Einstellungen")
        self.btn_settings.setStyleSheet("padding: 8px; font-size: 14px;")
        
        self.btn_play_pause = QPushButton("▶ Abspielen")
        self.btn_play_pause.setStyleSheet("padding: 8px; font-size: 14px;")
        self.btn_play_pause.setEnabled(False)
        
        self.lbl_status = QLabel("Bereit")
        self.lbl_status.setStyleSheet("color: #666; font-size: 12px;")
        
        toolbar_layout.addWidget(self.btn_settings)
        toolbar_layout.addWidget(self.btn_play_pause)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.lbl_status)
        
        main_layout.addLayout(toolbar_layout)
        
        # Video display
        self.video_container = QFrame()
        self.video_container.setStyleSheet("background-color: #222; border-radius: 5px;")
        video_layout = QVBoxLayout(self.video_container)
        
        self.label = QLabel("Bitte wählen Sie Videos und ein YOLO-Modell in den Einstellungen aus")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("color: #aaa; font-size: 16px;")
        video_layout.addWidget(self.label)
        
        main_layout.addWidget(self.video_container, 1)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(3)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        main_layout.addWidget(self.progress)
    
    def connect_signals(self):
        self.btn_settings.clicked.connect(self.open_settings)
        self.btn_play_pause.clicked.connect(self.toggle_playback)
    
    def open_settings(self):
        dialog = SettingsDialog(
            self.model_path, 
            self.class_config, 
            self.display_config,
            self.video_files,
            self
        )
        
        if dialog.exec():
            settings = dialog.get_settings()
            
            self.model_path = settings['model_path']
            self.class_config = settings['class_config']
            self.display_config = settings['display_config']
            self.video_files = settings['video_files']
            
            # Modell neu laden falls geändert
            if self.model_path:
                try:
                    self.model = YOLO(self.model_path)
                    model_name = os.path.basename(self.model_path)
                    self.lbl_status.setText(f"Modell geladen: {model_name}")
                except Exception as e:
                    QMessageBox.critical(self, "Fehler", f"Modell konnte nicht geladen werden: {str(e)}")
                    return
            
            # Videos vorbereiten
            if self.video_files:
                self.current_video_idx = 0
                self.btn_play_pause.setEnabled(True)
                self.label.setText("Bereit zum Abspielen")
                video_count = len(self.video_files)
                self.lbl_status.setText(f"Modell geladen, {video_count} Video(s) bereit")
            else:
                self.btn_play_pause.setEnabled(False)
                self.label.setText("Keine Videos ausgewählt")
    
    def start_video(self):
        if not self.model:
            QMessageBox.warning(self, "Warnung", "Bitte wählen Sie zuerst ein YOLO-Modell aus.")
            return
            
        if not self.video_files:
            QMessageBox.warning(self, "Warnung", "Bitte wählen Sie Videos aus.")
            return
        
        if self.cap:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(self.video_files[self.current_video_idx])
        
        fps = self.display_config.get('playback_speed', 30)
        delay = int(1000 / fps)
        self.timer.setInterval(delay)
        self.timer.start()
        
        self.btn_play_pause.setText("⏸ Pausieren")
        
        # Status aktualisieren
        video_name = os.path.basename(self.video_files[self.current_video_idx])
        current_num = self.current_video_idx + 1
        total_num = len(self.video_files)
        self.setWindowTitle(f"YOLO Video Annotator - {video_name} ({current_num}/{total_num})")
        self.lbl_status.setText(f"Spielt ab: {video_name} ({current_num}/{total_num})")
    
    def toggle_playback(self):
        if not self.cap:
            self.start_video()
            return
            
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play_pause.setText("▶ Abspielen")
            self.lbl_status.setText("Pausiert")
        else:
            self.timer.start()
            self.btn_play_pause.setText("⏸ Pausieren")
            video_name = os.path.basename(self.video_files[self.current_video_idx])
            self.lbl_status.setText(f"Spielt ab: {video_name}")
    
    def pulse_alarm(self):
        """Animation für den Alarmzustand - ohne Border für stabiles Layout"""
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
        """Process detection results from the worker thread"""
        if not result:
            return
            
        frame, detections = result
        self.current_frame = frame
        self.last_detections = detections
        
        # Check for alarm condition
        alarm_class_id = self.display_config.get('alarm_class')
        if alarm_class_id:
            alarm_triggered = any(str(d['class_id']) == alarm_class_id for d in detections)
            
            if alarm_triggered and not self.alarm_active:
                self.alarm_active = True
                self.alarm_timer.start(50)
            elif not alarm_triggered:
                self.alarm_active = False
                
        # Render the frame with detections
        self.render_frame()
        
        # Release the processing lock
        self.processing_frame = False
    
    def render_frame(self):
        """Draw detections on the frame and display it"""
        if self.current_frame is None:
            return
            
        # Create a copy to draw on
        frame = self.current_frame.copy()
        
        # Get display settings
        box_thickness = self.display_config.get('box_thickness', 2)
        font_scale = self.display_config.get('font_scale', 5) / 10.0
        text_thickness = self.display_config.get('text_thickness', 1)
        
        # Draw detections
        for detection in self.last_detections:
            cls_id = str(detection['class_id'])
            cfg = self.class_config.get(cls_id)
            if not cfg:
                continue
                
            box = detection['box']
            conf = detection['conf']
            
            color = cfg['color']
            label = f"{cfg['name']} {conf:.2f}"
            
            # Draw bounding box
            cv2.rectangle(frame, (box['x1'], box['y1']), (box['x2'], box['y2']), color, box_thickness)
            
            # Draw label
            cv2.putText(frame, label, (box['x1'], box['y1'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness)
        
        # Convert to Qt format for display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.AspectRatioMode.KeepAspectRatio))
    
    def next_frame(self):
        """Get the next video frame and process it"""
        if not self.cap or self.processing_frame:
            return
            
        # Get frame
        ret, frame = self.cap.read()
        if not ret:
            # End of current video, go to next one in endless loop
            self.current_video_idx = (self.current_video_idx + 1) % len(self.video_files)
            
            # Update status for next video
            video_name = os.path.basename(self.video_files[self.current_video_idx])
            current_num = self.current_video_idx + 1
            total_num = len(self.video_files)
            self.setWindowTitle(f"YOLO Video Annotator - {video_name} ({current_num}/{total_num})")
            self.lbl_status.setText(f"Spielt ab: {video_name} ({current_num}/{total_num})")
            
            # Start next video
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.video_files[self.current_video_idx])
            return
        
        # Set processing lock
        self.processing_frame = True
        
        # Submit to thread pool
        worker = DetectionWorker(frame, self.model, self.class_config)
        worker.signals.result.connect(self.handle_detection_result)
        worker.signals.error.connect(lambda err: print(f"Error: {err}"))
        self.threadpool.start(worker)
    
    def load_default_config(self):
        """Load default configuration if config.json exists"""
        config_path = 'config.json'
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                self.model_path = config.get('model_path', '')
                self.class_config = config.get('class_config', {})
                
                # Display config aus alter Struktur migrieren
                display = config.get('display', {})
                self.display_config = {
                    'box_thickness': display.get('box_thickness', 2),
                    'font_scale': display.get('font_scale', 5),
                    'text_thickness': display.get('text_thickness', 1),
                    'playback_speed': display.get('playback_speed', 30),
                    'alarm_class': display.get('alarm_class', None)
                }
                
                # Video files aus Konfiguration laden (falls vorhanden)
                self.video_files = config.get('video_files', [])
                
                # Farben konvertieren
                for cls_id, cfg in self.class_config.items():
                    if 'color' in cfg and isinstance(cfg['color'], list):
                        cfg['color'] = tuple(cfg['color'])
                
                # Modell laden falls Pfad vorhanden
                if self.model_path and os.path.exists(self.model_path):
                    try:
                        self.model = YOLO(self.model_path)
                        model_name = os.path.basename(self.model_path)
                        self.lbl_status.setText(f"Modell geladen: {model_name}")
                        
                        if self.video_files:
                            self.btn_play_pause.setEnabled(True)
                            video_count = len(self.video_files)
                            self.lbl_status.setText(f"Modell geladen, {video_count} Video(s) bereit")
                            self.label.setText("Bereit zum Abspielen")
                    except Exception as e:
                        print(f"Fehler beim Laden des Modells: {e}")
                        
            except Exception as e:
                print(f"Fehler beim Laden der Konfiguration: {e}")
                # Initialisiere mit Standard-Konfiguration
                self.class_config = {
                    "0": {"name": "GEFAHR", "color": COLORS["Red"], "conf": 0.5, "iou": 0.4},
                    "1": {"name": "Chair", "color": COLORS["Green"], "conf": 0.6, "iou": 0.4},
                    "2": {"name": "Human", "color": COLORS["Blue"], "conf": 0.6, "iou": 0.4}
                }
    
    def save_config(self):
        """Save current configuration to config.json"""
        config = {
            'model_path': self.model_path,
            'class_config': self.class_config,
            'display_config': self.display_config,
            'video_files': self.video_files
        }
        
        try:
            with open('config.json', 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Fehler beim Speichern der Konfiguration: {e}")
    
    def closeEvent(self, event):
        # Clean up resources
        self.timer.stop()
        self.alarm_timer.stop()
        if self.cap:
            self.cap.release()
            
        # Save config on exit
        self.save_config()
        event.accept()

if __name__ == "__main__":
   app = QApplication(sys.argv)
   player = VideoPlayer()
   player.show()
   sys.exit(app.exec())        