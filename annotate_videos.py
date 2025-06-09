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
    QDialog, QGridLayout, QListWidget, QListWidgetItem, QCheckBox
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

# Standard Pose-Verbindungen für YOLO Pose (17 Keypoints)
POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Kopf
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arme
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Beine
]

class WorkerSignals(QObject):
    """Defines the signals available from the worker thread."""
    result = pyqtSignal(object)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

class DualDetectionWorker(QRunnable):
    """Worker thread for processing video frames with detection first, then pose on detected objects."""
    def __init__(self, frame, detection_model, pose_model, class_config, pose_config):
        super().__init__()
        self.frame = frame.copy()
        self.detection_model = detection_model
        self.pose_model = pose_model
        self.class_config = class_config
        self.pose_config = pose_config
        self.signals = WorkerSignals()
        
    def run(self):
        try:
            detections = []
            poses = []
            
            # Step 1: Object Detection
            if self.detection_model:
                det_results = self.detection_model.predict(self.frame, verbose=False)[0]
                
                pose_detect_class = self.pose_config.get('pose_detect_class')
                
                for box in det_results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    cls_id = str(cls)
                    cfg = self.class_config.get(cls_id)
                    if not cfg or conf < float(cfg.get("conf", 0.5)):
                        continue
                    
                    detection = {
                        'box': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                        'conf': conf,
                        'class_id': cls,
                        'class_name': cfg.get('name', f"Class {cls}")
                    }
                    detections.append(detection)
                    
                    # Step 2: Pose Detection auf ausgeschnittenen Bereichen
                    if (self.pose_model and pose_detect_class and 
                        cls_id == pose_detect_class):
                        
                        # Bounding Box erweitern und beschränken
                        h, w = self.frame.shape[:2]
                        margin = 20  # Pixel Spielraum um die Box
                        x1_exp = max(0, x1 - margin)
                        y1_exp = max(0, y1 - margin)
                        x2_exp = min(w, x2 + margin)
                        y2_exp = min(h, y2 + margin)
                        
                        # Ausschnitt extrahieren
                        roi = self.frame[y1_exp:y2_exp, x1_exp:x2_exp]
                        
                        if roi.shape[0] > 0 and roi.shape[1] > 0:
                            # Pose Detection auf ROI
                            pose_results = self.pose_model.predict(roi, verbose=False)[0]
                            
                            if hasattr(pose_results, 'keypoints') and pose_results.keypoints is not None:
                                for person_idx, keypoints in enumerate(pose_results.keypoints.xy):
                                    if pose_results.keypoints.conf is not None:
                                        confs = pose_results.keypoints.conf[person_idx]
                                    else:
                                        confs = [1.0] * len(keypoints)
                                    
                                    # Filter keypoints by confidence
                                    min_conf = self.pose_config.get('min_confidence', 0.3)
                                    valid_keypoints = []
                                    
                                    for i, (kp, conf_kp) in enumerate(zip(keypoints, confs)):
                                        if conf_kp >= min_conf and kp[0] > 0 and kp[1] > 0:
                                            # Koordinaten zurück ins Vollbild transformieren
                                            global_x = float(kp[0]) + x1_exp
                                            global_y = float(kp[1]) + y1_exp
                                            
                                            valid_keypoints.append({
                                                'id': i,
                                                'x': global_x,
                                                'y': global_y,
                                                'conf': float(conf_kp)
                                            })
                                    
                                    if valid_keypoints:  # Nur hinzufügen wenn gültige Keypoints
                                        poses.append({
                                            'person_id': f"{cls_id}_{person_idx}",
                                            'detection_box': detection['box'],
                                            'keypoints': valid_keypoints
                                        })
            
            # Emit the result
            self.signals.result.emit((self.frame, detections, poses))
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()

class SettingsDialog(QDialog):
    """Großer übersichtlicher Dialog für alle Einstellungen."""
    def __init__(self, detection_model_path, pose_model_path, class_config, pose_config, display_config, video_files, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Einstellungen")
        self.resize(1200, 800)
        
        self.detection_model_path = detection_model_path
        self.pose_model_path = pose_model_path
        self.class_config = class_config.copy()
        self.pose_config = pose_config.copy()
        self.display_config = display_config.copy()
        self.video_files = video_files.copy()
        
        self.setup_ui()
        self.load_settings()
    
    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        
        # Linke Seite: Modelle und Videos
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Detection Modell-Auswahl
        detection_model_group = QGroupBox("YOLO Detection Modell")
        detection_model_layout = QVBoxLayout(detection_model_group)
        
        detection_model_select_layout = QHBoxLayout()
        self.txt_detection_model_path = QLineEdit(self.detection_model_path)
        self.txt_detection_model_path.setReadOnly(True)
        self.btn_select_detection_model = QPushButton("Detection Modell auswählen")
        detection_model_select_layout.addWidget(self.txt_detection_model_path, 3)
        detection_model_select_layout.addWidget(self.btn_select_detection_model, 1)
        detection_model_layout.addLayout(detection_model_select_layout)
        
        self.lbl_detection_model_info = QLabel("Kein Detection Modell geladen")
        detection_model_layout.addWidget(self.lbl_detection_model_info)
        
        left_layout.addWidget(detection_model_group)
        
        # Pose Modell-Auswahl
        pose_model_group = QGroupBox("YOLO Pose Modell")
        pose_model_layout = QVBoxLayout(pose_model_group)
        
        pose_model_select_layout = QHBoxLayout()
        self.txt_pose_model_path = QLineEdit(self.pose_model_path)
        self.txt_pose_model_path.setReadOnly(True)
        self.btn_select_pose_model = QPushButton("Pose Modell auswählen")
        pose_model_select_layout.addWidget(self.txt_pose_model_path, 3)
        pose_model_select_layout.addWidget(self.btn_select_pose_model, 1)
        pose_model_layout.addLayout(pose_model_select_layout)
        
        self.lbl_pose_model_info = QLabel("Kein Pose Modell geladen")
        pose_model_layout.addWidget(self.lbl_pose_model_info)
        
        left_layout.addWidget(pose_model_group)
        
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
        
        # Pose-Einstellungen
        pose_settings_group = QGroupBox("Pose-Einstellungen")
        pose_form = QFormLayout(pose_settings_group)
        
        self.pose_detect_class_dropdown = QComboBox()
        pose_form.addRow("Pose Detect Klasse:", self.pose_detect_class_dropdown)
        
        self.pose_min_confidence = QDoubleSpinBox()
        self.pose_min_confidence.setRange(0.1, 1.0)
        self.pose_min_confidence.setSingleStep(0.05)
        self.pose_min_confidence.setValue(self.pose_config.get('min_confidence', 0.3))
        pose_form.addRow("Min. Pose Konfidenz:", self.pose_min_confidence)
        
        self.pose_line_thickness = QSpinBox()
        self.pose_line_thickness.setRange(1, 10)
        self.pose_line_thickness.setValue(self.pose_config.get('line_thickness', 2))
        pose_form.addRow("Skelett-Liniendicke:", self.pose_line_thickness)
        
        self.pose_keypoint_radius = QSpinBox()
        self.pose_keypoint_radius.setRange(1, 20)
        self.pose_keypoint_radius.setValue(self.pose_config.get('keypoint_radius', 3))
        pose_form.addRow("Keypoint-Radius:", self.pose_keypoint_radius)
        
        self.pose_show_keypoints = QCheckBox()
        self.pose_show_keypoints.setChecked(self.pose_config.get('show_keypoints', True))
        pose_form.addRow("Keypoints anzeigen:", self.pose_show_keypoints)
        
        self.pose_show_skeleton = QCheckBox()
        self.pose_show_skeleton.setChecked(self.pose_config.get('show_skeleton', True))
        pose_form.addRow("Skelett anzeigen:", self.pose_show_skeleton)
        
        left_layout.addWidget(pose_settings_group)
        left_layout.addStretch()
        
        # Rechte Seite: Klasseneinstellungen
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        classes_group = QGroupBox("Detection Klasseneinstellungen")
        classes_layout = QVBoxLayout(classes_group)
        
        # Tabelle für Klasseneinstellungen
        self.class_table = QTableWidget()
        self.class_table.setColumnCount(5)  # Name, Farbe, Konfidenz, IoU, Aktiv
        self.class_table.setHorizontalHeaderLabels(['Klassen-ID', 'Name', 'Farbe', 'Konfidenz', 'IoU'])
        self.class_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        classes_layout.addWidget(self.class_table)
        
        note_label = QLabel("Hinweis: Klassen werden automatisch aus dem YOLO-Detection-Modell geladen.")
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
        self.btn_select_detection_model.clicked.connect(self.select_detection_model)
        self.btn_select_pose_model.clicked.connect(self.select_pose_model)
        self.btn_add_videos.clicked.connect(self.add_videos)
        self.btn_remove_video.clicked.connect(self.remove_video)
        self.btn_clear_videos.clicked.connect(self.clear_videos)
        self.speed_slider.valueChanged.connect(lambda v: self.speed_label.setText(str(v)))
        
        self.btn_save.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_load_config.clicked.connect(self.load_config)
        self.btn_save_config.clicked.connect(self.save_config)
    
    def select_detection_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "YOLO-Detection-Modell auswählen", "", "YOLO Models (*.pt)"
        )
        if file_path:
            try:
                model = YOLO(file_path)
                self.detection_model_path = file_path
                self.txt_detection_model_path.setText(file_path)
                
                model_name = os.path.basename(file_path)
                self.lbl_detection_model_info.setText(f"Detection Modell: {model_name}")
                
                # Klassen aus Modell extrahieren
                self.extract_model_classes(model)
                
            except Exception as e:
                QMessageBox.critical(self, "Fehler", f"Detection Modell konnte nicht geladen werden: {str(e)}")
    
    def select_pose_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "YOLO-Pose-Modell auswählen", "", "YOLO Models (*.pt)"
        )
        if file_path:
            try:
                model = YOLO(file_path)
                self.pose_model_path = file_path
                self.txt_pose_model_path.setText(file_path)
                
                model_name = os.path.basename(file_path)
                self.lbl_pose_model_info.setText(f"Pose Modell: {model_name}")
                
            except Exception as e:
                QMessageBox.critical(self, "Fehler", f"Pose Modell konnte nicht geladen werden: {str(e)}")
    
    def extract_model_classes(self, model):
        """Extract class information from the loaded YOLO detection model"""
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
        self.update_pose_detect_classes()
    
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
        self.alarm_class_dropdown.addItem("Keine Alarmklasse", userData=None)
        
        for cls_id, cfg in self.class_config.items():
            self.alarm_class_dropdown.addItem(f"{cls_id}: {cfg['name']}", userData=cls_id)
        
        # Setze die gespeicherte Auswahl zurück
        if current_data is not None:
            index = self.alarm_class_dropdown.findData(current_data)
            if index >= 0:
                self.alarm_class_dropdown.setCurrentIndex(index)
        elif self.display_config.get('alarm_class') is not None:
            index = self.alarm_class_dropdown.findData(self.display_config.get('alarm_class'))
            if index >= 0:
                self.alarm_class_dropdown.setCurrentIndex(index)
    
    def update_pose_detect_classes(self):
        current_data = self.pose_detect_class_dropdown.currentData()
        self.pose_detect_class_dropdown.clear()
        self.pose_detect_class_dropdown.addItem("Keine Pose Detection", userData=None)
        
        for cls_id, cfg in self.class_config.items():
            self.pose_detect_class_dropdown.addItem(f"{cls_id}: {cfg['name']}", userData=cls_id)
        
        # Setze die gespeicherte Auswahl zurück
        if current_data is not None:
            index = self.pose_detect_class_dropdown.findData(current_data)
            if index >= 0:
                self.pose_detect_class_dropdown.setCurrentIndex(index)
        elif self.pose_config.get('pose_detect_class') is not None:
            index = self.pose_detect_class_dropdown.findData(self.pose_config.get('pose_detect_class'))
            if index >= 0:
                self.pose_detect_class_dropdown.setCurrentIndex(index)
    
    def load_settings(self):
        # Videos laden
        for video_file in self.video_files:
            item = QListWidgetItem(os.path.basename(video_file))
            item.setToolTip(video_file)
            self.video_list.addItem(item)
        
        # Klassen laden
        self.load_class_table()
        self.update_alarm_classes()
        self.update_pose_detect_classes()
        
        # Modell-Info aktualisieren
        if self.detection_model_path:
            model_name = os.path.basename(self.detection_model_path)
            self.lbl_detection_model_info.setText(f"Detection Modell: {model_name}")
            
        if self.pose_model_path:
            model_name = os.path.basename(self.pose_model_path)
            self.lbl_pose_model_info.setText(f"Pose Modell: {model_name}")
    
    def get_settings(self):
        # WICHTIG: Klasseneinstellungen aus Tabelle sammeln VOR dem Speichern
        updated_class_config = {}
        for row in range(self.class_table.rowCount()):
            cls_id = self.class_table.item(row, 0).text()
            name = self.class_table.item(row, 1).text()
            
            color_combo = self.class_table.cellWidget(row, 2)
            color_name = color_combo.currentText()
            color = COLORS[color_name]
            
            conf = self.class_table.cellWidget(row, 3).value()
            iou = self.class_table.cellWidget(row, 4).value()
            
            updated_class_config[cls_id] = {
                'name': name,
                'color': color,
                'conf': conf,
                'iou': iou
            }
        
        # Display-Einstellungen sammeln
        display_config = {
            'box_thickness': self.box_thickness.value(),
            'font_scale': self.font_scale.value(),
            'text_thickness': self.text_thickness.value(),
            'playback_speed': self.speed_slider.value(),
            'alarm_class': self.alarm_class_dropdown.currentData()
        }
        
        # Pose-Einstellungen sammeln
        pose_config = {
            'pose_detect_class': self.pose_detect_class_dropdown.currentData(),
            'min_confidence': self.pose_min_confidence.value(),
            'line_thickness': self.pose_line_thickness.value(),
            'keypoint_radius': self.pose_keypoint_radius.value(),
            'show_keypoints': self.pose_show_keypoints.isChecked(),
            'show_skeleton': self.pose_show_skeleton.isChecked()
        }
        
        return {
            'detection_model_path': self.detection_model_path,
            'pose_model_path': self.pose_model_path,
            'class_config': updated_class_config,
            'pose_config': pose_config,
            'display_config': display_config,
            'video_files': self.video_files
        }
    
    def save_config(self):
        # Immer in die aktuelle config.json speichern
        config_path = self.find_or_create_config_file()
        try:
            config = self.get_settings()
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "Erfolg", f"Konfiguration wurde in {config_path} gespeichert.")
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Fehler beim Speichern: {str(e)}")
    
    def load_config(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Konfiguration laden", "", "JSON (*.json)"
        )
        if file_path:
            self._load_config_from_file(file_path)
    
    def _load_config_from_file(self, file_path):
        """Lädt Konfiguration aus der angegebenen Datei"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Konfiguration laden mit Fallback-Werten
            self.detection_model_path = config.get('detection_model_path', config.get('model_path', ''))
            self.pose_model_path = config.get('pose_model_path', '')
            self.class_config = config.get('class_config', {})
            self.pose_config = config.get('pose_config', {
                'pose_detect_class': None,
                'min_confidence': 0.3,
                'line_thickness': 2,
                'keypoint_radius': 3,
                'show_keypoints': True,
                'show_skeleton': True
            })
            
            # Display config aus alter Struktur migrieren
            display = config.get('display_config', config.get('display', {}))
            self.display_config = {
                'box_thickness': display.get('box_thickness', 2),
                'font_scale': display.get('font_scale', 5),
                'text_thickness': display.get('text_thickness', 1),
                'playback_speed': display.get('playback_speed', 30),
                'alarm_class': display.get('alarm_class', None)
            }
            
            self.video_files = config.get('video_files', [])
            
            # Farben konvertieren
            for cls_id, cfg in self.class_config.items():
                if 'color' in cfg and isinstance(cfg['color'], list):
                    cfg['color'] = tuple(cfg['color'])
            
            # UI aktualisieren
            self.txt_detection_model_path.setText(self.detection_model_path)
            self.txt_pose_model_path.setText(self.pose_model_path)
            
            if self.detection_model_path:
                model_name = os.path.basename(self.detection_model_path)
                self.lbl_detection_model_info.setText(f"Detection Modell: {model_name}")
            
            if self.pose_model_path:
                model_name = os.path.basename(self.pose_model_path)
                self.lbl_pose_model_info.setText(f"Pose Modell: {model_name}")
            
            # Display-Einstellungen
            self.box_thickness.setValue(self.display_config.get('box_thickness', 2))
            self.font_scale.setValue(self.display_config.get('font_scale', 5))
            self.text_thickness.setValue(self.display_config.get('text_thickness', 1))
            self.speed_slider.setValue(self.display_config.get('playback_speed', 30))
            
            # Pose-Einstellungen
            self.pose_min_confidence.setValue(self.pose_config.get('min_confidence', 0.3))
            self.pose_line_thickness.setValue(self.pose_config.get('line_thickness', 2))
            self.pose_keypoint_radius.setValue(self.pose_config.get('keypoint_radius', 3))
            self.pose_show_keypoints.setChecked(self.pose_config.get('show_keypoints', True))
            self.pose_show_skeleton.setChecked(self.pose_config.get('show_skeleton', True))
            
            # Videos neu laden
            self.video_list.clear()
            for video_file in self.video_files:
                if os.path.exists(video_file):  # Nur existierende Dateien
                    item = QListWidgetItem(os.path.basename(video_file))
                    item.setToolTip(video_file)
                    self.video_list.addItem(item)
                else:
                    # Entferne nicht existierende Videos aus der Liste
                    self.video_files.remove(video_file)
            
            self.load_class_table()
            self.update_alarm_classes()
            self.update_pose_detect_classes()
            
            QMessageBox.information(self, "Erfolg", f"Konfiguration wurde aus {file_path} geladen.")
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Fehler beim Laden: {str(e)}")
    
    def find_or_create_config_file(self):
        """Findet oder erstellt eine config.json Datei im App-Verzeichnis"""
        app_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(app_dir, "config.json")
        
        # Wenn keine config.json existiert, erstelle eine
        if not os.path.exists(config_path):
            default_config = {
                'detection_model_path': '',
                'pose_model_path': '',
                'class_config': {},
                'pose_config': {
                    'pose_detect_class': None,
                    'min_confidence': 0.3,
                    'line_thickness': 2,
                    'keypoint_radius': 3,
                    'show_keypoints': True,
                    'show_skeleton': True
                },
                'display_config': {
                    'box_thickness': 2,
                    'font_scale': 5,
                    'text_thickness': 1,
                    'playback_speed': 30,
                    'alarm_class': None
                },
                'video_files': []
            }
            try:
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2)
            except Exception as e:
                print(f"Konnte config.json nicht erstellen: {e}")
        
        return config_path

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Dual Model Video Annotator")
        
        # Vollbild-Modus aktivieren
        self.showMaximized()
        
        # Thread pool for parallel processing
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(4)
        
        # Initial default values
        self.detection_model = None
        self.pose_model = None
        self.detection_model_path = ""
        self.pose_model_path = ""
        self.class_config = {}
        self.pose_config = {
            'pose_detect_class': None,
            'min_confidence': 0.3,
            'line_thickness': 2,
            'keypoint_radius': 3,
            'show_keypoints': True,
            'show_skeleton': True
        }
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
        self.last_poses = []
        
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
        
        self.label = QLabel("Bitte wählen Sie Videos und YOLO-Modelle in den Einstellungen aus")
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
            self.detection_model_path,
            self.pose_model_path,
            self.class_config,
            self.pose_config,
            self.display_config,
            self.video_files,
            self
        )
        
        if dialog.exec():
            settings = dialog.get_settings()
            
            # WICHTIG: Alle Einstellungen übernehmen
            self.detection_model_path = settings['detection_model_path']
            self.pose_model_path = settings['pose_model_path']
            self.class_config = settings['class_config']  # Übernehme aktualisierte Klassen
            self.pose_config = settings['pose_config']
            self.display_config = settings['display_config']
            self.video_files = settings['video_files']
            
            # Detection Modell neu laden falls geändert
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
            
            # Pose Modell neu laden falls geändert
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
            
            # Status aktualisieren
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
            
            # Automatisch speichern
            self.save_config()
    
    def start_video(self):
        if not (self.detection_model or self.pose_model):
            QMessageBox.warning(self, "Warnung", "Bitte wählen Sie mindestens ein YOLO-Modell aus.")
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
        self.setWindowTitle(f"YOLO Dual Model Video Annotator - {video_name} ({current_num}/{total_num})")
        
        status_parts = []
        if self.detection_model:
            status_parts.append(f"Detection: {os.path.basename(self.detection_model_path)}")
        if self.pose_model:
            status_parts.append(f"Pose: {os.path.basename(self.pose_model_path)}")
        status_parts.append(f"Video: {video_name} ({current_num}/{total_num})")
        
        self.lbl_status.setText(" | ".join(status_parts))
    
    def toggle_playback(self):
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
            
        frame, detections, poses = result
        self.current_frame = frame
        self.last_detections = detections
        self.last_poses = poses
        
        # Check for alarm condition
        alarm_class_id = self.display_config.get('alarm_class')
        if alarm_class_id:
            alarm_triggered = any(str(d['class_id']) == alarm_class_id for d in detections)
            
            if alarm_triggered and not self.alarm_active:
                self.alarm_active = True
                self.alarm_timer.start(50)
            elif not alarm_triggered:
                self.alarm_active = False
                
        # Render the frame with detections and poses
        self.render_frame()
        
        # Release the processing lock
        self.processing_frame = False
    
    def render_frame(self):
        """Draw detections and poses on the frame and display it"""
        if self.current_frame is None:
            return
            
        # Create a copy to draw on
        frame = self.current_frame.copy()
        
        # Get display settings
        box_thickness = self.display_config.get('box_thickness', 2)
        font_scale = self.display_config.get('font_scale', 5) / 10.0
        text_thickness = self.display_config.get('text_thickness', 1)
        
        # Draw detection boxes
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
        
        # Draw poses - FIX: Überprüfe auf gültige Keypoints
        if self.pose_config.get('show_keypoints', True) or self.pose_config.get('show_skeleton', True):
            line_thickness = self.pose_config.get('line_thickness', 2)
            keypoint_radius = self.pose_config.get('keypoint_radius', 3)
            show_keypoints = self.pose_config.get('show_keypoints', True)
            show_skeleton = self.pose_config.get('show_skeleton', True)
            
            for pose in self.last_poses:
                keypoints = pose['keypoints']
                
                if not keypoints:  # Skip wenn keine gültigen Keypoints
                    continue
                
                # Erstelle Keypoint-Array für einfacheren Zugriff
                kp_array = [None] * 17  # COCO hat 17 Keypoints
                for kp in keypoints:
                    if kp['id'] < 17 and kp['x'] > 0 and kp['y'] > 0:  # Gültige Koordinaten
                        kp_array[kp['id']] = (int(kp['x']), int(kp['y']))
                
                # Zeichne Skelett-Verbindungen - FIX: Validiere Verbindungen
                if show_skeleton:
                    for connection in POSE_CONNECTIONS:
                        pt1_idx, pt2_idx = connection
                        if (pt1_idx < len(kp_array) and pt2_idx < len(kp_array) and 
                            kp_array[pt1_idx] is not None and kp_array[pt2_idx] is not None):
                            
                            # Zusätzliche Validierung der Koordinaten
                            pt1 = kp_array[pt1_idx]
                            pt2 = kp_array[pt2_idx]
                            
                            if (pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0 and
                                pt1[0] < frame.shape[1] and pt1[1] < frame.shape[0] and
                                pt2[0] < frame.shape[1] and pt2[1] < frame.shape[0]):
                                cv2.line(frame, pt1, pt2, (0, 255, 0), line_thickness)
                
                # Zeichne Keypoints - FIX: Validiere Koordinaten
                if show_keypoints:
                    for kp in keypoints:
                        x, y = int(kp['x']), int(kp['y'])
                        if (x > 0 and y > 0 and 
                            x < frame.shape[1] and y < frame.shape[0]):  # Innerhalb Bildgrenzen
                            cv2.circle(frame, (x, y), keypoint_radius, (0, 0, 255), -1)
        
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
            self.setWindowTitle(f"YOLO Dual Model Video Annotator - {video_name} ({current_num}/{total_num})")
            
            status_parts = []
            if self.detection_model:
                status_parts.append(f"Detection: {os.path.basename(self.detection_model_path)}")
            if self.pose_model:
                status_parts.append(f"Pose: {os.path.basename(self.pose_model_path)}")
            status_parts.append(f"Video: {video_name} ({current_num}/{total_num})")
            
            self.lbl_status.setText(" | ".join(status_parts))
            
            # Start next video
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.video_files[self.current_video_idx])
            return
        
        # Set processing lock
        self.processing_frame = True
        
        # Submit to thread pool
        worker = DualDetectionWorker(frame, self.detection_model, self.pose_model, 
                                   self.class_config, self.pose_config)
        worker.signals.result.connect(self.handle_detection_result)
        worker.signals.error.connect(lambda err: print(f"Error: {err}"))
        self.threadpool.start(worker)
    
    def find_or_create_config_file(self):
        """Findet oder erstellt eine config.json Datei im App-Verzeichnis"""
        app_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(app_dir, "config.json")
        
        # Schaue auch nach anderen JSON-Dateien im Verzeichnis
        if not os.path.exists(config_path):
            json_files = [f for f in os.listdir(app_dir) if f.endswith('.json')]
            
            # Versuche eine passende Konfigurationsdatei zu finden
            for json_file in json_files:
                try:
                    with open(os.path.join(app_dir, json_file), 'r', encoding='utf-8') as f:
                        content = json.load(f)
                        # Prüfe ob es eine YOLO-Konfigurationsdatei ist
                        if any(key in content for key in ['model_path', 'detection_model_path', 'class_config']):
                            config_path = os.path.join(app_dir, json_file)
                            print(f"Gefundene Konfigurationsdatei: {config_path}")
                            break
                except:
                    continue
        
        # Wenn keine config.json existiert, erstelle eine
        if not os.path.exists(config_path):
            default_config = {
                'detection_model_path': '',
                'pose_model_path': '',
                'class_config': {},
                'pose_config': {
                    'pose_detect_class': None,
                    'min_confidence': 0.3,
                    'line_thickness': 2,
                    'keypoint_radius': 3,
                    'show_keypoints': True,
                    'show_skeleton': True
                },
                'display_config': {
                    'box_thickness': 2,
                    'font_scale': 5,
                    'text_thickness': 1,
                    'playback_speed': 30,
                    'alarm_class': None
                },
                'video_files': []
            }
            try:
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2)
                print(f"Neue Konfigurationsdatei erstellt: {config_path}")
            except Exception as e:
                print(f"Konnte config.json nicht erstellen: {e}")
        
        return config_path
    
    def load_default_config(self):
        """Load default configuration if config.json exists or create one"""
        config_path = self.find_or_create_config_file()
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Konfiguration laden mit Fallback-Werten und Migration von alten Formaten
                self.detection_model_path = config.get('detection_model_path', config.get('model_path', ''))
                self.pose_model_path = config.get('pose_model_path', '')
                self.class_config = config.get('class_config', {})
                
                # Pose config mit Standardwerten
                self.pose_config = config.get('pose_config', {
                    'pose_detect_class': None,
                    'min_confidence': 0.3,
                    'line_thickness': 2,
                    'keypoint_radius': 3,
                    'show_keypoints': True,
                    'show_skeleton': True
                })
                
                # Display config aus alter Struktur migrieren
                display = config.get('display_config', config.get('display', {}))
                self.display_config = {
                    'box_thickness': display.get('box_thickness', 2),
                    'font_scale': display.get('font_scale', 5),
                    'text_thickness': display.get('text_thickness', 1),
                    'playback_speed': display.get('playback_speed', 30),
                    'alarm_class': display.get('alarm_class', None)
                }
                
                # Video files aus Konfiguration laden (falls vorhanden)
                self.video_files = config.get('video_files', [])
                # Entferne nicht existierende Videos
                self.video_files = [vf for vf in self.video_files if os.path.exists(vf)]
                
                # Farben konvertieren
                for cls_id, cfg in self.class_config.items():
                    if 'color' in cfg and isinstance(cfg['color'], list):
                        cfg['color'] = tuple(cfg['color'])
                
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
                    self.lbl_status.setText("Keine Modelle geladen")
                        
            except Exception as e:
                print(f"Fehler beim Laden der Konfiguration: {e}")
                # Verwende Standard-Konfiguration
                self.class_config = {
                    "0": {"name": "GEFAHR", "color": COLORS["Red"], "conf": 0.5, "iou": 0.4}
                }
    
    def save_config(self):
        """Save current configuration to config.json"""
        config_path = self.find_or_create_config_file()
        
        config = {
            'detection_model_path': self.detection_model_path,
            'pose_model_path': self.pose_model_path,
            'class_config': self.class_config,
            'pose_config': self.pose_config,
            'display_config': self.display_config,
            'video_files': self.video_files
        }
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
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