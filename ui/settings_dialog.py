import os
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout, 
    QPushButton, QLineEdit, QLabel, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
    QListWidget, QListWidgetItem, QCheckBox, QFileDialog,
    QMessageBox, QWidget
)
from PyQt6.QtCore import Qt
from ultralytics import YOLO
from config.constants import COLORS
from config.config_manager import ConfigManager

class SettingsDialog(QDialog):
    """Großer übersichtlicher Dialog für alle Einstellungen."""
    
    def __init__(self, detection_model_path, pose_model_path, class_config, 
                 pose_config, display_config, foi_config, video_files, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Einstellungen")
        self.resize(1200, 800)
        
        self.detection_model_path = detection_model_path
        self.pose_model_path = pose_model_path
        self.class_config = class_config.copy()
        self.pose_config = pose_config.copy()
        self.display_config = display_config.copy()
        self.foi_config = foi_config.copy()
        self.video_files = video_files.copy()
        
        self.config_manager = ConfigManager()
        
        self.setup_ui()
        self.load_settings()
    
    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        
        # Linke Seite: Modelle und Videos
        left_widget = self._create_left_panel()
        
        # Rechte Seite: Klasseneinstellungen
        right_widget = self._create_right_panel()
        
        # Layout zusammenfügen
        main_layout.addWidget(left_widget, 1)
        main_layout.addWidget(right_widget, 2)
        
        # Buttons unten
        button_layout = self._create_button_layout()
        
        # Hauptlayout erweitern
        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        
        dialog_layout = QVBoxLayout(self)
        dialog_layout.addWidget(main_widget)
        dialog_layout.addLayout(button_layout)
        
        # Signal-Verbindungen
        self.connect_signals()
    
    def _create_left_panel(self):
        """Erstellt das linke Panel mit Modell-, Video- und Display-Einstellungen"""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Detection Modell-Auswahl
        left_layout.addWidget(self._create_detection_model_group())
        
        # Pose Modell-Auswahl
        left_layout.addWidget(self._create_pose_model_group())
        
        # Video-Auswahl
        left_layout.addWidget(self._create_video_group())
        
        # Darstellungs-Einstellungen
        left_layout.addWidget(self._create_display_group())
        
        # Pose-Einstellungen
        left_layout.addWidget(self._create_pose_settings_group())
        
        # FOI-Einstellungen
        left_layout.addWidget(self._create_foi_settings_group())
        
        left_layout.addStretch()
        return left_widget
    
    def _create_detection_model_group(self):
        """Erstellt die Detection Model Gruppe"""
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
        
        return detection_model_group
    
    def _create_pose_model_group(self):
        """Erstellt die Pose Model Gruppe"""
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
        
        return pose_model_group
    
    def _create_video_group(self):
        """Erstellt die Video-Gruppe"""
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
        
        return video_group
    
    def _create_display_group(self):
        """Erstellt die Display-Gruppe"""
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
        
        self.alarm_class_dropdown = QComboBox()
        display_form.addRow("Alarmklasse:", self.alarm_class_dropdown)
        
        return display_group
    
    def _create_pose_settings_group(self):
        """Erstellt die Pose-Einstellungen Gruppe"""
        pose_settings_group = QGroupBox("Pose-Einstellungen")
        pose_form = QFormLayout(pose_settings_group)
        
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
        
        return pose_settings_group
    
    def _create_foi_settings_group(self):
        """Erstellt die FOI-Einstellungen Gruppe"""
        foi_settings_group = QGroupBox("Field of Interest (FOI) - Skilift Überwachung")
        foi_form = QFormLayout(foi_settings_group)
        
        self.foi_enabled = QCheckBox()
        self.foi_enabled.setChecked(self.foi_config.get('enabled', True))
        foi_form.addRow("FOI aktiviert:", self.foi_enabled)
        
        self.foi_count_class_dropdown = QComboBox()
        foi_form.addRow("Zählklasse (FOI):", self.foi_count_class_dropdown)
        
        self.foi_alert_class_dropdown = QComboBox()
        foi_form.addRow("Alert-Klasse (Lift):", self.foi_alert_class_dropdown)
        
        self.foi_alert_timeout = QDoubleSpinBox()
        self.foi_alert_timeout.setRange(1.0, 60.0)
        self.foi_alert_timeout.setSingleStep(1.0)
        self.foi_alert_timeout.setSuffix(" s")
        self.foi_alert_timeout.setValue(self.foi_config.get('alert_timeout', 10.0))
        foi_form.addRow("Alert Timeout:", self.foi_alert_timeout)
        
        self.foi_thickness = QSpinBox()
        self.foi_thickness.setRange(1, 10)
        self.foi_thickness.setValue(self.foi_config.get('foi_thickness', 3))
        foi_form.addRow("FOI Rahmendicke:", self.foi_thickness)
        
        info_label = QLabel(
            "Das FOI kann durch Ziehen der Eckpunkte im Video angepasst werden. "
            "Zählklasse wird im FOI gezählt, Alert-Klasse löst Lift-Verlangsamung aus."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; font-style: italic; font-size: 11px;")
        foi_form.addRow(info_label)
        
        return foi_settings_group
    
    def _create_right_panel(self):
        """Erstellt das rechte Panel mit Klasseneinstellungen"""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        classes_group = QGroupBox("Detection Klasseneinstellungen")
        classes_layout = QVBoxLayout(classes_group)
        
        # Tabelle für Klasseneinstellungen (mit Pose Detection Checkbox)
        self.class_table = QTableWidget()
        self.class_table.setColumnCount(6)  # ID, Name, Farbe, Konfidenz, IoU, Pose Detection
        self.class_table.setHorizontalHeaderLabels([
            'Klassen-ID', 'Name', 'Farbe', 'Konfidenz', 'IoU', 'Pose Detection'
        ])
        self.class_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        classes_layout.addWidget(self.class_table)
        
        note_label = QLabel(
            "Hinweis: Klassen werden automatisch aus dem YOLO-Detection-Modell geladen. "
            "Aktivieren Sie 'Pose Detection' für Klassen, bei denen Sturzerkennung durchgeführt werden soll."
        )
        note_label.setWordWrap(True)
        note_label.setStyleSheet("color: #666; font-style: italic;")
        classes_layout.addWidget(note_label)
        
        right_layout.addWidget(classes_group)
        return right_widget
    
    def _create_button_layout(self):
        """Erstellt das Button-Layout"""
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
        
        return button_layout
    
    def connect_signals(self):
        """Verbindet alle Signale mit ihren Slots"""
        self.btn_select_detection_model.clicked.connect(self.select_detection_model)
        self.btn_select_pose_model.clicked.connect(self.select_pose_model)
        self.btn_add_videos.clicked.connect(self.add_videos)
        self.btn_remove_video.clicked.connect(self.remove_video)
        self.btn_clear_videos.clicked.connect(self.clear_videos)
        
        self.btn_save.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_load_config.clicked.connect(self.load_config)
        self.btn_save_config.clicked.connect(self.save_config)
    
    def select_detection_model(self):
        """Wählt ein Detection-Modell aus"""
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
        """Wählt ein Pose-Modell aus"""
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
        """Extrahiert Klasseninformationen aus dem geladenen YOLO Detection Modell"""
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
        self.update_foi_classes()
    
    def load_class_table(self):
        """Lädt die Klassentabelle mit den aktuellen Einstellungen"""
        self.class_table.setRowCount(0)
        pose_detect_classes = self.pose_config.get('pose_detect_classes', [])
        
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
            
            # Pose Detection Checkbox
            pose_checkbox = QCheckBox()
            pose_checkbox.setChecked(cls_id in pose_detect_classes)
            self.class_table.setCellWidget(row, 5, pose_checkbox)
    
    def find_closest_color(self, bgr_color):
        """Findet den nächstgelegenen vordefinierten Farbnamen für einen BGR-Wert"""
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
        """Fügt Videos zur Liste hinzu"""
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
        """Entfernt ein Video aus der Liste"""
        current_row = self.video_list.currentRow()
        if current_row >= 0:
            self.video_files.pop(current_row)
            self.video_list.takeItem(current_row)
    
    def clear_videos(self):
        """Entfernt alle Videos aus der Liste"""
        self.video_files.clear()
        self.video_list.clear()
    
    def update_alarm_classes(self):
        """Aktualisiert die Alarmklassen-Dropdown"""
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
    
    def update_foi_classes(self):
        """Aktualisiert die FOI-Klassen-Dropdowns"""
        # Count Class Dropdown
        current_count_data = self.foi_count_class_dropdown.currentData()
        self.foi_count_class_dropdown.clear()
        self.foi_count_class_dropdown.addItem("Keine Zählklasse", userData=None)
        
        # Alert Class Dropdown
        current_alert_data = self.foi_alert_class_dropdown.currentData()
        self.foi_alert_class_dropdown.clear()
        self.foi_alert_class_dropdown.addItem("Keine Alert-Klasse", userData=None)
        
        for cls_id, cfg in self.class_config.items():
            # Für beide Dropdowns die gleichen Klassen hinzufügen
            self.foi_count_class_dropdown.addItem(f"{cls_id}: {cfg['name']}", userData=cls_id)
            self.foi_alert_class_dropdown.addItem(f"{cls_id}: {cfg['name']}", userData=cls_id)
        
        # Gespeicherte Auswahl wiederherstellen
        if current_count_data is not None:
            index = self.foi_count_class_dropdown.findData(current_count_data)
            if index >= 0:
                self.foi_count_class_dropdown.setCurrentIndex(index)
        elif self.foi_config.get('count_class') is not None:
            index = self.foi_count_class_dropdown.findData(self.foi_config.get('count_class'))
            if index >= 0:
                self.foi_count_class_dropdown.setCurrentIndex(index)
        
        if current_alert_data is not None:
            index = self.foi_alert_class_dropdown.findData(current_alert_data)
            if index >= 0:
                self.foi_alert_class_dropdown.setCurrentIndex(index)
        elif self.foi_config.get('alert_class') is not None:
            index = self.foi_alert_class_dropdown.findData(self.foi_config.get('alert_class'))
            if index >= 0:
                self.foi_alert_class_dropdown.setCurrentIndex(index)
    
    def load_settings(self):
        """Lädt die Einstellungen in die UI"""
        # Videos laden
        for video_file in self.video_files:
            item = QListWidgetItem(os.path.basename(video_file))
            item.setToolTip(video_file)
            self.video_list.addItem(item)
        
        # Klassen laden
        self.load_class_table()
        self.update_alarm_classes()
        self.update_foi_classes()
        
        # Modell-Info aktualisieren
        if self.detection_model_path:
            model_name = os.path.basename(self.detection_model_path)
            self.lbl_detection_model_info.setText(f"Detection Modell: {model_name}")
            
        if self.pose_model_path:
            model_name = os.path.basename(self.pose_model_path)
            self.lbl_pose_model_info.setText(f"Pose Modell: {model_name}")
    
    def get_settings(self):
        """Sammelt alle Einstellungen aus der UI"""
        # Klasseneinstellungen aus Tabelle sammeln
        updated_class_config = {}
        pose_detect_classes = []
        
        for row in range(self.class_table.rowCount()):
            cls_id = self.class_table.item(row, 0).text()
            name = self.class_table.item(row, 1).text()
            
            color_combo = self.class_table.cellWidget(row, 2)
            color_name = color_combo.currentText()
            color = COLORS[color_name]
            
            conf = self.class_table.cellWidget(row, 3).value()
            iou = self.class_table.cellWidget(row, 4).value()
            
            # Pose Detection Checkbox auslesen
            pose_checkbox = self.class_table.cellWidget(row, 5)
            if pose_checkbox.isChecked():
                pose_detect_classes.append(cls_id)
            
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
            'alarm_class': self.alarm_class_dropdown.currentData()
        }
        
        # Pose-Einstellungen sammeln
        pose_config = {
            'pose_detect_classes': pose_detect_classes,
            'min_confidence': self.pose_min_confidence.value(),
            'line_thickness': self.pose_line_thickness.value(),
            'keypoint_radius': self.pose_keypoint_radius.value(),
            'show_keypoints': self.pose_show_keypoints.isChecked(),
            'show_skeleton': self.pose_show_skeleton.isChecked()
        }
        
        # FOI-Einstellungen sammeln
        foi_config = {
            'enabled': self.foi_enabled.isChecked(),
            'points': self.foi_config.get('points', [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]),
            'count_class': self.foi_count_class_dropdown.currentData(),
            'alert_class': self.foi_alert_class_dropdown.currentData(),
            'alert_timeout': self.foi_alert_timeout.value(),
            'foi_color': self.foi_config.get('foi_color', (0, 255, 255)),
            'foi_thickness': self.foi_thickness.value()
        }
        
        return {
            'detection_model_path': self.detection_model_path,
            'pose_model_path': self.pose_model_path,
            'class_config': updated_class_config,
            'pose_config': pose_config,
            'display_config': display_config,
            'foi_config': foi_config,
            'video_files': self.video_files
        }
    
    def save_config(self):
        """Speichert die Konfiguration in die Standard-Konfigurationsdatei"""
        try:
            config = self.get_settings()
            if self.config_manager.save_config(config):
                QMessageBox.information(
                    self, "Erfolg", 
                    f"Konfiguration wurde in {self.config_manager.get_config_path()} gespeichert."
                )
            else:
                QMessageBox.critical(self, "Fehler", "Fehler beim Speichern der Konfiguration.")
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Fehler beim Speichern: {str(e)}")
    
    def load_config(self):
        """Lädt eine Konfiguration aus einer Datei"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Konfiguration laden", "", "JSON (*.json)"
        )
        if file_path:
            self._load_config_from_file(file_path)
        
    def _load_config_from_file(self, file_path):
        """Lädt Konfiguration aus der angegebenen Datei"""
        try:
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Verwende ConfigManager für Migration
            temp_manager = ConfigManager()
            migrated_config = temp_manager._migrate_config(config)
            
            # Konfiguration übernehmen
            self.detection_model_path = migrated_config['detection_model_path']
            self.pose_model_path = migrated_config['pose_model_path']
            self.class_config = migrated_config['class_config']
            self.pose_config = migrated_config['pose_config']
            self.display_config = migrated_config['display_config']
            self.foi_config = migrated_config.get('foi_config', self.foi_config)
            self.video_files = migrated_config['video_files']
            
            # Farben konvertieren
            for cls_id, cfg in self.class_config.items():
                if 'color' in cfg and isinstance(cfg['color'], list):
                    cfg['color'] = tuple(cfg['color'])
            
            # UI aktualisieren
            self._update_ui_after_config_load()
            
            QMessageBox.information(self, "Erfolg", f"Konfiguration wurde aus {file_path} geladen.")
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Fehler beim Laden: {str(e)}")
    
    def _update_ui_after_config_load(self):
        """Aktualisiert die UI nach dem Laden einer Konfiguration"""
        # Model paths
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
        
        # Pose-Einstellungen
        self.pose_min_confidence.setValue(self.pose_config.get('min_confidence', 0.3))
        self.pose_line_thickness.setValue(self.pose_config.get('line_thickness', 2))
        self.pose_keypoint_radius.setValue(self.pose_config.get('keypoint_radius', 3))
        self.pose_show_keypoints.setChecked(self.pose_config.get('show_keypoints', True))
        self.pose_show_skeleton.setChecked(self.pose_config.get('show_skeleton', True))
        
        # FOI-Einstellungen
        self.foi_enabled.setChecked(self.foi_config.get('enabled', True))
        self.foi_alert_timeout.setValue(self.foi_config.get('alert_timeout', 10.0))
        self.foi_thickness.setValue(self.foi_config.get('foi_thickness', 3))
        
        # Videos neu laden
        self.video_list.clear()
        for video_file in self.video_files:
            if os.path.exists(video_file):
                item = QListWidgetItem(os.path.basename(video_file))
                item.setToolTip(video_file)
                self.video_list.addItem(item)
            else:
                if video_file in self.video_files:
                    self.video_files.remove(video_file)
        
        self.load_class_table()
        self.update_alarm_classes()
        self.update_foi_classes()