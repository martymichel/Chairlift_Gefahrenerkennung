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
    QDialog, QGridLayout
)
from PyQt6.QtGui import QPixmap, QImage, QColor, QIcon
from PyQt6.QtCore import QTimer, Qt, QPropertyAnimation, QThread, pyqtSignal, QRunnable, QThreadPool, QObject
from ultralytics import YOLO

# Predefined colors for bounding boxes (in BGR format for OpenCV)
COLORS = {
    "Red": (0, 0, 255),      # BGR order
    "Green": (0, 255, 0),    # BGR order
    "Blue": (255, 0, 0),     # BGR order
    "Yellow": (0, 255, 255), # BGR order
    "Cyan": (255, 255, 0),   # BGR order
    "Magenta": (255, 0, 255),# BGR order
    "Orange": (0, 165, 255), # BGR order
    "Purple": (128, 0, 128), # BGR order
    "Brown": (42, 42, 165),  # BGR order
    "Pink": (203, 192, 255)  # BGR order
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

class ClassConfigDialog(QDialog):
    """Dialog for configuring class settings."""
    def __init__(self, class_config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Klasseneinstellungen")
        self.resize(600, 400)
        self.class_config = class_config.copy()
        
        layout = QVBoxLayout(self)
        
        # Table for class configuration
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(['Klassen-ID', 'Name', 'Farbe', 'Konfidenz'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        
        # Note label
        note_label = QLabel("Hinweis: Klassen werden automatisch aus dem YOLO-Modell geladen und können nicht hinzugefügt oder entfernt werden.")
        note_label.setWordWrap(True)
        note_label.setStyleSheet("color: #666;")
        layout.addWidget(note_label)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        self.btn_save = QPushButton("Speichern")
        self.btn_cancel = QPushButton("Abbrechen")
        
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.btn_save)
        buttons_layout.addWidget(self.btn_cancel)
        layout.addLayout(buttons_layout)
        
        # Connect signals
        self.btn_save.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        
        # Load existing classes
        self.load_classes()
    
    def load_classes(self):
        self.table.setRowCount(0)
        for cls_id, cfg in sorted(self.class_config.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999):
            row = self.table.rowCount()
            self.table.insertRow(row)
            
            # Class ID
            id_item = QTableWidgetItem(cls_id)
            id_item.setFlags(id_item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # Make ID non-editable
            self.table.setItem(row, 0, id_item)
            
            # Name
            self.table.setItem(row, 1, QTableWidgetItem(cfg.get('name', f"Class {cls_id}")))
            
            # Color dropdown
            color_combo = QComboBox()
            for color_name in COLORS:
                color_combo.addItem(color_name)
            
            # Set current color
            current_color = cfg.get('color', (0, 0, 255))  # Default red in BGR
            # Find closest color in our predefined colors
            closest_color = self.find_closest_color(current_color)
            color_combo.setCurrentText(closest_color)
            
            self.table.setCellWidget(row, 2, color_combo)
            
            # Confidence
            conf_spin = QDoubleSpinBox()
            conf_spin.setRange(0.1, 1.0)
            conf_spin.setSingleStep(0.05)
            conf_spin.setValue(float(cfg.get('conf', 0.5)))
            self.table.setCellWidget(row, 3, conf_spin)
    
    def find_closest_color(self, bgr_color):
        """Find the closest predefined color name for a BGR value"""
        if not bgr_color:
            return "Red"  # Default
            
        min_distance = float('inf')
        closest_name = "Red"
        
        for name, color in COLORS.items():
            # Calculate Euclidean distance between colors
            distance = sum((a - b) ** 2 for a, b in zip(bgr_color, color))
            if distance < min_distance:
                min_distance = distance
                closest_name = name
                
        return closest_name
    
    def get_config(self):
        config = {}
        for row in range(self.table.rowCount()):
            cls_id = self.table.item(row, 0).text()
            name = self.table.item(row, 1).text()
            
            # Get color from dropdown
            color_combo = self.table.cellWidget(row, 2)
            color_name = color_combo.currentText()
            color = COLORS[color_name]
            
            conf = self.table.cellWidget(row, 3).value()
            
            # Keep other existing settings like IoU
            existing_cfg = self.class_config.get(cls_id, {})
            iou = existing_cfg.get('iou', 0.5)
            
            config[cls_id] = {
                'name': name,
                'color': color,
                'conf': conf,
                'iou': iou
            }
        return config

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Video Annotator")
        self.resize(1200, 720)
        
        # Thread pool for parallel processing
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(4)  # Limit threads to avoid overwhelming the CPU
        
        # Initial default values
        self.model = None
        self.model_path = ""
        self.class_config = {}
        self.config_path = ""
        
        # Setup UI first, before loading config
        self.setup_ui()
        
        # Connect events
        self.connect_signals()
        
        # Video
        self.video_files = []
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
        
        # Load config at the end, after UI is setup
        self.load_config()
    
    def setup_ui(self):
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Split view
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.splitter)
        
        # Main area for video
        self.main_area = QFrame()
        self.main_layout = QVBoxLayout(self.main_area)
        
        # Video display
        self.video_container = QFrame()
        self.video_container.setStyleSheet("background-color: #222; border-radius: 5px;")
        video_layout = QVBoxLayout(self.video_container)
        
        self.label = QLabel("Video wird hier angezeigt")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setMinimumHeight(400)
        self.label.setStyleSheet("color: #aaa;")
        video_layout.addWidget(self.label)
        
        self.main_layout.addWidget(self.video_container, 1)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(5)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        self.main_layout.addWidget(self.progress)
        
        # Settings toggle button in main area (will show when sidebar is hidden)
        self.btn_toggle_settings_main = QPushButton("≫ Einstellungen einblenden")
        self.btn_toggle_settings_main.setVisible(False)
        self.main_layout.addWidget(self.btn_toggle_settings_main)
        
        # Sidebar for settings
        self.sidebar = QFrame()
        self.sidebar.setFrameShape(QFrame.Shape.StyledPanel)
        self.sidebar.setMinimumWidth(320)
        self.sidebar.setMaximumWidth(400)
        
        # Add to splitter
        self.splitter.addWidget(self.main_area)
        self.splitter.addWidget(self.sidebar)
        self.splitter.setSizes([700, 300])
        
        # Sidebar Layout
        sidebar_layout = QVBoxLayout(self.sidebar)
        
        # Settings toggle button (for sidebar)
        self.btn_toggle_settings = QPushButton("≪ Einstellungen ausblenden")
        sidebar_layout.addWidget(self.btn_toggle_settings)
        
        # Video control group
        video_control_group = QGroupBox("Video Steuerung")
        video_control_layout = QVBoxLayout(video_control_group)
        
        # Video selection button
        self.btn_select_videos = QPushButton("Videos auswählen")
        self.btn_select_videos.setStyleSheet("padding: 8px;")
        video_control_layout.addWidget(self.btn_select_videos)
        
        # Play/Pause button
        self.btn_play_pause = QPushButton("▶ Abspielen")
        self.btn_play_pause.setStyleSheet("padding: 8px;")
        self.btn_play_pause.setEnabled(False)
        video_control_layout.addWidget(self.btn_play_pause)
        
        sidebar_layout.addWidget(video_control_group)
        
        # Settings Tabs
        self.tabs = QTabWidget()
        sidebar_layout.addWidget(self.tabs)
        
        # Tab 1: Model
        model_tab = QWidget()
        model_layout = QVBoxLayout(model_tab)
        
        model_form = QFormLayout()
        
        # Model selection
        model_box = QHBoxLayout()
        self.txt_model_path = QLineEdit(self.model_path)
        self.txt_model_path.setReadOnly(True)
        self.btn_select_model = QPushButton("...")
        self.btn_select_model.setFixedWidth(30)
        model_box.addWidget(self.txt_model_path)
        model_box.addWidget(self.btn_select_model)
        model_form.addRow("YOLO-Modell (.pt):", model_box)
        
        model_layout.addLayout(model_form)
        
        # Model info
        self.lbl_model_info = QLabel("Kein Modell geladen")
        model_layout.addWidget(self.lbl_model_info)
        
        # Classes configuration
        self.btn_configure_classes = QPushButton("Klasseneinstellungen konfigurieren")
        model_layout.addWidget(self.btn_configure_classes)
        model_layout.addStretch()
        
        # Tab 2: Display
        display_tab = QWidget()
        display_layout = QVBoxLayout(display_tab)
        
        display_group = QGroupBox("Darstellungs-Einstellungen")
        display_form = QFormLayout(display_group)
        
        # Box thickness
        self.box_thickness = QSpinBox()
        self.box_thickness.setRange(1, 10)
        self.box_thickness.setValue(2)
        display_form.addRow("Rahmendicke:", self.box_thickness)
        
        # Font size
        self.font_scale = QSpinBox()
        self.font_scale.setRange(1, 20)
        self.font_scale.setValue(5)
        display_form.addRow("Textgrösse (x10):", self.font_scale)
        
        # Text thickness
        self.text_thickness = QSpinBox()
        self.text_thickness.setRange(1, 5)
        self.text_thickness.setValue(1)
        display_form.addRow("Textdicke:", self.text_thickness)
        
        # Playback speed
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(60)
        self.speed_slider.setValue(30)
        self.speed_slider.setTickInterval(5)
        self.speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        display_form.addRow("FPS:", self.speed_slider)
        
        # Alarm class
        self.alarm_class_dropdown = QComboBox()
        display_form.addRow("Alarmklasse:", self.alarm_class_dropdown)
        
        display_layout.addWidget(display_group)
        display_layout.addStretch()
        
        # Add tabs
        self.tabs.addTab(model_tab, "Modell")
        self.tabs.addTab(display_tab, "Anzeige")
        
        # Save/Load configuration
        config_layout = QHBoxLayout()
        self.btn_save_config = QPushButton("Konfiguration speichern")
        self.btn_load_config = QPushButton("Konfiguration laden")
        config_layout.addWidget(self.btn_save_config)
        config_layout.addWidget(self.btn_load_config)
        sidebar_layout.addLayout(config_layout)
    
    def connect_signals(self):
        # Model tab
        self.btn_select_model.clicked.connect(self.select_model)
        self.btn_configure_classes.clicked.connect(self.configure_classes)
        
        # Video controls
        self.btn_select_videos.clicked.connect(self.load_videos)
        self.btn_play_pause.clicked.connect(self.toggle_playback)
        
        # Settings toggle
        self.btn_toggle_settings.clicked.connect(self.toggle_settings)
        self.btn_toggle_settings_main.clicked.connect(self.toggle_settings)
        
        # Configuration
        self.btn_save_config.clicked.connect(lambda: self.save_config(True))
        self.btn_load_config.clicked.connect(lambda: self.load_config(True))
        
        # Playback speed
        self.speed_slider.valueChanged.connect(self.update_timer_interval)
    
    def select_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "YOLO-Modell auswählen", "", "YOLO Models (*.pt)"
        )
        if file_path:
            try:
                # Show loading progress
                self.progress.setVisible(True)
                QApplication.processEvents()
                
                # Load model in separate thread to prevent UI freezing
                def load_model_thread():
                    try:
                        self.model = YOLO(file_path)
                        self.model_path = file_path
                        self.txt_model_path.setText(file_path)
                        
                        # Update model info
                        model_name = os.path.basename(file_path)
                        info_text = f"Modell: {model_name}"
                        self.lbl_model_info.setText(info_text)
                        
                        # Extract classes from the model
                        self.extract_model_classes()
                        
                        # Hide progress when done
                        self.progress.setVisible(False)
                    except Exception as e:
                        self.progress.setVisible(False)
                        QMessageBox.critical(self, "Fehler", f"Modell konnte nicht geladen werden: {str(e)}")
                
                thread = threading.Thread(target=load_model_thread)
                thread.daemon = True
                thread.start()
            except Exception as e:
                self.progress.setVisible(False)
                QMessageBox.critical(self, "Fehler", f"Modell konnte nicht geladen werden: {str(e)}")
    
    def extract_model_classes(self):
        """Extract class information from the loaded YOLO model"""
        if not self.model:
            return
        
        # Get classes from model
        class_names = self.model.names
        
        # Create/update class config
        new_config = {}
        
        # Assign colors to the classes
        color_list = list(COLORS.values())
        
        for cls_id, name in class_names.items():
            # Convert to string key
            cls_id_str = str(cls_id)
            
            # If we already have config for this class, preserve settings
            if cls_id_str in self.class_config:
                new_config[cls_id_str] = self.class_config[cls_id_str]
                new_config[cls_id_str]['name'] = name  # Update name from model
            else:
                # Assign a color from our predefined list
                color_idx = cls_id % len(color_list)
                color = color_list[color_idx]
                
                new_config[cls_id_str] = {
                    'name': name,
                    'color': color,
                    'conf': 0.5,
                    'iou': 0.5
                }
        
        self.class_config = new_config
        self.update_alarm_classes()
    
    def configure_classes(self):
        if not self.model:
            QMessageBox.warning(self, "Warnung", "Bitte wählen Sie zuerst ein YOLO-Modell aus.")
            return
            
        dialog = ClassConfigDialog(self.class_config, self)
        if dialog.exec():
            self.class_config = dialog.get_config()
            self.update_alarm_classes()
    
    def update_alarm_classes(self):
        # Store current selection
        current_data = None
        if self.alarm_class_dropdown.count() > 0:
            current_idx = self.alarm_class_dropdown.currentIndex()
            if current_idx >= 0:
                current_data = self.alarm_class_dropdown.itemData(current_idx)
        
        # Clear and repopulate
        self.alarm_class_dropdown.clear()
        
        # Add classes from config
        for cls_id, cfg in self.class_config.items():
            self.alarm_class_dropdown.addItem(f"{cls_id}: {cfg['name']}", userData=cls_id)
        
        # Restore selection
        if current_data:
            index = self.alarm_class_dropdown.findData(current_data)
            if index >= 0:
                self.alarm_class_dropdown.setCurrentIndex(index)
    
    def load_videos(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Videos auswählen", "", "Videos (*.mp4 *.avi *.mov)"
        )
        if files:
            self.video_files = files
            self.current_video_idx = 0
            self.btn_play_pause.setEnabled(True)
            self.start_video()
    
    def start_video(self):
        if not self.model:
            QMessageBox.warning(self, "Warnung", "Bitte wählen Sie zuerst ein YOLO-Modell aus.")
            return
            
        if self.cap:
            self.cap.release()
        
        if not self.video_files:
            return
            
        self.cap = cv2.VideoCapture(self.video_files[self.current_video_idx])
        self.update_timer_interval()
        self.timer.start()
        self.btn_play_pause.setText("⏸ Pausieren")
        
        # Update video info in UI
        video_name = os.path.basename(self.video_files[self.current_video_idx])
        self.setWindowTitle(f"YOLO Video Annotator - {video_name}")
    
    def toggle_playback(self):
        if not self.cap:
            return
            
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play_pause.setText("▶ Abspielen")
        else:
            self.timer.start()
            self.btn_play_pause.setText("⏸ Pausieren")
    
    def update_timer_interval(self):
        fps = self.speed_slider.value()
        delay = int(1000 / fps)
        self.timer.setInterval(delay)
    
    def toggle_settings(self):
        if self.sidebar.isVisible():
            self.sidebar.hide()
            self.btn_toggle_settings_main.setVisible(True)
        else:
            self.sidebar.show()
            self.btn_toggle_settings_main.setVisible(False)
    
    def pulse_alarm(self):
        """Animation für den Alarmzustand"""
        if not self.alarm_active:
            self.alarm_timer.stop()
            self.video_container.setStyleSheet("background-color: #222; border-radius: 5px;")
            return
            
        # Pulseffekt (0-100)
        self.pulse_value += (5 * self.pulse_direction)
        if self.pulse_value >= 100:
            self.pulse_value = 100
            self.pulse_direction = -1
        elif self.pulse_value <= 0:
            self.pulse_value = 0
            self.pulse_direction = 1
            
        # Intensivere Animation bei Alarm - stärkere Farbübergänge
        intensity = 155 + int(self.pulse_value)  # 155-255
        border_intensity = 255 - intensity
        self.video_container.setStyleSheet(f"""
            background-color: rgb({intensity}, 0, 0);
            border: 5px solid rgb(255, {border_intensity}, 0);
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
        alarm_class_id = self.alarm_class_dropdown.currentData()
        if alarm_class_id:
            alarm_triggered = any(str(d['class_id']) == alarm_class_id for d in detections)
            
            if alarm_triggered and not self.alarm_active:
                self.alarm_active = True
                self.alarm_timer.start(50)  # 50ms for smooth animation
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
        box_thickness = self.box_thickness.value()
        font_scale = self.font_scale.value() / 10.0
        text_thickness = self.text_thickness.value()
        
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
            
            # Draw label with separate text thickness
            cv2.putText(frame, label, (box['x1'], box['y1'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness)
        
        # Remove the warning text overlay - as requested
        # No cv2.putText for alarm text anymore
        
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
            # End of video, go to next one
            self.current_video_idx = (self.current_video_idx + 1) % len(self.video_files)
            self.start_video()
            return
        
        # Set processing lock
        self.processing_frame = True
        
        # Submit to thread pool
        worker = DetectionWorker(frame, self.model, self.class_config)
        worker.signals.result.connect(self.handle_detection_result)
        worker.signals.error.connect(lambda err: print(f"Error: {err}"))
        self.threadpool.start(worker)
    
    def save_config(self, show_dialog=False):
        """Save current configuration to a JSON file"""
        config = {
            'model_path': self.model_path,
            'class_config': self.class_config,
            'display': {
                'box_thickness': self.box_thickness.value(),
                'font_scale': self.font_scale.value(),
                'text_thickness': self.text_thickness.value(),
                'playback_speed': self.speed_slider.value()
            }
        }
        
        # Get save path
        if show_dialog or not self.config_path:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Konfiguration speichern", "", "JSON (*.json)"
            )
            if not file_path:
                return
                
            self.config_path = file_path
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            if show_dialog:
                QMessageBox.information(self, "Erfolg", "Konfiguration wurde gespeichert.")
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Konfiguration konnte nicht gespeichert werden: {str(e)}")
    
    def load_config(self, show_dialog=False):
        """Load configuration from a JSON file"""
        config_path = 'config.json'
        
        if show_dialog or not os.path.exists(config_path):
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Konfiguration laden", "", "JSON (*.json)"
            )
            if not file_path:
                # Initialize with default config if no file selected
                if not self.class_config:
                    self.class_config = {
                        "0": {"name": "GEFAHR", "color": COLORS["Red"], "conf": 0.5, "iou": 0.4},
                        "1": {"name": "Chair", "color": COLORS["Green"], "conf": 0.6, "iou": 0.4},
                        "2": {"name": "Human", "color": COLORS["Blue"], "conf": 0.6, "iou": 0.4}
                    }
                    self.update_alarm_classes()
                return
                
            config_path = file_path
        
        self.config_path = config_path
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Load model path
            model_path = config.get('model_path', '')
            if model_path and os.path.exists(model_path):
                self.model_path = model_path
                self.txt_model_path.setText(model_path)
                
                # Load model in separate thread
                def load_model_thread():
                    try:
                        self.model = YOLO(model_path)
                        model_name = os.path.basename(model_path)
                        info_text = f"Modell: {model_name}"
                        self.lbl_model_info.setText(info_text)
                        
                        # Extract classes from the model
                        self.extract_model_classes()
                    except Exception as e:
                        QMessageBox.critical(self, "Fehler", f"Modell konnte nicht geladen werden: {str(e)}")
                
                thread = threading.Thread(target=load_model_thread)
                thread.daemon = True
                thread.start()
            
            # Load class config and convert color tuples if needed
            class_config = config.get('class_config', {})
            for cls_id, cfg in class_config.items():
                if 'color' in cfg and isinstance(cfg['color'], list):
                    cfg['color'] = tuple(cfg['color'])
            
            # Only update class config if model hasn't been loaded yet
            # This ensures model classes take precedence
            if not self.model:
                self.class_config = class_config
                self.update_alarm_classes()
            else:
                # If model is loaded, only update settings like color and confidence
                for cls_id, cfg in class_config.items():
                    if cls_id in self.class_config:
                        if 'color' in cfg:
                            self.class_config[cls_id]['color'] = cfg['color']
                        if 'conf' in cfg:
                            self.class_config[cls_id]['conf'] = cfg['conf']
                        if 'iou' in cfg:
                            self.class_config[cls_id]['iou'] = cfg['iou']
            
            # Load display settings
            display = config.get('display', {})
            self.box_thickness.setValue(display.get('box_thickness', 2))
            self.font_scale.setValue(display.get('font_scale', 5))
            self.text_thickness.setValue(display.get('text_thickness', 1))
            self.speed_slider.setValue(display.get('playback_speed', 30))
            
            if show_dialog:
                QMessageBox.information(self, "Erfolg", "Konfiguration wurde geladen.")
        except Exception as e:
            if show_dialog:
                QMessageBox.critical(self, "Fehler", f"Konfiguration konnte nicht geladen werden: {str(e)}")
            
            # Initialize with default config on error
            if not self.class_config:
                self.class_config = {
                    "0": {"name": "GEFAHR", "color": COLORS["Red"], "conf": 0.5, "iou": 0.6},
                    "1": {"name": "Chair", "color": COLORS["Green"], "conf": 0.6, "iou": 0.3},
                    "2": {"name": "Human", "color": COLORS["Blue"], "conf": 0.6, "iou": 0.2}
                }
                self.update_alarm_classes()
    
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
