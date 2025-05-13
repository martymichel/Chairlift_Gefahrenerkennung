import sys
import os
import cv2
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QSlider, QGroupBox, QFormLayout, QSpinBox, QComboBox
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QTimer, Qt
from ultralytics import YOLO
from playsound import playsound

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Video Annotator")
        self.resize(1000, 720)

        # Modellpfad laden
        model_path = r"C:\Users\miche\OneDrive - Flex\4.4_CAS Machine Intelligence\03_Deep_Learning\Leistungsnachweis\Dataset\runs\train\yolo11n_custom2\weights\best.pt"
        self.model = YOLO(model_path)

        # Konfiguration pro Klasse
        self.class_config = {
            0: {"name": "GEFAHR", "color": (0, 0, 255), "conf": 0.5, "iou": 0.6},
            1: {"name": "Chair", "color": (0, 200, 0), "conf": 0.6, "iou": 0.3},
            2: {"name": "Human", "color": (200, 0, 0), "conf": 0.6, "iou": 0.2}
        }

        # UI-Elemente
        self.label = QLabel("Video wird hier angezeigt")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.btn_select = QPushButton("Videos auswählen & starten")
        self.btn_toggle_settings = QPushButton("🔧 Einstellungen ein-/ausblenden")

        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(60)
        self.speed_slider.setValue(30)
        self.speed_slider.setTickInterval(5)
        self.speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)

        self.box_thickness = QSpinBox()
        self.box_thickness.setRange(1, 10)
        self.box_thickness.setValue(2)

        self.font_scale = QSpinBox()
        self.font_scale.setRange(1, 10)
        self.font_scale.setValue(5)

        self.alarm_class_dropdown = QComboBox()
        for cls_id, cfg in self.class_config.items():
            self.alarm_class_dropdown.addItem(f"{cls_id}: {cfg['name']}", userData=cls_id)
        self.alarm_class_dropdown.setCurrentIndex(0)

        # Einstellungsgruppe
        self.settings_group = QGroupBox("Darstellungs-Einstellungen")
        form = QFormLayout()
        form.addRow("Rahmendicke:", self.box_thickness)
        form.addRow("Textgrösse (x10):", self.font_scale)
        form.addRow("FPS (rechts = schneller):", self.speed_slider)
        form.addRow("Alarmklasse:", self.alarm_class_dropdown)
        self.settings_group.setLayout(form)

        # Layout
        buttons = QHBoxLayout()
        buttons.addWidget(self.btn_select)
        buttons.addWidget(self.btn_toggle_settings)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addLayout(buttons)
        layout.addWidget(self.settings_group)
        self.setLayout(layout)

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

        # Video
        self.video_files = []
        self.cap = None
        self.current_video_idx = 0

        # Events
        self.btn_select.clicked.connect(self.load_videos)
        self.speed_slider.valueChanged.connect(self.update_timer_interval)
        self.btn_toggle_settings.clicked.connect(self.toggle_settings)

    def load_videos(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Videos auswählen", "", "Videos (*.mp4 *.avi *.mov)"
        )
        if files:
            self.video_files = files
            self.current_video_idx = 0
            self.start_video()

    def start_video(self):
        if self.cap:
            self.cap.release()
        if not self.video_files:
            return
        self.cap = cv2.VideoCapture(self.video_files[self.current_video_idx])
        self.update_timer_interval()
        self.timer.start()

    def update_timer_interval(self):
        fps = self.speed_slider.value()
        delay = int(1000 / fps)
        self.timer.setInterval(delay)

    def toggle_settings(self):
        self.settings_group.setVisible(not self.settings_group.isVisible())

    def next_frame(self):
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.current_video_idx = (self.current_video_idx + 1) % len(self.video_files)
            self.start_video()
            return

        alarm_class_id = self.alarm_class_dropdown.currentData()
        alarm_triggered = False

        results = self.model.predict(frame, verbose=False)[0]
        thickness = self.box_thickness.value()
        font_scale = self.font_scale.value() / 10.0

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            cfg = self.class_config.get(cls, {})
            if not cfg or conf < cfg.get("conf", 0.5):
                continue

            if cls == alarm_class_id:
                alarm_triggered = True

            color = cfg["color"]
            label = f"{cfg['name']} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, max(1, thickness - 1))

        # UI Alarmaktion
        if alarm_triggered:
            self.setStyleSheet("background-color: red;")
            try:
                playsound("alarm.wav", block=False)
            except Exception as e:
                print("Fehler beim Abspielen des Tons:", e)
        else:
            self.setStyleSheet("")

        # Qt Anzeige
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec())
