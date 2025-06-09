import sys
import os
from PyQt6.QtWidgets import QApplication
from ui.video_player import VideoPlayer

def main():
    """Hauptfunktion der Anwendung"""
    app = QApplication(sys.argv)
    app.setApplicationName("YOLO Dual Model Video Annotator")
    app.setApplicationVersion("2.0")
    
    player = VideoPlayer()
    player.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()