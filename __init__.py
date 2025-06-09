
# config/__init__.py
"""Konfigurationspaket für die YOLO Video Annotator Anwendung"""

from .config_manager import ConfigManager
from .constants import COLORS, POSE_CONNECTIONS, DEFAULT_CONFIG

__all__ = ['ConfigManager', 'COLORS', 'POSE_CONNECTIONS', 'DEFAULT_CONFIG']

# core/__init__.py
"""Kernfunktionalitäten für die YOLO Video Annotator Anwendung"""

from .detection_worker import DualDetectionWorker, WorkerSignals
from .frame_renderer import FrameRenderer
from .foi_manager import FOIManager

__all__ = ['DualDetectionWorker', 'WorkerSignals', 'FrameRenderer', 'FOIManager']

# ui/__init__.py
"""Benutzeroberflächen-Komponenten für die YOLO Video Annotator Anwendung"""

from .video_player import VideoPlayer
from .settings_dialog import SettingsDialog

__all__ = ['VideoPlayer', 'SettingsDialog']