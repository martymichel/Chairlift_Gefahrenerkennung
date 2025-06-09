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

# Standard-Konfigurationswerte
DEFAULT_CONFIG = {
    'detection_model_path': '',
    'pose_model_path': '',
    'class_config': {},
    'pose_config': {
        'pose_detect_classes': [],  # Liste der Klassen-IDs für Pose Detection
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
        'alarm_class': None
    },
    'foi_config': {
        'enabled': True,
        'points': [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]],  # Relative Koordinaten (0-1)
        'count_class': None,  # Klasse für Personenzählung
        'alert_class': None,  # Klasse für Lift-Verlangsamung
        'alert_timeout': 10.0,  # Sekunden bis Lift-Stopp
        'foi_color': (0, 255, 255),  # Gelb für FOI-Rahmen
        'foi_thickness': 3
    },
    'video_files': []
}

# FOI (Field of Interest) Konstanten
FOI_CORNER_SIZE = 15  # Größe der Ecken-Griffe
FOI_CORNER_COLOR = (0, 0, 255)  # Rot für aktive Ecken
FOI_HOVER_COLOR = (255, 255, 0)  # Cyan für Hover-Effekt