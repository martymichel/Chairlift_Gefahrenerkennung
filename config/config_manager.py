import os
import json
from .constants import DEFAULT_CONFIG, COLORS

class ConfigManager:
    """Verwaltet das Laden und Speichern der Anwendungskonfiguration"""
    
    def __init__(self):
        self.config_path = self._find_or_create_config_file()
    
    def _find_or_create_config_file(self):
        """Findet oder erstellt eine config.json Datei im App-Verzeichnis"""
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
            try:
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(DEFAULT_CONFIG, f, indent=2)
                print(f"Neue Konfigurationsdatei erstellt: {config_path}")
            except Exception as e:
                print(f"Konnte config.json nicht erstellen: {e}")
        
        return config_path
    
    def load_config(self):
        """Lädt die Konfiguration aus der Datei"""
        if not os.path.exists(self.config_path):
            return DEFAULT_CONFIG.copy()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Migration von alten Konfigurationsformaten
            migrated_config = self._migrate_config(config)
            
            # Farben konvertieren
            for cls_id, cfg in migrated_config.get('class_config', {}).items():
                if 'color' in cfg and isinstance(cfg['color'], list):
                    cfg['color'] = tuple(cfg['color'])
            
            # Entferne nicht existierende Videos
            if 'video_files' in migrated_config:
                migrated_config['video_files'] = [
                    vf for vf in migrated_config['video_files'] 
                    if os.path.exists(vf)
                ]
            
            return migrated_config
            
        except Exception as e:
            print(f"Fehler beim Laden der Konfiguration: {e}")
            return DEFAULT_CONFIG.copy()
    
    def _migrate_config(self, config):
        """Migriert alte Konfigurationsformate"""
        migrated = DEFAULT_CONFIG.copy()
        
        # Detection Model Path Migration
        migrated['detection_model_path'] = config.get('detection_model_path', 
                                                     config.get('model_path', ''))
        migrated['pose_model_path'] = config.get('pose_model_path', '')
        migrated['class_config'] = config.get('class_config', {})
        migrated['video_files'] = config.get('video_files', [])
        
        # Pose Config Migration
        old_pose_config = config.get('pose_config', {})
        migrated['pose_config'] = {
            'pose_detect_classes': self._migrate_pose_detect_classes(old_pose_config),
            'min_confidence': old_pose_config.get('min_confidence', 0.3),
            'line_thickness': old_pose_config.get('line_thickness', 2),
            'keypoint_radius': old_pose_config.get('keypoint_radius', 3),
            'show_keypoints': old_pose_config.get('show_keypoints', True),
            'show_skeleton': old_pose_config.get('show_skeleton', True)
        }
        
        # Display config aus alter Struktur migrieren (ohne playback_speed)
        old_display = config.get('display_config', config.get('display', {}))
        migrated['display_config'] = {
            'box_thickness': old_display.get('box_thickness', 2),
            'font_scale': old_display.get('font_scale', 5),
            'text_thickness': old_display.get('text_thickness', 1),
            'alarm_class': old_display.get('alarm_class', None)
        }
        
        # FOI Config Migration - NEUE Konfiguration
        old_foi = config.get('foi_config', {})
        migrated['foi_config'] = {
            'enabled': old_foi.get('enabled', True),
            'points': old_foi.get('points', [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]),
            'count_class': old_foi.get('count_class', None),
            'alert_class': old_foi.get('alert_class', None),
            'alert_timeout': old_foi.get('alert_timeout', 10.0),
            'foi_color': old_foi.get('foi_color', (0, 255, 255)),
            'foi_thickness': old_foi.get('foi_thickness', 3)
        }
        
        return migrated
    
    def _migrate_pose_detect_classes(self, old_pose_config):
        """Migriert alte pose_detect_class zu pose_detect_classes Liste"""
        old_class = old_pose_config.get('pose_detect_class')
        if old_class:
            return [old_class]
        return []
    
    def save_config(self, config):
        """Speichert die Konfiguration in die Datei"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Fehler beim Speichern der Konfiguration: {e}")
            return False
    
    def get_config_path(self):
        """Gibt den Pfad zur Konfigurationsdatei zurück"""
        return self.config_path