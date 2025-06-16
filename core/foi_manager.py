import cv2
import numpy as np
import time
from config.constants import FOI_CORNER_SIZE, FOI_CORNER_COLOR, FOI_HOVER_COLOR

class FOIManager:
    """Verwaltet das Field of Interest (FOI) für die Skilift-Überwachung"""
    
    def __init__(self, foi_config):
        self.foi_config = foi_config
        self.frame_width = 1
        self.frame_height = 1
        
        # FOI-Status
        self.dragging_corner = -1
        self.hover_corner = -1
        
        # Lift-Status und Timing - VERBESSERT
        self.lift_status = "Lift Normalbetrieb"
        self.alert_start_time = None
        self.alert_active = False
        self.manual_reset_requested = False  # NEU: Flag für manuellen Reset
        
        # Zählung und Erkennung
        self.current_count = 0
        self.alert_object_in_foi = False
        
    def update_config(self, foi_config):
        """Aktualisiert die FOI-Konfiguration"""
        self.foi_config = foi_config
        
    def set_frame_dimensions(self, width, height):
        """Setzt die Frame-Dimensionen für die Koordinatenumrechnung"""
        self.frame_width = width
        self.frame_height = height
        
    def get_absolute_points(self):
        """Konvertiert relative FOI-Punkte zu absoluten Bildkoordinaten"""
        points = []
        for rel_point in self.foi_config['points']:
            x = int(rel_point[0] * self.frame_width)
            y = int(rel_point[1] * self.frame_height)
            points.append([x, y])
        return np.array(points, dtype=np.int32)
    
    def set_relative_points(self, absolute_points):
        """Konvertiert absolute Koordinaten zu relativen FOI-Punkten"""
        self.foi_config['points'] = []
        for point in absolute_points:
            rel_x = point[0] / self.frame_width
            rel_y = point[1] / self.frame_height
            self.foi_config['points'].append([rel_x, rel_y])
    
    def point_in_polygon(self, point, polygon):
        """Überprüft ob ein Punkt im Polygon liegt"""
        return cv2.pointPolygonTest(polygon, point, False) >= 0
    
    def get_corner_at_position(self, x, y):
        """Findet die Ecke an der gegebenen Position"""
        points = self.get_absolute_points()
        for i, point in enumerate(points):
            distance = np.sqrt((x - point[0])**2 + (y - point[1])**2)
            if distance <= FOI_CORNER_SIZE:
                return i
        return -1
    
    def move_corner(self, corner_idx, x, y):
        """Bewegt eine Ecke des FOI"""
        if 0 <= corner_idx < len(self.foi_config['points']):
            # Koordinaten auf Frame begrenzen
            x = max(0, min(self.frame_width - 1, x))
            y = max(0, min(self.frame_height - 1, y))
            
            # Relative Koordinaten aktualisieren
            rel_x = x / self.frame_width
            rel_y = y / self.frame_height
            self.foi_config['points'][corner_idx] = [rel_x, rel_y]
    
    def count_objects_in_foi(self, detections):
        """Zählt Objekte der definierten Klasse im FOI"""
        if not self.foi_config.get('enabled', False):
            return 0
            
        count_class = self.foi_config.get('count_class')
        if not count_class:
            return 0
            
        foi_polygon = self.get_absolute_points()
        count = 0
        
        for detection in detections:
            if str(detection['class_id']) == count_class:
                # Zentrum der Bounding Box berechnen
                box = detection['box']
                center_x = (box['x1'] + box['x2']) // 2
                center_y = (box['y1'] + box['y2']) // 2
                
                if self.point_in_polygon((center_x, center_y), foi_polygon):
                    count += 1
        
        self.current_count = count
        return count
    
    def check_alert_objects_in_foi(self, detections):
        """Überprüft Alert-Objekte im FOI und aktualisiert Lift-Status - VERBESSERT"""
        if not self.foi_config.get('enabled', False):
            return
            
        alert_class = self.foi_config.get('alert_class')
        if not alert_class:
            return
            
        foi_polygon = self.get_absolute_points()
        alert_object_found = False
        
        # Prüfe ob Alert-Objekt im FOI ist
        for detection in detections:
            if str(detection['class_id']) == alert_class:
                box = detection['box']
                center_x = (box['x1'] + box['x2']) // 2
                center_y = (box['y1'] + box['y2']) // 2
                
                if self.point_in_polygon((center_x, center_y), foi_polygon):
                    alert_object_found = True
                    break
        
        # Status-Management - VERBESSERT
        current_time = time.time()
        
        # Manueller Reset wurde angefordert
        if self.manual_reset_requested:
            self._reset_to_normal()
            self.manual_reset_requested = False
            return
        
        if alert_object_found:
            if not self.alert_active:
                # Erstes Auftreten des Alert-Objekts
                self.alert_active = True
                self.alert_start_time = current_time
                self.lift_status = "Lift verlangsamt"
            # Wenn bereits aktiv, Status beibehalten und Timer prüfen
            elif self.alert_start_time and (current_time - self.alert_start_time) > self.foi_config.get('alert_timeout', 10.0):
                self.lift_status = "Lift wird gestoppt. Personal informiert"
        else:
            if self.alert_active:
                # Alert-Objekt ist verschwunden
                if self.alert_start_time and (current_time - self.alert_start_time) <= self.foi_config.get('alert_timeout', 10.0):
                    # VERBESSERUNG: Vollständiger Reset zu Normalbetrieb
                    self.lift_status = "Lift wieder auf Normalgeschwindigkeit"
                    # Timer für kurze Anzeige, dann Reset zu Normalbetrieb
                    self._schedule_normal_reset()
                else:
                    # Zeit überschritten, Status bleibt "gestoppt" bis manueller Reset
                    pass
            else:
                # Kein Alert aktiv - sicherstellen dass Status normal ist
                if self.lift_status != "Lift Normalbetrieb" and "Normalgeschwindigkeit" not in self.lift_status:
                    self._reset_to_normal()
        
        self.alert_object_in_foi = alert_object_found
    
    def _schedule_normal_reset(self):
        """Hilfsmethode: Plant Reset zu Normalbetrieb nach kurzer Anzeige"""
        # Nach 3 Sekunden "Normalgeschwindigkeit" zurück zu "Normalbetrieb"
        import threading
        def delayed_reset():
            time.sleep(3.0)
            if self.lift_status == "Lift wieder auf Normalgeschwindigkeit":
                self._reset_to_normal()
        
        reset_thread = threading.Thread(target=delayed_reset, daemon=True)
        reset_thread.start()
    
    def _reset_to_normal(self):
        """NEUE METHODE: Setzt alle Timer und Status zurück auf Normalbetrieb"""
        self.lift_status = "Lift Normalbetrieb"
        self.alert_active = False
        self.alert_start_time = None
        self.alert_object_in_foi = False
    
    def manual_reset(self):
        """NEUE METHODE: Ermöglicht manuellen Reset des Lift-Status"""
        self.manual_reset_requested = True
        # Sofortiger Reset
        self._reset_to_normal()
    
    def get_remaining_timeout_seconds(self):
        """NEUE METHODE: Gibt verbleibende Sekunden bis Lift-Stopp zurück"""
        if not self.alert_active or not self.alert_start_time:
            return None
        
        current_time = time.time()
        elapsed = current_time - self.alert_start_time
        timeout = self.foi_config.get('alert_timeout', 10.0)
        remaining = timeout - elapsed
        
        return max(0, remaining)
    
    def get_alert_duration(self):
        """NEUE METHODE: Gibt die Dauer des aktuellen Alerts zurück"""
        if not self.alert_active or not self.alert_start_time:
            return 0
        
        current_time = time.time()
        return current_time - self.alert_start_time
    
    def draw_foi_on_frame(self, frame):
        """Zeichnet das FOI auf den Frame"""
        if not self.foi_config.get('enabled', False):
            return frame
            
        # FOI-Polygon zeichnen
        points = self.get_absolute_points()
        foi_color = self.foi_config.get('foi_color', (0, 255, 255))
        foi_thickness = self.foi_config.get('foi_thickness', 3)
        
        # Polygon-Linien zeichnen
        cv2.polylines(frame, [points], True, foi_color, foi_thickness)
        
        # Ecken-Griffe zeichnen
        for i, point in enumerate(points):
            if i == self.hover_corner:
                color = FOI_HOVER_COLOR
                size = FOI_CORNER_SIZE + 2
            elif i == self.dragging_corner:
                color = FOI_CORNER_COLOR
                size = FOI_CORNER_SIZE + 2
            else:
                color = foi_color
                size = FOI_CORNER_SIZE
                
            cv2.circle(frame, tuple(point), size, color, -1)
            cv2.circle(frame, tuple(point), size, (0, 0, 0), 2)  # Schwarzer Rand
        
        return frame
    
    def draw_count_display(self, frame):
        """Zeichnet die Objektzählung und Timer-Info oberhalb des FOI - ERWEITERT"""
        if not self.foi_config.get('enabled', False):
            return frame
            
        points = self.get_absolute_points()
        
        # Finde den höchsten Punkt (kleinste Y-Koordinate)
        top_y = min(point[1] for point in points)
        center_x = int(np.mean([point[0] for point in points]))
        
        # Text-Position oberhalb des FOI
        text_x = center_x
        base_y = max(50, top_y - 20)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Hauptzählung (nur wenn count_class definiert ist)
        count_class = self.foi_config.get('count_class')
        if count_class:
            count_text = f"Personen im FOI: {self.current_count}"
            self._draw_text_with_background(frame, count_text, text_x, base_y, font, font_scale, thickness, (255, 255, 255), (0, 0, 0))
            
        # Timer-Info hinzufügen wenn Alert aktiv
        if self.alert_active and self.alert_start_time:
            remaining = self.get_remaining_timeout_seconds()
            if remaining is not None and remaining > 0:
                timer_text = f"Lift-Stopp in: {remaining:.1f}s"
                timer_y = base_y + 35 if count_class else base_y
                timer_color = (0, 255, 255) if remaining > 5 else (0, 0, 255)  # Gelb oder Rot
                self._draw_text_with_background(frame, timer_text, text_x, timer_y, font, font_scale - 0.1, thickness, timer_color, (0, 0, 0))
            elif "gestoppt" in self.lift_status:
                timer_text = "Lift gestoppt - Manueller Reset erforderlich"
                timer_y = base_y + 35 if count_class else base_y
                self._draw_text_with_background(frame, timer_text, text_x, timer_y, font, font_scale - 0.1, thickness, (0, 0, 255), (0, 0, 0))
        
        return frame
    
    def _draw_text_with_background(self, frame, text, center_x, y, font, font_scale, thickness, text_color, bg_color):
        """Hilfsmethode: Zeichnet Text mit Hintergrund"""
        # Textgröße ermitteln
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Hintergrund zeichnen
        bg_x1 = center_x - text_width // 2 - 10
        bg_y1 = y - text_height - 10
        bg_x2 = center_x + text_width // 2 + 10
        bg_y2 = y + 5
        
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), 2)
        
        # Text zeichnen
        cv2.putText(frame, text, (center_x - text_width // 2, y), 
                   font, font_scale, text_color, thickness)
    
    def get_lift_status(self):
        """Gibt den aktuellen Lift-Status zurück"""
        return self.lift_status
    
    def reset_status(self):
        """VERBESSERUNG: Vollständiger Reset des Lift-Status"""
        self._reset_to_normal()
    
    def is_alert_active(self):
        """NEUE METHODE: Prüft ob ein Alert aktiv ist"""
        return self.alert_active
    
    def get_status_info(self):
        """NEUE METHODE: Gibt detaillierte Status-Informationen zurück"""
        info = {
            'status': self.lift_status,
            'alert_active': self.alert_active,
            'alert_duration': self.get_alert_duration(),
            'remaining_timeout': self.get_remaining_timeout_seconds(),
            'person_count': self.current_count,
            'alert_object_present': self.alert_object_in_foi
        }
        return info