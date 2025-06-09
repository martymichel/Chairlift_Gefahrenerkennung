import cv2
import numpy as np
import time
from config.constants import FOI_CORNER_SIZE, FOI_CORNER_COLOR, FOI_HOVER_COLOR
from PyQt6.QtGui import QPixmap, QImage, QMouseEvent
from PyQt6.QtCore import Qt, QRectF

class FOIManager:
    """Verwaltet das Field of Interest (FOI) für die Skilift-Überwachung"""
    
    def __init__(self, foi_config):
        self.foi_config = foi_config
        self.frame_width = 1
        self.frame_height = 1
        
        # FOI-Status
        self.dragging_corner = -1
        self.hover_corner = -1
        
        # Lift-Status und Timing
        self.lift_status = "Normalbetrieb"
        self.alert_start_time = None
        self.alert_active = False
        
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
        """Überprüft Alert-Objekte im FOI und aktualisiert Lift-Status"""
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
        
        # Status-Management
        current_time = time.time()
        
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
                    self.lift_status = "Lift wieder auf Normalgeschwindigkeit"
                    self.alert_active = False
                    self.alert_start_time = None
                else:
                    # Zeit überschritten, Status bleibt "gestoppt"
                    pass
            else:
                # Kein Alert aktiv
                self.lift_status = "Lift Normalbetrieb"
                self.alert_start_time = None
        
        self.alert_object_in_foi = alert_object_found
    
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
        """Zeichnet die Objektzählung oberhalb des FOI"""
        if not self.foi_config.get('enabled', False) or not self.foi_config.get('count_class'):
            return frame
            
        points = self.get_absolute_points()
        
        # Finde den höchsten Punkt (kleinste Y-Koordinate)
        top_y = min(point[1] for point in points)
        center_x = int(np.mean([point[0] for point in points]))
        
        # Text-Position oberhalb des FOI
        text_x = center_x
        text_y = max(30, top_y - 20)
        
        # Text zeichnen
        text = f"Personen im FOI: {self.current_count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Textgröße ermitteln für Hintergrund
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Hintergrund zeichnen
        bg_x1 = text_x - text_width // 2 - 10
        bg_y1 = text_y - text_height - 10
        bg_x2 = text_x + text_width // 2 + 10
        bg_y2 = text_y + 5
        
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), 2)
        
        # Text zeichnen
        cv2.putText(frame, text, (text_x - text_width // 2, text_y), 
                   font, font_scale, (255, 255, 255), thickness)
        
        return frame
    
    def get_lift_status(self):
        """Gibt den aktuellen Lift-Status zurück"""
        return self.lift_status
    
    def reset_status(self):
        """Setzt den Lift-Status zurück"""
        self.lift_status = "Lift Normalbetrieb"
        self.alert_active = False
        self.alert_start_time = None
        self.alert_object_in_foi = False