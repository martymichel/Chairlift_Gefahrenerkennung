import cv2
from config.constants import POSE_CONNECTIONS

class FrameRenderer:
    """Verantwortlich für das Zeichnen von Detections und Poses auf Frames"""
    
    def __init__(self, class_config, pose_config, display_config):
        self.class_config = class_config
        self.pose_config = pose_config
        self.display_config = display_config
    
    def update_config(self, class_config, pose_config, display_config):
        """Aktualisiert die Konfiguration"""
        self.class_config = class_config
        self.pose_config = pose_config
        self.display_config = display_config
    
    def render_frame(self, frame, detections, poses):
        """Zeichnet Detections und Poses auf den Frame"""
        if frame is None:
            return frame
            
        # Create a copy to draw on
        rendered_frame = frame.copy()
        
        # Draw detection boxes
        self._draw_detections(rendered_frame, detections)
        
        # Draw poses
        self._draw_poses(rendered_frame, poses)
        
        return rendered_frame
    
    def _draw_detections(self, frame, detections):
        """Zeichnet Detection-Bounding-Boxes"""
        box_thickness = self.display_config.get('box_thickness', 2)
        font_scale = self.display_config.get('font_scale', 5) / 10.0
        text_thickness = self.display_config.get('text_thickness', 1)
        
        for detection in detections:
            cls_id = str(detection['class_id'])
            cfg = self.class_config.get(cls_id)
            if not cfg:
                continue
                
            box = detection['box']
            conf = detection['conf']
            
            color = cfg['color']
            label = f"{cfg['name']} {conf:.2f}"
            
            # Draw bounding box
            cv2.rectangle(frame, (box['x1'], box['y1']), (box['x2'], box['y2']), 
                         color, box_thickness)
            
            # Draw label
            cv2.putText(frame, label, (box['x1'], box['y1'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness)
    
    def _draw_poses(self, frame, poses):
        """Zeichnet Pose-Keypoints und Skelett"""
        if not (self.pose_config.get('show_keypoints', True) or 
                self.pose_config.get('show_skeleton', True)):
            return
        
        line_thickness = self.pose_config.get('line_thickness', 2)
        keypoint_radius = self.pose_config.get('keypoint_radius', 3)
        show_keypoints = self.pose_config.get('show_keypoints', True)
        show_skeleton = self.pose_config.get('show_skeleton', True)
        
        for pose in poses:
            keypoints = pose['keypoints']
            
            if not keypoints:  # Skip wenn keine gültigen Keypoints
                continue
            
            # Erstelle Keypoint-Array für einfacheren Zugriff
            kp_array = [None] * 17  # COCO hat 17 Keypoints
            for kp in keypoints:
                if kp['id'] < 17 and kp['x'] > 0 and kp['y'] > 0:  # Gültige Koordinaten
                    kp_array[kp['id']] = (int(kp['x']), int(kp['y']))
            
            # Zeichne Skelett-Verbindungen
            if show_skeleton:
                self._draw_skeleton(frame, kp_array, line_thickness)
            
            # Zeichne Keypoints
            if show_keypoints:
                self._draw_keypoints(frame, keypoints, keypoint_radius)
    
    def _draw_skeleton(self, frame, kp_array, line_thickness):
        """Zeichnet Skelett-Verbindungen"""
        for connection in POSE_CONNECTIONS:
            pt1_idx, pt2_idx = connection
            if (pt1_idx < len(kp_array) and pt2_idx < len(kp_array) and 
                kp_array[pt1_idx] is not None and kp_array[pt2_idx] is not None):
                
                # Zusätzliche Validierung der Koordinaten
                pt1 = kp_array[pt1_idx]
                pt2 = kp_array[pt2_idx]
                
                if self._are_valid_coordinates(pt1, frame) and self._are_valid_coordinates(pt2, frame):
                    cv2.line(frame, pt1, pt2, (0, 255, 0), line_thickness)
    
    def _draw_keypoints(self, frame, keypoints, keypoint_radius):
        """Zeichnet Keypoints"""
        for kp in keypoints:
            x, y = int(kp['x']), int(kp['y'])
            if self._are_valid_coordinates((x, y), frame):
                cv2.circle(frame, (x, y), keypoint_radius, (0, 0, 255), -1)
    
    def _are_valid_coordinates(self, point, frame):
        """Überprüft ob die Koordinaten gültig sind"""
        x, y = point
        return (x > 0 and y > 0 and 
                x < frame.shape[1] and y < frame.shape[0])  # Innerhalb Bildgrenzen