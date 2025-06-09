from PyQt6.QtCore import QRunnable, QObject, pyqtSignal

class WorkerSignals(QObject):
    """Defines the signals available from the worker thread."""
    result = pyqtSignal(object)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

class DualDetectionWorker(QRunnable):
    """Worker thread for processing video frames with detection first, then pose on detected objects."""
    
    def __init__(self, frame, detection_model, pose_model, class_config, pose_config):
        super().__init__()
        self.frame = frame.copy()
        self.detection_model = detection_model
        self.pose_model = pose_model
        self.class_config = class_config
        self.pose_config = pose_config
        self.signals = WorkerSignals()
        
    def run(self):
        try:
            detections = []
            poses = []
            
            # Step 1: Object Detection
            if self.detection_model:
                det_results = self.detection_model.predict(self.frame, verbose=False)[0]
                
                pose_detect_classes = self.pose_config.get('pose_detect_classes', [])
                
                for box in det_results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    cls_id = str(cls)
                    cfg = self.class_config.get(cls_id)
                    if not cfg or conf < float(cfg.get("conf", 0.5)):
                        continue
                    
                    detection = {
                        'box': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                        'conf': conf,
                        'class_id': cls,
                        'class_name': cfg.get('name', f"Class {cls}")
                    }
                    detections.append(detection)
                    
                    # Step 2: Pose Detection auf ausgeschnittenen Bereichen
                    if (self.pose_model and pose_detect_classes and 
                        cls_id in pose_detect_classes):
                        
                        pose_data_list = self._detect_pose_in_roi(x1, y1, x2, y2, detection)
                        if pose_data_list:
                            poses.extend(pose_data_list)  # Erweitern statt einzeln hinzufügen
            
            # Emit the result
            self.signals.result.emit((self.frame, detections, poses))
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()
    
    def _detect_pose_in_roi(self, x1, y1, x2, y2, detection):
        """Führt Pose Detection in einem bestimmten Bereich durch - KANN MEHRERE PERSONEN ERKENNEN"""
        # Bounding Box erweitern und beschränken
        h, w = self.frame.shape[:2]
        margin = 20  # Pixel Spielraum um die Box
        x1_exp = max(0, x1 - margin)
        y1_exp = max(0, y1 - margin)
        x2_exp = min(w, x2 + margin)
        y2_exp = min(h, y2 + margin)
        
        # Ausschnitt extrahieren
        roi = self.frame[y1_exp:y2_exp, x1_exp:x2_exp]
        
        if roi.shape[0] <= 0 or roi.shape[1] <= 0:
            return None
        
        # Pose Detection auf ROI
        pose_results = self.pose_model.predict(roi, verbose=False)[0]
        
        if not hasattr(pose_results, 'keypoints') or pose_results.keypoints is None:
            return None
        
        poses_in_roi = []  # Liste für mehrere erkannte Personen
        
        # Iteriere über alle erkannten Personen in der Bounding Box
        for person_idx, keypoints in enumerate(pose_results.keypoints.xy):
            if pose_results.keypoints.conf is not None:
                confs = pose_results.keypoints.conf[person_idx]
            else:
                confs = [1.0] * len(keypoints)
            
            # Filter keypoints by confidence
            min_conf = self.pose_config.get('min_confidence', 0.3)
            valid_keypoints = []
            
            for i, (kp, conf_kp) in enumerate(zip(keypoints, confs)):
                if conf_kp >= min_conf and kp[0] > 0 and kp[1] > 0:
                    # Koordinaten zurück ins Vollbild transformieren
                    global_x = float(kp[0]) + x1_exp
                    global_y = float(kp[1]) + y1_exp
                    
                    valid_keypoints.append({
                        'id': i,
                        'x': global_x,
                        'y': global_y,
                        'conf': float(conf_kp)
                    })
            
            if valid_keypoints:  # Nur hinzufügen wenn gültige Keypoints
                pose_data = {
                    'person_id': f"{detection['class_id']}_{person_idx}",
                    'detection_box': detection['box'],
                    'keypoints': valid_keypoints
                }
                poses_in_roi.append(pose_data)
        
        return poses_in_roi if poses_in_roi else None