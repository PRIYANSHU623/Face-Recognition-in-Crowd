"""
YOLOv8 Face Detection Engine
Integrates trained YOLOv8 model for face detection.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class YOLOv8FaceDetector:
    """
    Face detector using trained YOLOv8 model.
    Compatible with face recognition engine interface.
    """
    
    def __init__(self, model_path='models/best.pt', conf_threshold=0.15, img_size=640):
        """
        Initialize YOLOv8 face detector.
        
        Args:
            model_path: Path to trained YOLOv8 model (.pt file)
            conf_threshold: Confidence threshold for detection (0.0-1.0)
            img_size: Input image size for inference (320, 640, 1280)
        """
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        
        # Get absolute path to model
        base_dir = Path(__file__).resolve().parent.parent
        self.model_path = base_dir / model_path
        
        # Check if model exists
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLOv8 model not found at: {self.model_path}\n"
                f"Please place your trained model at: models/best.pt"
            )
        
        # Load YOLOv8 model
        try:
            logger.info(f"Loading YOLOv8 model from: {self.model_path}")
            self.model = YOLO(str(self.model_path))
            logger.info("✓ YOLOv8 model loaded successfully")
            
            # Print model info
            if hasattr(self.model, 'names'):
                logger.info(f"Model classes: {self.model.names}")
            else:
                logger.warning("Model has no class names - this might cause issues")
            
            # Run a dummy inference to warm up the model
            logger.info("Warming up model...")
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_frame, conf=0.5, verbose=False)
            logger.info("✓ Model ready")
            
        except Exception as e:
            logger.error(f"Error loading YOLOv8 model: {e}")
            raise
    
    def detect_faces(self, frame):
        """
        Detect faces in frame using YOLOv8.
        
        Args:
            frame: OpenCV BGR image (numpy array)
            
        Returns:
            List of face dictionaries compatible with MTCNN format:
            [
                {
                    'box': [x, y, width, height],
                    'confidence': 0.95,
                    'keypoints': {...}  # Optional
                },
                ...
            ]
        """
        try:
            # Run YOLOv8 inference
            results = self.model(
                frame,
                conf=self.conf_threshold,
                verbose=False,
                device='cpu'  # Change to 'cuda' if you have GPU
            )
            
            faces = []
            
            # Process results
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Convert to [x, y, width, height] format (MTCNN compatible)
                    x = int(x1)
                    y = int(y1)
                    w = int(x2 - x1)
                    h = int(y2 - y1)
                    
                    # Get confidence score
                    conf = float(box.conf[0])
                    
                    # Create face dictionary (MTCNN compatible format)
                    face_dict = {
                        'box': [x, y, w, h],
                        'confidence': conf
                    }
                    
                    # Add keypoints if available (YOLOv8-pose)
                    if hasattr(result, 'keypoints') and result.keypoints is not None:
                        kpts = result.keypoints[0].xy.cpu().numpy()
                        face_dict['keypoints'] = {
                            'left_eye': tuple(kpts[0]) if len(kpts) > 0 else None,
                            'right_eye': tuple(kpts[1]) if len(kpts) > 1 else None,
                            'nose': tuple(kpts[2]) if len(kpts) > 2 else None,
                            'mouth_left': tuple(kpts[3]) if len(kpts) > 3 else None,
                            'mouth_right': tuple(kpts[4]) if len(kpts) > 4 else None,
                        }
                    
                    faces.append(face_dict)
            
            logger.debug(f"YOLOv8 detected {len(faces)} face(s)")
            return faces
            
        except Exception as e:
            logger.error(f"Error in YOLOv8 face detection: {e}")
            return []
    
    def detect_and_draw(self, frame, draw=True, color=(0, 255, 0)):
        """
        Detect faces and optionally draw bounding boxes.
        
        Args:
            frame: OpenCV BGR image
            draw: Whether to draw boxes on frame
            color: Box color (B, G, R)
            
        Returns:
            Tuple of (faces_list, annotated_frame)
        """
        faces = self.detect_faces(frame)
        
        if draw:
            annotated_frame = frame.copy()
            
            for face in faces:
                x, y, w, h = face['box']
                conf = face['confidence']
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw confidence label
                label = f"Face: {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                
                # Background for text
                cv2.rectangle(
                    annotated_frame,
                    (x, y - label_size[1] - 10),
                    (x + label_size[0], y),
                    color,
                    -1
                )
                
                # Text
                cv2.putText(
                    annotated_frame,
                    label,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )
                
                # Draw keypoints if available
                if 'keypoints' in face:
                    for kp_name, kp_coords in face['keypoints'].items():
                        if kp_coords:
                            cv2.circle(annotated_frame, (int(kp_coords[0]), int(kp_coords[1])), 
                                     3, (0, 0, 255), -1)
            
            return faces, annotated_frame
        
        return faces, frame
    
    def get_model_info(self):
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_path': str(self.model_path),
            'confidence_threshold': self.conf_threshold,
            'model_type': 'YOLOv8',
            'classes': self.model.names if hasattr(self.model, 'names') else {},
            'device': str(self.model.device) if hasattr(self.model, 'device') else 'cpu'
        }
    
    def set_confidence(self, threshold):
        """
        Update confidence threshold.
        
        Args:
            threshold: New confidence threshold (0.0-1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.conf_threshold = threshold
            logger.info(f"Confidence threshold updated to: {threshold}")
        else:
            logger.warning(f"Invalid threshold: {threshold}. Must be between 0.0 and 1.0")
    
    def benchmark(self, frame, iterations=100):
        """
        Benchmark detection speed.
        
        Args:
            frame: Test frame
            iterations: Number of iterations
            
        Returns:
            Average FPS
        """
        import time
        
        logger.info(f"Running benchmark with {iterations} iterations...")
        
        start_time = time.time()
        for _ in range(iterations):
            _ = self.detect_faces(frame)
        end_time = time.time()
        
        total_time = end_time - start_time
        fps = iterations / total_time
        
        logger.info(f"Benchmark results: {fps:.2f} FPS (avg: {(total_time/iterations)*1000:.2f}ms per frame)")
        
        return fps