"""
Camera management and video streaming functionality.
"""

import cv2
import threading
import time
import logging
from datetime import datetime
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
import base64
import numpy as np

logger = logging.getLogger(__name__)


class CameraManager:
    """
    Manages multiple camera connections and video streaming.
    """
    
    def __init__(self):
        self.cameras = {}  # {camera_number: cv2.VideoCapture}
        self.camera_threads = {}  # {camera_number: threading.Thread}
        self.running = False
        self.face_engine = None
        self.known_embeddings = {}
        self.channel_layer = get_channel_layer()
        self.lock = threading.Lock()
        logger.info("CameraManager initialized")
    
    def detect_available_cameras(self, max_cameras=5, prioritize_builtin=True):
        """
        Detect all available cameras on the system.
        
        Args:
            max_cameras: Maximum number of cameras to check
            prioritize_builtin: If True, laptop webcam (camera 0) is checked first
            
        Returns:
            List of available camera indices (laptop webcam first if available)
        """
        available_cameras = []
        
        # Always check camera 0 first (laptop's built-in webcam)
        if prioritize_builtin:
            logger.info("Checking laptop's built-in webcam (Camera 0)...")
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(0)
                    logger.info("✓ Laptop webcam (Camera 0) detected and working")
                else:
                    logger.warning("✗ Camera 0 opened but failed to read frame")
                cap.release()
            else:
                logger.warning("✗ Camera 0 (laptop webcam) not available")
        
        # Then check other cameras (1, 2, 3, 4...)
        for i in range(1, max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(i)
                    logger.info(f"✓ Camera {i} detected and working")
                cap.release()
        
        if not available_cameras:
            logger.error("No cameras detected on the system!")
        else:
            logger.info(f"Total cameras detected: {available_cameras}")
        
        return available_cameras
    
    def start_camera(self, camera_number):
        """
        Initialize and start a specific camera.
        
        Args:
            camera_number: Index of the camera to start
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if camera_number in self.cameras:
                logger.warning(f"Camera {camera_number} already running")
                return True
            
            cap = cv2.VideoCapture(camera_number)
            
            if not cap.isOpened():
                logger.error(f"Failed to open camera {camera_number}")
                return False
            
            # Set camera properties for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            with self.lock:
                self.cameras[camera_number] = cap
            
            logger.info(f"Camera {camera_number} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera {camera_number}: {e}")
            return False
    
    def stop_camera(self, camera_number):
        """
        Stop a specific camera.
        
        Args:
            camera_number: Index of the camera to stop
        """
        try:
            with self.lock:
                if camera_number in self.cameras:
                    self.cameras[camera_number].release()
                    del self.cameras[camera_number]
                    logger.info(f"Camera {camera_number} stopped")
        except Exception as e:
            logger.error(f"Error stopping camera {camera_number}: {e}")
    
    def start_all_cameras(self, laptop_only=False):
        """
        Start all available cameras and begin processing.
        
        Args:
            laptop_only: If True, only use laptop webcam (camera 0)
        """
        if laptop_only:
            logger.info("Laptop-only mode: Attempting to start camera 0 only")
            available_cameras = []
            
            # Try to start only camera 0
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras = [0]
                    logger.info("✓ Laptop webcam (Camera 0) will be used")
                else:
                    logger.error("✗ Laptop webcam opened but failed to read")
                cap.release()
            else:
                logger.error("✗ Laptop webcam (Camera 0) not available")
        else:
            # Detect all available cameras
            available_cameras = self.detect_available_cameras()
        
        if not available_cameras:
            logger.warning("No cameras detected")
            return []
        
        self.running = True
        
        for cam_num in available_cameras:
            if self.start_camera(cam_num):
                # Start processing thread for this camera
                thread = threading.Thread(
                    target=self.process_camera_feed,
                    args=(cam_num,),
                    daemon=True
                )
                thread.start()
                self.camera_threads[cam_num] = thread
        
        logger.info(f"Started {len(available_cameras)} camera(s): {available_cameras}")
        return available_cameras
    
    def stop_all_cameras(self):
        """
        Stop all cameras and processing threads.
        """
        self.running = False
        
        # Wait for threads to finish
        for thread in self.camera_threads.values():
            thread.join(timeout=2)
        
        self.camera_threads.clear()
        
        # Release all cameras
        camera_numbers = list(self.cameras.keys())
        for cam_num in camera_numbers:
            self.stop_camera(cam_num)
        
        logger.info("All cameras stopped")
    
    def process_camera_feed(self, camera_number):
        """
        Process video feed from a specific camera (runs in separate thread).
        
        Args:
            camera_number: Index of the camera to process
        """
        logger.info(f"Started processing thread for camera {camera_number}")
        frame_count = 0
        last_detection_time = {}  # Track last detection time for each person
        detection_cooldown = 5  # Seconds between duplicate detections
        
        while self.running:
            try:
                with self.lock:
                    if camera_number not in self.cameras:
                        break
                    cap = self.cameras[camera_number]
                
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning(f"Failed to read from camera {camera_number}")
                    time.sleep(0.1)
                    continue
                
                # Process every 3rd frame for performance
                frame_count += 1
                if frame_count % 3 != 0:
                    # Still send frame to frontend
                    self.send_frame_to_frontend(camera_number, frame)
                    continue
                
                # Detect faces
                if self.face_engine:
                    faces = self.face_engine.detect_faces(frame)
                    
                    for face in faces:
                        # Recognize face
                        name, confidence = self.face_engine.recognize_face(
                            frame, face, self.known_embeddings
                        )
                        
                        # Draw face box
                        frame = self.face_engine.draw_face_box(
                            frame, face, name, confidence
                        )
                        
                        # Log detection if recognized
                        if name:
                            current_time = time.time()
                            last_time = last_detection_time.get(name, 0)
                            
                            # Only log if cooldown period has passed
                            if current_time - last_time > detection_cooldown:
                                self.log_detection(name, camera_number, confidence, frame, face)
                                last_detection_time[name] = current_time
                
                # Send processed frame to frontend
                self.send_frame_to_frontend(camera_number, frame)
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.01)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error processing camera {camera_number}: {e}")
                time.sleep(0.1)
        
        logger.info(f"Stopped processing thread for camera {camera_number}")
    
    def send_frame_to_frontend(self, camera_number, frame):
        """
        Send video frame to frontend via WebSocket.
        
        Args:
            camera_number: Camera index
            frame: OpenCV BGR image
        """
        try:
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            
            # Convert to base64
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            # Send to WebSocket group
            async_to_sync(self.channel_layer.group_send)(
                'recognition_updates',
                {
                    'type': 'frame_update',
                    'camera_number': camera_number,
                    'frame': jpg_as_text
                }
            )
        except Exception as e:
            logger.error(f"Error sending frame to frontend: {e}")
    
    def log_detection(self, person_name, camera_number, confidence, frame, face):
        """
        Log a detection event to database and notify frontend.
        
        Args:
            person_name: Name of detected person
            camera_number: Camera that detected the person
            confidence: Confidence score
            frame: Current video frame
            face: Face detection dictionary
        """
        try:
            from .models import DetectionLog
            from django.core.files.base import ContentFile
            import io
            from PIL import Image
            
            # Extract face snapshot
            x, y, w, h = face['box']
            face_snapshot = frame[y:y+h, x:x+w]
            
            # Convert to JPEG
            _, buffer = cv2.imencode('.jpg', face_snapshot)
            image_bytes = buffer.tobytes()
            
            # Create detection log
            detection = DetectionLog.objects.create(
                person_name=person_name,
                camera_number=camera_number,
                confidence_score=confidence
            )
            
            # Save snapshot
            detection.snapshot_image.save(
                f'detection_{detection.id}.jpg',
                ContentFile(image_bytes),
                save=True
            )
            
            logger.info(f"Logged detection: {person_name} on camera {camera_number}")
            
            # Notify frontend via WebSocket
            async_to_sync(self.channel_layer.group_send)(
                'recognition_updates',
                {
                    'type': 'detection_alert',
                    'person_name': person_name,
                    'camera_number': camera_number,
                    'confidence': round(confidence, 2),
                    'timestamp': detection.detection_time.isoformat(),
                    'detection_id': detection.id
                }
            )
            
        except Exception as e:
            logger.error(f"Error logging detection: {e}")
    
    def set_face_engine(self, engine):
        """Set the face recognition engine."""
        self.face_engine = engine
    
    def update_known_embeddings(self, embeddings):
        """Update the dictionary of known face embeddings."""
        with self.lock:
            self.known_embeddings = embeddings.copy()
        logger.info(f"Updated known embeddings: {len(embeddings)} persons")