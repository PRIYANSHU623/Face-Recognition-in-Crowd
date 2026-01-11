
# """
# Face recognition engine using MTCNN for detection and DeepFace for recognition.
# """

import cv2
import numpy as np
from mtcnn import MTCNN

from ultralytics import YOLO

from deepface import DeepFace
import logging
from PIL import Image
import io
import os
from .yolo_face_detector import YOLOv8FaceDetector


logger = logging.getLogger(__name__)

class FaceRecognitionEngine:
#     """
#     Handles face detection and recognition using MTCNN and DeepFace.
#     """
    
#     def __init__(self, threshold=0.5):
#         print("✅ FaceRecognitionEngine initialized (Facenet512)")
#         """
#         Initialize the face recognition engine.
        
#         Args:
#             threshold: Distance threshold for face matching (lower = stricter)
#         """
#         self.threshold = threshold
        
#         # self.detector = YOLO("FaceRecognitionSystem/recognition/best.pt")
#         self.detector = MTCNN()
#         self.model_name = 'Facenet512'  # Use consistent model for all embeddings
#         logger.info(f"Face Recognition Engine initialized with threshold: {threshold}")
    
#     def detect_faces(self, frame):
#         """
#         Detect faces in a frame using MTCNN.
#         """
#         try:
#             # Convert BGR to RGB for MTCNN
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             faces = self.detector.detect_faces(rgb_frame)
#             logger.info(f"✅ Detected {len(faces)} faces in frame")
#             return faces
#         except Exception as e:
#             logger.error(f"❌ Error detecting faces: {e}")
#             return []

    def __init__(self, threshold=0.6, model_name='Facenet512', use_yolo=True):
        """
        Initialize the face recognition engine.
        
        Args:
            threshold: Distance threshold for face matching
            model_name: Face recognition model name
            use_yolo: If True, use YOLOv8 for detection, else use MTCNN
        """
        self.threshold = threshold
        self.model_name = model_name
        self.use_yolo = use_yolo
        
        # Initialize detector
        if use_yolo:
            logger.info("Using YOLOv8 for face detection")
            try:
                print("successfully to load yolo++++++++++++++++++++++++++++++") 

                self.detector = YOLOv8FaceDetector(
                    model_path='models/best.pt',
                    conf_threshold=0.5
                )
                self.detector_type = 'yolo'
            except Exception as e:
                print("failed to load yolo++++++++++++++++++++++++++++++") 
                logger.error(f"Failed to load YOLOv8: {e}")
                logger.info("Falling back to MTCNN")
                from mtcnn import MTCNN
                self.detector = MTCNN()
                self.detector_type = 'mtcnn'
        else:
            logger.info("Using MTCNN for face detection")
            from mtcnn import MTCNN
            self.detector = MTCNN()
            self.detector_type = 'mtcnn'
        
        # Initialize face recognition (DeepFace)
        try:
            from deepface import DeepFace
            self.DeepFace = DeepFace
            logger.info(f"Loading {model_name} model...")
            _ = DeepFace.represent(
                img_path=np.zeros((224, 224, 3), dtype=np.uint8),
                model_name=model_name,
                enforce_detection=False
            )
            logger.info(f"✓ Face Recognition Engine initialized")
        except Exception as e:
            logger.error(f"Error loading DeepFace: {e}")
            self.DeepFace = None
    
    def detect_faces(self, frame):
        """
        Detect faces in frame using YOLOv8 or MTCNN.
        
        Args:
            frame: OpenCV BGR image
            
        Returns:
            List of face dictionaries with 'box' and 'confidence'
        """
        try:
            if self.detector_type == 'yolo':
                # YOLOv8 returns faces in correct format
                return self.detector.detect_faces(frame)
            else:
                # MTCNN - convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return self.detector.detect_faces(rgb_frame)
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def extract_embedding(self, image_path=None, frame=None):
        """
        Extract face embedding from an image or frame.
        """
        try:
            if image_path:
                embedding_objs = DeepFace.represent(
                    img_path=image_path,
                    model_name=self.model_name,
                    enforce_detection=True,
                    detector_backend='mtcnn'
                )
                if embedding_objs and len(embedding_objs) > 0:
                    return np.array(embedding_objs[0]['embedding'])
            
            elif frame is not None:
                print("🧠 Creating embedding for detected face region")
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                buffer = io.BytesIO()
                pil_image.save(buffer, format='JPEG')
                buffer.seek(0)
                
                embedding_objs = DeepFace.represent(
                    img_path=np.array(pil_image),
                    model_name=self.model_name,
                    enforce_detection=False,
                    detector_backend='skip'
                )
                
                if embedding_objs and len(embedding_objs) > 0:
                    return np.array(embedding_objs[0]['embedding'])
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error extracting embedding: {e}")
            return None
    
    def compare_faces(self, embedding1, embedding2):
        """
        Compare two face embeddings using cosine distance.
        """
        print("🔍 Comparing faces ...")
        try:
            if embedding1 is None or embedding2 is None:
                return float('inf')
            
            emb1 = np.array(embedding1)
            
            emb2 = np.array(embedding2)

            # Debug print to check dimensions
            print(f"[DEBUG] Embedding1 shape: {emb1.shape}, Embedding2 shape: {emb2.shape}")
            
            # Compute cosine similarity (1 - cosine = distance)
            cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            distance = 1-cosine_sim
            print(f"{distance}  , {cosine_sim}========================================")
            return cosine_sim  #distance###############################################3
            
        except Exception as e:
            logger.error(f"❌ Error comparing faces: {e}")
            return float('inf')
    
    def recognize_face(self, frame, face_box, known_embeddings):
        """
        Recognize a face in a frame by comparing with known embeddings.
        """
        try:
            print("🔎 Recognizing face ...")
            x, y, w, h = face_box['box']
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            face_region = frame[y1:y2, x1:x2]
            if face_region.size == 0:
                return None, None
            
            face_embedding = self.extract_embedding(frame=face_region)
            if face_embedding is None:
                return None, None
            
            # Debug print to show embedding dimension
            print(f"[DEBUG] Face embedding shape: {face_embedding.shape}")
            
            best_match = None
            min_distance = float('inf')
            
            for name, known_embedding in known_embeddings.items():
                print(f"[DEBUG] Comparing with {name}")
                distance = self.compare_faces(face_embedding, known_embedding)
                print(f"[DEBUG] Distance from {name}: {distance:.4f}")
                
                if distance > self.threshold:#min_distance:#################################
                    min_distance = distance
                    best_match = name
            
            if min_distance >  self.threshold:
                print(f"✅ Recognized as {best_match} (distance={min_distance:.3f})")
                return best_match, min_distance
            
            print(f"❌ Unknown face (min distance={min_distance:.3f})")
            return None, None
            
        except Exception as e:
            logger.error(f"❌ Error recognizing face: {e}")
            return None, None
    
    def draw_face_box(self, frame, face_box, name=None, confidence=None):
        """
        Draw bounding box and label on frame.
        """
        x, y, w, h = face_box['box']
        color = (0, 255, 0) if name else (0, 165, 255)
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        if name:
            label = f"{name}"
            if confidence is not None:
                label += f" ({confidence:.2f})"
        else:
            label = "Unknown"
        
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width, y), color, -1)
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame




