"""
Test face recognition functionality.
Run this to verify DeepFace and face recognition are working.

Usage:
    python test_face_recognition.py
"""

import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'FaceRecognitionSystem.settings')
django.setup()

from recognition.face_recognition_engine import FaceRecognitionEngine
from recognition.models import Person
import cv2
import numpy as np

print("=" * 60)
print("Face Recognition Test")
print("=" * 60)

# Test 1: Initialize Engine
print("\n1. Initializing Face Recognition Engine...")
try:
    engine = FaceRecognitionEngine(threshold=0.6, model_name='Facenet512')
    print("✓ Engine initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize engine: {e}")
    print("\nTrying alternative model (Facenet)...")
    try:
        engine = FaceRecognitionEngine(threshold=0.6, model_name='Facenet')
        print("✓ Engine initialized with Facenet")
    except Exception as e2:
        print(f"✗ All models failed: {e2}")
        sys.exit(1)

# Test 2: Check Database
print("\n2. Checking Database...")
persons = Person.objects.filter(is_active=True, face_embedding__isnull=False)
print(f"✓ Found {persons.count()} person(s) with embeddings")

if persons.count() == 0:
    print("\n⚠ No persons found in database!")
    print("Please upload at least one person's photo first:")
    print("1. Go to: http://localhost:8000")
    print("2. Upload a photo with person's name")
    print("3. Run this test again")
    sys.exit(0)

for person in persons:
    print(f"  - {person.name} (embedding shape: {len(person.face_embedding)})")

# Test 3: Test Face Detection
print("\n3. Testing Face Detection with Webcam...")
print("Press 'q' to quit, 's' to test recognition")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("✗ Could not open webcam")
    sys.exit(1)

print("✓ Webcam opened")
print("\nLook at the camera. Press 's' to test recognition...")

# Load known embeddings
known_embeddings = {}
for person in persons:
    known_embeddings[person.name] = np.array(person.face_embedding)

recognition_count = 0
unknown_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to read frame")
        break
    
    # Detect faces
    faces = engine.detect_faces(frame)
    
    # Draw boxes
    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {face['confidence']:.2f}", 
                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display info
    cv2.putText(frame, f"Faces detected: {len(faces)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Press 's' to test recognition, 'q' to quit", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Recognized: {recognition_count}, Unknown: {unknown_count}", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imshow('Face Recognition Test', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('s') and len(faces) > 0:
        print("\n4. Testing Recognition...")
        
        for i, face in enumerate(faces):
            print(f"\n  Face {i+1}:")
            name, confidence = engine.recognize_face(frame, face, known_embeddings)
            
            if name:
                print(f"    ✓ Recognized: {name} (distance: {confidence:.4f})")
                recognition_count += 1
            else:
                print(f"    ✗ Unknown person")
                unknown_count += 1
            
            # Draw result on frame
            frame_copy = frame.copy()
            engine.draw_face_box(frame_copy, face, name, confidence)
            cv2.imshow('Recognition Result', frame_copy)
            cv2.waitKey(2000)  # Show for 2 seconds
            cv2.destroyWindow('Recognition Result')

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("Test Summary:")
print(f"  - Total recognitions tested: {recognition_count + unknown_count}")
print(f"  - Successfully recognized: {recognition_count}")
print(f"  - Unknown: {unknown_count}")
print("=" * 60)

if recognition_count > 0:
    print("\n✓ Face recognition is working!")
else:
    print("\n⚠ No faces were recognized. Possible issues:")
    print("  1. Person not in database")
    print("  2. Poor lighting or image quality")
    print("  3. Threshold too strict (try increasing to 0.7-0.8)")
    print("  4. Different facial expression or angle")