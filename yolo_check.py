from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt
import os

# Load your fine-tuned model
model = YOLO("/home/priyanshu/Desktop/study/DesignProject/face_recognition_project/FaceRecognitionSystem/models/best.pt")

# Set your image path
img_path = "/home/priyanshu/Desktop/study/DesignProject/face_recognition_project/img_check.jpg"   # ← CHANGE THIS to any real image name

# Run detection
results = model(img_path)

# Draw bounding boxes
annotated_img = results[0].plot()

# Count predicted faces
predicted_faces = len(results[0].boxes)

# Find corresponding label file
label_path = img_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"

# Count actual faces in label file
if os.path.exists(label_path):
    with open(label_path, "r") as f:
        gt_faces = len(f.readlines())
else:
    gt_faces = 0

print("✅ Detection Results")
print("Predicted faces :", predicted_faces)
print("Actual faces    :", gt_faces)

# Show the image with bounding boxes
plt.figure(figsize=(8,8))
plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
