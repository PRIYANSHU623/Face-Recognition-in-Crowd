# Face-Recognition-in-Crowd

Monitoring crowded places manually is a tedious and time-consuming process, and human intervention makes it prone to errors. Therefore, automated face recognition has become one of the most widely used technologies in security and surveillance systems, as it can identify individuals efficiently even in crowded environments and less prone to error. It uses Django-based web application to manage multiple camera feeds, store suspect data, process recognition events, and send alerts to security authorities in real time.
Read Documentation : [Documentation.pdf](https://github.com/user-attachments/files/26600042/DESIGN_PROJECT.10.pdf)

## Models Used
### YOLOv8 — Face Detection

YOLOv8 is a state-of-the-art object detection model known for its high accuracy and low latency. In this project, YOLOv8 was trained on the WIDERFACE dataset to specialize in face detection, especially in crowded and challenging scenes.
The model detects multiple faces in each video frame efficiently, making it suitable for real-time surveillance applications. Its fast inference speed and strong performance in dense environments make it ideal for deployment in live monitoring systems.


### FaceNet512 — Face Recognition

FaceNet512 is a deep learning model used for face recognition that converts a face image into a 512-dimensional feature embedding. Instead of directly classifying faces, the model learns a numerical representation that captures the unique facial characteristics of each individual.
In this system, once faces are detected using YOLOv8, the cropped face images are passed to FaceNet512 to generate embeddings. These embeddings are then compared with the stored embeddings of suspect images in the database using a distance metric such as Euclidean distance or cosine similarity.
If the similarity score exceeds a predefined threshold, the system identifies the person as a match and triggers an alert. This embedding-based approach makes the recognition process scalable and effective, even when new suspects are added without retraining the entire model.

## Poster with Flow Digram

[Poster.pdf](https://github.com/user-attachments/files/26599833/DesignPoster.pdf)

## Model Output
During testing on a group image containing Dhoni, Rohit, and Sachin, the system accurately detects multiple faces, generates bounding boxes for each individual, and assigns confidence scores to each detection, demonstrating effective multi-face recognition in crowded scenarios.
<img width="1024" height="576" alt="Multi-Suspect Detection (Gray + RGB Boxes)_screenshot_12 11 2025" src="https://github.com/user-attachments/assets/20dfb759-9859-46a4-acad-bcd1300a135d" />
