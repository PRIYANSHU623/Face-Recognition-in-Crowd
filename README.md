### Face-Recognition-in-Crowd

Monitoring crowded places manually is a tedious and time-consuming process, and human intervention makes it prone to errors. Therefore, automated face recognition has become one of the most widely used technologies in security and surveillance systems, as it can identify individuals efficiently even in crowded environments and less prone to error. It uses Django-based web application to manage multiple camera feeds, store suspect data, process recognition events, and send alerts to security authorities in real time.


## Models Used
#YOLOv8 — Face Detection

YOLOv8 is a state-of-the-art object detection model known for its high accuracy and low latency. In this project, YOLOv8 was trained on the WIDERFACE dataset to specialize in face detection, especially in crowded and challenging scenes.
The model detects multiple faces in each video frame efficiently, making it suitable for real-time surveillance applications. Its fast inference speed and strong performance in dense environments make it ideal for deployment in live monitoring systems.


#FaceNet512 — Face Recognition

FaceNet512 is a deep learning model used for face recognition that converts a face image into a 512-dimensional feature embedding. Instead of directly classifying faces, the model learns a numerical representation that captures the unique facial characteristics of each individual.
In this system, once faces are detected using YOLOv8, the cropped face images are passed to FaceNet512 to generate embeddings. These embeddings are then compared with the stored embeddings of suspect images in the database using a distance metric such as Euclidean distance or cosine similarity.
If the similarity score exceeds a predefined threshold, the system identifies the person as a match and triggers an alert. This embedding-based approach makes the recognition process scalable and effective, even when new suspects are added without retraining the entire model.
