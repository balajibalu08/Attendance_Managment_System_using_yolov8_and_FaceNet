import cv2
import torch
import numpy as np
from scipy.spatial.distance import cosine
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO
import os

# Check for CUDA availability and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the FaceNet model
try:
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
except Exception as e:
    print(f"Error loading FaceNet model: {e}")
    exit()

# Load YOLOv8 model
try:
    yolo_model = YOLO('yolov8n.pt')  # Load a pretrained YOLOv8n model. You can change this to 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', or 'yolov8x.pt' for different sizes/performance.
    yolo_model.to(device)  # Move model to device
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    exit()

def preprocess_image(image):
    """Preprocess the face image for embedding generation."""
    image = cv2.resize(image, (160, 160))
    image = image.astype('float32') / 127.5 - 1  # Normalize to [-1, 1]
    image = np.transpose(image, (2, 0, 1))  # Convert to channel-first format
    return torch.tensor(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

def detect_and_crop_face(image):
    """Detect face using YOLOv8 and crop it."""
    results = yolo_model(image)  # Perform object detection on the image

    faces = []
    for result in results:
        boxes = result.boxes  # Get the boxes of detected objects
        for box in boxes:
            if box.cls == 0:  # Check if the detected object is a face (class 0 for COCO dataset)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf.item()
                if conf > 0.5:  # Only consider detections with confidence > 0.5
                    faces.append([x1, y1, x2, y2, conf])
    
    if not faces:
        raise ValueError("No face detected in the image.")

    # Get the face with the highest confidence
    best_face = max(faces, key=lambda x: x[4])
    x1, y1, x2, y2, _ = best_face
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image.shape[1], x2), min(image.shape[0], y2)
    cropped_face = image[y1:y2, x1:x2]
    return cropped_face

def create_embedding(image_path):
    """Create face embedding for a given image."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        cropped_face = detect_and_crop_face(image)
        preprocessed_face = preprocess_image(cropped_face)
        with torch.no_grad():
            embedding = model(preprocessed_face)
        return embedding.cpu().squeeze(0).numpy()  # Convert to numpy array and move back to CPU
    except Exception as e:
        print(f"Error creating embedding for {image_path}: {e}")
        return None

def average_embeddings(image_paths):
    """Compute average embeddings from multiple images."""
    embeddings = []
    for image_path in image_paths:
        embedding = create_embedding(image_path)
        if embedding is not None:
            embeddings.append(embedding)

    if not embeddings:
        raise ValueError("No valid embeddings could be created.")
    
    return np.mean(embeddings, axis=0)

def optimize_face_comparison(person_image_paths, new_face_path, threshold=0.75):
    """Optimize face comparison."""
    stored_embeddings = {}
    stored_embeddings["Balaji"] = average_embeddings(person_image_paths)

    new_face_embedding = create_embedding(new_face_path)

    if new_face_embedding is not None:
        for name, embedding in stored_embeddings.items():
            similarity = 1 - cosine(new_face_embedding, embedding)
            print(f"Similarity with {name}: {similarity}")
            if similarity > threshold:  # Adjust threshold as needed
                print(f"Match found: {name}")
            else:
                print("Not a match.")
    else:
        print("Could not create embedding for the new face.")

# Example usage
person_image_paths = [
    "C:\\Users\\balaj\\Downloads\\BJ-275.JPG",
    "C:\\Users\\balaj\\Downloads\\b.jpg",
    "C:\\Users\\balaj\\Downloads\\bb.jpg",
    "C:\\Users\\balaj\\Downloads\\bala.jpg"
]

new_face_path = "C:\\Users\\balaj\\Downloads\\WhatsApp Image 2025-01-07 at 17.27.23_6ae097b0.jpg"

# Optimize face comparison
optimize_face_comparison(person_image_paths, new_face_path)

