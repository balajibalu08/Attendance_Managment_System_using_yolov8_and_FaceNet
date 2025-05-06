from facenet_pytorch import InceptionResnetV1
import torch
from PIL import Image
import numpy as np
import os
from torchvision import transforms
import cv2
from scipy.spatial.distance import cosine
import time
from ultralytics import YOLO

# Load pre-trained FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Load YOLOv8 face detection model
face_detector = YOLO('yolov8n.pt')

# Define a transformation pipeline for images
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def create_embedding(image):
    # Convert numpy array to PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Preprocess the image
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Generate embedding
    with torch.no_grad():
        embedding = model(image)
    return embedding[0].numpy()

def process_directory(directory_path):
    embeddings = {}
    for file_name in os.listdir(directory_path):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(directory_path, file_name)
            try:
                name = input(f"Enter name for {file_name}: ")
                embedding = create_embedding(file_path)
                embeddings[name] = embedding
                print(f"Successfully processed: {name}")
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
    return embeddings

def compare_faces(new_face_embedding, stored_embeddings, threshold=0.7):
    matches = []
    for name, embedding in stored_embeddings.items():
        similarity = 1 - cosine(new_face_embedding, embedding)
        if similarity > threshold:
            matches.append((name, similarity))
    return matches

def detect_faces(frame):
    # Run YOLOv8 inference with confidence threshold of 0.7
    results = face_detector(frame, classes=0, conf=0.7)  # class 0 is person, confidence threshold 0.7
    
    boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            boxes.append([x1, y1, x2, y2])
    
    return boxes

def save_embeddings_with_names(embeddings_dict, filename="face_embeddings.npy"):
    """
    Save embeddings with names in a structured format
    embeddings_dict should be a dictionary with names as keys and embeddings as values
    """
    # Get absolute path to ensure we're saving in the right place
    abs_path = os.path.abspath(filename)
    print(f"Saving embeddings to: {abs_path}")
    
    # Convert embeddings to numpy arrays and create a structured array
    names = list(embeddings_dict.keys())
    embeddings = np.array([embeddings_dict[name] for name in names])
    
    # Create a structured array with names and embeddings
    structured_data = {
        'names': np.array(names, dtype='U50'),  # Unicode string array for names
        'embeddings': embeddings
    }
    
    try:
        # Save the structured data
        np.save(abs_path, structured_data)
        print(f"Successfully saved embeddings for {len(names)} faces")
        print(f"File saved at: {abs_path}")
    except Exception as e:
        print(f"Error saving embeddings: {str(e)}")
        return False
    return True

def load_embeddings_with_names(filename="face_embeddings.npy"):
    """
    Load embeddings with names from the saved file
    Returns a dictionary with names as keys and embeddings as values
    """
    # Get absolute path to ensure we're looking in the right place
    abs_path = os.path.abspath(filename)
    print(f"Looking for embeddings file at: {abs_path}")
    
    if not os.path.exists(abs_path):
        print(f"Error: Embeddings file not found at {abs_path}")
        print("Please make sure you have created embeddings first using options 1 or 2")
        return None
    
    try:
        data = np.load(abs_path, allow_pickle=True).item()
        if 'names' not in data or 'embeddings' not in data:
            print("Error: Invalid embeddings file format")
            return None
            
        embeddings_dict = {name: embedding for name, embedding in zip(data['names'], data['embeddings'])}
        print(f"Successfully loaded {len(embeddings_dict)} embeddings")
        return embeddings_dict
    except Exception as e:
        print(f"Error loading embeddings: {str(e)}")
        return None

def capture_embeddings_from_camera():
    # Try to initialize camera with retry logic
    max_retries = 3
    cap = None
    
    for attempt in range(max_retries):
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend
            if cap.isOpened():
                # Set some basic properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if cap is not None:
                cap.release()
            time.sleep(1)  # Wait before retrying
    
    if cap is None or not cap.isOpened():
        print("Error: Could not initialize camera after multiple attempts")
        print("Please check if:")
        print("1. Camera is properly connected")
        print("2. No other application is using the camera")
        print("3. Camera drivers are properly installed")
        return

    embeddings = {}
    print("\nCamera feed started. Press:")
    print("'c' to capture a face")
    print("'s' to save all captured faces")
    print("'q' to quit without saving")

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame. Retrying...")
                time.sleep(0.1)  # Small delay before retry
                continue

            # Detect faces using YOLOv8
            boxes = detect_faces(frame)
            
            if boxes:
                for box in boxes:
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Put text inside the rectangle
                    cv2.putText(frame, "Press 'c' to capture", (x1 + 5, y1 + 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow('Capture Faces', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                if boxes:
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = [int(coord) for coord in box]
                        face = frame[y1:y2, x1:x2]
                        
                        try:
                            name = input(f"Enter name for person {i+1}: ")
                            embedding = create_embedding(face)
                            embeddings[name] = embedding
                            print(f"Successfully captured face for {name}")
                        except Exception as e:
                            print(f"Error capturing face: {str(e)}")
            
            elif key == ord('s'):
                if embeddings:
                    save_embeddings_with_names(embeddings)
                    break
                else:
                    print("No faces captured yet!")
            
            elif key == ord('q'):
                print("Quitting without saving...")
                break

        except Exception as e:
            print(f"Error in camera feed: {str(e)}")
            time.sleep(0.1)  # Small delay before retry
            continue

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

def process_camera_feed():
    stored_embeddings = load_embeddings_with_names()
    if stored_embeddings is None:
        print("No stored embeddings found. Please create embeddings first.")
        return
    
    # Try to initialize camera with retry logic
    max_retries = 3
    cap = None
    
    for attempt in range(max_retries):
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend
            if cap.isOpened():
                # Set some basic properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if cap is not None:
                cap.release()
            time.sleep(1)  # Wait before retrying
    
    if cap is None or not cap.isOpened():
        print("Error: Could not initialize camera after multiple attempts")
        print("Please check if:")
        print("1. Camera is properly connected")
        print("2. No other application is using the camera")
        print("3. Camera drivers are properly installed")
        return

    print("Press 'q' to quit camera feed")
    
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame. Retrying...")
                time.sleep(0.1)  # Small delay before retry
                continue

            # Detect faces using YOLOv8
            boxes = detect_faces(frame)
            
            if boxes:
                for box in boxes:
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    face = frame[y1:y2, x1:x2]
                    
                    try:
                        # Create embedding for the detected face
                        face_embedding = create_embedding(face)

# Compare with stored embeddings
                        matches = compare_faces(face_embedding, stored_embeddings)
                        
                        # Draw rectangle
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        if matches:
                            name, similarity = matches[0]  # Get the best match
                            label = f"{name} ({similarity:.2f})"
                        else:
                            label = "Unknown"
                        
                        # Put text inside the rectangle
                        cv2.putText(frame, label, (x1 + 5, y1 + 20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    except Exception as e:
                        print(f"Error processing face: {str(e)}")
            
            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Error in camera feed: {str(e)}")
            time.sleep(0.1)  # Small delay before retry
            continue

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

def main():
    while True:
        print("\nFace Recognition System")
        print("1. Create embeddings from directory")
        print("2. Create embeddings from camera")
        print("3. Compare a new face")
        print("4. Start camera feed for recognition")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == "1":
            directory_path = input("Enter the directory path containing face images: ")
            if not os.path.exists(directory_path):
                print("Directory does not exist!")
                continue
                
            embeddings = process_directory(directory_path)
            if embeddings:
                save_embeddings_with_names(embeddings)
            else:
                print("No valid images found in the directory")
                
        elif choice == "2":
            capture_embeddings_from_camera()
                
        elif choice == "3":
            stored_embeddings = load_embeddings_with_names()
            if stored_embeddings is None:
                print("No stored embeddings found. Please create embeddings first.")
                continue
                
            new_face_path = input("Enter the path of the new face image: ")
            
            if not os.path.exists(new_face_path):
                print("Image file does not exist!")
                continue
                
            matches = compare_faces(create_embedding(new_face_path), stored_embeddings)
            if matches:
                print("\nMatches found:")
                for name, similarity in matches:
                    print(f"Name: {name}, Similarity: {similarity:.4f}")
            else:
                print("No matches found above the threshold")
                
        elif choice == "4":
            process_camera_feed()
            
        elif choice == "5":
            print("Exiting...")
            break
            
        else:
            print("Invalid choice! Please try again.")

if __name__ == "__main__":
    main()
