import cv2
import numpy as np
import os
import time
from datetime import datetime
import pandas as pd
from facenet_pytorch import InceptionResnetV1
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from scipy.spatial.distance import cosine

class AttendanceSystem:
    def __init__(self):
        # Initialize FaceNet model
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        
        # Initialize YOLOv8 face detector
        self.face_detector = YOLO('yolov8n.pt')
        
        # Define image transformation
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize camera
        self.cap = None
        self.initialize_camera()
        
        # Initialize attendance records
        self.attendance_file = "attendance_records.csv"
        self.initialize_attendance_file()

    def initialize_camera(self):
        """Initialize camera with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    print("Camera initialized successfully")
                    return True
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if self.cap is not None:
                    self.cap.release()
                time.sleep(1)
        
        print("Error: Could not initialize camera")
        return False

    def initialize_attendance_file(self):
        """Create attendance file if it doesn't exist"""
        if not os.path.exists(self.attendance_file):
            df = pd.DataFrame(columns=['Name', 'Date', 'Time', 'Status'])
            df.to_csv(self.attendance_file, index=False)
            print("Created new attendance file")

    def create_embedding(self, image):
        """Create face embedding from image"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        image = self.transform(image)
        image = image.unsqueeze(0)
        
        with torch.no_grad():
            embedding = self.model(image)
        return embedding[0].numpy()

    def detect_faces(self, frame):
        """Detect faces in frame using YOLOv8"""
        results = self.face_detector(frame, classes=0, conf=0.7)
        boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                boxes.append([x1, y1, x2, y2])
        return boxes

    def save_embeddings(self, embeddings_dict, filename="face_embeddings.npy"):
        """Save embeddings with names"""
        abs_path = os.path.abspath(filename)
        print(f"Saving embeddings to: {abs_path}")
        
        names = list(embeddings_dict.keys())
        embeddings = np.array([embeddings_dict[name] for name in names])
        
        structured_data = {
            'names': np.array(names, dtype='U50'),
            'embeddings': embeddings
        }
        
        try:
            np.save(abs_path, structured_data)
            print(f"Successfully saved embeddings for {len(names)} faces")
            return True
        except Exception as e:
            print(f"Error saving embeddings: {str(e)}")
            return False

    def load_embeddings(self, filename="face_embeddings.npy"):
        """Load embeddings with names"""
        abs_path = os.path.abspath(filename)
        print(f"Looking for embeddings file at: {abs_path}")
        
        if not os.path.exists(abs_path):
            print("No stored embeddings found")
            return None
        
        try:
            data = np.load(abs_path, allow_pickle=True).item()
            if 'names' not in data or 'embeddings' not in data:
                print("Invalid embeddings file format")
                return None
            
            embeddings_dict = {name: embedding for name, embedding in zip(data['names'], data['embeddings'])}
            print(f"Successfully loaded {len(embeddings_dict)} embeddings")
            return embeddings_dict
        except Exception as e:
            print(f"Error loading embeddings: {str(e)}")
            return None

    def compare_faces(self, new_face_embedding, stored_embeddings, threshold=0.7):
        """Compare new face with stored embeddings"""
        matches = []
        for name, embedding in stored_embeddings.items():
            similarity = 1 - cosine(new_face_embedding, embedding)
            if similarity > threshold:
                matches.append((name, similarity))
        return matches

    def mark_attendance(self, name):
        """Mark attendance for a person"""
        current_time = datetime.now()
        date = current_time.strftime("%Y-%m-%d")
        time = current_time.strftime("%H:%M:%S")
        
        # Check if already marked attendance today
        df = pd.read_csv(self.attendance_file)
        today_records = df[(df['Name'] == name) & (df['Date'] == date)]
        
        if not today_records.empty:
            print(f"{name} has already marked attendance today")
            return False
        
        # Add new attendance record
        new_record = pd.DataFrame({
            'Name': [name],
            'Date': [date],
            'Time': [time],
            'Status': ['Present']
        })
        
        df = pd.concat([df, new_record], ignore_index=True)
        df.to_csv(self.attendance_file, index=False)
        print(f"Attendance marked for {name} at {time}")
        return True

    def capture_faces(self):
        """Capture faces from camera and save embeddings"""
        if not self.cap or not self.cap.isOpened():
            print("Camera not initialized")
            return

        embeddings = {}
        print("\nCamera feed started. Press:")
        print("'c' to capture a face")
        print("'s' to save all captured faces")
        print("'q' to quit without saving")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            boxes = self.detect_faces(frame)
            
            for box in boxes:
                x1, y1, x2, y2 = [int(coord) for coord in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
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
                            embedding = self.create_embedding(face)
                            embeddings[name] = embedding
                            print(f"Successfully captured face for {name}")
                        except Exception as e:
                            print(f"Error capturing face: {str(e)}")
            
            elif key == ord('s'):
                if embeddings:
                    self.save_embeddings(embeddings)
                    break
                else:
                    print("No faces captured yet!")
            
            elif key == ord('q'):
                print("Quitting without saving...")
                break

        cv2.destroyAllWindows()

    def take_attendance(self):
        """Take attendance using face recognition"""
        stored_embeddings = self.load_embeddings()
        if stored_embeddings is None:
            print("No stored embeddings found. Please create embeddings first.")
            return

        if not self.cap or not self.cap.isOpened():
            print("Camera not initialized")
            return

        print("Press 'q' to quit attendance taking")
        print("Looking for faces to mark attendance...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            boxes = self.detect_faces(frame)
            
            for box in boxes:
                x1, y1, x2, y2 = [int(coord) for coord in box]
                face = frame[y1:y2, x1:x2]
                
                try:
                    face_embedding = self.create_embedding(face)
                    matches = self.compare_faces(face_embedding, stored_embeddings)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    if matches:
                        name, similarity = matches[0]
                        label = f"{name} ({similarity:.2f})"
                        # Mark attendance for recognized person
                        self.mark_attendance(name)
                    else:
                        label = "Unknown"
                    
                    cv2.putText(frame, label, (x1 + 5, y1 + 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                except Exception as e:
                    print(f"Error processing face: {str(e)}")
            
            cv2.imshow('Attendance System', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def view_attendance(self):
        """View attendance records"""
        if not os.path.exists(self.attendance_file):
            print("No attendance records found")
            return
        
        df = pd.read_csv(self.attendance_file)
        if df.empty:
            print("No attendance records found")
            return
        
        print("\nAttendance Records:")
        print(df)
        
        # Option to save to Excel
        save_excel = input("\nDo you want to save the records to Excel? (y/n): ")
        if save_excel.lower() == 'y':
            excel_file = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            df.to_excel(excel_file, index=False)
            print(f"Records saved to {excel_file}")

    def __del__(self):
        """Cleanup when object is destroyed"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    system = AttendanceSystem()
    
    while True:
        print("\nAttendance Management System")
        print("1. Register new faces")
        print("2. Take attendance")
        print("3. View attendance records")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == "1":
            system.capture_faces()
        elif choice == "2":
            system.take_attendance()
        elif choice == "3":
            system.view_attendance()
        elif choice == "4":
            print("Exiting...")
            break
        else:
            print("Invalid choice! Please try again.")

if __name__ == "__main__":
    main() 