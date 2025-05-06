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
import json

class StudentRegistration:
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
        
        # Initialize student database
        self.students_file = "students_database.csv"
        self.embeddings_file = "student_embeddings.npy"
        self.initialize_database()

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

    def initialize_database(self):
        """Initialize student database if it doesn't exist"""
        if not os.path.exists(self.students_file):
            df = pd.DataFrame(columns=[
                'Student_ID', 'Name', 'Roll_Number', 'Class', 'Section',
                'Date_of_Birth', 'Gender', 'Contact_Number', 'Email',
                'Address', 'Registration_Date'
            ])
            df.to_csv(self.students_file, index=False)
            print("Created new student database")

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

    def save_embeddings(self, embeddings_dict):
        """Save embeddings with student IDs"""
        abs_path = os.path.abspath(self.embeddings_file)
        print(f"Saving embeddings to: {abs_path}")
        
        student_ids = list(embeddings_dict.keys())
        embeddings = np.array([embeddings_dict[student_id] for student_id in student_ids])
        
        structured_data = {
            'student_ids': np.array(student_ids, dtype='U50'),
            'embeddings': embeddings
        }
        
        try:
            np.save(abs_path, structured_data)
            print(f"Successfully saved embeddings for {len(student_ids)} students")
            return True
        except Exception as e:
            print(f"Error saving embeddings: {str(e)}")
            return False

    def load_embeddings(self):
        """Load embeddings with student IDs"""
        abs_path = os.path.abspath(self.embeddings_file)
        print(f"Looking for embeddings file at: {abs_path}")
        
        if not os.path.exists(abs_path):
            print("No stored embeddings found")
            return None
        
        try:
            data = np.load(abs_path, allow_pickle=True).item()
            if 'student_ids' not in data or 'embeddings' not in data:
                print("Invalid embeddings file format")
                return None
            
            embeddings_dict = {student_id: embedding for student_id, embedding in zip(data['student_ids'], data['embeddings'])}
            print(f"Successfully loaded {len(embeddings_dict)} embeddings")
            return embeddings_dict
        except Exception as e:
            print(f"Error loading embeddings: {str(e)}")
            return None

    def get_student_details(self):
        """Get student details from user input"""
        print("\nEnter Student Details:")
        student_id = input("Student ID: ")
        
        # Check if student ID already exists
        df = pd.read_csv(self.students_file)
        if not df.empty and student_id in df['Student_ID'].values:
            print("Error: Student ID already exists!")
            return None
        
        details = {
            'Student_ID': student_id,
            'Name': input("Full Name: "),
            'Roll_Number': input("Roll Number: "),
            'Class': input("Class: "),
            'Section': input("Section: "),
            'Date_of_Birth': input("Date of Birth (YYYY-MM-DD): "),
            'Gender': input("Gender: "),
            'Contact_Number': input("Contact Number: "),
            'Email': input("Email: "),
            'Address': input("Address: "),
            'Registration_Date': datetime.now().strftime("%Y-%m-%d")
        }
        return details

    def capture_student_face(self):
        """Capture student face from camera"""
        if not self.cap or not self.cap.isOpened():
            print("Camera not initialized")
            return None

        print("\nCamera feed started. Press:")
        print("'c' to capture face")
        print("'q' to quit")

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

            cv2.imshow('Capture Student Face', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                if boxes:
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    face = frame[y1:y2, x1:x2]
                    cv2.destroyAllWindows()
                    return face
                else:
                    print("No face detected! Please try again.")
            
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return None

    def register_student(self):
        """Register a new student with face capture"""
        # Get student details
        details = self.get_student_details()
        if details is None:
            return
        
        # Capture student face
        print("\nPlease look at the camera for face capture")
        face = self.capture_student_face()
        if face is None:
            print("Face capture cancelled")
            return
        
        try:
            # Create embedding
            embedding = self.create_embedding(face)
            
            # Save student details
            df = pd.read_csv(self.students_file)
            new_student = pd.DataFrame([details])
            df = pd.concat([df, new_student], ignore_index=True)
            df.to_csv(self.students_file, index=False)
            
            # Save embedding
            embeddings_dict = self.load_embeddings() or {}
            embeddings_dict[details['Student_ID']] = embedding
            self.save_embeddings(embeddings_dict)
            
            print(f"\nStudent {details['Name']} (ID: {details['Student_ID']}) registered successfully!")
            
        except Exception as e:
            print(f"Error registering student: {str(e)}")

    def view_students(self):
        """View registered students"""
        if not os.path.exists(self.students_file):
            print("No student records found")
            return
        
        df = pd.read_csv(self.students_file)
        if df.empty:
            print("No student records found")
            return
        
        print("\nRegistered Students:")
        print(df)
        
        # Option to save to Excel
        save_excel = input("\nDo you want to save the records to Excel? (y/n): ")
        if save_excel.lower() == 'y':
            excel_file = f"students_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            df.to_excel(excel_file, index=False)
            print(f"Records saved to {excel_file}")

    def __del__(self):
        """Cleanup when object is destroyed"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    registration = StudentRegistration()
    
    while True:
        print("\nStudent Registration System")
        print("1. Register new student")
        print("2. View registered students")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == "1":
            registration.register_student()
        elif choice == "2":
            registration.view_students()
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice! Please try again.")

if __name__ == "__main__":
    main() 