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
        
        # Initialize attendance records with absolute paths
        self.attendance_file = os.path.abspath("attendance_records.csv")
        self.students_file = os.path.abspath("students_database.csv")
        self.embeddings_file = os.path.abspath("student_embeddings.npy")
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
            df = pd.DataFrame(columns=['Student_ID', 'Name', 'Class', 'Section', 'Date', 'Time', 'Status'])
            df.to_csv(self.attendance_file, index=False)
            print(f"Created new attendance file at {self.attendance_file}")
        else:
            # Check if Student_ID column exists, if not add it
            df = pd.read_csv(self.attendance_file)
            if 'Student_ID' not in df.columns:
                print(f"Adding Student_ID column to attendance file at {self.attendance_file}")
                df['Student_ID'] = None
                df.to_csv(self.attendance_file, index=False)

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

    def load_embeddings(self):
        """Load embeddings with student IDs"""
        if not os.path.exists(self.embeddings_file):
            print("No stored embeddings found")
            return None
        
        try:
            data = np.load(self.embeddings_file, allow_pickle=True).item()
            
            # Handle new format (list-based)
            if 'student_ids' in data and 'embeddings' in data:
                if isinstance(data['student_ids'], list):
                    # Convert all student IDs to integers
                    embeddings_dict = {int(student_id): embedding for student_id, embedding in zip(data['student_ids'], data['embeddings'])}
                    print(f"Successfully loaded {len(embeddings_dict)} embeddings")
                    return embeddings_dict
            
            # Handle old format (array-based)
            if 'student_ids' in data and 'embeddings' in data:
                if isinstance(data['student_ids'], np.ndarray):
                    # Convert all student IDs to integers
                    embeddings_dict = {int(student_id): embedding for student_id, embedding in zip(data['student_ids'], data['embeddings'])}
                    print(f"Successfully loaded {len(embeddings_dict)} embeddings")
                    return embeddings_dict
            
            print("Invalid embeddings file format")
            return None
            
        except Exception as e:
            print(f"Error loading embeddings: {str(e)}")
            return None

    def compare_faces(self, new_face_embedding, stored_embeddings, threshold=0.3):
        """Compare new face with stored embeddings"""
        matches = []
        for student_id, embedding in stored_embeddings.items():
            try:
                similarity = 1 - cosine(new_face_embedding, embedding)
                if similarity > threshold:
                    # Ensure student_id is integer
                    matches.append((int(student_id), similarity))
            except Exception as e:
                print(f"Error comparing face with student ID {student_id}: {str(e)}")
                continue
        return matches

    def get_student_details(self, student_id):
        """Get student details from database"""
        if not os.path.exists(self.students_file):
            print("Student database not found")
            return None
        
        try:
            df = pd.read_csv(self.students_file)
            # Convert student_id to integer for comparison
            student_id = int(student_id)
            student = df[df['Student_ID'] == student_id]
            if student.empty:
                print(f"Student ID {student_id} not found in database")
                return None
            
            return student.iloc[0].to_dict()
        except Exception as e:
            print(f"Error getting student details: {str(e)}")
            return None

    def mark_attendance(self, student_id):
        """Mark attendance for a student"""
        # Get student details
        student = self.get_student_details(student_id)
        if student is None:
            return False
        
        current_time = datetime.now()
        date = current_time.strftime("%Y-%m-%d")
        time = current_time.strftime("%H:%M:%S")
        
        try:
            # First, check if file exists and create it if it doesn't
            if not os.path.exists(self.attendance_file):
                print(f"Creating new attendance file at {self.attendance_file}")
                df = pd.DataFrame(columns=['Name', 'Date', 'Time', 'Status', 'Student_ID'])
                df.to_csv(self.attendance_file, index=False)
            
            # Read existing records
            print(f"Reading attendance file from {self.attendance_file}")
            df = pd.read_csv(self.attendance_file)
            print(f"Current records in file: {len(df)}")
            
            # Check if Student_ID column exists
            if 'Student_ID' not in df.columns:
                print("Adding Student_ID column to attendance file")
                df['Student_ID'] = None
                df.to_csv(self.attendance_file, index=False)
            
            # Convert student_id to integer for comparison
            student_id = int(student_id)
            
            # Check for existing attendance
            today_records = df[(df['Student_ID'] == student_id) & (df['Date'] == date)]
            
            if not today_records.empty:
                print(f"Student {student['Name']} (ID: {student_id}) has already marked attendance today")
                return False
            
            # Create new record with columns in the same order as the file
            new_record = pd.DataFrame({
                'Name': [student['Name']],
                'Date': [date],
                'Time': [time],
                'Status': ['Present'],
                'Student_ID': [student_id]
            })
            
            # Append new record
            df = pd.concat([df, new_record], ignore_index=True)
            
            # Save to CSV with explicit encoding and line terminator
            print(f"Saving {len(df)} records to {self.attendance_file}")
            df.to_csv(self.attendance_file, index=False, encoding='utf-8', line_terminator='\n')
            
            # Verify the save
            if os.path.exists(self.attendance_file):
                # Read back the file to verify
                verify_df = pd.read_csv(self.attendance_file)
                print(f"Verified: File contains {len(verify_df)} records")
                if len(verify_df) == len(df):
                    print("Successfully saved attendance record")
                    return True
                else:
                    print("Warning: Record count mismatch after save")
                    return False
            else:
                print("Error: File not found after saving")
                return False
                
        except Exception as e:
            print(f"Error marking attendance: {str(e)}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Attempted to save to: {self.attendance_file}")
            return False

    def take_attendance(self):
        """Take attendance using face recognition"""
        stored_embeddings = self.load_embeddings()
        if stored_embeddings is None:
            print("No stored embeddings found. Please register students first.")
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
                    
                    # Initialize label as Unknown by default
                    label = "Unknown"
                    
                    if matches:
                        student_id, similarity = matches[0]
                        try:
                            student = self.get_student_details(student_id)
                            if student:
                                label = f"ID: {student_id} ({similarity:.2f})"
                                # Mark attendance for recognized student
                                self.mark_attendance(student_id)
                        except Exception as e:
                            print(f"Error processing student ID {student_id}: {str(e)}")
                            continue
                    
                    cv2.putText(frame, label, (x1 + 5, y1 + 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                except Exception as e:
                    print(f"Error processing face: {str(e)}")
                    continue
            
            cv2.imshow('Attendance System', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def view_attendance(self):
        """View attendance records"""
        if not os.path.exists(self.attendance_file):
            print(f"No attendance records found at {self.attendance_file}")
            return
        
        try:
            df = pd.read_csv(self.attendance_file)
            if df.empty:
                print("No attendance records found")
                return
            
            print(f"\nAttendance Records from {self.attendance_file}:")
            print(df)
            
            # Option to save to Excel
            save_excel = input("\nDo you want to save the records to Excel? (y/n): ")
            if save_excel.lower() == 'y':
                # Make sure the DataFrame is properly formatted
                if 'Student_ID' not in df.columns:
                    df['Student_ID'] = None
                
                # Save to CSV first to ensure data is properly stored
                df.to_csv(self.attendance_file, index=False, mode='a',header=not os.path.exists(self.attendance_file))
                print(f"Records saved to {self.attendance_file}")
                
                # Then export to Excel
                excel_file = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                df.to_excel(excel_file, index=False)
                print(f"Records also exported to {excel_file}")
                
                # Verify the save was successful
                if os.path.exists(self.attendance_file):
                    print(f"Verified: {self.attendance_file} exists and contains {len(df)} records")
                else:
                    print(f"Warning: {self.attendance_file} was not found after saving")
        except Exception as e:
            print(f"Error viewing attendance: {str(e)}")

    def view_student_attendance(self, student_id):
        """View attendance records for a specific student"""
        if not os.path.exists(self.attendance_file):
            print("No attendance records found")
            return
        
        df = pd.read_csv(self.attendance_file)
        student_records = df[df['Student_ID'] == int(student_id)]
        
        if student_records.empty:
            print(f"No attendance records found for student ID {student_id}")
            return
        
        student = self.get_student_details(student_id)
        if student:
            print(f"\nAttendance Records for {student['Name']} (ID: {student_id}):")
            print(f"Class: {student['Class']} {student['Section']}")
            print("\nAttendance History:")
            print(student_records[['Date', 'Time', 'Status']])
            
            # Calculate attendance statistics
            total_days = len(student_records)
            present_days = len(student_records[student_records['Status'] == 'Present'])
            attendance_percentage = (present_days / total_days) * 100 if total_days > 0 else 0
            
            print(f"\nAttendance Statistics:")
            print(f"Total Days: {total_days}")
            print(f"Present Days: {present_days}")
            print(f"Attendance Percentage: {attendance_percentage:.2f}%")

    def __del__(self):
        """Cleanup when object is destroyed"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    system = AttendanceSystem()
    
    while True:
        print("\nAttendance Management System")
        print("1. Take attendance")
        print("2. View all attendance records")
        print("3. View student attendance")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == "1":
            system.take_attendance()
        elif choice == "2":
            system.view_attendance()
        elif choice == "3":
            student_id = int(input("Enter student ID: "))
            system.view_student_attendance(student_id)
        elif choice == "4":
            print("Exiting...")
            break
        else:
            print("Invalid choice! Please try again.")

if __name__ == "__main__":
    main() 