from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response, send_file
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
import threading
import queue
import base64
import io

app = Flask(__name__)
app.secret_key = 'face_recognition_attendance_system'

# Global variables
camera = None
camera_lock = threading.Lock()
frame_queue = queue.Queue(maxsize=10)
attendance_system = None

def initialize_attendance_system():
    """Initialize the attendance system"""
    global attendance_system
    if attendance_system is None:
        attendance_system = AttendanceSystem()
    return attendance_system

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
        
        # Initialize attendance records with absolute paths
        self.attendance_file = os.path.abspath("attendance_records.csv")
        self.students_file = os.path.abspath("students_database.csv")
        self.embeddings_file = os.path.abspath("student_embeddings.npy")
        self.initialize_attendance_file()
        self.initialize_students_file()

    def initialize_attendance_file(self):
        """Create attendance file if it doesn't exist"""
        if not os.path.exists(self.attendance_file):
            df = pd.DataFrame(columns=['Name', 'Date', 'Time', 'Status', 'Student_ID'])
            df.to_csv(self.attendance_file, index=False)
            print(f"Created new attendance file at {self.attendance_file}")
        else:
            # Check if Student_ID column exists, if not add it
            df = pd.read_csv(self.attendance_file)
            if 'Student_ID' not in df.columns:
                print("Adding Student_ID column to attendance file")
                df['Student_ID'] = None
                df.to_csv(self.attendance_file, index=False)

    def initialize_students_file(self):
        """Create students database file if it doesn't exist"""
        if not os.path.exists(self.students_file):
            print(f"Creating new students database at {self.students_file}")
            df = pd.DataFrame(columns=[
                'Student_ID', 'Name', 'Email', 'Phone', 'Gender',
                'Department', 'Course', 'Year', 'Section', 'Address'
            ])
            df.to_csv(self.students_file, index=False)
        else:
            # Check if all required columns exist
            df = pd.read_csv(self.students_file)
            required_columns = [
                'Student_ID', 'Name', 'Email', 'Phone', 'Gender',
                'Department', 'Course', 'Year', 'Section', 'Address'
            ]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Adding missing columns to students database: {missing_columns}")
                for col in missing_columns:
                    df[col] = None
                df.to_csv(self.students_file, index=False)

    def save_embedding(self, student_id, embedding):
        """Save face embedding for a student"""
        try:
            # Load existing embeddings if any
            if os.path.exists(self.embeddings_file):
                data = np.load(self.embeddings_file, allow_pickle=True).item()
                student_ids = data.get('student_ids', [])
                embeddings = data.get('embeddings', [])
            else:
                student_ids = []
                embeddings = []
            
            # Check if student ID already exists
            if student_id in student_ids:
                idx = student_ids.index(student_id)
                embeddings[idx] = embedding
            else:
                student_ids.append(student_id)
                embeddings.append(embedding)
            
            # Save updated embeddings
            np.save(self.embeddings_file, {
                'student_ids': student_ids,
                'embeddings': embeddings
            })
            return True
            
        except Exception as e:
            print(f"Error saving embedding: {str(e)}")
            return False

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
            print(f"Reading student database from {self.students_file}")
            df = pd.read_csv(self.students_file)
            print(f"Current students in database: {len(df)}")
            print(f"Looking for student ID: {student_id}")
            # Convert student_id to integer for comparison
            student_id = int(student_id)
            student = df[df['Student_ID'] == student_id]
            if student.empty:
                print(f"Student ID {student_id} not found in database")
                print("Available student IDs:", df['Student_ID'].tolist())
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
            
            # Save to CSV
            print(f"Saving {len(df)} records to {self.attendance_file}")
            df.to_csv(self.attendance_file, index=False)
            
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

    def process_frame(self, frame):
        """Process a single frame for face recognition and attendance marking"""
        stored_embeddings = self.load_embeddings()
        if stored_embeddings is None:
            return frame, "No stored embeddings found. Please register students first."
        
        boxes = self.detect_faces(frame)
        message = "No faces detected"
        
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
                            if self.mark_attendance(student_id):
                                message = f"Attendance marked for {student['Name']} (ID: {student_id})"
                            else:
                                message = f"Attendance already marked for {student['Name']} (ID: {student_id})"
                    except Exception as e:
                        print(f"Error processing student ID {student_id}: {str(e)}")
                        continue
                
                cv2.putText(frame, label, (x1 + 5, y1 + 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            except Exception as e:
                print(f"Error processing face: {str(e)}")
                continue
        
        return frame, message

    def get_attendance_records(self):
        """Get all attendance records"""
        if not os.path.exists(self.attendance_file):
            return pd.DataFrame()
        
        try:
            # Read the attendance records
            df = pd.read_csv(self.attendance_file)
            
            # Ensure all required columns exist
            required_columns = ['Name', 'Date', 'Time', 'Status', 'Student_ID']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = None
            
            # Convert Student_ID to integer
            df['Student_ID'] = pd.to_numeric(df['Student_ID'], errors='coerce').astype('Int64')
            
            # Sort by date and time in descending order (most recent first)
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            df = df.sort_values('DateTime', ascending=False)
            df = df.drop('DateTime', axis=1)
            
            # Reset index after sorting
            df = df.reset_index(drop=True)
            
            print(f"Retrieved {len(df)} attendance records")
            return df
            
        except Exception as e:
            print(f"Error reading attendance records: {str(e)}")
            return pd.DataFrame()

    def get_student_attendance(self, student_id):
        """Get attendance records for a specific student"""
        if not os.path.exists(self.attendance_file):
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.attendance_file)
            student_id = int(student_id)
            student_records = df[df['Student_ID'] == student_id]
            return student_records
        except Exception as e:
            print(f"Error reading student attendance: {str(e)}")
            return pd.DataFrame()

    def get_all_students(self):
        """Get all registered students"""
        if not os.path.exists(self.students_file):
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.students_file)
            return df
        except Exception as e:
            print(f"Error reading student database: {str(e)}")
            return pd.DataFrame()

def initialize_camera():
    """Initialize the camera"""
    global camera
    with camera_lock:
        if camera is None:
            print("Attempting to initialize camera...")
            try:
                # Try different camera indices
                for i in range(2):  # Try first two camera indices
                    print(f"Trying camera index {i}...")
                    try:
                        # Try without CAP_DSHOW first
                        camera = cv2.VideoCapture(i)
                        if not camera.isOpened():
                            # If that fails, try with CAP_DSHOW
                            camera = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                        
                        if camera.isOpened():
                            print(f"Camera {i} opened successfully")
                            # Set camera properties
                            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            camera.set(cv2.CAP_PROP_FPS, 30)
                            camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                            
                            # Test if we can actually read a frame
                            ret, frame = camera.read()
                            if ret and frame is not None:
                                print(f"Successfully captured frame from camera {i}")
                                return True
                            else:
                                print(f"Could not read frame from camera {i}")
                                camera.release()
                                camera = None
                        else:
                            print(f"Failed to open camera {i}")
                    except Exception as e:
                        print(f"Error initializing camera {i}: {str(e)}")
                        if camera is not None:
                            camera.release()
                            camera = None
                
                print("Failed to initialize any camera")
                return False
                
            except Exception as e:
                print(f"Error in camera initialization: {str(e)}")
                if camera is not None:
                    camera.release()
                    camera = None
                return False
    return True

def release_camera():
    """Release the camera"""
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None

def generate_frames():
    """Generate frames from the camera"""
    global attendance_system
    attendance_system = initialize_attendance_system()
    
    print("Starting frame generation...")
    while True:
        with camera_lock:
            if camera is None or not camera.isOpened():
                print("Camera not available, breaking frame generation loop")
                break
            
            success, frame = camera.read()
            if not success or frame is None:
                print("Failed to read frame from camera")
                break
            
            try:
                # Process the frame for face recognition
                processed_frame, message = attendance_system.process_frame(frame)
                
                # Add message to the frame
                cv2.putText(processed_frame, message, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Convert the frame to JPEG
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                if not ret:
                    print("Failed to encode frame")
                    continue
                    
                frame = buffer.tobytes()
                
                # Put the frame in the queue
                if not frame_queue.full():
                    frame_queue.put(frame)
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                continue
        
        time.sleep(0.03)  # Reduced sleep time for smoother video

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/take_attendance')
def take_attendance():
    """Take attendance page"""
    if not initialize_camera():
        return redirect(url_for('index'))
    
    return render_template('take_attendance.html')

@app.route('/video_feed')
def video_feed():
    """Video feed for face recognition"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/view_attendance')
def view_attendance():
    """View attendance records"""
    attendance_system = initialize_attendance_system()
    records = attendance_system.get_attendance_records()
    return render_template('view_attendance.html', records=records)

@app.route('/view_student_attendance/<int:student_id>')
def view_student_attendance(student_id):
    """View attendance records for a specific student"""
    attendance_system = initialize_attendance_system()
    student = attendance_system.get_student_details(student_id)
    if student is None:
        flash(f"Student ID {student_id} not found")
        return redirect(url_for('view_attendance'))
    
    records = attendance_system.get_student_attendance(student_id)
    return render_template('student_attendance.html', student=student, records=records)

@app.route('/students')
def students():
    """View all registered students"""
    attendance_system = initialize_attendance_system()
    students = attendance_system.get_all_students()
    return render_template('students.html', students=students)

@app.route('/export_attendance')
def export_attendance():
    """Export attendance records to Excel"""
    attendance_system = initialize_attendance_system()
    records = attendance_system.get_attendance_records()
    if records.empty:
        flash("No attendance records found")
        return redirect(url_for('view_attendance'))
    
    # Create Excel file
    excel_file = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    records.to_excel(excel_file, index=False)
    
    # Return the file as a download
    return send_file(excel_file, as_attachment=True)

@app.route('/stop_camera')
def stop_camera():
    """Stop the camera"""
    release_camera()
    return redirect(url_for('index'))

@app.route('/register_student', methods=['GET', 'POST'])
def register_student():
    """Register a new student"""
    release_camera()
    attendance_system = initialize_attendance_system()
    
    if request.method == 'POST':
        try:
            # Get form data
            student_id = request.form['student_id']
            name = request.form['name']
            roll_number = request.form['roll_number']
            class_name = request.form['class']
            section = request.form['section']
            date_of_birth = request.form['date_of_birth']
            gender = request.form['gender']
            contact_number = request.form['contact_number']
            email = request.form['email']
            phone = request.form['phone']
            department = request.form['department']
            course = request.form['course']
            year = request.form['year']
            registration_date = request.form['registration_date']
            address = request.form['address']
            
            # Validate student ID
            if not student_id.isdigit():
                flash('Student ID must be a number')
                return redirect(url_for('register_student'))
            
            student_id = int(student_id)
            
            # Check if student ID already exists
            if attendance_system.get_student_details(student_id):
                flash('Student ID already exists')
                return redirect(url_for('register_student'))
            
            # Get face image from webcam
            face_image = request.form.get('face_image')
            if not face_image:
                flash('No face image captured')
                return redirect(url_for('register_student'))
            
            # Convert base64 image to numpy array
            face_image = face_image.split(',')[1]
            face_image = base64.b64decode(face_image)
            face_image = Image.open(io.BytesIO(face_image))
            face_image = np.array(face_image)
            
            # Create face embedding
            face_embedding = attendance_system.create_embedding(face_image)
            
            # Save student details
            student_data = {
                'Student_ID': student_id,
                'Name': name,
                'Roll_Number': roll_number,
                'Class': class_name,
                'Section': section,
                'Date_of_Birth': date_of_birth,
                'Gender': gender,
                'Contact_Number': contact_number,
                'Email': email,
                'Address': address,
                'Registration_Date': registration_date,
                'Phone': phone,
                'Department': department,
                'Course': course,
                'Year': year
            }
            
            # Save to database
            df = pd.DataFrame([student_data])
            if os.path.exists(attendance_system.students_file):
                df.to_csv(attendance_system.students_file, mode='a', header=False, index=False)
            else:
                df.to_csv(attendance_system.students_file, index=False)
            
            # Save face embedding
            attendance_system.save_embedding(student_id, face_embedding)
            
            flash('Student registered successfully')
            return redirect(url_for('students'))
            
        except Exception as e:
            print(f"Error registering student: {str(e)}")
            flash('Error registering student')
            return redirect(url_for('register_student'))
    
    return render_template('register_student.html')

@app.route('/capture_face', methods=['POST'])
def capture_face():
    """Capture face image from webcam"""
    try:
        # Get base64 image from request
        face_image = request.json.get('image')
        if not face_image:
            return jsonify({'error': 'No image data received'}), 400
        
        # Convert base64 to image
        face_image = face_image.split(',')[1]
        face_image = base64.b64decode(face_image)
        face_image = Image.open(io.BytesIO(face_image))
        face_image = np.array(face_image)
        
        # Detect face
        attendance_system = initialize_attendance_system()
        boxes = attendance_system.detect_faces(face_image)
        
        if not boxes:
            return jsonify({'error': 'No face detected'}), 400
        
        # Get the first face
        x1, y1, x2, y2 = [int(coord) for coord in boxes[0]]
        face = face_image[y1:y2, x1:x2]
        
        # Convert back to base64
        face = Image.fromarray(face)
        buffered = io.BytesIO()
        face.save(buffered, format="JPEG")
        face_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({'image': face_base64})
        
    except Exception as e:
        print(f"Error capturing face: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        attendance_system = initialize_attendance_system()
    app.run(debug=False,port=1000,host="0.0.0.0") 
