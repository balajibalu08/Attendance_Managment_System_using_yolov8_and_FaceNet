# Face Recognition Attendance System

A web-based attendance management system that uses facial recognition to automatically mark student attendance.

## Features

- **Face Recognition**: Automatically recognizes students using facial recognition technology
- **Real-time Attendance**: Marks attendance in real-time as students are recognized
- **Student Database**: Stores student information and face embeddings
- **Attendance Records**: Tracks and displays attendance history for all students
- **Export Functionality**: Export attendance records to Excel
- **Web Interface**: User-friendly web interface built with Flask and Bootstrap

## Technologies Used

- **Backend**: Flask, Python
- **Face Recognition**: FaceNet, YOLOv8
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Data Storage**: CSV files, NumPy arrays
- **Data Processing**: Pandas, NumPy, SciPy

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd face-recognition-attendance
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Download the YOLOv8 model:
   ```
   # The model will be downloaded automatically when the app is run
   ```

## Usage

1. Start the Flask application:
   ```
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

3. Use the web interface to:
   - Take attendance using facial recognition
   - View attendance records
   - View student information
   - Export attendance data

## Student Registration

Before using the attendance system, you need to register students:

1. Use the student registration system to add students to the database
2. Capture face images for each student
3. The system will create embeddings for each student's face

## File Structure

- `app.py`: Main Flask application
- `templates/`: HTML templates for the web interface
- `attendance_records.csv`: Stores attendance records
- `students_database.csv`: Stores student information
- `student_embeddings.npy`: Stores face embeddings for recognition

## Troubleshooting

- **Camera Issues**: Make sure your camera is properly connected and accessible
- **Recognition Problems**: Ensure good lighting and clear face visibility
- **Database Errors**: Check if the CSV files are properly formatted

## License

This project is licensed under the MIT License - see the LICENSE file for details. 