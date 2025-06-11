import cv2
import numpy as np
from typing import Tuple, List, Optional

class FaceDetector:
    def __init__(self):
        # Initialize OpenCV's face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in the frame using OpenCV's face detector."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces
    
    def get_landmarks(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Get facial landmarks for a detected face."""
        x, y, w, h = face
        roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) >= 2:
            # Create simple landmarks (just eye centers)
            landmarks = []
            for (ex, ey, ew, eh) in eyes[:2]:  # Take first two eyes
                center_x = x + ex + ew//2
                center_y = y + ey + eh//2
                landmarks.append([center_x, center_y])
            
            # Add nose and mouth points (approximate)
            nose_x = x + w//2
            nose_y = y + h//2
            mouth_x = x + w//2
            mouth_y = y + int(h * 0.7)
            
            landmarks.append([nose_x, nose_y])
            landmarks.append([mouth_x, mouth_y])
            
            return np.array(landmarks)
        return None
    
    def get_face_region(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract the face region from the frame."""
        x, y, w, h = face
        return frame[y:y+h, x:x+w]
    
    def get_eye_regions(self, frame: np.ndarray, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract left and right eye regions using facial landmarks."""
        if len(landmarks) < 2:
            return None, None
            
        # Get eye centers
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        
        # Define eye region size
        eye_size = 30
        
        # Extract eye regions
        left_eye_region = frame[
            left_eye[1]-eye_size//2:left_eye[1]+eye_size//2,
            left_eye[0]-eye_size//2:left_eye[0]+eye_size//2
        ]
        right_eye_region = frame[
            right_eye[1]-eye_size//2:right_eye[1]+eye_size//2,
            right_eye[0]-eye_size//2:right_eye[0]+eye_size//2
        ]
        
        return left_eye_region, right_eye_region
    
    def draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Draw facial landmarks on the frame."""
        frame_copy = frame.copy()
        for (x, y) in landmarks:
            cv2.circle(frame_copy, (x, y), 3, (0, 255, 0), -1)
        return frame_copy 