import cv2
import numpy as np
from typing import Tuple, List, Optional

class FaceDetector:
    def __init__(self):
        # Load OpenCV's DNN face detector
        self.face_detector = cv2.dnn.readNetFromCaffe(
            "models/deploy.prototxt",
            "models/res10_300x300_ssd_iter_140000.caffemodel"
        )
        
        # Initialize face landmark detector
        self.landmark_detector = cv2.face.createFacemarkLBF()
        self.landmark_detector.loadModel("models/lbfmodel.yaml")
        
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in the frame using OpenCV's DNN face detector."""
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 
            1.0, 
            (300, 300), 
            (104, 177, 123)
        )
        
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.5:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                x, y, w, h = box.astype("int")
                faces.append((x, y, w-x, h-y))
        
        return faces
    
    def get_landmarks(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Get facial landmarks using OpenCV's face landmark detector."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x, y, w, h = face
            face_rect = np.array([(x, y, w, h)], dtype=np.int32)
            
            success, landmarks = self.landmark_detector.fit(gray, face_rect)
            if success and len(landmarks) > 0:
                # Convert landmarks to the correct format
                landmarks = landmarks[0].reshape(-1, 2)
                return landmarks
            return None
        except Exception as e:
            print(f"Error detecting landmarks: {str(e)}")
            return None
    
    def get_face_region(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract the face region from the frame."""
        x, y, w, h = face
        return frame[y:y+h, x:x+w]
    
    def get_eye_regions(self, frame: np.ndarray, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract eye regions from facial landmarks."""
        try:
            if landmarks is None or len(landmarks) == 0:
                return None, None
            
            # Convert landmarks to numpy array if it's not already
            landmarks = np.array(landmarks)
            
            # Get eye landmarks (approximate indices for left and right eyes)
            # Using a simpler approach with fewer points
            left_eye = landmarks[36:42]  # Left eye points
            right_eye = landmarks[42:48]  # Right eye points
            
            # Get bounding boxes for eyes with padding
            padding = 5
            left_eye_box = cv2.boundingRect(left_eye.astype(np.float32))
            right_eye_box = cv2.boundingRect(right_eye.astype(np.float32))
            
            # Add padding to the boxes
            x, y, w, h = left_eye_box
            left_eye_box = (max(0, x-padding), max(0, y-padding), 
                          min(frame.shape[1]-x, w+2*padding), 
                          min(frame.shape[0]-y, h+2*padding))
            
            x, y, w, h = right_eye_box
            right_eye_box = (max(0, x-padding), max(0, y-padding), 
                           min(frame.shape[1]-x, w+2*padding), 
                           min(frame.shape[0]-y, h+2*padding))
            
            # Extract eye regions
            x, y, w, h = left_eye_box
            left_eye_region = frame[y:y+h, x:x+w]
            
            x, y, w, h = right_eye_box
            right_eye_region = frame[y:y+h, x:x+w]
            
            # Ensure regions are not empty
            if left_eye_region.size == 0 or right_eye_region.size == 0:
                return None, None
                
            return left_eye_region, right_eye_region
            
        except Exception as e:
            print(f"Error extracting eye regions: {str(e)}")
            return None, None
    
    def draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Draw facial landmarks on the frame."""
        if landmarks is not None:
            for point in landmarks:
                x, y = point
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
        return frame 