import cv2
import numpy as np
from typing import Tuple, List, Dict
from src.utils.face_detection import FaceDetector

class EyeMovementDetector:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.eye_aspect_ratio_threshold = 0.2
        self.consecutive_frames_threshold = 3
        self.blink_history = []
        self.movement_history = []
        
    def analyze_eye_movement(self, frame: np.ndarray, face: np.ndarray, landmarks: np.ndarray) -> Dict:
        """Analyze eye movements and blinks in the frame."""
        left_eye, right_eye = self.face_detector.get_eye_regions(frame, landmarks)
        
        if left_eye is None or right_eye is None:
            return {
                'is_blinking': False,
                'eye_aspect_ratio': 0.0,
                'movement_score': 0.0,
                'is_natural': False
            }
        
        # Calculate eye aspect ratios
        left_ear = self._calculate_eye_aspect_ratio(left_eye)
        right_ear = self._calculate_eye_aspect_ratio(right_eye)
        
        # Average EAR
        ear = (left_ear + right_ear) / 2.0
        
        # Detect blink
        is_blinking = ear < self.eye_aspect_ratio_threshold
        
        # Analyze eye movement
        movement_score = self._analyze_eye_movement_smoothness(left_eye, right_eye)
        
        # Update history
        self.blink_history.append(is_blinking)
        self.movement_history.append(movement_score)
        
        # Keep history limited
        if len(self.blink_history) > 30:
            self.blink_history.pop(0)
        if len(self.movement_history) > 30:
            self.movement_history.pop(0)
        
        return {
            'is_blinking': is_blinking,
            'eye_aspect_ratio': ear,
            'movement_score': movement_score,
            'is_natural': self._is_natural_eye_behavior()
        }
    
    def _calculate_eye_aspect_ratio(self, eye_region: np.ndarray) -> float:
        """Calculate the eye aspect ratio (EAR) for an eye region."""
        # Convert to grayscale
        gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Get the largest contour (should be the eye)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate aspect ratio
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / h if h != 0 else 0.0
        
        return aspect_ratio
    
    def _analyze_eye_movement_smoothness(self, left_eye: np.ndarray, right_eye: np.ndarray) -> float:
        """Analyze the smoothness of eye movement."""
        # Convert to grayscale
        left_gray = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow if we have previous frames
        if hasattr(self, 'prev_left_eye') and hasattr(self, 'prev_right_eye'):
            # Calculate optical flow
            left_flow = cv2.calcOpticalFlowFarneback(
                self.prev_left_eye, left_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            right_flow = cv2.calcOpticalFlowFarneback(
                self.prev_right_eye, right_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Calculate movement magnitude
            left_magnitude = np.sqrt(left_flow[..., 0]**2 + left_flow[..., 1]**2)
            right_magnitude = np.sqrt(right_flow[..., 0]**2 + right_flow[..., 1]**2)
            
            # Calculate smoothness score (lower is smoother)
            smoothness = np.mean(np.abs(np.diff(left_magnitude))) + np.mean(np.abs(np.diff(right_magnitude)))
        else:
            smoothness = 0.0
        
        # Store current frames for next iteration
        self.prev_left_eye = left_gray
        self.prev_right_eye = right_gray
        
        return smoothness
    
    def _is_natural_eye_behavior(self) -> bool:
        """Determine if the eye behavior appears natural based on history."""
        if len(self.blink_history) < 10:
            return True
        
        # Check blink pattern
        blink_count = sum(self.blink_history)
        if blink_count == 0 or blink_count == len(self.blink_history):
            return False  # No blinks or constant blinking is suspicious
        
        # Check movement smoothness
        movement_scores = np.array(self.movement_history)
        if np.std(movement_scores) < 0.01:
            return False  # Too consistent movement is suspicious
        
        return True 