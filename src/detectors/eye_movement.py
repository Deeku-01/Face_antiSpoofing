import cv2
import numpy as np
from typing import Tuple, List, Dict
from src.utils.face_detection import FaceDetector
from collections import deque

class EyeMovementDetector:
    def __init__(self, history_size=10, movement_threshold=0.1):
        self.face_detector = FaceDetector()
        self.eye_aspect_ratio_threshold = 0.2
        self.consecutive_frames_threshold = 3
        self.blink_history = []
        self.movement_history = []
        self.history_size = history_size
        self.movement_threshold = movement_threshold
        self.eye_movement_history = deque(maxlen=history_size)
        self.prev_eye_regions = None
        self.optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
        
    def _preprocess_eye_region(self, eye_region):
        """Preprocess eye region for optical flow."""
        if eye_region is None or eye_region.size == 0:
            return None
            
        # Convert to grayscale
        if len(eye_region.shape) == 3:
            eye_region = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            
        # Ensure minimum size
        min_size = (32, 32)
        if eye_region.shape[0] < min_size[0] or eye_region.shape[1] < min_size[1]:
            eye_region = cv2.resize(eye_region, min_size)
            
        # Normalize
        eye_region = cv2.normalize(eye_region, None, 0, 255, cv2.NORM_MINMAX)
        
        return eye_region
        
    def _calculate_movement(self, prev_region, curr_region):
        """Calculate movement between two eye regions."""
        if prev_region is None or curr_region is None:
            return 0.0
            
        try:
            # Ensure both regions have the same size
            if prev_region.shape != curr_region.shape:
                curr_region = cv2.resize(curr_region, (prev_region.shape[1], prev_region.shape[0]))
            
            # Calculate optical flow
            flow = self.optical_flow.calc(prev_region, curr_region, None)
            
            if flow is None:
                return 0.0
                
            # Calculate magnitude of movement
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            return np.mean(magnitude)
            
        except Exception as e:
            print(f"Error calculating movement: {str(e)}")
            return 0.0
            
    def analyze_eye_movement(self, frame: np.ndarray, face: np.ndarray, landmarks: np.ndarray) -> Dict:
        """Analyze eye movements and blinks in the frame."""
        try:
            x, y, w, h = face
            
            # Extract eye regions using landmarks
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            
            # Get bounding boxes for eyes
            left_eye_box = cv2.boundingRect(left_eye.astype(np.float32))
            right_eye_box = cv2.boundingRect(right_eye.astype(np.float32))
            
            # Extract eye regions
            left_eye_region = frame[y + left_eye_box[1]:y + left_eye_box[1] + left_eye_box[3],
                                  x + left_eye_box[0]:x + left_eye_box[0] + left_eye_box[2]]
            right_eye_region = frame[y + right_eye_box[1]:y + right_eye_box[1] + right_eye_box[3],
                                   x + right_eye_box[0]:x + right_eye_box[0] + right_eye_box[2]]
            
            # Preprocess eye regions
            left_eye_region = self._preprocess_eye_region(left_eye_region)
            right_eye_region = self._preprocess_eye_region(right_eye_region)
            
            if self.prev_eye_regions is None:
                self.prev_eye_regions = (left_eye_region, right_eye_region)
                return {
                    'is_natural': True,
                    'movement_score': 0.0,
                    'error': None
                }
            
            # Calculate movement for both eyes
            left_movement = self._calculate_movement(self.prev_eye_regions[0], left_eye_region)
            right_movement = self._calculate_movement(self.prev_eye_regions[1], right_eye_region)
            
            # Average movement
            movement = (left_movement + right_movement) / 2.0
            
            # Update history
            self.eye_movement_history.append(movement)
            
            # Update previous regions
            self.prev_eye_regions = (left_eye_region, right_eye_region)
            
            # Calculate movement score
            if len(self.eye_movement_history) > 1:
                movement_score = np.mean(list(self.eye_movement_history))
                is_natural = movement_score > self.movement_threshold
            else:
                movement_score = 0.0
                is_natural = True
            
            return {
                'is_natural': is_natural,
                'movement_score': movement_score,
                'error': None
            }
            
        except Exception as e:
            print(f"Error in eye movement analysis: {str(e)}")
            return {
                'is_natural': False,
                'movement_score': 0.0,
                'error': str(e)
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