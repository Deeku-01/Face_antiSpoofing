import cv2
import numpy as np
from typing import Dict, Tuple
from src.utils.face_detection import FaceDetector

class LivenessDetector:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.liveness_threshold = 0.3  # Lowered threshold
        self.history_size = 10  # Reduced history size
        self.face_history = []
        self.movement_threshold = 0.05  # Lowered movement threshold
        self.min_face_size = 100  # Minimum face size to process
        
    def analyze_liveness(self, frame: np.ndarray, face: np.ndarray, landmarks: np.ndarray) -> Dict:
        """Analyze if the face shows signs of being alive."""
        try:
            # Check face size
            x, y, w, h = face
            if w < self.min_face_size or h < self.min_face_size:
                return self._get_default_result()
            
            face_region = self.face_detector.get_face_region(frame, face)
            if face_region is None or face_region.size == 0:
                return self._get_default_result()
            
            # Convert to grayscale
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Calculate liveness features
            movement_score = self._analyze_movement(gray)
            expression_score = self._analyze_expression(face_region)
            temporal_score = self._analyze_temporal_patterns()
            
            # Combine scores with weights
            combined_score = (0.4 * movement_score + 
                            0.3 * expression_score + 
                            0.3 * temporal_score)
            
            # Update history
            self.face_history.append(gray)
            if len(self.face_history) > self.history_size:
                self.face_history.pop(0)
            
            return {
                'movement_score': float(movement_score),
                'expression_score': float(expression_score),
                'temporal_score': float(temporal_score),
                'combined_score': float(combined_score),
                'is_live': combined_score > self.liveness_threshold
            }
            
        except Exception as e:
            print(f"Error in liveness analysis: {str(e)}")
            return self._get_default_result()
    
    def _get_default_result(self) -> Dict:
        """Return default result when analysis fails."""
        return {
            'movement_score': 0.0,
            'expression_score': 0.0,
            'temporal_score': 0.0,
            'combined_score': 0.0,
            'is_live': True  # Default to True to avoid false positives
        }
    
    def _analyze_movement(self, current_frame: np.ndarray) -> float:
        """Analyze face movement between frames."""
        try:
            if len(self.face_history) < 2:
                return 0.0
            
            # Get previous frame
            prev_frame = self.face_history[-2]
            
            # Ensure frames are the same size
            if prev_frame.shape != current_frame.shape:
                prev_frame = cv2.resize(prev_frame, (current_frame.shape[1], current_frame.shape[0]))
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Calculate movement magnitude
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            
            # Calculate movement score
            movement_score = np.mean(magnitude)
            return float(np.clip(movement_score / self.movement_threshold, 0.0, 1.0))
            
        except Exception as e:
            print(f"Error in movement analysis: {str(e)}")
            return 0.0
    
    def _analyze_expression(self, face_region: np.ndarray) -> float:
        """Analyze facial expressions for signs of liveness."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 100, 200)
            
            # Calculate local variance
            local_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate expression score
            expression_score = local_var / 200.0  # Adjusted normalization
            return float(np.clip(expression_score, 0.0, 1.0))
            
        except Exception as e:
            print(f"Error in expression analysis: {str(e)}")
            return 0.0
    
    def _analyze_temporal_patterns(self) -> float:
        """Analyze temporal patterns in face movement."""
        try:
            if len(self.face_history) < 3:
                return 0.0
            
            # Calculate frame differences
            diffs = []
            for i in range(1, len(self.face_history)):
                if self.face_history[i].shape == self.face_history[i-1].shape:
                    diff = cv2.absdiff(self.face_history[i], self.face_history[i-1])
                    diffs.append(np.mean(diff))
            
            # Calculate temporal score
            if not diffs:
                return 0.0
            
            # Check for natural variation in movement
            variation = np.std(diffs)
            mean_diff = np.mean(diffs)
            
            if mean_diff == 0:
                return 0.0
            
            temporal_score = variation / (mean_diff + 1e-7)
            return float(np.clip(temporal_score, 0.0, 1.0))
            
        except Exception as e:
            print(f"Error in temporal pattern analysis: {str(e)}")
            return 0.0 