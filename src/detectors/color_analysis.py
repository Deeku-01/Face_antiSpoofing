import cv2
import numpy as np
from typing import Dict, Tuple
from src.utils.face_detection import FaceDetector

class ColorAnalyzer:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.color_threshold = 0.7  # Increased threshold
        self.history_size = 10
        self.color_history = []
        self.min_face_size = 100  # Minimum face size to process
        
    def analyze_color(self, frame: np.ndarray, face: np.ndarray, landmarks: np.ndarray) -> Dict:
        """Analyze color patterns in the face region."""
        try:
            # Check face size
            x, y, w, h = face
            if w < self.min_face_size or h < self.min_face_size:
                return self._get_default_result()
            
            face_region = self.face_detector.get_face_region(frame, face)
            if face_region is None or face_region.size == 0:
                return self._get_default_result()
            
            # Calculate color features
            skin_tone_score = self._analyze_skin_tone(face_region)
            color_variance_score = self._analyze_color_variance(face_region)
            reflection_score = self._analyze_reflections(face_region)
            
            # Combine scores with weights
            combined_score = (0.4 * skin_tone_score + 
                            0.3 * color_variance_score + 
                            0.3 * reflection_score)
            
            # Update history
            self.color_history.append(combined_score)
            if len(self.color_history) > self.history_size:
                self.color_history.pop(0)
            
            # Calculate temporal consistency
            temporal_score = self._calculate_temporal_consistency()
            
            return {
                'skin_tone_score': float(skin_tone_score),
                'color_variance_score': float(color_variance_score),
                'reflection_score': float(reflection_score),
                'temporal_score': float(temporal_score),
                'combined_score': float(combined_score),
                'is_spoofed': combined_score > self.color_threshold
            }
            
        except Exception as e:
            print(f"Error in color analysis: {str(e)}")
            return self._get_default_result()
    
    def _get_default_result(self) -> Dict:
        """Return default result when analysis fails."""
        return {
            'skin_tone_score': 0.0,
            'color_variance_score': 0.0,
            'reflection_score': 0.0,
            'temporal_score': 0.0,
            'combined_score': 0.0,
            'is_spoofed': False  # Default to False to avoid false positives
        }
    
    def _analyze_skin_tone(self, face_region: np.ndarray) -> float:
        """Analyze if the skin tone appears natural."""
        try:
            # Convert to YCrCb color space (better for skin tone analysis)
            ycrcb = cv2.cvtColor(face_region, cv2.COLOR_BGR2YCR_CB)
            
            # Define skin tone ranges (adjusted for better detection)
            min_YCrCb = np.array([0, 130, 70], dtype=np.uint8)
            max_YCrCb = np.array([255, 180, 140], dtype=np.uint8)
            
            # Create skin mask
            skin_mask = cv2.inRange(ycrcb, min_YCrCb, max_YCrCb)
            
            # Calculate skin tone score
            skin_pixels = np.sum(skin_mask > 0)
            total_pixels = face_region.shape[0] * face_region.shape[1]
            skin_ratio = skin_pixels / total_pixels
            
            # Calculate color distribution
            hist = cv2.calcHist([ycrcb], [1, 2], skin_mask, [32, 32], [0, 256, 0, 256])
            hist = hist.astype("float")
            hist_sum = hist.sum()
            if hist_sum > 0:
                hist /= hist_sum
            
            # Calculate entropy of color distribution
            valid_hist = hist[hist > 0]
            if len(valid_hist) == 0:
                return 0.0
                
            entropy = -np.sum(valid_hist * np.log2(valid_hist + 1e-7))
            max_entropy = np.log2(len(valid_hist))
            if max_entropy == 0:
                return 0.0
            
            # Combine skin ratio and entropy
            skin_tone_score = (skin_ratio * (1 - entropy/max_entropy))
            return float(np.clip(skin_tone_score, 0.0, 1.0))
            
        except Exception as e:
            print(f"Error in skin tone analysis: {str(e)}")
            return 0.0
    
    def _analyze_color_variance(self, face_region: np.ndarray) -> float:
        """Analyze color variance in the face region."""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
            
            # Calculate variance for each channel
            variances = []
            for i in range(3):
                channel = lab[:, :, i]
                variance = np.var(channel)
                variances.append(variance)
            
            # Calculate color variance score
            color_variance_score = np.mean(variances) / 500.0  # Adjusted normalization
            return float(np.clip(color_variance_score, 0.0, 1.0))
            
        except Exception as e:
            print(f"Error in color variance analysis: {str(e)}")
            return 0.0
    
    def _analyze_reflections(self, face_region: np.ndarray) -> float:
        """Analyze reflections and highlights in the face region."""
        try:
            # Convert to HSV color space
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            
            # Extract value channel
            value = hsv[:, :, 2]
            
            # Calculate local contrast
            blur = cv2.GaussianBlur(value, (5, 5), 0)
            contrast = cv2.absdiff(value, blur)
            
            # Analyze contrast distribution
            hist, _ = np.histogram(contrast.ravel(), bins=50)
            hist = hist.astype("float")
            hist_sum = hist.sum()
            if hist_sum > 0:
                hist /= hist_sum
            
            # Calculate reflection score
            reflection_score = np.std(contrast) / (np.mean(contrast) + 1e-7)
            return float(np.clip(reflection_score, 0.0, 1.0))
            
        except Exception as e:
            print(f"Error in reflection analysis: {str(e)}")
            return 0.0
    
    def _calculate_temporal_consistency(self) -> float:
        """Calculate temporal consistency of color measurements."""
        try:
            if len(self.color_history) < 2:
                return 0.0
            
            # Calculate variation in color measurements
            variations = np.diff(self.color_history)
            consistency = 1.0 - np.std(variations)
            
            return float(np.clip(consistency, 0.0, 1.0))
            
        except Exception as e:
            print(f"Error in temporal consistency calculation: {str(e)}")
            return 0.0 