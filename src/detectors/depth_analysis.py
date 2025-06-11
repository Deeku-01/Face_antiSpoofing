import cv2
import numpy as np
from typing import Dict, Tuple
from src.utils.face_detection import FaceDetector

class DepthAnalyzer:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.depth_threshold = 0.6
        self.history_size = 10
        self.depth_history = []
        
    def analyze_depth(self, frame: np.ndarray, face: np.ndarray, landmarks: np.ndarray) -> Dict:
        """Analyze depth characteristics of the face."""
        try:
            face_region = self.face_detector.get_face_region(frame, face)
            if face_region is None or face_region.size == 0:
                return self._get_default_result()
            
            # Convert to grayscale
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Calculate depth features
            gradient_score = self._analyze_gradient_depth(gray)
            shadow_score = self._analyze_shadow_patterns(face_region)
            contour_score = self._analyze_contour_variation(gray)
            
            # Combine scores
            combined_score = (gradient_score + shadow_score + contour_score) / 3.0
            
            # Update history
            self.depth_history.append(combined_score)
            if len(self.depth_history) > self.history_size:
                self.depth_history.pop(0)
            
            # Calculate temporal consistency
            temporal_score = self._calculate_temporal_consistency()
            
            return {
                'gradient_score': float(gradient_score),
                'shadow_score': float(shadow_score),
                'contour_score': float(contour_score),
                'temporal_score': float(temporal_score),
                'combined_score': float(combined_score),
                'is_spoofed': combined_score > self.depth_threshold
            }
            
        except Exception as e:
            print(f"Error in depth analysis: {str(e)}")
            return self._get_default_result()
    
    def _get_default_result(self) -> Dict:
        """Return default result when analysis fails."""
        return {
            'gradient_score': 0.0,
            'shadow_score': 0.0,
            'contour_score': 0.0,
            'temporal_score': 0.0,
            'combined_score': 0.0,
            'is_spoofed': False
        }
    
    def _analyze_gradient_depth(self, gray_image: np.ndarray) -> float:
        """Analyze depth using gradient information."""
        try:
            # Calculate gradients
            sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate gradient magnitude
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            # Calculate gradient direction
            direction = np.arctan2(sobely, sobelx)
            
            # Analyze gradient distribution
            hist, _ = np.histogram(direction.ravel(), bins=36, range=(-np.pi, np.pi))
            hist = hist.astype("float")
            hist_sum = hist.sum()
            if hist_sum > 0:
                hist /= hist_sum
            
            # Calculate gradient score
            gradient_score = np.std(magnitude) / (np.mean(magnitude) + 1e-7)
            return float(np.clip(gradient_score, 0.0, 1.0))
            
        except Exception as e:
            print(f"Error in gradient depth analysis: {str(e)}")
            return 0.0
    
    def _analyze_shadow_patterns(self, face_region: np.ndarray) -> float:
        """Analyze shadow patterns for depth information."""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            
            # Calculate local contrast
            blur = cv2.GaussianBlur(l_channel, (5, 5), 0)
            contrast = cv2.absdiff(l_channel, blur)
            
            # Analyze contrast distribution
            hist, _ = np.histogram(contrast.ravel(), bins=50)
            hist = hist.astype("float")
            hist_sum = hist.sum()
            if hist_sum > 0:
                hist /= hist_sum
            
            # Calculate shadow score
            shadow_score = np.std(contrast) / (np.mean(contrast) + 1e-7)
            return float(np.clip(shadow_score, 0.0, 1.0))
            
        except Exception as e:
            print(f"Error in shadow pattern analysis: {str(e)}")
            return 0.0
    
    def _analyze_contour_variation(self, gray_image: np.ndarray) -> float:
        """Analyze contour variations for depth information."""
        try:
            # Apply edge detection
            edges = cv2.Canny(gray_image, 100, 200)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 0.0
            
            # Calculate contour variations
            variations = []
            for contour in contours:
                if len(contour) > 5:  # Only consider contours with enough points
                    # Fit ellipse
                    ellipse = cv2.fitEllipse(contour)
                    center, axes, angle = ellipse
                    
                    # Calculate variation
                    variation = abs(axes[0] - axes[1]) / (axes[0] + axes[1] + 1e-7)
                    variations.append(variation)
            
            if not variations:
                return 0.0
            
            # Calculate contour score
            contour_score = np.mean(variations)
            return float(np.clip(contour_score, 0.0, 1.0))
            
        except Exception as e:
            print(f"Error in contour variation analysis: {str(e)}")
            return 0.0
    
    def _calculate_temporal_consistency(self) -> float:
        """Calculate temporal consistency of depth measurements."""
        try:
            if len(self.depth_history) < 2:
                return 0.0
            
            # Calculate variation in depth measurements
            variations = np.diff(self.depth_history)
            consistency = 1.0 - np.std(variations)
            
            return float(np.clip(consistency, 0.0, 1.0))
            
        except Exception as e:
            print(f"Error in temporal consistency calculation: {str(e)}")
            return 0.0 