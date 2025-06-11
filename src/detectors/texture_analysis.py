import cv2
import numpy as np
from typing import Dict, Tuple
from src.utils.face_detection import FaceDetector

class TextureAnalyzer:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.texture_threshold = 0.5
        self.edge_threshold = 100
        
    def analyze_texture(self, frame: np.ndarray, face: np.ndarray, landmarks: np.ndarray) -> Dict:
        """Analyze texture patterns in the face region."""
        try:
            face_region = self.face_detector.get_face_region(frame, face)
            if face_region is None or face_region.size == 0:
                return self._get_default_result()
            
            # Convert to grayscale
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Perform texture analysis
            texture_score = self._analyze_texture_patterns(gray)
            edge_score = self._analyze_edges(gray)
            frequency_score = self._analyze_frequency_domain(gray)
            
            # Combine scores
            combined_score = (texture_score + edge_score + frequency_score) / 3.0
            
            return {
                'texture_score': float(texture_score),
                'edge_score': float(edge_score),
                'frequency_score': float(frequency_score),
                'combined_score': float(combined_score),
                'is_spoofed': combined_score > self.texture_threshold
            }
        except Exception as e:
            print(f"Error in texture analysis: {str(e)}")
            return self._get_default_result()
    
    def _get_default_result(self) -> Dict:
        """Return default result when analysis fails."""
        return {
            'texture_score': 0.0,
            'edge_score': 0.0,
            'frequency_score': 0.0,
            'combined_score': 0.0,
            'is_spoofed': False
        }
    
    def _analyze_texture_patterns(self, gray_image: np.ndarray) -> float:
        """Analyze texture patterns using Local Binary Patterns (LBP)."""
        try:
            # Ensure image is not empty
            if gray_image is None or gray_image.size == 0:
                return 0.0
                
            # Calculate LBP
            radius = 1
            n_points = 8
            lbp = self._local_binary_pattern(gray_image, n_points, radius)
            
            # Calculate histogram with fixed number of bins
            hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
            hist = hist.astype("float")
            hist_sum = hist.sum()
            if hist_sum > 0:
                hist /= hist_sum
            
            # Calculate texture score based on histogram distribution
            valid_hist = hist[hist > 0]
            if len(valid_hist) == 0:
                return 0.0
                
            entropy = -np.sum(valid_hist * np.log2(valid_hist + 1e-7))
            max_entropy = np.log2(len(valid_hist))
            if max_entropy == 0:
                return 0.0
                
            texture_score = 1 - (entropy / max_entropy)
            return float(np.clip(texture_score, 0.0, 1.0))
            
        except Exception as e:
            print(f"Error in texture pattern analysis: {str(e)}")
            return 0.0
    
    def _local_binary_pattern(self, image: np.ndarray, n_points: int, radius: int) -> np.ndarray:
        """Calculate Local Binary Pattern."""
        try:
            rows, cols = image.shape
            output = np.zeros((rows, cols), dtype=np.uint8)
            
            for i in range(radius, rows - radius):
                for j in range(radius, cols - radius):
                    center = image[i, j]
                    pattern = 0
                    
                    for k in range(n_points):
                        # Calculate coordinates of the neighboring pixel
                        r = radius
                        x = i + int(round(r * np.cos(2.0 * np.pi * k / n_points)))
                        y = j - int(round(r * np.sin(2.0 * np.pi * k / n_points)))
                        
                        # Get the pixel value
                        if 0 <= x < rows and 0 <= y < cols:
                            if image[x, y] >= center:
                                pattern |= (1 << k)
                    
                    # Ensure pattern fits in uint8
                    output[i, j] = pattern & 0xFF
            
            return output
            
        except Exception as e:
            print(f"Error in LBP calculation: {str(e)}")
            return np.zeros_like(image, dtype=np.uint8)
    
    def _analyze_edges(self, gray_image: np.ndarray) -> float:
        """Analyze edge patterns in the image."""
        try:
            if gray_image is None or gray_image.size == 0:
                return 0.0
                
            # Apply Canny edge detection
            edges = cv2.Canny(gray_image, 100, 200)
            
            # Calculate edge density
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Calculate edge direction histogram
            sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            direction = np.arctan2(sobely, sobelx)
            
            # Calculate direction histogram
            hist, _ = np.histogram(direction[edges > 0], bins=36, range=(-np.pi, np.pi))
            hist = hist.astype("float")
            hist_sum = hist.sum()
            if hist_sum > 0:
                hist /= hist_sum
            
            # Calculate edge score based on density and direction distribution
            edge_score = edge_density * (1 - np.std(hist))
            return float(np.clip(edge_score, 0.0, 1.0))
            
        except Exception as e:
            print(f"Error in edge analysis: {str(e)}")
            return 0.0
    
    def _analyze_frequency_domain(self, gray_image: np.ndarray) -> float:
        """Analyze frequency domain patterns."""
        try:
            if gray_image is None or gray_image.size == 0:
                return 0.0
                
            # Apply 2D FFT
            f = np.fft.fft2(gray_image)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
            
            # Calculate frequency distribution
            hist, _ = np.histogram(magnitude_spectrum.ravel(), bins=50)
            hist = hist.astype("float")
            hist_sum = hist.sum()
            if hist_sum > 0:
                hist /= hist_sum
            
            # Calculate frequency score based on distribution
            valid_hist = hist[hist > 0]
            if len(valid_hist) == 0:
                return 0.0
                
            entropy = -np.sum(valid_hist * np.log2(valid_hist + 1e-7))
            max_entropy = np.log2(len(valid_hist))
            if max_entropy == 0:
                return 0.0
                
            frequency_score = 1 - (entropy / max_entropy)
            return float(np.clip(frequency_score, 0.0, 1.0))
            
        except Exception as e:
            print(f"Error in frequency domain analysis: {str(e)}")
            return 0.0 