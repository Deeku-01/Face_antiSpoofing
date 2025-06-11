import cv2
import numpy as np
from typing import Tuple

class ImagePreprocessor:
    def __init__(self):
        self.target_size = (224, 224)
        self.histogram_clip_limit = 2.0
        self.histogram_grid_size = (8, 8)
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing steps to the image."""
        # Resize image
        resized = self._resize_image(image)
        
        # Apply histogram equalization
        equalized = self._apply_clahe(resized)
        
        # Normalize image
        normalized = self._normalize_image(equalized)
        
        return normalized
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size while maintaining aspect ratio."""
        h, w = image.shape[:2]
        
        # Calculate scaling factor
        scale = min(self.target_size[0] / w, self.target_size[1] / h)
        new_size = (int(w * scale), int(h * scale))
        
        # Resize image
        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        
        # Create black canvas of target size
        canvas = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
        
        # Calculate position to paste resized image
        x_offset = (self.target_size[0] - new_size[0]) // 2
        y_offset = (self.target_size[1] - new_size[1]) // 2
        
        # Paste resized image onto canvas
        canvas[y_offset:y_offset + new_size[1], x_offset:x_offset + new_size[0]] = resized
        
        return canvas
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(
            clipLimit=self.histogram_clip_limit,
            tileGridSize=self.histogram_grid_size
        )
        
        # Apply CLAHE to L channel
        cl = clahe.apply(l)
        
        # Merge channels
        merged = cv2.merge((cl, a, b))
        
        # Convert back to BGR
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image values to [0, 1] range."""
        return image.astype(np.float32) / 255.0
    
    def enhance_edges(self, image: np.ndarray) -> np.ndarray:
        """Enhance edges in the image."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        return dilated
    
    def analyze_lighting(self, image: np.ndarray) -> Tuple[float, float]:
        """Analyze lighting conditions in the image."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Get L channel
        l_channel = lab[:, :, 0]
        
        # Calculate mean and standard deviation of brightness
        mean_brightness = np.mean(l_channel)
        std_brightness = np.std(l_channel)
        
        return mean_brightness, std_brightness
    
    def detect_noise(self, image: np.ndarray) -> float:
        """Detect noise level in the image."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply median blur
        blurred = cv2.medianBlur(gray, 3)
        
        # Calculate difference
        diff = cv2.absdiff(gray, blurred)
        
        # Calculate noise level
        noise_level = np.mean(diff)
        
        return noise_level 