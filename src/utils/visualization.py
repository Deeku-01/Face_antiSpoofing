import cv2
import numpy as np
from typing import Dict, List, Tuple

class Visualizer:
    def __init__(self):
        self.colors = {
            'real': (0, 255, 0),      # Green
            'spoofed': (0, 0, 255),   # Red
            'warning': (0, 255, 255), # Yellow
            'info': (255, 255, 255)   # White
        }
        
    def draw_detection_results(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """Draw detection results on the frame."""
        frame_copy = frame.copy()
        
        # Draw face rectangle and label
        if 'face_bbox' in results:
            bbox = results['face_bbox']
            color = self.colors['spoofed'] if results.get('is_spoofed', False) else self.colors['real']
            cv2.rectangle(frame_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Add spoofing status
            status = "SPOOFED" if results.get('is_spoofed', False) else "REAL"
            cv2.putText(frame_copy, status, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Draw landmarks if available
        if 'landmarks' in results:
            landmarks = results['landmarks']
            for (x, y) in landmarks:
                cv2.circle(frame_copy, (x, y), 1, self.colors['info'], -1)
        
        # Draw eye regions if available
        if 'eye_regions' in results:
            left_eye, right_eye = results['eye_regions']
            cv2.rectangle(frame_copy, 
                        (left_eye[0], left_eye[1]),
                        (left_eye[2], left_eye[3]),
                        self.colors['info'], 1)
            cv2.rectangle(frame_copy,
                        (right_eye[0], right_eye[1]),
                        (right_eye[2], right_eye[3]),
                        self.colors['info'], 1)
        
        # Draw detection scores
        self._draw_scores(frame_copy, results)
        
        return frame_copy
    
    def _draw_scores(self, frame: np.ndarray, results: Dict) -> None:
        """Draw detection scores and metrics."""
        y_offset = 30
        line_height = 25
        
        # Draw eye analysis results
        if 'eye_analysis' in results:
            eye_results = results['eye_analysis']
            cv2.putText(frame, f"Eye Movement: {eye_results['movement_score']:.2f}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       self.colors['info'], 2)
            y_offset += line_height
            
            cv2.putText(frame, f"Blink Status: {'Blinking' if eye_results['is_blinking'] else 'Open'}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       self.colors['info'], 2)
            y_offset += line_height
        
        # Draw texture analysis results
        if 'texture_analysis' in results:
            texture_results = results['texture_analysis']
            cv2.putText(frame, f"Texture Score: {texture_results['texture_score']:.2f}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       self.colors['info'], 2)
            y_offset += line_height
            
            cv2.putText(frame, f"Edge Score: {texture_results['edge_score']:.2f}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       self.colors['info'], 2)
            y_offset += line_height
            
            cv2.putText(frame, f"Frequency Score: {texture_results['frequency_score']:.2f}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       self.colors['info'], 2)
    
    def create_debug_view(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """Create a debug view with multiple analysis windows."""
        # Create a larger canvas for debug view
        h, w = frame.shape[:2]
        debug_view = np.zeros((h, w * 2, 3), dtype=np.uint8)
        
        # Original frame with detection results
        debug_view[:, :w] = self.draw_detection_results(frame, results)
        
        # Analysis view
        analysis_view = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw eye regions
        if 'eye_regions' in results:
            left_eye, right_eye = results['eye_regions']
            left_eye_img = frame[left_eye[1]:left_eye[3], left_eye[0]:left_eye[2]]
            right_eye_img = frame[right_eye[1]:right_eye[3], right_eye[0]:right_eye[2]]
            
            # Resize eye images to fit in analysis view
            eye_h = h // 3
            eye_w = w // 4
            
            left_eye_resized = cv2.resize(left_eye_img, (eye_w, eye_h))
            right_eye_resized = cv2.resize(right_eye_img, (eye_w, eye_h))
            
            # Place eye images in analysis view
            analysis_view[10:10 + eye_h, 10:10 + eye_w] = left_eye_resized
            analysis_view[10:10 + eye_h, w - eye_w - 10:w - 10] = right_eye_resized
            
            # Add labels
            cv2.putText(analysis_view, "Left Eye", (10, 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['info'], 1)
            cv2.putText(analysis_view, "Right Eye", (w - eye_w - 10, 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['info'], 1)
        
        # Draw scores and metrics
        self._draw_scores(analysis_view, results)
        
        # Add analysis view to debug view
        debug_view[:, w:] = analysis_view
        
        return debug_view 