import cv2
import numpy as np
import argparse
from typing import Dict, Tuple
from utils.face_detection import FaceDetector
from detectors.eye_movement import EyeMovementDetector
from detectors.texture_analysis import TextureAnalyzer
from utils.visualization import Visualizer

class AntiSpoofingSystem:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.eye_detector = EyeMovementDetector()
        self.texture_analyzer = TextureAnalyzer()
        self.visualizer = Visualizer()
        self.spoof_threshold = 0.7
        
    def process_frame(self, frame: np.ndarray) -> Tuple[bool, Dict]:
        """Process a single frame and determine if it's spoofed."""
        # Detect faces
        faces = self.face_detector.detect_faces(frame)
        
        if len(faces) == 0:
            return False, {'error': 'No face detected'}
        
        # Process the first detected face
        face = faces[0]
        landmarks = self.face_detector.get_landmarks(frame, face)
        
        if landmarks is None:
            return False, {'error': 'Could not detect facial landmarks'}
        
        # Get eye regions
        left_eye, right_eye = self.face_detector.get_eye_regions(frame, landmarks)
        
        if left_eye is None or right_eye is None:
            return False, {'error': 'Could not detect eye regions'}
        
        # Get detection results from each module
        eye_results = self.eye_detector.analyze_eye_movement(frame, face, landmarks)
        texture_results = self.texture_analyzer.analyze_texture(frame, face, landmarks)
        
        # Combine results
        is_spoofed = self._combine_results(eye_results, texture_results)
        
        # Prepare detailed results
        results = {
            'face_bbox': face,
            'landmarks': landmarks,
            'eye_regions': (left_eye, right_eye),
            'eye_analysis': eye_results,
            'texture_analysis': texture_results,
            'is_spoofed': is_spoofed
        }
        
        return is_spoofed, results
    
    def _combine_results(self, eye_results: Dict, texture_results: Dict) -> bool:
        """Combine results from different detectors to make final decision."""
        # Weight the different factors
        eye_weight = 0.4
        texture_weight = 0.6
        
        # Calculate combined score
        eye_score = 0.0 if eye_results['is_natural'] else 1.0
        texture_score = texture_results['combined_score']
        
        combined_score = (eye_weight * eye_score + texture_weight * texture_score)
        
        return combined_score > self.spoof_threshold

def main():
    parser = argparse.ArgumentParser(description='Anti-Face Spoofing Detection System')
    parser.add_argument('--video', type=str, help='Path to video file (optional)')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with additional visualization')
    args = parser.parse_args()
    
    # Initialize system
    system = AntiSpoofingSystem()
    
    # Open video source
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    print("Press 'q' to quit")
    print("Press 'd' to toggle debug mode")
    
    debug_mode = args.debug
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        is_spoofed, results = system.process_frame(frame)
        
        # Draw results
        if debug_mode:
            output_frame = system.visualizer.create_debug_view(frame, results)
        else:
            output_frame = system.visualizer.draw_detection_results(frame, results)
        
        # Display results
        cv2.imshow('Anti-Spoofing Detection', output_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug mode: {'enabled' if debug_mode else 'disabled'}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 