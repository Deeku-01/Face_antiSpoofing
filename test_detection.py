import cv2
import numpy as np
from src.detectors.eye_movement import EyeMovementDetector
from src.detectors.texture_analysis import TextureAnalyzer
from src.utils.face_detection import FaceDetector

def main():
    # Initialize components
    face_detector = FaceDetector()
    eye_analyzer = EyeMovementDetector()
    texture_analyzer = TextureAnalyzer()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit")
    print("Press 't' to toggle test mode")
    print("Press 'r' to reset test mode")
    
    test_mode = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect face
        faces = face_detector.detect_faces(frame)
        
        if len(faces) > 0:
            face = faces[0]
            landmarks = face_detector.get_landmarks(frame, face)
            
            if landmarks is not None:
                # Analyze eye movement
                eye_result = eye_analyzer.analyze_eye_movement(frame, face, landmarks)
                
                # Analyze texture
                texture_result = texture_analyzer.analyze_texture(frame, face, landmarks)
                
                # Draw results
                x, y, w, h = face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Display results
                is_spoofed = not eye_result['is_natural'] or texture_result['is_spoofed']
                status = "SPOOFED" if is_spoofed else "REAL"
                color = (0, 0, 255) if is_spoofed else (0, 255, 0)
                
                cv2.putText(frame, f"Status: {status}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                cv2.putText(frame, f"Eye Score: {eye_result['movement_score']:.2f}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.putText(frame, f"Texture Score: {texture_result['combined_score']:.2f}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display test mode status
                if test_mode:
                    cv2.putText(frame, "TEST MODE: ON", (10, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display frame
        cv2.imshow('Anti-Spoof Detection', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            test_mode = not test_mode
            if test_mode:
                eye_analyzer.enable_test_mode()
                print("Test mode enabled")
            else:
                eye_analyzer.disable_test_mode()
                print("Test mode disabled")
        elif key == ord('r'):
            test_mode = False
            eye_analyzer.disable_test_mode()
            print("Test mode reset")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 