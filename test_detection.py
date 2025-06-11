import cv2
import numpy as np
from src.detectors.eye_movement import EyeMovementDetector
from src.detectors.texture_analysis import TextureAnalyzer
from src.detectors.depth_analysis import DepthAnalyzer
from src.detectors.color_analysis import ColorAnalyzer
from src.detectors.liveness_detection import LivenessDetector
from src.utils.face_detection import FaceDetector

def draw_text_with_background(frame, text, position, scale=0.7, thickness=2, color=(0, 255, 0)):
    """Draw text with a black background for better visibility."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    x, y = position
    
    # Draw black background rectangle
    cv2.rectangle(frame, 
                 (x, y - text_size[1] - 10),
                 (x + text_size[0] + 10, y + 10),
                 (0, 0, 0),
                 -1)
    
    # Draw text
    cv2.putText(frame, text, (x + 5, y - 5), font, scale, color, thickness)

def main():
    # Initialize components
    face_detector = FaceDetector()
    eye_detector = EyeMovementDetector()
    texture_analyzer = TextureAnalyzer()
    depth_analyzer = DepthAnalyzer()
    color_analyzer = ColorAnalyzer()
    liveness_detector = LivenessDetector()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set to 30 FPS
    
    # Initialize variables for smooth updates
    frame_count = 0
    last_results = {
        'eye_score': 0.0,
        'texture_score': 0.0,
        'depth_score': 0.0,
        'color_score': 0.0,
        'liveness_score': 0.0,
        'is_spoofed': False
    }
    alpha = 0.3  # Smoothing factor for score updates
    
    # Initialize display parameters
    line_height = 30
    base_y_offset = 60
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process every frame for better responsiveness
        frame_count += 1
        
        # Initialize y_offset for this frame
        y_offset = base_y_offset
        
        # Detect face
        faces = face_detector.detect_faces(frame)
        
        if len(faces) > 0:
            face = faces[0]
            x, y, w, h = face
            
            # Get face landmarks
            landmarks = face_detector.get_landmarks(frame, face)
            
            if landmarks is not None:
                # Analyze eye movements
                eye_result = eye_detector.analyze_eye_movement(frame, face, landmarks)
                
                # Analyze texture
                texture_result = texture_analyzer.analyze_texture(frame, face, landmarks)
                
                # Analyze depth
                depth_result = depth_analyzer.analyze_depth(frame, face, landmarks)
                
                # Analyze color
                color_result = color_analyzer.analyze_color(frame, face, landmarks)
                
                # Analyze liveness
                liveness_result = liveness_detector.analyze_liveness(frame, face, landmarks)
                
                # Smoothly update results
                last_results['eye_score'] = (1 - alpha) * last_results['eye_score'] + alpha * eye_result['movement_score']
                last_results['texture_score'] = (1 - alpha) * last_results['texture_score'] + alpha * texture_result['texture_score']
                last_results['depth_score'] = (1 - alpha) * last_results['depth_score'] + alpha * depth_result['combined_score']
                last_results['color_score'] = (1 - alpha) * last_results['color_score'] + alpha * color_result['combined_score']
                last_results['liveness_score'] = (1 - alpha) * last_results['liveness_score'] + alpha * liveness_result['combined_score']
                
                # Determine if face is spoofed
                is_spoofed = (not eye_result['is_natural'] or  # Inverted is_natural to get is_spoofed
                            texture_result['is_spoofed'] or 
                            depth_result['is_spoofed'] or
                            color_result['is_spoofed'] or
                            not liveness_result['is_live'])  # Inverted is_live to get is_spoofed
                
                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw status with background
                status = "SPOOFED" if is_spoofed else "REAL"
                color = (0, 0, 255) if is_spoofed else (0, 255, 0)
                draw_text_with_background(frame, f"Status: {status}", (10, 30), color=color)
                
                # Draw scores with background
                scores = [
                    f"Eye Movement: {last_results['eye_score']:.2f}",
                    f"Texture: {last_results['texture_score']:.2f}",
                    f"Depth: {last_results['depth_score']:.2f}",
                    f"Color: {last_results['color_score']:.2f}",
                    f"Liveness: {last_results['liveness_score']:.2f}"
                ]
                
                for score in scores:
                    draw_text_with_background(frame, score, (10, y_offset))
                    y_offset += line_height
                
                # Draw landmarks
                for (x, y) in landmarks:
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
        
        # Display FPS (always at the bottom)
        fps = cap.get(cv2.CAP_PROP_FPS)
        draw_text_with_background(frame, f"FPS: {fps:.1f}", (10, y_offset + line_height))
        
        # Show frame
        cv2.imshow('Anti-Spoofing Detection', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 