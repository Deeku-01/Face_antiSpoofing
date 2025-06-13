import cv2
import numpy as np
from collections import deque
from src.detectors.eye_movement import EyeMovementDetector
from src.detectors.texture_analysis import TextureAnalyzer
from src.detectors.depth_analysis import DepthAnalyzer
from src.detectors.color_analysis import ColorAnalyzer
from src.detectors.liveness_detection import LivenessDetector
from src.detectors.deep_learning import DeepLearningDetector
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

class ScoreAccumulator:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.scores = {
            'eye_score': deque(maxlen=window_size),
            'texture_score': deque(maxlen=window_size),
            'depth_score': deque(maxlen=window_size),
            'color_score': deque(maxlen=window_size),
            'liveness_score': deque(maxlen=window_size),
            'deep_learning_score': deque(maxlen=window_size)
        }
        self.weights = {
            'eye_score': 0.15,
            'texture_score': 0.15,
            'depth_score': 0.15,
            'color_score': 0.15,
            'liveness_score': 0.15,
            'deep_learning_score': 0.25  # Higher weight for deep learning
        }
        
    def add_scores(self, scores):
        for key in self.scores:
            if key in scores:
                self.scores[key].append(scores[key])
    
    def get_decision(self):
        if not any(self.scores.values()):
            return False, 0.0
            
        weighted_sum = 0.0
        total_weight = 0.0
        
        for key, weight in self.weights.items():
            if self.scores[key]:
                avg_score = np.mean(list(self.scores[key]))
                weighted_sum += avg_score * weight
                total_weight += weight
        
        if total_weight == 0:
            return False, 0.0
            
        final_score = weighted_sum / total_weight
        return final_score > 0.5, final_score

def main():
    # Initialize components
    face_detector = FaceDetector()
    eye_detector = EyeMovementDetector()
    texture_analyzer = TextureAnalyzer()
    depth_analyzer = DepthAnalyzer()
    color_analyzer = ColorAnalyzer()
    liveness_detector = LivenessDetector()
    deep_learning_detector = DeepLearningDetector()
    
    # Initialize score accumulator
    score_accumulator = ScoreAccumulator(window_size=30)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Initialize display parameters
    line_height = 30
    base_y_offset = 60
    
    print("Press 'q' to quit")
    print("Analyzing frames... Please wait for stable results...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera")
            break
            
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
                try:
                    # Analyze all features
                    eye_result = eye_detector.analyze_eye_movement(frame, face, landmarks)
                    texture_result = texture_analyzer.analyze_texture(frame, face, landmarks)
                    depth_result = depth_analyzer.analyze_depth(frame, face, landmarks)
                    color_result = color_analyzer.analyze_color(frame, face, landmarks)
                    liveness_result = liveness_detector.analyze_liveness(frame, face, landmarks)
                    deep_learning_result = deep_learning_detector.analyze_frame(frame, face)
                    
                    # Collect scores
                    current_scores = {
                        'eye_score': eye_result['movement_score'],
                        'texture_score': texture_result['texture_score'],
                        'depth_score': depth_result['combined_score'],
                        'color_score': color_result['combined_score'],
                        'liveness_score': liveness_result['combined_score'],
                        'deep_learning_score': deep_learning_result['confidence']
                    }
                    
                    # Add scores to accumulator
                    score_accumulator.add_scores(current_scores)
                    
                    # Get final decision
                    is_real, confidence = score_accumulator.get_decision()
                    
                    # Draw face rectangle
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Draw status with background
                    status = "REAL" if is_real else "SPOOFED"
                    color = (0, 255, 0) if is_real else (0, 0, 255)
                    draw_text_with_background(frame, f"Status: {status} ({confidence:.2f})", (10, 30), color=color)
                    
                    # Draw individual scores
                    scores = [
                        f"Eye Movement: {current_scores['eye_score']:.2f}",
                        f"Texture: {current_scores['texture_score']:.2f}",
                        f"Depth: {current_scores['depth_score']:.2f}",
                        f"Color: {current_scores['color_score']:.2f}",
                        f"Liveness: {current_scores['liveness_score']:.2f}",
                        f"Deep Learning: {current_scores['deep_learning_score']:.2f}"
                    ]
                    
                    for score in scores:
                        draw_text_with_background(frame, score, (10, y_offset))
                        y_offset += line_height
                    
                    # Draw landmarks
                    for point in landmarks:
                        x, y = point
                        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
                
                except Exception as e:
                    print(f"Error processing frame: {str(e)}")
        
        # Display FPS
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