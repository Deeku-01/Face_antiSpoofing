# Anti-Face Spoofing Detection System

A robust anti-spoofing system that uses multiple detection techniques to identify fake faces in real-time video streams.

## Features

- **Multi-Modal Detection**: Combines multiple detection methods for higher accuracy
- **Real-time Processing**: Optimized for live video streams
- **Multiple Analysis Techniques**:
  - Eye Movement Analysis
  - Texture Analysis
  - Depth Analysis
  - Color Analysis
  - Liveness Detection

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Webcam
- dlib shape predictor file

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Deeku-01/anti-face-spoofing.git
cd anti-face-spoofing
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the shape predictor file:
   - Download the 68-point facial landmark predictor file from [dlib's official website](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
   - Extract the downloaded file
   - Place the `shape_predictor_68_face_landmarks.dat` file in the project root directory

   Or use the following commands:
   ```bash
   # For Windows
   curl -L http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -o shape_predictor_68_face_landmarks.dat.bz2
   bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

   # For Linux/Mac
   wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
   ```

## Usage

Run the test script to start the anti-spoofing detection:
```bash
python test_detection.py
```

### Controls
- Press 'q' to quit the application

### Display Information
The system shows:
- Real-time status (REAL/SPOOFED)
- Individual scores for each detection method
- FPS counter
- Face landmarks
- Face bounding box

## Project Structure

```
anti-face-spoofing/
├── src/
│   ├── detectors/
│   │   ├── eye_movement.py
│   │   ├── texture_analysis.py
│   │   ├── depth_analysis.py
│   │   ├── color_analysis.py
│   │   └── liveness_detection.py
│   └── utils/
│       ├── face_detection.py
│       └── visualization.py
├── test_detection.py
├── requirements.txt
├── shape_predictor_68_face_landmarks.dat
└── README.md
```

## Detection Methods

### 1. Eye Movement Analysis
- Detects natural eye movements and blinks
- Analyzes eye aspect ratio
- Tracks eye movement smoothness

### 2. Texture Analysis
- Analyzes facial texture patterns
- Detects unnatural edges
- Examines frequency domain characteristics

### 3. Depth Analysis
- Analyzes facial depth characteristics
- Detects gradient depth patterns
- Examines shadow patterns
- Analyzes contour variations

### 4. Color Analysis
- Analyzes skin tone patterns
- Detects color variance
- Examines reflections and highlights
- Tracks temporal consistency

### 5. Liveness Detection
- Analyzes facial movements
- Detects natural expressions
- Examines temporal patterns
- Tracks face dynamics

## Performance Optimization

- Reduced resolution (640x480)
- Frame skipping for better performance
- Optimized processing pipeline
- Smooth score updates
- Efficient memory usage

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 