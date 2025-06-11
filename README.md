# Anti-Face Spoofing Detection System

A comprehensive system for detecting face spoofing attempts in images and videos, including deepfake detection.

## Features

- Real-time face liveness detection
- Multiple spoofing detection methods:
  - Eye movement analysis
  - Facial expression symmetry
  - Head movement consistency
  - Texture analysis
  - Color consistency
  - Edge detection
  - Frequency domain analysis
  - Reflection analysis
  - Micro-expression detection

## Project Structure

```
anti_face_spoofing/
├── src/
│   ├── detectors/
│   │   ├── eye_movement.py
│   │   ├── texture_analysis.py
│   │   ├── color_analysis.py
│   │   ├── frequency_analysis.py
│   │   └── micro_expressions.py
│   ├── utils/
│   │   ├── face_detection.py
│   │   ├── preprocessing.py
│   │   └── visualization.py
│   └── main.py
├── models/
│   └── saved_models/
├── data/
│   ├── raw/
│   └── processed/
├── tests/
└── requirements.txt
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main detection system:
```bash
python src/main.py
```

2. For video processing:
```bash
python src/main.py --video path/to/video.mp4
```

## Detection Methods

The system uses multiple detection methods to ensure robust spoofing detection:

1. **Eye Movement Analysis**
   - Detects natural eye blinking patterns
   - Analyzes eye movement smoothness
   - Checks for proper eye reflections

2. **Texture Analysis**
   - Analyzes skin texture patterns
   - Detects unnatural edges
   - Identifies printing artifacts

3. **Color Analysis**
   - Checks for consistent skin tones
   - Analyzes specular highlights
   - Detects color artifacts

4. **Frequency Analysis**
   - Performs FFT analysis
   - Detects unnatural frequency patterns
   - Identifies compression artifacts

5. **Micro-expression Detection**
   - Analyzes subtle facial movements
   - Detects natural skin deformation
   - Identifies static face regions

## Contributing

Feel free to submit issues and enhancement requests! 