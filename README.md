# Anti Face Spoofing System

A robust face anti-spoofing system that combines multiple detection methods to prevent presentation attacks. The system uses a combination of traditional computer vision techniques and deep learning to achieve high accuracy in detecting fake faces.

## Features

- **Multi-Modal Detection**:
  - Eye Movement Analysis
  - Texture Analysis
  - Depth Analysis
  - Color Analysis
  - Liveness Detection
  - Deep Learning-based Detection

- **Real-time Processing**:
  - Live webcam feed analysis
  - Real-time visualization
  - Frame-by-frame scoring

- **Robust Decision Making**:
  - Weighted scoring system
  - Temporal averaging
  - Confidence scoring
  - Multiple frame analysis

## Requirements

- Python 3.8+
- OpenCV (with contrib modules)
- PyTorch
- Transformers
- NumPy
- dlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/anti-face-spoofing.git
cd anti-face-spoofing
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download and set up required models:

   a. Download the dlib shape predictor:
   ```bash
   # For Windows
   curl -L http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -o shape_predictor_68_face_landmarks.dat.bz2
   bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

   # For Linux/Mac
   wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
   ```

   b. Download the OpenCV face detection model:
   ```bash
   # For Windows
   curl -L https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt -o models/deploy.prototxt
   curl -L https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel -o models/res10_300x300_ssd_iter_140000.caffemodel

   # For Linux/Mac
   wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt -O models/deploy.prototxt
   wget https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel -O models/res10_300x300_ssd_iter_140000.caffemodel
   ```

   c. Download the LBF model:
   ```bash
   # For Windows
   curl -L https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml -o models/lbfmodel.yaml

   # For Linux/Mac
   wget https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml -O models/lbfmodel.yaml
   ```

   d. Download the ResNet-50 model (this will be downloaded automatically on first run):
   ```bash
   python -c "from transformers import AutoModelForImageClassification; AutoModelForImageClassification.from_pretrained('microsoft/resnet-50', cache_dir='models/saved_models')"
   ```

4. Verify the model directory structure:
```
models/
├── deploy.prototxt
├── res10_300x300_ssd_iter_140000.caffemodel
├── lbfmodel.yaml
└── saved_models/
    └── microsoft/
        └── resnet-50/
            └── (model files)
```

## Project Structure

```
anti-face-spoofing/
├── src/
│   ├── detectors/
│   │   ├── eye_movement.py
│   │   ├── texture_analysis.py
│   │   ├── depth_analysis.py
│   │   ├── color_analysis.py
│   │   ├── liveness_detection.py
│   │   └── deep_learning.py
│   ├── utils/
│   │   ├── face_detection.py
│   │   ├── preprocessing.py
│   │   └── visualization.py
│   ├── main.py
│   └── __init__.py
├── models/
|   ├── shape_predictor_68_face_landmarks.dat
│   ├── deploy.prototxt
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   ├── lbfmodel.yaml
│   └── saved_models/
│       └── microsoft/
│           └── resnet-50/
│               └── (model files)
├── data/
├── tests/
├── test_detection.py
├── download_models.py
└── requirements.txt
```

## Usage

1. Run the test script:
```bash
python test_detection.py
```

2. The system will:
   - Open your webcam
   - Start analyzing frames
   - Display real-time results
   - Show confidence scores
   - Indicate REAL/SPOOFED status

3. Controls:
   - Press 'q' to quit
   - Wait for stable results (about 1 second)

## How It Works

The system uses a multi-stage approach to detect spoofing attempts:

1. **Face Detection**: Locates faces in the frame using OpenCV's DNN face detector
2. **Landmark Detection**: Identifies facial landmarks using dlib
3. **Feature Analysis**: Multiple detectors analyze different aspects:
   - Eye movement patterns
   - Surface texture
   - Depth information
   - Color distribution
   - Liveness indicators
   - Deep learning features

4. **Decision Making**:
   - Scores from each detector are weighted
   - Temporal averaging over 30 frames
   - Confidence-based final decision

## Performance

- Processing Speed: 20-30 FPS on standard hardware
- Accuracy: High accuracy with multiple detection methods
- False Positive Rate: Reduced through temporal averaging
- False Negative Rate: Minimized by combining multiple features

## Troubleshooting

1. If models are not found:
   - Verify the models directory structure
   - Check if all model files are downloaded
   - Ensure correct file permissions

2. If face detection fails:
   - Check if OpenCV is properly installed with contrib modules
   - Verify the face detection model files are in the correct location
   - Ensure good lighting conditions

3. If deep learning model fails:
   - Check internet connection for model download
   - Verify PyTorch installation
   - Check GPU availability if using CUDA

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Uses ResNet-50 from Microsoft for deep learning detection
- Implements OpenCV's DNN face detector
- Uses dlib for facial landmark detection
- OpenCV for computer vision operations 