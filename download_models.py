import os
import urllib.request
import cv2

def download_file(url, filename):
    """Download a file from URL to filename."""
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename)
    print(f"Downloaded {filename}")

def main():
    # Create models directory if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")
    
    # Download face detection model files
    deploy_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    
    download_file(deploy_url, "models/deploy.prototxt")
    download_file(model_url, "models/res10_300x300_ssd_iter_140000.caffemodel")
    
    # Download face landmark model
    landmark_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"
    download_file(landmark_url, "models/lbfmodel.yaml")
    
    print("All model files downloaded successfully!")

if __name__ == "__main__":
    main() 