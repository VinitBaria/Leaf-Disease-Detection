import cv2
import numpy as np
import joblib
import os
from PIL import Image

# Feature extraction function
def extract_features(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image not found - {image_path}")
        return None
    try:
        with Image.open(image_path) as img:
            img.verify()
    except (IOError, SyntaxError):
        print(f"Error: Invalid or corrupted image - {image_path}")
        return None
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image - {image_path}")
        return None
    img = cv2.resize(img, (64, 64))
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return hist.flatten()

# Print current working directory for debugging
print(f"Current working directory: {os.getcwd()}")

# Load model and label encoder with raw strings
model_path = r"v:\CODING\flsk_test\models\rice_disease_model.pkl"
encoder_path = r"v:\CODING\flsk_test\models\label_encoder.pkl"
print(f"Looking for model at: {os.path.abspath(model_path)}")
print(f"Looking for label encoder at: {os.path.abspath(encoder_path)}")

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    exit(1)
if not os.path.exists(encoder_path):
    print(f"Error: Label encoder file not found at {encoder_path}")
    exit(1)

try:
    model = joblib.load(model_path)
    le = joblib.load(encoder_path)
    print("Model and label encoder loaded successfully.")
except Exception as e:
    print(f"Error loading model or encoder: {e}")
    exit(1)

# Load test data
test_dir = r"v:\CODING\flsk_test\data\test"
print(f"Looking for test data at: {os.path.abspath(test_dir)}")

if not os.path.exists(test_dir):
    print(f"Error: Test directory not found - {test_dir}")
    exit(1)

X_test, y_test = [], []
for class_folder in os.listdir(test_dir):
    class_path = os.path.join(test_dir, class_folder)
    if os.path.isdir(class_path):
        print(f"Processing test class: {class_folder}")
        for img_file in os.listdir(class_path):  # Fixed typo here
            img_path = os.path.join(class_path, img_file)
            print(f"Processing image: {img_path}")
            features = extract_features(img_path)
            if features is not None:
                X_test.append(features)
                y_test.append(class_folder)

# Check if test data is valid
if not X_test or not y_test:
    print("Error: No valid images found for testing in 'data/test'.")
    exit(1)

# Convert to numpy arrays
X_test = np.array(X_test)
y_test_encoded = le.transform(y_test)

# Calculate accuracy
accuracy = model.score(X_test, y_test_encoded)
print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")
print(f"Number of test samples: {len(y_test)}")