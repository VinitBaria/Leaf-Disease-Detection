import cv2
import numpy as np
import joblib
import os
from PIL import Image  # Use PIL instead of imghdr

# Feature extraction function
def extract_features(image_path):
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found - {image_path}")
        return None

    # Validate image using PIL
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify that the image is valid
    except (IOError, SyntaxError):
        print(f"Error: Invalid or corrupted image - {image_path}")
        return None

    # Read and process the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image - {image_path}")
        return None

    img = cv2.resize(img, (64, 64))
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return hist.flatten()

# Load model and label encoder
try:
    model = joblib.load("models/rice_disease_model.pkl")
    le = joblib.load("models/label_encoder.pkl")
except FileNotFoundError:
    print("Error: Model or label encoder file not found. Please train the model first.")
    exit()

# Load test data
test_dir = "data/test"
X_test, y_test = [], []

for class_folder in os.listdir(test_dir):
    class_path = os.path.join(test_dir, class_folder)
    if os.path.isdir(class_path):
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            features = extract_features(img_path)
            if features is not None:  # Ensure only valid images are processed
                X_test.append(features)
                y_test.append(class_folder)

# Convert to numpy arrays
if not X_test:
    print("Error: No valid images found for testing.")
    exit()

X_test = np.array(X_test)
y_test_encoded = le.transform(y_test)

# Evaluate model
accuracy = model.score(X_test, y_test_encoded)
print(f"Test Accuracy: {accuracy:.2f}")

# Example single prediction
def predict_disease(img_path):
    features = extract_features(img_path)
    if features is None:
        return "Invalid image"
    features = np.array([features])
    prediction = model.predict(features)
    return le.inverse_transform(prediction)[0]

# Replace with an actual test image path
test_image = "data/test/Bacterial leaf blight/IMG_20190419_165218.jpg"
result = predict_disease(test_image)
print(f"Predicted Disease for test image: {result}")
