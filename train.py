import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
import joblib

# Define training directory
train_dir = "data/train"

# Check if training directory exists
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Training directory not found: {os.path.abspath(train_dir)}")

# Feature extraction function
def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Skipping invalid or unreadable image: {image_path}")
        return None
    img = cv2.resize(img, (64, 64))
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return hist.flatten()

# Load training data
X_train, y_train = [], []

for class_folder in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_folder)
    
    if os.path.isdir(class_path):
        print(f"Processing class: {class_folder}")
        
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            
            features = extract_features(img_path)
            if features is not None:  # Skip invalid images
                X_train.append(features)
                y_train.append(class_folder)

# Convert lists to numpy arrays
if len(X_train) == 0 or len(y_train) == 0:
    raise ValueError("No valid training data found. Ensure 'data/train' contains image folders.")

X_train = np.array(X_train)
y_train = np.array(y_train)

# Encode labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train_encoded)

# Ensure the 'models' directory exists
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

# Save model and label encoder
try:
    model_path = os.path.join(models_dir, "rice_disease_model.pkl")
    encoder_path = os.path.join(models_dir, "label_encoder.pkl")
    
    joblib.dump(model, model_path)
    joblib.dump(le, encoder_path)
    
    print(f"Training complete. Model saved at {model_path}")
    print(f"Label encoder saved at {encoder_path}")

except PermissionError as e:
    print(f"Permission denied: {e}. Try running as administrator or moving the project to a writable directory.")
except Exception as e:
    print(f"An error occurred while saving: {e}")
