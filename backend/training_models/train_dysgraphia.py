import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))  # ../data
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Image settings
IMG_SIZE = (64, 64)
X = []
y = []

# Filter only folders (classes) inside data directory
class_names = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
label_map = {class_name: idx for idx, class_name in enumerate(class_names)}

for class_name in class_names:
    class_dir = os.path.join(DATA_DIR, class_name)
    for file in os.listdir(class_dir):
        file_path = os.path.join(class_dir, file)
        try:
            img = cv2.imread(file_path)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            img = img / 255.0
            X.append(img.flatten())
            y.append(label_map[class_name])
        except Exception as e:
            print(f"Skipping {file_path} due to error: {e}")

X = np.array(X)
y = np.array(y)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=class_names))

# Save model
model_path = os.path.join(MODEL_DIR, "dysgraphia_handwritten_model.joblib")
dump(model, model_path)
print(f"Model saved to {model_path}")
