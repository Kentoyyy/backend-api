import os
import pandas as pd
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

# ======= Setup paths =======
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/spelling_audio_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "dyslexia_spelling_audio_model.joblib")
os.makedirs(MODEL_DIR, exist_ok=True)

# ======= Feature Extraction =======
def extract_features(audio_path, max_len=100):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
        else:
            mfcc = mfcc[:, :max_len]
        return mfcc.flatten()
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# ======= Load Data =======
df = pd.read_csv(DATA_PATH)
X = []
y = []

for _, row in df.iterrows():
    audio_file = os.path.abspath(os.path.join(BASE_DIR, "..", row["audio_file"]))
    label = 1 if "correct" in row["audio_file"].lower() else 0
    features = extract_features(audio_file)
    if features is not None:
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

# ======= Train/Test Split =======
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ======= Train Model =======
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# ======= Evaluation =======
y_pred = clf.predict(X_test)
print("✅ Classification Report:")
print(classification_report(y_test, y_pred))
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# ======= Save Model =======
dump(clf, MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
