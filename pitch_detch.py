!pip uninstall -y mediapipe
!pip install mediapipe
!pip install pybaseball opencv-python

import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import site
import sys
from importlib import reload
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pybaseball import statcast


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360 - angle if angle > 180.0 else angle

def extract_biomechanics(video_path):
    cap = cv2.VideoCapture(video_path)
    features = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            elbow = [lm[mp_pose.PoseLandmark.LEFT_ELBOW].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW].y]
            wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST].x, lm[mp_pose.PoseLandmark.LEFT_WRIST].y]
            features.append({'arm_angle': calculate_angle(shoulder, elbow, wrist)})
    cap.release()
    return pd.DataFrame(features)

# Load a sample of pitching data
# Identifies pitch types and results
data = statcast(start_dt='2023-04-01', end_dt='2023-04-01')

# Filtering for Fastball vs Offspeed for a general analysis
data['pitch_category'] = data['pitch_type'].map({
    'FF': 'Fastball', 'SI': 'Fastball', 'FC': 'Fastball',
    'SL': 'Offspeed', 'CU': 'Offspeed', 'CH': 'Offspeed', 'KC': 'Offspeed', 'ST': 'Offspeed', 'SV': 'Offspeed'
})

display(data[['pitch_type', 'pitch_category', 'release_speed', 'release_pos_x', 'release_pos_z']].head())

try:
    import mediapipe as mp
    from mediapipe.solutions import pose as mp_pose
    print("Success: MediaPipe solutions found.")
except ImportError:
    print("CRITICAL: MediaPipe solutions not found. Colab: Please go to 'Runtime' -> 'Restart session' and run the install cell again.")


if 'mp_pose' in locals():
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

print("Pipeline setup complete.")

# Load Data
if 'data' not in locals():
    data = statcast(start_dt='2023-04-01', end_dt='2023-04-01')
    data['pitch_category'] = data['pitch_type'].map({
        'FF': 'Fastball', 'SI': 'Fastball', 'FC': 'Fastball',
        'SL': 'Offspeed', 'CU': 'Offspeed', 'CH': 'Offspeed', 'KC': 'Offspeed', 'ST': 'Offspeed', 'SV': 'Offspeed'
    })

# For demonstration: Synthetic biomechanical features
np.random.seed(42)
n_samples = len(data)
data['feat_arm_angle'] = np.random.normal(90, 15, n_samples)
data['feat_release_extension'] = np.random.normal(6, 0.5, n_samples)
data['feat_torso_rotation'] = np.random.normal(45, 10, n_samples)

# Drop NaN rows for pitch_category
model_df = data.dropna(subset=['pitch_category'])

X = model_df[['feat_arm_angle', 'feat_release_extension', 'feat_torso_rotation']]
y = model_df['pitch_category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)
model_accuracy = accuracy_score(y_test, y_pred)

# Random Baseline
random_preds = np.random.choice(y_train.unique(), size=len(y_test))
baseline_accuracy = accuracy_score(y_test, random_preds)

print(f"Model Accuracy: {model_accuracy:.2%}")
print(f"Random Baseline Accuracy: {baseline_accuracy:.2%}")

# Feature Importance
importances = clf.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 5))
plt.barh(feature_names, importances)
plt.title('Determining the Tells: Feature Importance')
plt.show()
