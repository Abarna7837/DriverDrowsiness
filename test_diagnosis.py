"""Test script to diagnose drowsiness detection"""
import cv2
import numpy as np
import sys
sys.path.insert(0, ".")

from src.vision.feature_extractor import FeatureExtractor
from src.prediction.predictor import Predictor

print("="*70)
print("DROWSINESS DETECTION DIAGNOSTIC")
print("="*70)

# Initialize
extractor = FeatureExtractor()
predictor = Predictor(threshold=0.6, use_ml=False)

# Try to capture from webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam!")
    sys.exit(1)

print("✓ Webcam opened")
print("\nCapturing 10 frames for diagnosis...\n")

for i in range(10):
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame!")
        break
    
    # Extract features
    features = extractor.extract(frame)
    
    if features is None:
        print(f"Frame {i+1}: ❌ Features = None")
        continue
    
    features_flat = features.reshape(-1)
    current_ear = features_flat[0]
    avg_ear = features_flat[1]
    blink_rate = features_flat[2]
    closure_duration = features_flat[3]
    
    # Predict
    prediction, score = predictor.predict_with_score(features)
    status = "DROWSY" if prediction == 1 else "AWAKE"
    
    print(f"Frame {i+1}:")
    print(f"  EAR (current):    {current_ear:.3f}")
    print(f"  EAR (avg):        {avg_ear:.3f}")
    print(f"  Blink rate:       {blink_rate:.3f}")
    print(f"  Closure duration: {closure_duration:.3f}")
    print(f"  Score:            {score:.3f}")
    print(f"  Prediction:       {status}")
    print()

cap.release()

print("="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
print("\nExpected values when AWAKE:")
print("  EAR (current): 0.4-1.0 (high)")
print("  Score:        < 0.4 (low)")
print("  Prediction:   AWAKE")
print("\nExpected values when DROWSY:")
print("  EAR (current): 0.0-0.3 (low)")
print("  Score:        > 0.6 (high)")
print("  Prediction:   DROWSY")
