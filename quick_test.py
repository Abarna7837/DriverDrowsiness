"""Quick diagnostic to see what's actually happening"""
import cv2
import sys
sys.path.insert(0, ".")

from src.vision.feature_extractor import FeatureExtractor
from src.prediction.predictor import Predictor

print("Starting quick diagnostic...")
print("="*60)

# Try to open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ CAMERA FAILED TO OPEN!")
    sys.exit(1)

print("✓ Camera opened")

# Test extraction and prediction
extractor = FeatureExtractor()
predictor = Predictor(threshold=0.5, use_ml=False)

print("\nTesting 5 frames:")
print("="*60)

for i in range(5):
    ret, frame = cap.read()
    if not ret:
        print(f"Frame {i}: ❌ Failed to read")
        continue
    
    features = extractor.extract(frame, verbose=False)
    if features is None:
        print(f"Frame {i}: ❌ Features is None")
        continue
    
    features_flat = features.reshape(-1)
    current_ear = features_flat[0]
    avg_ear = features_flat[1]
    blink_rate = features_flat[2]
    closure_duration = features_flat[3]
    
    prediction, score = predictor.predict_with_score(features)
    status = "DROWSY" if prediction == 1 else "AWAKE"
    
    print(f"\nFrame {i+1}:")
    print(f"  EAR:          {current_ear:.4f}")
    print(f"  Avg EAR:      {avg_ear:.4f}")
    print(f"  Blink Rate:   {blink_rate:.4f}")
    print(f"  Closure Dur:  {closure_duration:.4f}")
    print(f"  Drowsiness Score: {score:.4f}")
    print(f"  Status: {status} {'✓' if status == 'AWAKE' else '✗'}")

cap.release()
print("\n" + "="*60)
print("Analysis complete")
