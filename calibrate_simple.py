"""Simple EAR calibration - collects frame data and shows statistics"""
import cv2
import numpy as np
import sys
sys.path.insert(0, ".")

from src.vision.eye_detector import EyeDetector
from src.vision.feature_extractor import FeatureExtractor

print("EAR Calibration - Collecting 10 frames")
print("="*60)
print("Open your eyes normally during this test")
print()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera failed")
    sys.exit(1)

detector = EyeDetector()
extractor = FeatureExtractor()

def get_ear_components(eye_roi):
    """Extract the components of EAR calculation"""
    brightness = np.mean(eye_roi) / 255.0
    variance = np.var(eye_roi) / 10000.0
    
    hist = cv2.calcHist([eye_roi], [0], None, [256], [0, 256])
    hist_normalized = hist / hist.sum()
    entropy = -np.sum(hist_normalized * np.log(hist_normalized + 1e-10))
    entropy_score = entropy / 8.0
    
    combined = brightness * 0.4 + variance * 0.3 + entropy_score * 0.3
    
    return brightness, variance, entropy_score, combined

all_ears = []
frame_num = 0

print("Collecting frames...")
while frame_num < 10:
    ret, frame = cap.read()
    if not ret:
        continue
    
    left_eye, right_eye, face_bbox = detector.extract_eye_regions(frame)
    
    if left_eye is None:
        print(f"Frame {frame_num+1}: ⚠️  No face detected, skipping")
        continue
    
    # Extract eyes from frame
    x, y, w, h = face_bbox
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w]
    
    ex1, ey1, ew1, eh1 = left_eye
    ex2, ey2, ew2, eh2 = right_eye
    
    left_roi = gray[ey1:ey1+eh1, ex1:ex1+ew1]
    right_roi = gray[ey2:ey2+eh2, ex2:ex2+ew2]
    
    # Get components for both eyes
    left_bright, left_var, left_ent, left_ear = get_ear_components(left_roi)
    right_bright, right_var, right_ent, right_ear = get_ear_components(right_roi)
    
    avg_ear = (left_ear + right_ear) / 2
    
    all_ears.append({
        'brightness': (left_bright + right_bright) / 2,
        'variance': (left_var + right_var) / 2,
        'entropy': (left_ent + right_ent) / 2,
        'combined': avg_ear
    })
    
    print(f"Frame {frame_num+1}: Brightness={left_bright:.3f} Variance={left_var:.3f} Entropy={left_ent:.3f} → EAR={avg_ear:.4f}")
    frame_num += 1

cap.release()

print("\n" + "="*60)
print("ANALYSIS (10 frames):")
print("="*60)

for key in ['brightness', 'variance', 'entropy', 'combined']:
    vals = [e[key] for e in all_ears]
    avg = np.mean(vals)
    std = np.std(vals)
    print(f"{key:12}: avg={avg:.4f}  std={std:.4f}  range=[{min(vals):.4f}, {max(vals):.4f}]")

print("\n" + "="*60)
print("CURRENT THRESHOLDS IN CODE:")
print("  - Drowsy threshold (< this): 0.2")
print("  - Awake threshold (> this):  0.4")
print("  - Interpolate between: 0.2-0.4")
print("="*60)

avg_combined = np.mean([e['combined'] for e in all_ears])
print(f"\nYour average EAR (eyes open): {avg_combined:.4f}")

if avg_combined < 0.4:
    print(f"⚠️  Your EAR is below the 'awake' threshold (0.4)")
    print(f"   You will be classified as DROWSY!")
    print(f"   Suggested new 'awake' threshold: {avg_combined + 0.1:.4f}")
elif avg_combined < 0.2:
    print(f"❌ Your EAR is way too low (< 0.2 = drowsy threshold)")
    print(f"   Suggested new 'drowsy' threshold: {avg_combined - 0.1:.4f}")
else:
    print(f"✓ You will be classified as AWAKE (good!)")

print(f"\nNeed to test with eyes CLOSED to get full calibration")
