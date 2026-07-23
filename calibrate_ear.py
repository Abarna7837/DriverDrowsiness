"""Calibration: measure EAR components with different eye states"""
import cv2
import numpy as np
import sys
sys.path.insert(0, ".")

from src.vision.eye_detector import EyeDetector
from src.vision.feature_extractor import FeatureExtractor

print("EAR Calibration Tool")
print("="*60)
print("Keep your camera on and follow instructions")
print()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera failed")
    sys.exit(1)

detector = EyeDetector()
extractor = FeatureExtractor()

def analyze_eye_components(eye_roi):
    """Show brightness, variance, entropy for an eye region"""
    mean_brightness = np.mean(eye_roi) / 255.0
    variance_score = np.var(eye_roi) / 10000.0
    
    hist = cv2.calcHist([eye_roi], [0], None, [256], [0, 256])
    hist_normalized = hist / hist.sum()
    entropy = -np.sum(hist_normalized * np.log(hist_normalized + 1e-10))
    entropy_score = entropy / 8.0
    
    combined = mean_brightness * 0.4 + variance_score * 0.3 + entropy_score * 0.3
    
    return {
        'brightness': mean_brightness,
        'variance': variance_score,
        'entropy': entropy_score,
        'combined': combined
    }

states = ["EYES OPEN (5 frames)", "EYES CLOSED (5 frames)"]
frame_count = 0
state_idx = 0

print(f"Step 1/2: {states[0]}")
print("Press SPACE to start, ESC to end")
print()

collecting = False
open_values = []
closed_values = []

while state_idx < 2:
    ret, frame = cap.read()
    if not ret:
        continue
    
    left_eye, right_eye, face_bbox = detector.extract_eye_regions(frame)
    
    if left_eye is None:
        cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        x, y, w, h = face_bbox
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract eyes from frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w]
        ex1, ey1, ew1, eh1 = left_eye
        left_roi = gray[ey1:ey1+eh1, ex1:ex1+ew1]
        
        if collecting and len(left_roi) > 0:
            vals = analyze_eye_components(left_roi)
            if state_idx == 0:
                open_values.append(vals)
            else:
                closed_values.append(vals)
            
            frame_count += 1
            if frame_count >= 5:
                frame_count = 0
                state_idx += 1
                collecting = False
                if state_idx < 2:
                    print(f"\nStep {state_idx+1}/2: {states[state_idx]}")
                    print("Press SPACE to start, ESC to end")
    
    # Show UI
    if collecting:
        status = f"Recording {frame_count}/5"
        color = (0, 255, 0)
    else:
        status = "Press SPACE to start"
        color = (0, 165, 255)
    
    cv2.putText(frame, f"State {state_idx+1}/2: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow('Calibration', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        collecting = True
        frame_count = 0
    elif key == 27:  # ESC
        break

cv2.destroyAllWindows()
cap.release()

if len(open_values) > 0 and len(closed_values) > 0:
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    
    print("\nEYES OPEN:")
    avg_open = {}
    for key in ['brightness', 'variance', 'entropy', 'combined']:
        vals = [v[key] for v in open_values]
        avg = np.mean(vals)
        avg_open[key] = avg
        print(f"  {key:12} avg={avg:.4f}  (range: {min(vals):.4f}-{max(vals):.4f})")
    
    print("\nEYES CLOSED:")
    avg_closed = {}
    for key in ['brightness', 'variance', 'entropy', 'combined']:
        vals = [v[key] for v in closed_values]
        avg = np.mean(vals)
        avg_closed[key] = avg
        print(f"  {key:12} avg={avg:.4f}  (range: {min(vals):.4f}-{max(vals):.4f})")
    
    print("\nRECOMMENDED THRESHOLDS:")
    # Calculate midpoint between open and closed
    threshold_brightness = (avg_open['brightness'] + avg_closed['brightness']) / 2
    threshold_combined = (avg_open['combined'] + avg_closed['combined']) / 2
    
    print(f"  Brightness threshold: {threshold_brightness:.4f}")
    print(f"  Combined EAR threshold: {threshold_combined:.4f}")
    print(f"  Current drowsy threshold in code: 0.2")
    print(f"  Current awake threshold in code: 0.4")
