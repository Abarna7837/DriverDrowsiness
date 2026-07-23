"""Diagnostic: Check if face/eye detection is working"""
import cv2
import sys
sys.path.insert(0, ".")

from src.vision.eye_detector import EyeDetector

print("Testing face and eye detection...")
print("="*60)

# Try to open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ CAMERA FAILED!")
    sys.exit(1)

detector = EyeDetector()

print("\nTesting 5 frames for face/eye detection:\n")

for i in range(5):
    ret, frame = cap.read()
    if not ret:
        print(f"Frame {i+1}: ❌ Failed to read")
        continue
    
    # Try to detect face
    result = detector.detect_face(frame)
    if result is None:
        print(f"Frame {i+1}: ❌ NO FACE DETECTED (using fallback)")
    else:
        face_roi, face_bbox = result
        x, y, w, h = face_bbox
        print(f"Frame {i+1}: ✓ Face detected at ({x}, {y}) size {w}x{h}")
        
        # Try to detect eyes in face
        eyes = detector.detect_eyes(face_roi)
        print(f"           Eyes detected: {len(eyes)}")
        if len(eyes) > 0:
            for j, (ex, ey, ew, eh) in enumerate(eyes[:2]):
                print(f"             Eye {j+1}: ({ex}, {ey}) size {ew}x{eh}")

cap.release()
print("\n" + "="*60)
print("If you see 'NO FACE DETECTED' for all frames:")
print("  → Camera/lighting issue OR faces at unusual angles")
print("  → Current fallback method should still work (analyzes frame region)")
