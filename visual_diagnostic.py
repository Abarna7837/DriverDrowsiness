"""Visual diagnostic - draw eye regions on screen to see what's being detected"""
import cv2
import sys
sys.path.insert(0, ".")

from src.vision.eye_detector import EyeDetector

cap = cv2.VideoCapture(0)
detector = EyeDetector()

print("Visual Eye Detection Diagnostic")
print("Close this window to exit")
print()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect face and eyes
    result = detector.detect_face(frame)
    if result is None:
        cv2.putText(frame, "NO FACE DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        face_roi, (fx, fy, fw, fh) = result
        
        # Draw face box
        cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)
        cv2.putText(frame, "FACE", (fx, fy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Detect eyes
        eyes = detector.detect_eyes(face_roi)
        
        # Draw eyes
        for i, (ex, ey, ew, eh) in enumerate(eyes[:2]):
            # Convert from face ROI coordinates to frame coordinates
            frame_x = fx + ex
            frame_y = fy + ey
            
            cv2.rectangle(frame, (frame_x, frame_y), (frame_x+ew, frame_y+eh), (0, 255, 0), 2)
            cv2.putText(frame, f"EYE{i+1}", (frame_x, frame_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if len(eyes) < 1:
            cv2.putText(frame, "NO EYES DETECTED", (fx+10, fy+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"Eyes: {len(eyes)}", (fx+10, fy+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Eye Detection Debug', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
