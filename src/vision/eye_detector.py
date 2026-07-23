import cv2
import numpy as np


class EyeDetector:
    """Detect eyes using face cascade classifiers and dlib landmarks"""
    
    def __init__(self):
        # Load cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
    
    def detect_face(self, frame):
        """Detect face in frame
        
        Returns:
            face ROI or None if no face detected
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
        
        # Return largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return gray[y:y+h, x:x+w], (x, y, w, h)
    
    def detect_eyes(self, face_roi):
        """Detect eyes in face ROI with fallback strategies
        
        Returns:
            List of eye ROIs (attempts to return 2 eyes if possible)
        """
        # Try aggressive detection first (more permissive)
        eyes = self.eye_cascade.detectMultiScale(
            face_roi,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(10, 10)
        )
        
        # If we got at least 2 eyes, return them
        if len(eyes) >= 2:
            return eyes[:2]  # Return top 2 detections
        
        # If not enough, try more permissive settings
        if len(eyes) < 2:
            eyes = self.eye_cascade.detectMultiScale(
                face_roi,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(15, 15)
            )
        
        return eyes
    
    def extract_eye_regions(self, frame):
        """Extract both eye regions from frame
        
        Returns:
            (left_eye, right_eye, face_bbox) or (None, None, None)
        """
        face_result = self.detect_face(frame)
        if face_result is None:
            return None, None, None
        
        face_roi, face_bbox = face_result
        eyes = self.detect_eyes(face_roi)
        
        if len(eyes) == 0:
            # No eyes detected
            return None, None, None
        
        if len(eyes) == 1:
            # Only one eye detected - use it for both eyes as fallback
            eye = eyes[0]
            return eye, eye, face_bbox
        
        # Sort eyes by x coordinate (left eye first)
        eyes = sorted(eyes, key=lambda e: e[0])
        left_eye = eyes[0]
        right_eye = eyes[1]
        
        return left_eye, right_eye, face_bbox
