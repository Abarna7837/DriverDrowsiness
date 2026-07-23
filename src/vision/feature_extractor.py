import cv2
import numpy as np
from .eye_detector import EyeDetector


class FeatureExtractor:
	"""Extract drowsiness features from video frames
	
	Features extracted:
	- Eye Aspect Ratio (EAR): Detects eye closure
	- Blink rate: Tracks blinking frequency
	- Eye closure duration: Time eyes are closed
	- Pupil position: Eye gaze direction
	"""
	
	def __init__(self):
		self.eye_detector = EyeDetector()
		self.frame_history = []
		self.ear_threshold = 0.2  # Below this = eyes closed
		self.blink_threshold = 5  # Consecutive frames = blink
		self.max_history = 30  # Keep last 30 frames
	
	def calculate_eye_aspect_ratio(self, eye_roi):
		"""Calculate Eye Aspect Ratio (EAR)
		
		Open eyes: Dark iris/pupil visible → Low brightness in eye region
		Closed eyes: Only skin tone eyelid → Higher brightness
		
		Also uses histogram to detect presence of dark iris
		
		Returns value in [0, 1] where higher = more likely open
		"""
		if eye_roi is None or eye_roi.size == 0:
			return 0.5
		
		try:
			# Key insight: Open eyes have DARK iris/pupil
			# Closed eyes have only skin tone (no dark areas)
			
			# Metric 1: Darkness in the eye region
			# Lower mean brightness = more likely open (dark iris present)
			# Higher mean brightness = more likely closed (only skin)
			mean_brightness = np.mean(eye_roi) / 255.0
			
			# Invert: darker = higher score
			darkness_score = 1.0 - mean_brightness
			
			# Metric 2: Histogram concentration (does it have dark pixels?)
			# Open eyes: bimodal histogram (dark iris + light sclera)
			# Closed eyes: uniform histogram (skin tone)
			hist = cv2.calcHist([eye_roi], [0], None, [256], [0, 256])
			hist_normalized = hist / hist.sum()
			
			# Check how much of histogram is in dark range (0-100)
			dark_pixels = np.sum(hist_normalized[:100])  # Pixels with value < 100
			
			# Combined: Darkness (60%) + Dark pixel presence (40%)
			combined_ear = darkness_score * 0.6 + dark_pixels * 0.4
			
			return min(combined_ear, 1.0)
		except:
			return 0.5
	
	def extract(self, frame, verbose=False):
		"""Extract drowsiness feature tensor from frame
		
		Returns feature vector (1, 4, 1) containing:
		- Current EAR
		- Average EAR (trend)
		- Blink rate (last 30 frames)
		- Eye closure duration
		"""
		try:
			# Try primary method: face and eye detection
			left_eye, right_eye, face_bbox = self.eye_detector.extract_eye_regions(frame)
			
			if left_eye is None or right_eye is None:
				# Fallback: use simple frame analysis
				if verbose:
					print("  [Feature] ⚠️ No face/eyes detected - using frame analysis")
				
				current_ear = self._analyze_frame_simple(frame)
			else:
				# Primary method: extract eye ROIs and calculate EAR
				if verbose:
					print(f"  [Feature] ✓ Face detected, eyes found")
				
				# Extract eye ROIs from frame
				x, y, w, h = face_bbox
				face_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w]
				
				x1, y1, w1, h1 = left_eye
				x2, y2, w2, h2 = right_eye
				
				left_roi = face_gray[y1:y1+h1, x1:x1+w1]
				right_roi = face_gray[y2:y2+h2, x2:x2+w2]
				
				# Calculate EAR for each eye
				left_ear = self.calculate_eye_aspect_ratio(left_roi)
				right_ear = self.calculate_eye_aspect_ratio(right_roi)
				current_ear = (left_ear + right_ear) / 2
			
			if verbose:
				print(f"  [Feature] Current EAR: {current_ear:.3f}")
			
			# Track history
			self.frame_history.append(current_ear)
			if len(self.frame_history) > self.max_history:
				self.frame_history.pop(0)
			
			# Calculate features
			avg_ear = np.mean(self.frame_history) if self.frame_history else 0.5
			
			# Blink detection (rapid EAR drop)
			blink_rate = 0.0
			if len(self.frame_history) > 5:
				recent = self.frame_history[-5:]
				blink_count = sum(1 for e in recent if e < self.ear_threshold)
				blink_rate = blink_count / 5.0
			
			# Eye closure duration
			closure_duration = sum(1 for e in self.frame_history if e < self.ear_threshold) / len(self.frame_history) if self.frame_history else 0.0
			
			if verbose:
				print(f"  [Feature] Avg EAR: {avg_ear:.3f}, Blink: {blink_rate:.3f}, Closure: {closure_duration:.3f}")
			
			# Create feature vector: [current_ear, avg_ear, blink_rate, closure_duration]
			features = np.array([
				[current_ear],
				[avg_ear],
				[blink_rate],
				[closure_duration]
			], dtype='float32')
			
			return features.reshape(1, 4, 1)
		
		except Exception as e:
			# Return neutral features on error
			if verbose:
				print(f"  [Feature] ❌ Error: {str(e)}")
			return np.array([[0.5], [0.5], [0.0], [0.0]], dtype='float32').reshape(1, 4, 1)
	
	def _analyze_frame_simple(self, frame):
		"""Simple frame-level analysis as fallback
		
		Analyzes center region of frame (where eyes typically are)
		to estimate if they're open or closed
		"""
		try:
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			h, w = gray.shape
			
			# Focus on upper-middle region (eye area)
			y_start = h // 4
			y_end = h // 2
			x_start = w // 4
			x_end = 3 * w // 4
			
			eye_region = gray[y_start:y_end, x_start:x_end]
			
			# Use same EAR calculation on eye region
			return self.calculate_eye_aspect_ratio(eye_region)
		except:
			return 0.5
