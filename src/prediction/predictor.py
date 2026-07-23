from tensorflow.keras.models import load_model
import numpy as np


class Predictor:
	def __init__(self, model_path="models/drowsiness_model.keras", threshold=0.5, use_ml=False):
		"""Initialize predictor
		
		Args:
			model_path: Path to Keras model
			threshold: Sensitivity threshold (higher = less sensitive)
			use_ml: Use ML model or rule-based detection
		"""
		self.threshold = threshold
		self.use_ml = use_ml
		
		if use_ml:
			try:
				self.model = load_model(model_path)
			except:
				print("⚠️ Could not load model. Using rule-based detection.")
				self.use_ml = False
	
	def predict(self, features: np.ndarray) -> int:
		"""Return 1 for drowsy, 0 for awake"""
		if self.use_ml:
			return self._predict_ml(features)
		else:
			return self._predict_rule_based(features)
	
	def predict_with_score(self, features: np.ndarray) -> tuple:
		"""Return (prediction, confidence_score)"""
		if self.use_ml:
			return self._predict_ml_with_score(features)
		else:
			return self._predict_rule_based_with_score(features)
	
	def _predict_ml(self, features: np.ndarray) -> int:
		"""ML-based prediction (original approach)"""
		try:
			preds = self.model.predict(features, verbose=0)
			score = float(preds.ravel()[0])
			return int(score > self.threshold)
		except:
			return 0
	
	def _predict_ml_with_score(self, features: np.ndarray) -> tuple:
		"""ML-based prediction with confidence"""
		try:
			preds = self.model.predict(features, verbose=0)
			score = float(preds.ravel()[0])
			prediction = int(score > self.threshold)
			return prediction, score
		except:
			return 0, 0.5
	
	def _predict_rule_based(self, features: np.ndarray) -> int:
		"""Rule-based drowsiness detection using extracted features
		
		Features: [current_ear, avg_ear, blink_rate, closure_duration]
		
		Drowsiness indicators:
		- Low EAR (eyes closed) = drowsy
		- Low average EAR = drowsy trend
		- Low blink rate = drowsy
		- High closure duration = drowsy
		"""
		try:
			features = features.reshape(-1)
			current_ear = features[0]
			avg_ear = features[1]
			blink_rate = features[2]
			closure_duration = features[3]
			
			# Score calculation (0 = awake, 1 = drowsy)
			# Higher EAR = more likely awake, so we invert it
			
			# Current EAR: Most important (50%)
			# Threshold: EAR > 0.4 = awake, EAR < 0.2 = drowsy
			if current_ear > 0.4:
				ear_score = 0.0  # Clearly awake
			elif current_ear < 0.2:
				ear_score = 1.0  # Clearly drowsy
			else:
				ear_score = (0.4 - current_ear) / 0.2  # Interpolate
			
			# Average EAR trend (20%)
			if avg_ear > 0.35:
				avg_score = 0.0
			elif avg_ear < 0.15:
				avg_score = 1.0
			else:
				avg_score = (0.35 - avg_ear) / 0.2
			
			# Blink rate (15%)
			# Normal: 0.4-0.8 (16-20 blinks/min over 5 frames)
			# Low blink rate indicates drowsiness
			if blink_rate > 0.3:
				blink_score = 0.0  # Normal blinking
			elif blink_rate < 0.1:
				blink_score = 1.0  # Not blinking
			else:
				blink_score = (0.3 - blink_rate) / 0.2
			
			# Closure duration (15%)
			# Should normally be close to 0 (eyes almost always open)
			if closure_duration < 0.1:
				closure_score = 0.0
			elif closure_duration > 0.4:
				closure_score = 1.0
			else:
				closure_score = closure_duration / 0.4
			
			# Weighted combination
			drowsiness_score = (
				ear_score * 0.5 +
				avg_score * 0.2 +
				blink_score * 0.15 +
				closure_score * 0.15
			)
			
			# Clamp to [0, 1]
			drowsiness_score = max(0, min(1, drowsiness_score))
			
			# Apply threshold
			# threshold=0.5 means: detect drowsy if score > 0.5
			# threshold=0.6 means: detect drowsy if score > 0.6 (less sensitive)
			# threshold=0.7 means: detect drowsy if score > 0.7 (even less sensitive)
			return int(drowsiness_score > self.threshold)
		except:
			return 0
	
	def _predict_rule_based_with_score(self, features: np.ndarray) -> tuple:
		"""Rule-based prediction with score"""
		try:
			features = features.reshape(-1)
			current_ear = features[0]
			avg_ear = features[1]
			blink_rate = features[2]
			closure_duration = features[3]
			
			# Score calculation
			if current_ear > 0.4:
				ear_score = 0.0
			elif current_ear < 0.2:
				ear_score = 1.0
			else:
				ear_score = (0.4 - current_ear) / 0.2
			
			if avg_ear > 0.35:
				avg_score = 0.0
			elif avg_ear < 0.15:
				avg_score = 1.0
			else:
				avg_score = (0.35 - avg_ear) / 0.2
			
			if blink_rate > 0.3:
				blink_score = 0.0
			elif blink_rate < 0.1:
				blink_score = 1.0
			else:
				blink_score = (0.3 - blink_rate) / 0.2
			
			if closure_duration < 0.1:
				closure_score = 0.0
			elif closure_duration > 0.4:
				closure_score = 1.0
			else:
				closure_score = closure_duration / 0.4
			
			drowsiness_score = (
				ear_score * 0.5 +
				avg_score * 0.2 +
				blink_score * 0.15 +
				closure_score * 0.15
			)
			
			drowsiness_score = max(0, min(1, drowsiness_score))
			prediction = int(drowsiness_score > self.threshold)
			
			return prediction, drowsiness_score
		except:
			return 0, 0.5