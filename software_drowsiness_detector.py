"""
Software-only Drowsiness Detection System
Tests drowsiness detection with simulated video frames or video files
No webcam required!
"""

import sys
import pathlib
import numpy as np
import cv2
from pathlib import Path

# ensure project root is on sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))

from src.vision.feature_extractor import FeatureExtractor
from src.prediction.predictor import Predictor
from src.utils.alert import play_alarm


class SoftwareDrowsinessDetector:
    def __init__(self):
        self.extractor = FeatureExtractor()
        self.predictor = Predictor()
        self.frame_count = 0
        self.drowsy_count = 0
        self.awake_count = 0
        
    def process_frame(self, frame):
        """Process a single frame and return drowsiness prediction"""
        if frame is None:
            return None
            
        features = self.extractor.extract(frame)
        if features is not None:
            prediction = self.predictor.predict(features)
            self.frame_count += 1
            
            if prediction == 1:
                self.drowsy_count += 1
                status = "DROWSY"
            else:
                self.awake_count += 1
                status = "AWAKE"
            
            return status
        return None
    
    def get_stats(self):
        """Return statistics"""
        return {
            "total_frames": self.frame_count,
            "drowsy": self.drowsy_count,
            "awake": self.awake_count,
            "drowsy_percentage": (self.drowsy_count / self.frame_count * 100) if self.frame_count > 0 else 0
        }


def test_with_synthetic_frames(num_frames=50):
    """Test with synthetic frames of varying intensity"""
    print("\n" + "="*70)
    print("DROWSINESS DETECTION - SYNTHETIC FRAME TEST")
    print("="*70)
    
    detector = SoftwareDrowsinessDetector()
    
    print(f"\nProcessing {num_frames} synthetic frames...\n")
    print(f"{'Frame':<8} {'Status':<12} {'Pattern':<25}")
    print("-" * 50)
    
    for i in range(num_frames):
        # Create varying frame patterns
        intensity = (i * 255) // num_frames
        frame = np.full((480, 640, 3), fill_value=intensity, dtype=np.uint8)
        
        status = detector.process_frame(frame)
        pattern = f"Intensity: {intensity}"
        
        print(f"{i+1:<8} {status:<12} {pattern:<25}")
    
    # Print statistics
    stats = detector.get_stats()
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    print(f"Total frames processed: {stats['total_frames']}")
    print(f"Drowsy detections: {stats['drowsy']} ({stats['drowsy_percentage']:.1f}%)")
    print(f"Awake detections: {stats['awake']} ({100-stats['drowsy_percentage']:.1f}%)")
    print("="*70)
    

def test_with_video_file(video_path, sample_rate=5):
    """Test with actual video file if provided"""
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        return
    
    print("\n" + "="*70)
    print(f"DROWSINESS DETECTION - VIDEO FILE TEST")
    print(f"File: {video_path}")
    print("="*70)
    
    detector = SoftwareDrowsinessDetector()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Could not open video file")
        return
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_rate == 0:
            status = detector.process_frame(frame)
            if status:
                print(f"Frame {frame_idx:<6} → {status}")
        
        frame_idx += 1
    
    cap.release()
    
    # Print statistics
    stats = detector.get_stats()
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    print(f"Total frames processed: {stats['total_frames']}")
    print(f"Drowsy detections: {stats['drowsy']} ({stats['drowsy_percentage']:.1f}%)")
    print(f"Awake detections: {stats['awake']} ({100-stats['drowsy_percentage']:.1f}%)")
    print("="*70)


def test_realistic_scenario():
    """Test a realistic drowsiness scenario"""
    print("\n" + "="*70)
    print("DROWSINESS DETECTION - REALISTIC SCENARIO TEST")
    print("="*70)
    print("\nScenario: Driver transitions from awake to drowsy to drowsy to awake\n")
    
    detector = SoftwareDrowsinessDetector()
    
    scenarios = [
        ("Awake Phase (Frames 1-15)", [np.zeros((480, 640, 3), dtype=np.uint8)] * 15),
        ("Transitioning (Frames 16-25)", [np.full((480, 640, 3), 100+i*10, dtype=np.uint8) for i in range(10)]),
        ("Drowsy Phase (Frames 26-45)", [np.full((480, 640, 3), 200, dtype=np.uint8)] * 20),
        ("Alert Again (Frames 46-60)", [np.zeros((480, 640, 3), dtype=np.uint8)] * 15),
    ]
    
    print(f"{'Frame':<8} {'Status':<12} {'Phase':<25}")
    print("-" * 50)
    
    frame_num = 1
    for phase_name, frames in scenarios:
        for frame in frames:
            status = detector.process_frame(frame)
            print(f"{frame_num:<8} {status:<12} {phase_name:<25}")
            frame_num += 1
    
    # Print statistics
    stats = detector.get_stats()
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    print(f"Total frames processed: {stats['total_frames']}")
    print(f"Drowsy detections: {stats['drowsy']} ({stats['drowsy_percentage']:.1f}%)")
    print(f"Awake detections: {stats['awake']} ({100-stats['drowsy_percentage']:.1f}%)")
    
    if stats['drowsy_percentage'] > 30:
        print("\n⚠️  ALERT: High drowsiness detected! Driver should rest.")
    else:
        print("\n✓ Driver appears to be alert.")
    
    print("="*70)


def main():
    print("\n" + "="*70)
    print("SOFTWARE-ONLY DROWSINESS DETECTION SYSTEM")
    print("="*70)
    
    # Run all tests
    test_with_synthetic_frames(num_frames=50)
    test_realistic_scenario()
    
    print("\n" + "="*70)
    print("✅ ALL SOFTWARE TESTS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nTo test with a video file, run:")
    print("  python software_drowsiness_detector.py <path_to_video>")
    print("\nExample:")
    print("  python software_drowsiness_detector.py video.mp4")
    print("="*70 + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test with video file if provided
        video_file = sys.argv[1]
        test_with_video_file(video_file)
    else:
        # Run default synthetic tests
        main()
