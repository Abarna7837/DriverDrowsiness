import sys
import pathlib
import cv2
import os

# ensure project root is on sys.path when running `python app/main.py`
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.vision.feature_extractor import FeatureExtractor
from src.prediction.predictor import Predictor
from src.utils.alert import play_alarm

# ----------------------------
# Initialize
# ----------------------------

def check_camera_availability():
    """Check if webcam is available"""
    print("Checking camera availability...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ ERROR: Camera is not accessible!")
        return False
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        print("❌ ERROR: Could not read from camera!")
        return False
    
    print("✓ Camera is ready!")
    return True


def process_video_file(video_path):
    """Process drowsiness detection from a video file"""
    print(f"\nProcessing video file: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Cannot open video file: {video_path}")
        return
    
    extractor = FeatureExtractor()
    predictor = Predictor()
    
    frame_count = 0
    drowsy_count = 0
    
    print("Video Drowsiness Detection Started\nPress 'Q' to quit\n")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        features = extractor.extract(frame)
        
        if features is not None:
            prediction = predictor.predict(features)
            
            if prediction == 1:
                drowsy_count += 1
                cv2.putText(
                    frame,
                    "DROWSINESS DETECTED",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
                play_alarm()
            else:
                cv2.putText(
                    frame,
                    "AWAKE",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
        
        cv2.imshow("Driver Drowsiness Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nVideo Processing Complete!")
    print(f"Total frames: {frame_count}")
    print(f"Drowsy detections: {drowsy_count} ({drowsy_count/frame_count*100:.1f}%)")


def main_webcam(threshold=0.6, debug=False):
    """Process drowsiness detection from webcam
    
    Args:
        threshold: Sensitivity level (0.5-0.9). Higher = less sensitive to drowsiness
        debug: Show confidence scores on screen
    """
    cap = cv2.VideoCapture(0)
    
    extractor = FeatureExtractor()
    predictor = Predictor(threshold=threshold, use_ml=False)  # Use rule-based detection
    
    print(f"Driver Drowsiness Detection Started")
    print(f"Using: Rule-based detection (Eye Aspect Ratio)")
    print(f"Sensitivity Threshold: {threshold}")
    print(f"Debug Mode: {debug}")
    print("Press 'Q' to quit\n")
    
    frame_count = 0
    drowsy_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Failed to read from webcam!")
            break
        
        frame_count += 1
        features = extractor.extract(frame, verbose=debug)
        
        if features is not None:
            prediction, score = predictor.predict_with_score(features)
            
            if prediction == 1:
                drowsy_count += 1
                status_text = "DROWSY"
                color = (0, 0, 255)  # Red
                play_alarm()
            else:
                status_text = "AWAKE"
                color = (0, 255, 0)  # Green
            
            # Display status
            cv2.putText(
                frame,
                status_text,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )
            
            # Show detailed debug info
            if debug:
                feature_vals = features.reshape(-1)
                current_ear = feature_vals[0]
                avg_ear = feature_vals[1]
                blink_rate = feature_vals[2]
                closure_dur = feature_vals[3]
                
                debug_texts = [
                    f"Score: {score:.2f}",
                    f"EAR: {current_ear:.2f}",
                    f"Avg EAR: {avg_ear:.2f}",
                    f"Blink: {blink_rate:.2f}",
                    f"Close: {closure_dur:.2f}"
                ]
                
                print(f"Frame {frame_count}: Pred={prediction} Score={score:.3f} EAR={current_ear:.3f}")
                
                y_offset = 80
                for text in debug_texts:
                    cv2.putText(
                        frame,
                        text,
                        (30, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1
                    )
                    y_offset += 25
        
        cv2.imshow("Driver Drowsiness Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nWebcam Processing Complete!")
    print(f"Total frames: {frame_count}")
    print(f"Drowsy detections: {drowsy_count} ({drowsy_count/frame_count*100:.1f}%)" if frame_count > 0 else "No frames captured")


def main():
    """Main entry point with camera detection
    
    Usage:
        python app/main.py                           # Default (threshold 0.6)
        python app/main.py video.mp4                 # Video file
        python app/main.py --threshold 0.7           # Less sensitive (0.5-0.9)
        python app/main.py --threshold 0.7 --debug   # Show confidence scores
    """
    print("="*70)
    print("DRIVER DROWSINESS DETECTION SYSTEM")
    print("="*70 + "\n")
    
    # Parse command-line arguments
    threshold = 0.6  # Default threshold (less sensitive than original 0.5)
    debug = False
    video_file = None
    
    if len(sys.argv) > 1:
        i = 1
        while i < len(sys.argv):
            if sys.argv[i] == "--threshold" and i + 1 < len(sys.argv):
                try:
                    threshold = float(sys.argv[i + 1])
                    i += 2
                except ValueError:
                    print(f"❌ Invalid threshold: {sys.argv[i + 1]}")
                    return
            elif sys.argv[i] == "--debug":
                debug = True
                i += 1
            elif not sys.argv[i].startswith("--"):
                video_file = sys.argv[i]
                i += 1
            else:
                i += 1
    
    # Process video file if provided
    if video_file:
        if os.path.exists(video_file):
            process_video_file(video_file)
            return
        else:
            print(f"❌ File not found: {video_file}")
            return
    
    # Try to use webcam
    if check_camera_availability():
        main_webcam(threshold=threshold, debug=debug)
    else:
        print("\n" + "="*70)
        print("SOLUTIONS:")
        print("="*70)
        print("1. Update your camera drivers from manufacturer website")
        print("2. Check Privacy & Security → Camera permissions in Windows Settings")
        print("3. Try a USB/external webcam")
        print("4. Use a video file instead:\n")
        print("   python app/main.py video.mp4\n")
        print("5. Use software-only detection (no webcam needed):\n")
        print("   python software_drowsiness_detector.py")
        print("="*70)


if __name__ == "__main__":
    main()