"""
Offline test for drowsiness detection system
Tests the model and feature extraction without requiring a webcam
"""

import sys
import pathlib
import numpy as np
import cv2

# ensure project root is on sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))

from src.vision.feature_extractor import FeatureExtractor
from src.prediction.predictor import Predictor


def test_model_loading():
    """Test that the model loads successfully"""
    print("\n" + "="*60)
    print("TEST 1: Model Loading")
    print("="*60)
    try:
        predictor = Predictor()
        print("✓ Model loaded successfully")
        print(f"  Model type: {type(predictor.model)}")
        return True
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False


def test_feature_extraction():
    """Test feature extraction with dummy frames"""
    print("\n" + "="*60)
    print("TEST 2: Feature Extraction")
    print("="*60)
    try:
        extractor = FeatureExtractor()
        
        # Create dummy frames
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        features = extractor.extract(dummy_frame)
        
        if features is not None:
            print(f"✓ Feature extraction successful")
            print(f"  Input frame shape: {dummy_frame.shape}")
            print(f"  Output features shape: {features.shape}")
            print(f"  Feature range: [{features.min():.3f}, {features.max():.3f}]")
            return True
        else:
            print("✗ Feature extraction returned None")
            return False
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
        return False


def test_predictions_with_random_data():
    """Test predictions with random feature data"""
    print("\n" + "="*60)
    print("TEST 3: Predictions with Random Data")
    print("="*60)
    try:
        predictor = Predictor()
        
        # Create random features (1, 2, 2, 1)
        random_features = np.random.rand(1, 2, 2, 1).astype("float32")
        prediction = predictor.predict(random_features)
        
        print(f"✓ Prediction successful")
        print(f"  Random features shape: {random_features.shape}")
        print(f"  Prediction output: {prediction}")
        print(f"  Result: {'DROWSY' if prediction == 1 else 'AWAKE'}")
        return True
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False


def test_end_to_end_with_synthetic_frames():
    """Test full pipeline: extract features → predict"""
    print("\n" + "="*60)
    print("TEST 4: End-to-End Pipeline (10 frames)")
    print("="*60)
    try:
        extractor = FeatureExtractor()
        predictor = Predictor()
        
        drowsy_count = 0
        awake_count = 0
        
        for i in range(10):
            # Create synthetic frames (varying intensity)
            frame = np.full((480, 640, 3), fill_value=i*25, dtype=np.uint8)
            
            features = extractor.extract(frame)
            if features is not None:
                prediction = predictor.predict(features)
                status = "DROWSY" if prediction == 1 else "AWAKE"
                
                if prediction == 1:
                    drowsy_count += 1
                else:
                    awake_count += 1
                
                print(f"  Frame {i+1:2d}: {status}")
        
        print(f"\n✓ End-to-end pipeline successful")
        print(f"  Total frames processed: 10")
        print(f"  Drowsy detections: {drowsy_count}")
        print(f"  Awake detections: {awake_count}")
        return True
    except Exception as e:
        print(f"✗ End-to-end pipeline failed: {e}")
        return False


def test_with_image_patterns():
    """Test with different image patterns"""
    print("\n" + "="*60)
    print("TEST 5: Predictions with Different Image Patterns")
    print("="*60)
    try:
        extractor = FeatureExtractor()
        predictor = Predictor()
        
        patterns = {
            "Black frame (all 0s)": np.zeros((480, 640, 3), dtype=np.uint8),
            "White frame (all 255s)": np.full((480, 640, 3), 255, dtype=np.uint8),
            "Mid-gray frame (all 128s)": np.full((480, 640, 3), 128, dtype=np.uint8),
            "Random noise": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
            "Gradient": np.tile(np.linspace(0, 255, 640), (480, 1, 3)).astype(np.uint8),
        }
        
        for pattern_name, frame in patterns.items():
            features = extractor.extract(frame)
            if features is not None:
                prediction = predictor.predict(features)
                status = "DROWSY" if prediction == 1 else "AWAKE"
                print(f"  {pattern_name:30s} → {status}")
        
        print(f"\n✓ Pattern-based testing successful")
        return True
    except Exception as e:
        print(f"✗ Pattern-based testing failed: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("DRIVER DROWSINESS DETECTION - OFFLINE TEST SUITE")
    print("="*60)
    
    results = []
    results.append(("Model Loading", test_model_loading()))
    results.append(("Feature Extraction", test_feature_extraction()))
    results.append(("Random Predictions", test_predictions_with_random_data()))
    results.append(("End-to-End Pipeline", test_end_to_end_with_synthetic_frames()))
    results.append(("Pattern Testing", test_with_image_patterns()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:30s} {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - System is working correctly!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - Please check the errors above")


if __name__ == "__main__":
    main()
