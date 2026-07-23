import csv
import pathlib
import sys

import cv2


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.feature_extractor import FeatureExtractor


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "drowsiness.csv"


def ensure_header():
    if not DATA_PATH.exists() or DATA_PATH.stat().st_size == 0:
        with DATA_PATH.open("w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["EAR", "PUC", "MAR", "MOE", "drowsy"])


def append_sample(features, label):
    sample = features.reshape(-1)
    row = [float(sample[0]), float(sample[1]), float(sample[2]), float(sample[3]), int(label)]

    with DATA_PATH.open("a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(row)


def main():
    ensure_header()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera could not be opened.")
        return

    extractor = FeatureExtractor()
    saved_count = 0

    print("=" * 70)
    print("DROWSINESS DATA COLLECTION")
    print("=" * 70)
    print("Press 'o' to save an AWAKE sample (label 0)")
    print("Press 'd' to save a DROWSY sample (label 1)")
    print("Press 'q' to quit")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read from webcam.")
            break

        features = extractor.extract(frame, verbose=False)

        cv2.putText(
            frame,
            "o: awake  d: drowsy  q: quit",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Saved samples: {saved_count}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Collect Training Data", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("o"):
            append_sample(features, 0)
            saved_count += 1
            print(f"Saved awake sample #{saved_count}")
        elif key == ord("d"):
            append_sample(features, 1)
            saved_count += 1
            print(f"Saved drowsy sample #{saved_count}")

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nDone. Total saved samples: {saved_count}")
    print(f"Dataset file: {DATA_PATH}")


if __name__ == "__main__":
    main()