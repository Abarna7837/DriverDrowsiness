import pathlib
import sys
from collections import Counter

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_dataset():
    data_path = PROJECT_ROOT / "data" / "raw" / "drowsiness.csv"
    data = []

    with data_path.open(newline="") as csv_file:
        for line_number, line in enumerate(csv_file, start=1):
            line = line.strip()
            if not line or line_number == 1:
                continue

            values = line.split(",")
            if len(values) < 5:
                continue

            row = [float(values[i]) for i in range(4)]
            label = int(values[4])
            data.append(row + [label])

    if not data:
        raise ValueError(f"No usable rows found in {data_path}")

    array = __import__("numpy").array(data, dtype="float32")
    X = array[:, :-1]
    y = array[:, -1].astype(int)

    return X, y


def prepare_test_split(X, y):
    label_counts = Counter(int(label) for label in y)
    can_stratify = len(y) >= 4 and min(label_counts.values(), default=0) >= 2

    if can_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
        split_message = "Using a stratified 80/20 train-test split."
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=max(1, len(y) // 2),
            random_state=42,
            shuffle=True,
        )
        split_message = (
            "Dataset is too small for stratified evaluation; using a simple split "
            "and treating the result as a rough sanity check, not a reliable estimate."
        )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = X_train.reshape((-1, 2, 2, 1))
    X_test = X_test.reshape((-1, 2, 2, 1))

    return X_train, X_test, y_train, y_test, split_message


def main():
    model_path = PROJECT_ROOT / "models" / "drowsiness_model.keras"
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        print("Run models/train_model.py first to create the saved model.")
        return

    X, y = load_dataset()
    _, X_test, _, y_test, split_message = prepare_test_split(X, y)

    model = load_model(model_path, compile=False)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    probabilities = model.predict(X_test, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)

    print("=" * 70)
    print("MODEL ACCURACY REPORT")
    print("=" * 70)
    print(split_message)
    print()
    print(f"Test loss:     {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print()
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print()
    print("Classification report:")
    print(classification_report(y_test, predictions, labels=[0, 1], zero_division=0, digits=4))

    manual_accuracy = accuracy_score(y_test, predictions)
    print(f"Manual accuracy check: {manual_accuracy:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()