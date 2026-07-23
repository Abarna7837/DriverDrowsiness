import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

# --------------------------
# Load Dataset
# --------------------------


print("Current Working Directory:", os.getcwd())
print("CSV Exists:", os.path.exists("data/raw/drowsiness.csv"))

df = pd.read_csv("data/raw/drowsiness.csv")

print("=" * 50)
print("Dataset Loaded Successfully")
print("=" * 50)

print(df.head())
print("\nShape :", df.shape)
print("\nMissing Values")
print(df.isnull().sum())

# --------------------------
# Features & Labels
# --------------------------

X = df[["EAR", "PUC", "MAR", "MOE"]].values
y = df["drowsy"].values

# --------------------------
# Train Test Split
# --------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------
# Standardization
# --------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------
# Reshape for CNN
# --------------------------

X_train = X_train.reshape(-1, 2, 2, 1)
X_test = X_test.reshape(-1, 2, 2, 1)

# --------------------------
# CNN Model
# --------------------------

model = Sequential()

model.add(
    Conv2D(
        filters=32,
        kernel_size=(2,2),
        activation="relu",
        input_shape=(2,2,1)
    )
)

model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(128, activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(64, activation="relu"))

model.add(Dense(1, activation="sigmoid"))

# --------------------------
# Compile Model
# --------------------------

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\n")
model.summary()

# --------------------------
# Train Model
# --------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# --------------------------
# Evaluate Model
# --------------------------

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("\n")
print("=" * 50)
print(f"Test Accuracy : {accuracy * 100:.2f}%")
print("=" * 50)

# --------------------------
# Predictions
# --------------------------

y_pred = model.predict(X_test)

y_pred = (y_pred > 0.5).astype(int)

# --------------------------
# Classification Report
# --------------------------

print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

# --------------------------
# Confusion Matrix
# --------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix\n")

print(cm)

# --------------------------
# Save Model
# --------------------------

model.save("models/drowsiness_model.keras")

print("\nModel Saved Successfully!")

print("\nSaved as : models/drowsiness_model.keras")