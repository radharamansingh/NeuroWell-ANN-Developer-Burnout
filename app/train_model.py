
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "developer_wellbeing.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# 1. Load dataset
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["burnout_risk"])
y = df["burnout_risk"]

print("Dataset shape:", df.shape)
print("\nMissing values:", df.isnull().sum().sum())

# 2. Split before scaling to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save preprocessing objects
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

with open(MODEL_DIR / "features.json", "w") as f:
    json.dump(list(X.columns), f)

# 4. Build ANN
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(3, activation="softmax")
])

# 5. Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 6. Train the ANN
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=40,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# 7. Evaluate on unseen test data
probabilities = model.predict(X_test, verbose=0)
predictions = np.argmax(probabilities, axis=1)

accuracy = accuracy_score(y_test, predictions)

print("\nTEST ACCURACY:", round(accuracy * 100, 2), "%")
print("\nCLASSIFICATION REPORT:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1, 2],
    target_names=["Low", "Medium", "High"],
    zero_division=0
))

print("CONFUSION MATRIX:")
print(confusion_matrix(y_test, predictions, labels=[0, 1, 2]))

# 8. Save trained model
model.save(MODEL_DIR / "neurowell_ann.keras")

# Save training history for later visualization
history_data = {
    "accuracy": [float(v) for v in history.history["accuracy"]],
    "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
    "loss": [float(v) for v in history.history["loss"]],
    "val_loss": [float(v) for v in history.history["val_loss"]]
}

with open(MODEL_DIR / "training_history.json", "w") as f:
    json.dump(history_data, f)

print("\nSUCCESS! ANN TRAINED AND SAVED.")
print("Model:", MODEL_DIR / "neurowell_ann.keras")
print("Scaler:", MODEL_DIR / "scaler.pkl")
