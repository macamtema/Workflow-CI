import mlflow
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import tempfile
import os

mlflow.autolog(disable=True)  # Nonaktifkan autolog MLflow untuk menghindari konflik API

# Load dan praproses data
iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Membangun model
def build_model(input_shape, num_classes):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_shape,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_model(X_train.shape[1], len(np.unique(y)))

with mlflow.start_run() as run:
    model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), verbose=2)

    # Simpan model secara lokal
    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, "model.keras")
    model.save(model_path)

    # Log model ke MLflow (disimpan sebagai artefak)
    mlflow.log_artifacts(temp_dir, artifact_path="model")

    print(f"✅ Model berhasil disimpan sebagai artefak di run ID: {run.info.run_id}")
    print("ℹ️ Untuk mengunduh model, lihat artefak di UI Dagshub atau gunakan MLflow API.")
