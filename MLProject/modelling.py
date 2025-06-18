import argparse
import mlflow
import mlflow.tensorflow
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Argument parser untuk parameter eksternal
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=5)
args = parser.parse_args()

# Nonaktifkan autolog karena tidak stabil di DagsHub
mlflow.tensorflow.autolog(disable=True)

# Buat data dummy: 1000 gambar 28x28 grayscale
X = np.random.rand(1000, 28, 28, 1).astype(np.float32)
y = np.random.randint(0, 10, size=(1000,))

# Bagi data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Buat model CNN sederhana
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Jalankan training dalam MLflow run
with mlflow.start_run() as run:
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=args.epochs)

    # Simpan model lokal, lalu log manual
    model_dir = "saved_model"
    mlflow.tensorflow.save_model(model, model_dir)
    mlflow.log_artifacts(model_dir, artifact_path="model")

    print("MLflow run ID:", run.info.run_id)
