import argparse
import mlflow
import mlflow.tensorflow
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# Argument parser untuk parameter eksternal
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=5)
args = parser.parse_args()

# Aktifkan autologging
mlflow.tensorflow.autolog()

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

# Mulai run MLflow secara eksplisit (jika autolog tidak cukup)
with mlflow.start_run() as run:
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=args.epochs)
    mlflow.log_model(model, artifact_path="model", input_example=None, signature=None)
    print("MLflow run ID:", run.info.run_id)

