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
    # Dapatkan run_id dari run yang sedang aktif
    run_id = run.info.run_id
    
    # Simpan run_id ke file teks di direktori saat ini
    with open("run_id.txt", "w") as f:
        f.write(run_id)
    
    print(f"Successfully saved run_id: {run_id} to run_id.txt")

    # Lakukan training model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=args.epochs)

    # --- PERUBAHAN UTAMA DI SINI ---
    # Gunakan fungsi log_model yang dirancang khusus untuk ini.
    # Ini akan secara otomatis membuat file 'MLmodel' dan semua metadata yang diperlukan.
    print("Logging model using mlflow.tensorflow.log_model...")
    mlflow.tensorflow.log_model(
        model=model,
        artifact_path="model"  # Ini akan menjadi nama folder artefak di DagsHub
    )
    print("Model successfully logged.")