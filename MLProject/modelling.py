import argparse
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import mlflow
import mlflow.tensorflow
import dagshub

# --- 1. Inisialisasi DagsHub & MLflow ---
# Inisialisasi ini akan mengambil kredensial dari environment variables yang disediakan oleh GitHub Actions
print("Initializing DagsHub and MLflow...")
dagshub.init(repo_owner="macamtema", repo_name="smsml_tema", mlflow=True)

# Aktifkan autologging untuk mencatat semua parameter, metrik, dan model secara otomatis
mlflow.tensorflow.autolog()
print("MLflow Autologging Enabled.")

# --- 2. Setup Argumen dan Path Data ---
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10, help="Jumlah epoch training")
args = parser.parse_args()

# Path ke data, diasumsikan skrip dijalankan dari dalam folder MLProject
DATA_DIR = "data_split"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
print(f"Loading data from: {DATA_DIR}")

# --- 3. Persiapan Data Generator ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(128, 128), batch_size=32, class_mode='categorical', shuffle=True)
val_generator = val_datagen.flow_from_directory(
    VAL_DIR, target_size=(128, 128), batch_size=32, class_mode='categorical', shuffle=False)

# --- 4. Bangun Arsitektur Model CNN ---
print("Building CNN model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- 5. Training Model dengan Tracking Otomatis ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
]

print(f"\nStarting training for {args.epochs} epochs...")

with mlflow.start_run() as run:
    # Logika krusial untuk pipeline CI/CD: simpan run_id ke file
    run_id = run.info.run_id
    with open("run_id.txt", "w") as f:
        f.write(run_id)
    print(f"MLflow Run ID: {run_id} (disimpan ke run_id.txt)")

    # Mulai training. Autologger akan bekerja di background.
    model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=val_generator,
        callbacks=callbacks
    )

print("\nTraining and logging finished successfully.")