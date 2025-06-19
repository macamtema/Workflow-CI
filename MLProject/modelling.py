import argparse
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import mlflow
import mlflow.tensorflow
import dagshub

# --- 0. Setup Awal & Inisialisasi DagsHub ---
# Inisialisasi koneksi ke DagsHub, ini akan otomatis mengatur MLFLOW_TRACKING_URI
# Ganti dengan username dan nama repo DagsHub Anda
dagshub.init(repo_owner="macamtema", repo_name="smsml_tema", mlflow=True)

# Argumen input dari command line
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=5, help="Jumlah epoch training")
args = parser.parse_args()

# --- 1. Persiapan Data Generator ---
# Path ke dataset yang sudah dibagi, relatif dari folder MLProject
DATA_DIR = '../data_split'
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(128, 128), batch_size=32, class_mode='categorical', shuffle=True)
val_generator = val_datagen.flow_from_directory(
    VAL_DIR, target_size=(128, 128), batch_size=32, class_mode='categorical', shuffle=False)

# --- 2. Bangun Model CNN ---
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

# --- 3. Setup Callbacks ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
]

# --- 4. Training Model dengan MLflow Autologging ---
# Aktifkan autolog MLflow. Ini akan otomatis me-log parameter, metrik, dan model
mlflow.tensorflow.autolog()

with mlflow.start_run() as run:
    # Logika penting untuk pipeline CI/CD kita: simpan run_id
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

    # --- 5. Simpan Hasil Training secara Lokal (Opsional) ---
    # Langkah ini tidak berhubungan dengan MLflow, hanya untuk menyimpan file di lokal
    # jika diperlukan. Autologger sudah menyimpan model ke MLflow.
    print("Menyimpan model dan label secara lokal...")
    # Nama folder diubah agar sesuai dengan permintaan Anda
    local_model_dir = "Membangun_model"
    os.makedirs(local_model_dir, exist_ok=True)
    model.save(os.path.join(local_model_dir, "model.keras"))

    with open(os.path.join(local_model_dir, "label.txt"), "w") as f:
        # Menyimpan nama kelas sesuai dengan indeksnya
        class_indices = train_generator.class_indices
        sorted_labels = sorted(class_indices.keys(), key=lambda k: class_indices[k])
        for label in sorted_labels:
            f.write(f"{label}\n")
            
    print(f"Model dan label disimpan di folder '{local_model_dir}'")