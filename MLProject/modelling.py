import argparse
import os
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- 0. Setup Awal & Argumen ---
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=5, help="Jumlah epoch untuk training")
args = parser.parse_args()

# --- 1. Persiapan Data Generator (Versi Lebih Ringan) ---
print("Mempersiapkan Data Generator...")
DATA_DIR = '../data_split'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')

TARGET_SIZE = (96, 96) # Ukuran gambar diperkecil agar training lebih cepat
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)
val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Jalankan proses training dan logging di dalam MLflow run
with mlflow.start_run() as run:
    run_id = run.info.run_id
    with open("run_id.txt", "w") as f:
        f.write(run_id)
    print(f"MLflow Run ID: {run_id}")

    # Log parameter
    mlflow.log_param("epochs", args.epochs)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("target_size", f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}")

    # --- 2. Definisi Arsitektur Model (Versi Lebih Ringan) ---
    print("Membangun model CNN...")
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3)),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(train_generator.num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # --- 3. Setup Callbacks ---
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    ]

    # --- 4. Training Model ---
    print("Memulai training model...")
    history = model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=val_generator,
        callbacks=callbacks
    )

    # --- 5. Log Metrik dan Artefak ke MLflow (Manual & Eksplisit) ---
    print("Training selesai. Melakukan logging hasil ke MLflow...")
    # Log metrik performa final
    final_val_acc = history.history['val_accuracy'][-1]
    mlflow.log_metric("final_validation_accuracy", final_val_acc)

    # Simpan dan log label kelas
    with open("labels.txt", "w") as f:
        for label, index in train_generator.class_indices.items():
            f.write(f"{index}: {label}\n")
    mlflow.log_artifact("labels.txt")

    # Log file conda.yaml secara eksplisit
    mlflow.log_artifact("conda.yaml", artifact_path="environment")
    
    # Log model yang sudah dilatih
    mlflow.tensorflow.log_model(
        model=model,
        artifact_path="model"
    )
    
    print("\nProses selesai. Model, metrik, dan artefak berhasil di-log ke MLflow.")