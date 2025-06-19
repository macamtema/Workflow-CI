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
# Atur parser argumen untuk menerima jumlah epoch dari luar
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10, help="Jumlah epoch untuk training")
args = parser.parse_args()

# Nonaktifkan autologging MLflow default untuk kontrol manual
mlflow.tensorflow.autolog(disable=True)

# Definisikan path ke dataset yang sudah dibagi
# Diasumsikan folder 'data_split' berada di root proyek 'Workflow-CI'
DATA_DIR = '../data_split' # Path relatif dari dalam folder MLProject
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')

# --- 1. Persiapan Data Generator ---
print("Mempersiapkan Data Generator...")
# Augmentasi untuk data training untuk meningkatkan robustisitas model
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Hanya lakukan rescale untuk data validasi dan test
val_datagen = ImageDataGenerator(rescale=1./255)

# Load data dari direktori menggunakan flow_from_directory
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Jalankan proses training dan logging di dalam MLflow run
with mlflow.start_run() as run:
    run_id = run.info.run_id
    # Simpan run_id ke file untuk digunakan oleh langkah CI/CD selanjutnya
    with open("run_id.txt", "w") as f:
        f.write(run_id)
    print(f"MLflow Run ID: {run_id}")
    print(f"Run ID disimpan di run_id.txt")

    # Log parameter training ke MLflow
    mlflow.log_param("epochs", args.epochs)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("target_size", "128x128")

    # --- 2. Definisi Arsitektur Model ---
    print("Membangun model CNN...")
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
        # Jumlah output neuron disesuaikan dengan jumlah kelas yang terdeteksi oleh generator
        Dense(train_generator.num_classes, activation='softmax')
    ])

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # --- 3. Setup Callbacks ---
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, min_lr=1e-6)
    ]

    # --- 4. Training Model ---
    print("Memulai training model...")
    history = model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=val_generator,
        callbacks=callbacks
    )

    # --- 5. Log Metrik dan Model ke MLflow ---
    print("Training selesai. Melakukan logging hasil ke MLflow...")
    # Log metrik performa final ke MLflow
    final_val_loss = history.history['val_loss'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    mlflow.log_metric("final_validation_loss", final_val_loss)
    mlflow.log_metric("final_validation_accuracy", final_val_acc)

    # Log file conda.yaml secara eksplisit untuk memastikan lingkungan bisa direplikasi
    mlflow.log_artifact("conda.yaml", artifact_path="environment")
    
    # Log model yang sudah dilatih
    mlflow.tensorflow.log_model(
        model=model,
        artifact_path="model"
    )
    
    print("\nProses selesai. Model dan metrik berhasil di-log ke MLflow.")