import mlflow
import tensorflow as tf
import numpy as np
import os

# Dummy data (ganti dengan data real kalau sudah clone dataset)
x_train = np.random.rand(100, 32, 32, 3)
y_train = tf.keras.utils.to_categorical(np.random.randint(3, size=(100,)), num_classes=3)

x_val = np.random.rand(20, 32, 32, 3)
y_val = tf.keras.utils.to_categorical(np.random.randint(3, size=(20,)), num_classes=3)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

with mlflow.start_run():
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=3)

    mlflow.log_metric("val_accuracy", history.history["val_accuracy"][-1])
    mlflow.keras.log_model(model, "model")
