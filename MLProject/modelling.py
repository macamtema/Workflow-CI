# MLProject/mlflow_train.py

import mlflow
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

mlflow.autolog(disable=True)  # kita log manual

# Load data
iris = load_iris()
X = StandardScaler().fit_transform(iris.data)
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
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

    # Simpan model lokal
    os.makedirs("model", exist_ok=True)
    model.save("model/model.keras")

    mlflow.log_artifact("model/model.keras", artifact_path="model")
    print(f"✅ Model saved locally to model/model.keras")
