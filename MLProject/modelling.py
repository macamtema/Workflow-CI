import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

mlflow.keras.autolog()

# Load and preprocess data
iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Build a simple Keras model
def build_model(input_shape, num_classes):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_model(X_train.shape[1], len(np.unique(y)))

with mlflow.start_run() as run:
    model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), verbose=2)

    # ✅ Store in directory 'model', not 'model.keras'
    artifact_path = "model"
    mlflow.keras.log_model(model, artifact_path=artifact_path)

    # Register the model
    model_uri = f"runs:/{run.info.run_id}/{artifact_path}"
    result = mlflow.register_model(model_uri=model_uri, name="smsml_model")

    print(f"✅ Model registered to MLflow with name 'smsml_model' and version {result.version}")
