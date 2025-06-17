import argparse
import mlflow
import mlflow.tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Parse CLI args
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=5)
args = parser.parse_args()

# Jangan pakai autolog karena error dengan DAGsHub (optional)
# mlflow.tensorflow.autolog()

with mlflow.start_run():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train_cat, validation_data=(x_test, y_test_cat), epochs=args.epochs)

    # Logging manual (karena autolog error)
    mlflow.log_param("epochs", args.epochs)
    mlflow.log_metric("final_accuracy", history.history["val_accuracy"][-1])

    # Save and log model directory as artifact
    model.save("model.keras")              # Simpan ke format Keras terbaru
    mlflow.log_artifact("model.keras")     # Upload ke MLflow (DagsHub)
