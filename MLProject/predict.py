# MLProject/predict.py

import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("model/model.keras")

# Contoh input dummy
X_new = np.array([[5.1, 3.5, 1.4, 0.2]])
pred = model.predict(X_new)
print("Predicted class:", np.argmax(pred))
