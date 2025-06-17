import mlflow
import mlflow.keras

with mlflow.start_run() as run:
    # Latih model
    model = build_model(...)  # ganti dengan fungsi modelmu
    model.fit(...)

    # Log model (gunakan .keras)
    mlflow.keras.log_model(model, artifact_path="model.keras")

    # Daftarkan model ke registry
    result = mlflow.register_model(
        model_uri=f"runs:/{run.info.run_id}/model.keras",
        name="smsml_model"
    )
    print(f"✅ Model registered: {result.name}, version: {result.version}")
