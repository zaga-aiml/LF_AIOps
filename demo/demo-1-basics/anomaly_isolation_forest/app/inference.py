# inference.py

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import mlflow
import mlflow.sklearn
import os
import time
import json

# -----------------------------
# Initialize MLflow
# -----------------------------
def init_mlflow():
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")

    for attempt in range(10):
        try:
            print(f"Connecting to MLflow at {mlflow_uri} (attempt {attempt+1})")

            mlflow.set_tracking_uri(mlflow_uri)

            # Test connection
            mlflow.search_experiments()

            # ✅ Set inference experiment
            mlflow.set_experiment("anomaly-inference")

            print("✅ Connected to MLflow")
            return

        except Exception as e:
            print("❌ Waiting for MLflow...", e)
            time.sleep(3)

    raise RuntimeError("Could not connect to MLflow")


init_mlflow()

# -----------------------------
# Load model from MLflow
# -----------------------------
MODEL_NAME = "anomaly-detection-model"

try:
    print("Trying to load model using alias: production")
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@production")

except Exception as e:
    print("⚠️ Alias not found, loading latest version instead:", e)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/latest")

print("✅ Model loaded successfully")

# -----------------------------
# Request schema
# -----------------------------
class Metrics(BaseModel):
    cpu_usage: float
    memory_usage: float


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Anomaly Detection API")


# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict_anomaly(metrics: Metrics):

    print("🚀 /predict API HIT")

    try:
        # Prepare input
        features = np.array([[metrics.cpu_usage, metrics.memory_usage]])

        # Predict
        prediction = model.predict(features)
        result = "anomaly" if prediction[0] == -1 else "normal"

        print(f"Prediction: {result}")

        # -----------------------------
        # MLflow Logging
        # -----------------------------
        with mlflow.start_run(run_name="inference-run") as run:

            print(f"MLflow Run ID: {run.info.run_id}")

            # Tags
            mlflow.set_tag("stage", "inference")

            # Metrics
            mlflow.log_metric("cpu_usage", metrics.cpu_usage)
            mlflow.log_metric("memory_usage", metrics.memory_usage)
            mlflow.log_metric("is_anomaly", 1 if result == "anomaly" else 0)

            # Params
            mlflow.log_param("prediction_label", result)

            # Artifact
            inference_data = {
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "prediction": result
            }

            with open("inference.json", "w") as f:
                json.dump(inference_data, f)

            mlflow.log_artifact("inference.json")

            print("✅ Logged to MLflow")

        return {
            "cpu_usage": metrics.cpu_usage,
            "memory_usage": metrics.memory_usage,
            "result": result
        }

    except Exception as e:
        print("❌ ERROR in prediction:", str(e))
        return {"error": str(e)}
