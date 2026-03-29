# anomaly_api.py

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os
import mlflow
import time

# -----------------------------
# Initialize MLflow connection
# -----------------------------
def init_mlflow():
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")

    for attempt in range(10):
        try:
            print(f"Connecting to MLflow at {mlflow_uri} (attempt {attempt+1})")

            mlflow.set_tracking_uri(mlflow_uri)

            # Check connection
            mlflow.search_experiments()

            mlflow.set_experiment("anomaly-detection")

            print("Connected to MLflow successfully")
            return

        except Exception as e:
            print("Waiting for MLflow...", e)
            time.sleep(3)

    raise RuntimeError("Could not connect to MLflow")


# Initialize MLflow
init_mlflow()

# -----------------------------
# Load the model
# -----------------------------
model_path = "model/isolation_forest_model.joblib"

if not os.path.exists(model_path):
    raise FileNotFoundError("Model not found at model/isolation_forest_model.joblib")

model = joblib.load(model_path)

print("Model loaded successfully")

# -----------------------------
# Define request schema
# -----------------------------
class Metrics(BaseModel):
    cpu_usage: float
    memory_usage: float


# -----------------------------
# Initialize FastAPI app
# -----------------------------
app = FastAPI(title="Anomaly Detection API")


# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict_anomaly(metrics: Metrics):

    features = np.array([[metrics.cpu_usage, metrics.memory_usage]])
    prediction = model.predict(features)

    result = "anomaly" if prediction[0] == -1 else "normal"

    # -----------------------------
    # Log inference in MLflow
    # -----------------------------
    with mlflow.start_run(run_name="inference-run"):

        # Log inputs
        mlflow.log_param("cpu_usage", metrics.cpu_usage)
        mlflow.log_param("memory_usage", metrics.memory_usage)

        # Log output
        mlflow.log_param("prediction", result)

        # Log simple metric
        mlflow.log_metric("is_anomaly", 1 if result == "anomaly" else 0)

    # -----------------------------
    # Return response
    # -----------------------------
    return {
        "cpu_usage": metrics.cpu_usage,
        "memory_usage": metrics.memory_usage,
        "result": result
    }
