from flask import Flask, request, jsonify
import mlflow
import numpy as np
import os

app = Flask(__name__)

def get_latest_model_path():
    """Find and return the path to the latest model."""
    models_dir = "models"
    model_paths = [f for f in os.listdir(models_dir) if f.startswith("model_")]
    if not model_paths:
        raise FileNotFoundError("No models found in models directory")
    latest_model = sorted(model_paths)[-1]
    return os.path.join(models_dir, latest_model)

# Load the model
model = mlflow.sklearn.load_model(get_latest_model_path())

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})

@app.route("/predict", methods=["POST"])
def predict():
    """Prediction endpoint."""
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({"prediction": prediction.tolist(), "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001) 