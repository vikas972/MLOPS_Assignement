from flask import Flask, request, jsonify
import mlflow
import numpy as np
import os
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def get_latest_model_path():
    """Find and return the path to the latest model."""
    try:
        models_dir = "models"
        if not os.path.exists(models_dir):
            raise FileNotFoundError(f"Models directory '{models_dir}' does not exist")
        
        model_paths = [f for f in os.listdir(models_dir) if f.startswith("model_")]
        if not model_paths:
            raise FileNotFoundError("No models found in models directory")
        
        latest_model = sorted(model_paths)[-1]
        return os.path.join(models_dir, latest_model)
    except Exception as e:
        logger.error(f"Error finding latest model: {str(e)}")
        raise

# Load the model with retry logic
max_retries = 3
retry_delay = 5

for attempt in range(max_retries):
    try:
        logger.info(f"Attempting to load model (attempt {attempt + 1}/{max_retries})")
        model_path = get_latest_model_path()
        model = mlflow.sklearn.load_model(model_path)
        logger.info(f"Successfully loaded model from {model_path}")
        break
    except Exception as e:
        if attempt < max_retries - 1:
            logger.warning(f"Failed to load model: {str(e)}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            logger.error(f"Failed to load model after {max_retries} attempts")
            raise

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    try:
        # Verify model is loaded
        if model is None:
            raise RuntimeError("Model not loaded")
        
        return jsonify({
            "status": "healthy",
            "model_path": get_latest_model_path()
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route("/predict", methods=["POST"])
def predict():
    """Prediction endpoint."""
    try:
        data = request.get_json()
        if not data or "features" not in data:
            raise ValueError("Request must include 'features' field")
        
        features = np.array(data["features"]).reshape(1, -1)
        if features.shape[1] != 10:  # Ensure correct number of features
            raise ValueError(f"Expected 10 features, got {features.shape[1]}")
        
        prediction = model.predict(features)
        logger.info("Successfully made prediction")
        
        return jsonify({
            "prediction": prediction.tolist(),
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400

if __name__ == "__main__":
    logger.info("Starting Flask application...")
    app.run(host="0.0.0.0", port=5001) 