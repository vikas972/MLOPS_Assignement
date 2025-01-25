from flask import Flask, request, jsonify, render_template
import mlflow
import numpy as np
import os
import logging
import time
from sklearn.ensemble import RandomForestRegressor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')

def create_default_model():
    """Create and save a default model if none exists."""
    try:
        os.makedirs("models", exist_ok=True)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(np.random.randn(100, 10), np.random.randn(100))
        model_path = os.path.join("models", "model_default.pkl")
        mlflow.sklearn.save_model(model, model_path)
        logger.info(f"Created default model at {model_path}")
        return model_path
    except Exception as e:
        logger.error(f"Failed to create default model: {str(e)}")
        raise

def get_latest_model_path():
    """Find and return the path to the latest model."""
    try:
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        
        model_paths = [f for f in os.listdir(models_dir) if f.startswith("model_")]
        if not model_paths:
            logger.warning("No models found. Creating default model...")
            return create_default_model()
        
        latest_model = sorted(model_paths)[-1]
        return os.path.join(models_dir, latest_model)
    except Exception as e:
        logger.error(f"Error finding latest model: {str(e)}")
        raise

# Load the model with retry logic
max_retries = 3
retry_delay = 5
model = None

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
            # Create and load default model as last resort
            try:
                model_path = create_default_model()
                model = mlflow.sklearn.load_model(model_path)
                logger.info("Successfully loaded default model")
            except Exception as e:
                logger.error(f"Failed to create and load default model: {str(e)}")
                raise

@app.route("/", methods=["GET"])
def index():
    """Serve the frontend interface."""
    return render_template('index.html')

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