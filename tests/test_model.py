import os
import sys
import pytest
import numpy as np
from sklearn.metrics import r2_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generator import generate_dataset
from src.train_mlflow import generate_synthetic_data, train_model

def test_data_generation():
    """Test data generation functionality"""
    X, y = generate_synthetic_data(n_samples=100)
    assert X.shape == (100, 10)
    assert y.shape == (100,)
    assert not np.isnan(X).any()
    assert not np.isnan(y).any()

def test_model_training():
    """Test model training functionality"""
    # Generate test data
    X, y = generate_synthetic_data(n_samples=100)
    
    # Define test parameters
    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "min_samples_split": 2
    }
    
    # Train model
    model, mse, r2 = train_model(X, y, params)
    
    # Check if metrics are valid
    assert mse >= 0
    assert 0 <= r2 <= 1
    assert model is not None

def test_model_prediction():
    """Test model prediction functionality"""
    # Generate test data
    X, y = generate_synthetic_data(n_samples=100)
    
    # Train model
    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "min_samples_split": 2
    }
    model, _, _ = train_model(X, y, params)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Check predictions
    assert predictions.shape == (100,)
    assert not np.isnan(predictions).any()
    assert r2_score(y, predictions) > 0  # Should have some predictive power

def test_data_generator_output():
    """Test if data generator creates valid CSV file"""
    generate_dataset(n_samples=10, version="v1")
    assert os.path.exists("data/dataset.csv")
    
    # Clean up
    if os.path.exists("data/dataset.csv"):
        os.remove("data/dataset.csv") 