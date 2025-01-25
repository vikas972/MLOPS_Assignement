import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generator import generate_dataset

def test_data_generator():
    """Test if data generator creates data file"""
    generate_dataset(n_samples=10, version="v1")
    assert os.path.exists("data/dataset.csv")

def test_imports():
    """Test if all required packages are installed"""
    import numpy
    import pandas
    import sklearn
    import mlflow
    assert True 