import os
import mlflow
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import datetime

def load_data(data_path):
    """Load and split the dataset from CSV."""
    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def objective(trial, X_train, X_test, y_train, y_test):
    """Optuna objective function for hyperparameter optimization."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 100),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
    }
    
    model = RandomForestRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)

def train_model(data_path, experiment_name):
    """Train model with MLflow tracking."""
    # Set up MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data(data_path)
    
    # Hyperparameter optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, X_test, y_train, y_test), 
                  n_trials=20)
    
    # Train final model with best parameters
    with mlflow.start_run():
        best_params = study.best_params
        mlflow.log_params(best_params)
        
        # Log dataset info
        mlflow.log_param("data_version", os.path.basename(data_path))
        mlflow.log_param("dataset_size", len(X_train) + len(X_test))
        
        model = RandomForestRegressor(**best_params, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        })
        feature_importance.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")
        
        # Save model with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/model_{timestamp}"
        os.makedirs("models", exist_ok=True)
        mlflow.sklearn.save_model(model, model_path)
        
        print(f"Best parameters: {best_params}")
        print(f"MSE: {mse:.4f}")
        print(f"R2 Score: {r2:.4f}")

if __name__ == "__main__":
    # Train models on different dataset versions
    datasets = [
        ("data/diabetes_v1.csv", "experiment_v1"),
        ("data/diabetes_v2.csv", "experiment_v2"),
        ("data/diabetes_v3.csv", "experiment_v3")
    ]
    
    for data_path, experiment_name in datasets:
        print(f"\nTraining on {data_path}")
        train_model(data_path, experiment_name) 