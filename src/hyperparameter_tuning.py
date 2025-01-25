import optuna
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_data():
    """Load the dataset."""
    data = pd.read_csv("data/dataset.csv")
    X = data.drop('target', axis=1)
    y = data['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    """Optuna objective function."""
    # Define hyperparameters to optimize
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features': trial.suggest_float('max_features', 0.1, 1.0)
    }
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    try:
        # Train model with suggested parameters
        model = RandomForestRegressor(**params, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Log to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metrics({
                'mse': mse,
                'r2': r2
            })
        
        return mse
    except Exception as e:
        print(f"Trial failed: {str(e)}")
        raise optuna.exceptions.TrialPruned()

def run_optimization(n_trials=100):
    """Run the hyperparameter optimization."""
    mlflow.set_experiment("hyperparameter_optimization")
    
    with mlflow.start_run(run_name="optuna_optimization"):
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # Log best parameters and results
        mlflow.log_params(study.best_params)
        mlflow.log_metrics({
            'best_mse': study.best_value,
            'n_trials': n_trials
        })
        
        # Train final model with best parameters
        X_train, X_test, y_train, y_test = load_data()
        best_model = RandomForestRegressor(**study.best_params, random_state=42)
        best_model.fit(X_train, y_train)
        
        # Save the best model
        mlflow.sklearn.log_model(best_model, "best_model")
        
        print("\nBest trial:")
        print(f"  Value (MSE): {study.best_value:.4f}")
        print("  Params: ")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    run_optimization(n_trials=50) 