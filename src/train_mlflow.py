import mlflow
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def generate_synthetic_data(n_samples=442):
    """Generate synthetic regression data."""
    X = np.random.randn(n_samples, 10)
    y = (
        0.3 * X[:, 0]
        + 0.5 * X[:, 1]
        + 0.2 * X[:, 2]
        + np.random.normal(0, 0.1, n_samples)
    )
    return X, y


def train_model(X, y, params):
    """Train a model with given parameters."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(**params, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2


def run_experiment(experiment_name, params):
    """Run an MLflow experiment with given parameters."""
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)

        # Generate data
        X, y = generate_synthetic_data()

        # Train model and get metrics
        model, mse, r2 = train_model(X, y, params)

        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        return mse, r2


if __name__ == "__main__":
    # Set up MLflow tracking URI (local)
    mlflow.set_tracking_uri("http://localhost:5000")

    # Define different parameter sets for experiments
    experiment_configs = [
        {
            "name": "baseline_model",
            "params": {
                "n_estimators": 100,
                "max_depth": 5,
                "min_samples_split": 2,
            },
        },
        {
            "name": "deep_trees",
            "params": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 2,
            },
        },
        {
            "name": "more_trees",
            "params": {
                "n_estimators": 200,
                "max_depth": 5,
                "min_samples_split": 2,
            },
        },
    ]

    # Run experiments
    for config in experiment_configs:
        print(f"\nRunning experiment: {config['name']}")
        mse, r2 = run_experiment(config["name"], config["params"])
        print(f"Results - MSE: {mse:.4f}, R2: {r2:.4f}") 