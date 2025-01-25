# MLOps Project

This project demonstrates MLOps best practices using MLflow, DVC, Optuna, and Docker.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start MLflow server:
```bash
mlflow server --host 0.0.0.0 --port 5000
```

## Data Management with DVC

1. Generate and version the dataset:
```bash
dvc repro generate_data
```

2. Track changes and create a new version:
```bash
dvc add data/dataset.csv
git add data/dataset.csv.dvc
git commit -m "Add dataset version X"
```

3. Switch between dataset versions:
```bash
git checkout <commit-hash>
dvc checkout
```

## Model Training and Tracking

1. Run experiments with MLflow:
```bash
python src/train_mlflow.py
```

View experiments at http://localhost:5000

2. Perform hyperparameter optimization:
```bash
python src/hyperparameter_tuning.py
```

## Model Deployment

1. Build the Docker image:
```bash
docker build -t mlops-model:latest .
```

2. Run the container:
```bash
docker run -p 5001:5001 mlops-model:latest
```

3. Test the API:
```bash
curl -X POST http://localhost:5001/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}'
```

## Project Structure

- `src/`
  - `app.py`: Flask API for model serving
  - `data_generator.py`: Generate and version datasets
  - `train_mlflow.py`: Model training with MLflow tracking
  - `hyperparameter_tuning.py`: Optuna optimization
- `data/`: Versioned datasets
- `models/`: Saved models
- `dvc.yaml`: DVC pipeline configuration
- `Dockerfile`: Container configuration for deployment

## MLOps Features

1. **Experiment Tracking (MLflow)**
   - Parameter logging
   - Metric tracking
   - Model versioning
   - Experiment comparison

2. **Data Versioning (DVC)**
   - Dataset versioning
   - Pipeline tracking
   - Reproducible experiments

3. **Hyperparameter Optimization (Optuna)**
   - Automated tuning
   - Integration with MLflow
   - Best parameter tracking

4. **Model Deployment**
   - Containerized deployment
   - REST API
   - Health monitoring
   - Security best practices

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT 