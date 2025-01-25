# MLOps Project

This project demonstrates MLOps best practices including CI/CD, experiment tracking, and model deployment.

## Project Structure
```
.
├── .github/
│   └── workflows/      # CI/CD pipeline configurations
├── src/
│   ├── train.py       # Model training script
│   ├── app.py         # Flask application for model serving
│   └── utils.py       # Utility functions
├── tests/             # Unit tests
├── data/              # Dataset storage
├── models/            # Saved model artifacts
├── notebooks/         # Jupyter notebooks for exploration
├── Dockerfile         # Container configuration
└── requirements.txt   # Project dependencies
```

## Setup and Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Train the model:
   ```bash
   python src/train.py
   ```

2. Run the Flask application:
   ```bash
   python src/app.py
   ```

## MLOps Features

- CI/CD Pipeline using GitHub Actions
- Experiment tracking with MLflow
- Data versioning with DVC
- Model serving with Flask
- Containerization with Docker

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT 