import numpy as np
import pandas as pd
import os
import sys

def generate_dataset(n_samples=1000, version="v1"):
    """Generate a synthetic dataset and save it with versioning."""
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Generate synthetic features
    X = np.random.randn(n_samples, 10)
    
    # Generate target variable with different versions
    if version == "v1":
        y = 0.3 * X[:, 0] + 0.5 * X[:, 1] + 0.2 * X[:, 2] + np.random.normal(0, 0.1, n_samples)
    elif version == "v2":
        y = 0.4 * X[:, 0] + 0.4 * X[:, 1] + 0.3 * X[:, 2] + np.random.normal(0, 0.05, n_samples)
    else:
        y = 0.35 * X[:, 0] + 0.45 * X[:, 1] + 0.25 * X[:, 2] + np.random.normal(0, 0.075, n_samples)
    
    # Create a DataFrame
    columns = [f"feature_{i}" for i in range(10)]
    df = pd.DataFrame(X, columns=columns)
    df['target'] = y
    
    # Save the dataset
    output_path = "data/dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    
if __name__ == "__main__":
    try:
        # Ensure data directory exists
        print("Creating data directory...")
        os.makedirs("data", exist_ok=True)
        
        print("\nGenerating datasets...")
        
        # Version 1: Original dataset
        print("\nGenerating version 1 (original dataset)...")
        generate_dataset(n_samples=442, version="v1")
        
        # Version 2: Dataset with some noise
        print("\nGenerating version 2 (with noise)...")
        generate_dataset(n_samples=442, version="v2")
        
        # Version 3: Smaller dataset with different noise
        print("\nGenerating version 3 (smaller with noise)...")
        generate_dataset(n_samples=300, version="v3")
        
        print("\nVerifying files...")
        for version in range(1, 4):
            path = f"data/dataset.csv"
            if os.path.exists(path):
                size = os.path.getsize(path)
                print(f"Created {path} (size: {size} bytes)")
            else:
                print(f"Error: {path} was not created")
                
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
        
    print("\nDataset generation completed successfully!") 