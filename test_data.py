import numpy as np
import pandas as pd
import os

print("Script started")

# Create test data
data = {
    'A': np.random.rand(10),
    'B': np.random.rand(10)
}

print("Created test data")

# Create DataFrame
df = pd.DataFrame(data)
print("Created DataFrame")

# Create directory
os.makedirs("test_output", exist_ok=True)
print("Created directory")

# Save DataFrame
df.to_csv("test_output/test.csv", index=False)
print("Saved CSV")

# Verify file
if os.path.exists("test_output/test.csv"):
    print(f"File exists with size: {os.path.getsize('test_output/test.csv')} bytes")
else:
    print("File was not created") 