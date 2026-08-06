import pandas as pd
from pathlib import Path

# Path to your combined CSV
csv_path = Path("/home/amlpai05/erik/hackathon_2025/htrflow/outputs/russian/csv/htr_out_pred_only.csv")

# Load the CSV into a DataFrame
df = pd.read_csv(csv_path)

# Define pickle path (same name but .pkl extension)
feather_path = csv_path.with_suffix(".feather")

# Save DataFrame as a pickle
df.to_feather(feather_path)

print(f"Loaded {len(df)} rows from {csv_path}")
print(f"Saved pickle to {feather_path}")