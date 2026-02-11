import pandas as pd

df = pd.read_parquet("./data/project_a_samples.parquet")

print("Columns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

print("\nRow count:", len(df))