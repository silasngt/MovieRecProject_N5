import pandas as pd

print("ETL ca nhan cua Huong dang chay...")

# Extract
df = pd.read_csv("raw/movies.csv")
print("da doc du lieu tu raw/movies.csv")

# Transform
df_cleaned = df.dropna()
print("Da loai bo dong thieu du lieu")

# Load
df_cleaned.to_csv("Huong/movies_cleaned.csv", index=False)
print("Da ghi du lieu vao Huong/movies_cleaned.csv")