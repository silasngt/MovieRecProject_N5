import pandas as pd
from pathlib import Path

def clean_ratings():
    """
    Làm sạch dữ liệu ratings.csv
    - Chuyển kiểu dữ liệu
    - Lọc rating hợp lệ
    - Chuyển timestamp sang datetime
    """
    raw_path = Path("etl/raw/ratings.csv")
    out_path = Path("etl/intermediate/ratings.cleaned.parquet")

    # Đọc file CSV
    df = pd.read_csv(raw_path)

    # Loại bỏ dòng có NaN ở userId hoặc movieId
    df = df.dropna(subset=["userId", "movieId"])

    # Ép kiểu dữ liệu
    df["userId"] = df["userId"].astype(int)
    df["movieId"] = df["movieId"].astype(int)

    # Lọc rating hợp lệ
    df = df[(df["rating"] >= 0.5) & (df["rating"] <= 5.0)]
    df["rating"] = df["rating"].astype(float)

    # Chuyển timestamp epoch -> datetime UTC (nếu lỗi => NaT)
    def convert_timestamp(ts):
        try:
            return pd.to_datetime(ts, unit="s", utc=True)
        except Exception:
            return pd.NaT

    df["timestamp"] = df["timestamp"].apply(convert_timestamp)

    # Ghi ra file parquet
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    # In thông tin kiểm tra nhanh
    print("✅ ratings.cleaned.parquet saved:", out_path)
    print("rating.min():", df["rating"].min(), "| rating.max():", df["rating"].max())
    print("10 dòng mẫu:")
    print(df.head(10))



if __name__ == "__main__":
    clean_ratings()
