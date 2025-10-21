import pandas as pd
from pathlib import Path
import re

def clean_links():
    """
    Làm sạch dữ liệu links.csv
    - Chuẩn hoá imdbId thành định dạng tt#######
    - Giữ các cột: movieId, imdbId_tt, tmdbId
    - Đảm bảo movieId là duy nhất
    - Không loại bỏ dòng thiếu tmdbId
    """

    raw_path = Path("etl/raw/links.csv")
    out_path = Path("etl/intermediate/links.cleaned.parquet")

    # Đọc dữ liệu gốc
    df = pd.read_csv(raw_path)

    # Đảm bảo movieId là số nguyên
    df["movieId"] = pd.to_numeric(df["movieId"], errors="coerce").astype("Int64")

    # --- Chuẩn hóa imdbId ---
    def normalize_imdb(imdb):
        try:
            imdb_str = str(int(imdb))  # bỏ .0 nếu có
            if re.fullmatch(r"\d{7,8}", imdb_str):
                return f"tt{imdb_str}"
            else:
                return None
        except Exception:
            return None

    df["imdbId_tt"] = df["imdbId"].apply(normalize_imdb)

    # --- Chuẩn hóa tmdbId ---
    df["tmdbId"] = pd.to_numeric(df["tmdbId"], errors="coerce").astype("Int64")

    # --- Loại bỏ trùng movieId (giữ dòng đầu tiên) ---
    before = len(df)
    df = df.drop_duplicates(subset=["movieId"], keep="first")
    after = len(df)
    if before != after:
        print(f"⚠️  {before - after} dòng trùng movieId đã được loại bỏ.")

    # --- Chỉ giữ 3 cột cần thiết ---
    df = df[["movieId", "imdbId_tt", "tmdbId"]]

    # --- Ghi ra file parquet ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    # --- Kiểm thử nhanh ---
    print("✅ links.cleaned.parquet saved:", out_path)
    print("Số dòng:", len(df))
    print("movieId duy nhất:", df["movieId"].is_unique)
    print("20 dòng mẫu:")
    print(df.head(20))
    print("Kiểm tra imdbId_tt hợp lệ:",
        df["imdbId_tt"].dropna().apply(lambda x: bool(re.fullmatch(r"tt\d{7,8}", x))).all())

if __name__ == "__main__":
    clean_links()