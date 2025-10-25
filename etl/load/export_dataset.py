# ------------------------------------------------------------
# Nhiệm vụ:
#   - Đọc dữ liệu cleaned từ etl/intermediate/*.parquet
#   - Gộp thông tin ratings + movies + links
#   - Tính trung bình, số lượng, độ lệch chuẩn rating theo movie
#   - Lấy năm phát hành, thể loại đầu tiên làm label
#   - Xuất thành CSV tại etl/datasets/movie_features.csv
# ------------------------------------------------------------

from pathlib import Path
import pandas as pd
import numpy as np

# --------- Đường dẫn ---------
ROOT = Path(__file__).resolve().parents[2]         # thư mục gốc
INTERMEDIATE = ROOT / "etl" / "intermediate"
DATASETS = ROOT / "etl" / "datasets"
DATASETS.mkdir(parents=True, exist_ok=True)
OUTPUT = DATASETS / "movie_features.csv"

def export_dataset():
    print("[load] Bắt đầu gộp dữ liệu từ parquet...")

    # 1️⃣ Đọc dữ liệu parquet
    movies_path = INTERMEDIATE / "movies.cleaned.parquet"
    ratings_path = INTERMEDIATE / "ratings.cleaned.parquet"
    links_path = INTERMEDIATE / "links.cleaned.parquet"

    if not all(p.exists() for p in [movies_path, ratings_path, links_path]):
        raise FileNotFoundError("❌ Thiếu 1 trong 3 file parquet cần thiết (movies, ratings, links).")

    movies = pd.read_parquet(movies_path)
    ratings = pd.read_parquet(ratings_path)
    links = pd.read_parquet(links_path)

    print(f"[load] movies: {movies.shape}, ratings: {ratings.shape}, links: {links.shape}")

    # 2️⃣ Tính toán đặc trưng rating theo movieId
    rating_stats = ratings.groupby("movieId").agg(
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count"),
        rating_std=("rating", "std")
    ).reset_index()

    # Điền giá trị thiếu (std có thể bị NaN khi chỉ có 1 rating)
    rating_stats["rating_std"] = rating_stats["rating_std"].fillna(0)

    print(f"[load] rating_stats: {rating_stats.shape}")
    print(rating_stats.head(3))

    # 3️⃣ Lấy thông tin cơ bản từ movies
    # Chọn các cột cần: movieId, year, genres_list
    if not set(["movieId", "year", "genres_list"]).issubset(movies.columns):
        print("[load] ❗ Cảnh báo: movies không có đúng các cột 'movieId', 'year', 'genres_list'. Các cột hiện có:", movies.columns.tolist())

    movies_subset = movies[["movieId", "year", "genres_list"]].copy()

    # Tạo cột label_genre = thể loại đầu tiên (nếu có)
    def first_genre(genres):
        # 1) None hoặc scalar NA
        if genres is None:
            return np.nan
        # nếu là scalar NaN (pandas / numpy scalar)
        try:
            # pd.isna on scalar returns bool; on array returns array -> we guard with isinstance checks below
            if not hasattr(genres, "__iter__") and pd.isna(genres):
                return np.nan
        except Exception:
            pass

        # 2) Nếu là string -> xử lý chuỗi
        if isinstance(genres, str):
            s = genres.strip()
            if s == "":
                return np.nan
            # nếu chuỗi có ngoặc vuông/nhỏ -> loại ngoặc rồi split theo comma
            if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
                inner = s[1:-1].strip()
                if inner == "":
                    return np.nan
                parts = [p.strip().strip("\"'") for p in inner.split(",") if p.strip()]
                return parts[0] if parts else np.nan
            if "|" in s:
                parts = [p.strip() for p in s.split("|") if p.strip()]
                return parts[0] if parts else np.nan
            if "," in s:
                parts = [p.strip().strip("\"'") for p in s.split(",") if p.strip()]
                return parts[0] if parts else np.nan
            return s

        # 3) Nếu là iterable (list/tuple/ndarray/Series...), chuyển về list và lấy phần tử đầu
        if isinstance(genres, (list, tuple, np.ndarray, pd.Series)):
            try:
                seq = list(genres)
            except Exception:
                return np.nan
            if len(seq) == 0:
                return np.nan
            first = seq[0]
            if pd.isna(first):
                return np.nan
            if isinstance(first, str):
                return first.strip().strip("\"'")
            return first

        # 4) Fallback: chuyển sang string rồi xử lý như trên
        try:
            s = str(genres).strip()
            if s == "" or s.lower() == "nan":
                return np.nan
            return s
        except Exception:
            return np.nan

    movies_subset["label_genre"] = movies_subset["genres_list"].apply(first_genre)

    # debug info: types distribution of genres_list
    try:
        types_counts = movies_subset["genres_list"].apply(lambda x: type(x).__name__).value_counts().to_dict()
        print(f"[load] types in genres_list (sample): {types_counts}")
    except Exception:
        pass

    print(f"[load] movies_subset: {movies_subset.shape}")
    print(movies_subset.head(3))

    # 4️⃣ Gộp rating_stats + movies + links
    merged = (
        rating_stats
        .merge(movies_subset, on="movieId", how="left")
        .merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    )

    print(f"[load] merged trước khi xử lý: {merged.shape}")
    print(merged.head(3))

    # 5️⃣ Chuyển đổi kiểu dữ liệu & xử lý thiếu
    merged["avg_rating"] = merged["avg_rating"].round(2)
    merged["rating_std"] = merged["rating_std"].round(2)
    merged["rating_count"] = merged["rating_count"].astype(int)

    # Kiểm tra số hàng bị thiếu year / label_genre trước khi dropna
    missing_year = merged["year"].isna().sum()
    missing_genre = merged["label_genre"].isna().sum()
    total_before = merged.shape[0]
    print(f"[load] Trước dropna: tổng {total_before} dòng, missing year = {missing_year}, missing label_genre = {missing_genre}")

    # Lọc bỏ các dòng không có year hoặc genre (nếu muốn “dữ liệu sạch” cho ML)
    merged = merged.dropna(subset=["year", "label_genre"])
    total_after = merged.shape[0]
    print(f"[load] Sau dropna: tổng {total_after} dòng. Đã loại {total_before - total_after} dòng.")

    # 6️⃣ Ghi ra CSV
    merged.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f"[load] ✅ Xuất thành công -> {OUTPUT}")
    print(f"[load] {merged.shape[0]} dòng, {merged.shape[1]} cột")

if __name__ == "__main__":
    export_dataset()