# etl/schemas/validate_and_profile.py
# ------------------------------------------------------------
# Kiểm tra Schema + Profile dữ liệu cleaned trước khi đem đi Sanity/Load/ML
# Đầu vào : etl/intermediate/*.parquet
# Đầu ra  : etl/reports/validation_report.json, etl/reports/profile_summary.csv
# Yêu cầu : Khớp contract, khóa & nulls, giá trị hợp lệ, báo cáo tổng quan
# Lưu ý   : Thiếu tmdbId chỉ WARNING, không fail pipeline
# ------------------------------------------------------------

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json
import re

# ============ ĐƯỜNG DẪN ============
ROOT = Path(__file__).resolve().parents[2]     # .../MovieRecProject_N5
INTERMEDIATE = ROOT / "etl" / "intermediate"   # nơi chứa parquet cleaned
REPORTS = ROOT / "etl" / "reports"             # nơi ghi báo cáo
REPORTS.mkdir(parents=True, exist_ok=True)

VALIDATION_JSON = REPORTS / "validation_report.json"
PROFILE_CSV = REPORTS / "profile_summary.csv"

# ============ CÔNG CỤ HỖ TRỢ ============
def safe_read_parquet(path: Path) -> pd.DataFrame:
    """Đọc parquet và ném lỗi nếu thiếu file để người dùng biết rõ."""
    if not path.exists():
        raise FileNotFoundError(f"Thiếu file: {path}")
    return pd.read_parquet(path)

def is_list_like(val):
    """Xác định giá trị có phải list/tuple không (để check genres_list)."""
    return isinstance(val, (list, tuple))

def pct(x, total):
    return 0.0 if total == 0 else round(x / total * 100.0, 2)

# ============ CHECK CONTRACT ============
def validate_movies(df: pd.DataFrame) -> dict:
    """
    Contract:
      movieId:int (unique),
      title_clean:str (cho phép null),
      year:int|null (>=1900),
      genres_list:list[str]
    """
    report = {}
    rows = len(df)

    # Tồn tại cột?
    expected = ["movieId", "title_clean", "year", "genres_list"]
    missing_cols = [c for c in expected if c not in df.columns]
    report["missing_columns"] = missing_cols

    # Nulls từng cột
    nulls = {c: int(df[c].isna().sum()) if c in df.columns else rows for c in expected}
    report["missing_values"] = nulls

    # movieId unique?
    if "movieId" in df.columns:
        report["duplicate_keys"] = int(rows - df["movieId"].nunique())
    else:
        report["duplicate_keys"] = rows  # nếu không có cột thì xem như fail nặng

    # year hợp lệ (nếu có)
    current_year = datetime.utcnow().year + 1
    if "year" in df.columns:
        non_null_year = df["year"].dropna()
        bad_year = int(((non_null_year < 1900) | (non_null_year > current_year)).sum())
        report["invalid_year_range"] = bad_year
    else:
        report["invalid_year_range"] = rows

    # genres_list là list ở phần lớn bản ghi (không bắt buộc tuyệt đối)
    if "genres_list" in df.columns and rows > 0:
        sample = df["genres_list"].dropna().head(50)
        ok_list = int(sample.apply(is_list_like).sum())
        report["genres_list_listlike_in_sample"] = ok_list  # kỳ vọng ~ số mẫu
    else:
        report["genres_list_listlike_in_sample"] = 0

    # Đánh giá tổng thể
    # - Thiếu cột hoặc duplicate_keys>0 hoặc invalid_year_range>0 => WARNING/FAIL
    if missing_cols:
        schema = "FAIL"
    elif report["duplicate_keys"] > 0:
        schema = "FAIL"
    elif report["invalid_year_range"] > 0:
        schema = "WARNING"
    else:
        schema = "PASSED"

    report["rows"] = rows
    report["schema_check"] = schema
    return report

def validate_ratings(df: pd.DataFrame) -> dict:
    """
    Contract:
      userId:int (non-null),
      movieId:int (non-null),
      rating:float in [0.5,5.0],
      timestamp:datetime|null
    """
    report = {}
    rows = len(df)

    expected = ["userId", "movieId", "rating", "timestamp"]
    missing_cols = [c for c in expected if c not in df.columns]
    report["missing_columns"] = missing_cols

    # Nulls
    null_user = int(df["userId"].isna().sum()) if "userId" in df.columns else rows
    null_movie = int(df["movieId"].isna().sum()) if "movieId" in df.columns else rows
    null_rating = int(df["rating"].isna().sum()) if "rating" in df.columns else rows
    null_ts = int(df["timestamp"].isna().sum()) if "timestamp" in df.columns else rows
    report["missing_values"] = {
        "userId": null_user, "movieId": null_movie, "rating": null_rating, "timestamp": null_ts
    }

    # Giá trị hợp lệ
    if "rating" in df.columns:
        invalid_ratings = int((~df["rating"].between(0.5, 5.0)).sum())
    else:
        invalid_ratings = rows

    # Timestamp format: nếu có cột, thử convert 10 dòng đầu (best-effort)
    timestamp_format_errors = 0
    if "timestamp" in df.columns:
        sample = df["timestamp"].dropna().head(10)
        try:
            pd.to_datetime(sample, errors="raise", utc=True)
        except Exception:
            timestamp_format_errors = len(sample) or 1  # nếu convert lỗi, đánh dấu >0

    report["invalid_ratings"] = invalid_ratings
    report["timestamp_format_errors"] = timestamp_format_errors

    # Đánh giá
    if missing_cols:
        schema = "FAIL"
    elif null_user > 0 or null_movie > 0:
        schema = "FAIL"
    elif invalid_ratings > 0:
        schema = "FAIL"
    else:
        schema = "PASSED"

    report["rows"] = rows
    report["schema_check"] = schema
    return report

def validate_links(df: pd.DataFrame) -> dict:
    """
    Contract:
      movieId:int (unique),
      imdbId_tt:str|null (regex ^tt\\d{7,8}$),
      tmdbId:int|null   (thiếu -> WARNING, không fail)
    """
    report = {}
    rows = len(df)

    expected = ["movieId", "imdbId_tt", "tmdbId"]
    missing_cols = [c for c in expected if c not in df.columns]
    report["missing_columns"] = missing_cols

    # movieId unique?
    if "movieId" in df.columns:
        report["duplicate_keys"] = int(rows - df["movieId"].nunique())
    else:
        report["duplicate_keys"] = rows

    # imdb regex
    invalid_imdb = 0
    if "imdbId_tt" in df.columns:
        nn = df["imdbId_tt"].dropna().astype(str)
        invalid_imdb = int((~nn.str.match(r"^tt\d{7,8}$")).sum())
    else:
        invalid_imdb = rows

    # tmdbId missing (chỉ WARNING)
    if "tmdbId" in df.columns:
        missing_tmdb = int(df["tmdbId"].isna().sum())
    else:
        missing_tmdb = rows

    report["invalid_imdb_format"] = invalid_imdb
    report["missing_tmdbId"] = missing_tmdb
    report["rows"] = rows

    # Đánh giá:
    #  - Thiếu cột, duplicate movieId, hoặc imdb sai format -> FAIL
    #  - tmdbId thiếu -> WARNING
    if missing_cols or report["duplicate_keys"] > 0 or invalid_imdb > 0:
        schema = "FAIL"
    elif missing_tmdb > 0:
        schema = "WARNING"
    else:
        schema = "PASSED"

    report["schema_check"] = schema
    return report

# ============ PROFILE (TỔNG QUAN) ============
def profile_block(name: str, df: pd.DataFrame) -> dict:
    """Sinh thống kê tổng quan cho 1 bảng (đưa vào hàng của profile_summary.csv)."""
    row = {
        "file_name": name,
        "rows": len(df),
        "columns": df.shape[1],
        "duplicate_rows": int(len(df) - len(df.drop_duplicates())),
    }
    # Null tổng
    null_total = int(df.isna().sum().sum())
    row["null_values_total"] = null_total

    # Thống kê numeric cơ bản (nếu có)
    num_cols = df.select_dtypes(include=["number", "float", "int"]).columns.tolist()
    if "rating" in df.columns:
        row["avg_rating"] = round(float(df["rating"].mean()), 3)
        row["min_rating"] = float(df["rating"].min())
        row["max_rating"] = float(df["rating"].max())
    if "year" in df.columns:
        # dùng dropna để tránh NaN
        y = df["year"].dropna()
        row["min_year"] = int(y.min()) if not y.empty else None
        row["max_year"] = int(y.max()) if not y.empty else None
    row["numeric_cols"] = ";".join(num_cols)
    return row

# ============ MAIN ============
def main():
    # Đọc dữ liệu
    movies = safe_read_parquet(INTERMEDIATE / "movies.cleaned.parquet")
    ratings = safe_read_parquet(INTERMEDIATE / "ratings.cleaned.parquet")
    links = safe_read_parquet(INTERMEDIATE / "links.cleaned.parquet")

    # Validate theo contract
    movies_rep = validate_movies(movies)
    ratings_rep = validate_ratings(ratings)
    links_rep = validate_links(links)

    validation_report = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "movies.cleaned": movies_rep,
        "ratings.cleaned": ratings_rep,
        "links.cleaned": links_rep,
    }

    # Ghi JSON
    with open(VALIDATION_JSON, "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2, ensure_ascii=False)

    # Profile summary CSV
    rows = [
        profile_block("movies.cleaned", movies),
        profile_block("ratings.cleaned", ratings),
        profile_block("links.cleaned", links),
    ]
    pd.DataFrame(rows).to_csv(PROFILE_CSV, index=False, encoding="utf-8")

    # In console tóm tắt
    print(f"[schema] Wrote: {VALIDATION_JSON}")
    print(f"[schema] Wrote: {PROFILE_CSV}")
    print("[schema] Status:",
          "movies:", movies_rep["schema_check"],
          "| ratings:", ratings_rep["schema_check"],
          "| links:", links_rep["schema_check"])

if __name__ == "__main__":
    main()
