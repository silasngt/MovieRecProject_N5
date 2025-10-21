# etl/sanity/check_basic_quality.py
# ------------------------------------------------------------
# Sanity checks cơ bản cho dữ liệu cleaned (.parquet) trước khi dùng cho ML
# Đầu vào:  etl/intermediate/movies.cleaned.parquet
#           etl/intermediate/ratings.cleaned.parquet
#           etl/intermediate/links.cleaned.parquet
# Đầu ra:   etl/sanity/sanity_report.txt
# Quy tắc kiểm: duplicates, range/định dạng, coverage các giá trị thiếu
# ------------------------------------------------------------

from pathlib import Path                  # Quản lý đường dẫn theo cách an toàn hệ điều hành
import pandas as pd                       # Xử lý dữ liệu bảng
import re                                  # Kiểm tra định dạng chuỗi bằng regex
from datetime import datetime              # Lấy năm hiện tại để kiểm tra year

# --------- Cấu hình đường dẫn I/O ---------
ROOT = Path(__file__).resolve().parents[2]       # Thư mục gốc dự án (đi lên 2 lần: sanity -> etl -> root)
INTERMEDIATE = ROOT / "etl" / "intermediate"     # Nơi chứa các file parquet cleaned
REPORTS_DIR = ROOT / "etl" / "reports"             # Thư mục để xuất báo cáo sanity
REPORTS_DIR.mkdir(parents=True, exist_ok=True)    # Tạo thư mục nếu chưa có
REPORT_PATH = REPORTS_DIR / "sanity_report.txt"   # File báo cáo đầu ra

# --------- Hàm tiện ích để ghi log vào file ---------
def writeln(f, text=""):
    """Ghi 1 dòng vào file báo cáo (và đảm bảo xuống dòng)."""
    f.write(str(text) + "\n")

# --------- Hàm kiểm tra duplicates ---------
def check_duplicates_ratings(df):
    """
    ratings: không có duplicate theo cặp (userId, movieId).
    Trả về (is_ok, message, count_dup)
    """
    # duplicated(subset=...) -> True cho các dòng bị trùng theo cặp chỉ định
    dup_mask = df.duplicated(subset=["userId", "movieId"], keep=False)
    count_dup = int(dup_mask.sum())
    is_ok = count_dup == 0
    msg = "✅ No duplicate (userId, movieId)" if is_ok else f"⚠️ Found {count_dup} duplicated rows by (userId, movieId)"
    return is_ok, msg, count_dup

def check_unique(series, name):
    """
    Kiểm tra cột là unique (không trùng khoá).
    Trả về (is_ok, message, count_dup)
    """
    # is_unique là thuộc tính của Pandas Series cho biết cột có unique không
    is_ok = series.is_unique
    count_dup = 0 if is_ok else int(len(series) - len(series.drop_duplicates()))
    msg = f"✅ {name} unique" if is_ok else f"⚠️ {name} has {count_dup} duplicate(s)"
    return is_ok, msg, count_dup

# --------- Hàm kiểm tra range & định dạng ---------
def check_rating_range(df):
    """
    0.5 ≤ rating ≤ 5.0
    Trả về (is_ok, message, count_out)
    """
    out_mask = ~df["rating"].between(0.5, 5.0)     # between: nằm trong khoảng [0.5,5.0]
    count_out = int(out_mask.sum())
    is_ok = count_out == 0
    msg = "✅ Rating range OK (0.5–5.0)" if is_ok else f"⚠️ {count_out} rating(s) out of range [0.5, 5.0]"
    return is_ok, msg, count_out

def check_year_range(df):
    """
    year (nếu có) ≥ 1900 và ≤ năm hiện tại + 1
    Trả về (is_ok, message, count_bad, count_null)
    """
    current_year = datetime.utcnow().year + 1     # Cho phép tới năm hiện tại + 1
    # year có thể null: dùng dropna để chỉ kiểm trên giá trị có dữ liệu
    non_null = df["year"].dropna()
    bad_mask = (non_null < 1900) | (non_null > current_year)
    count_bad = int(bad_mask.sum())
    count_null = int(df["year"].isna().sum())
    is_ok = count_bad == 0
    if is_ok:
        msg = f"✅ Year range OK (nulls: {count_null})"
    else:
        msg = f"⚠️ {count_bad} year(s) out of range; nulls: {count_null}"
    return is_ok, msg, count_bad, count_null

def check_imdb_format(df):
    """
    imdbId_tt null hoặc khớp regex ^tt\\d{7,8}$
    Trả về (is_ok, message, count_bad, count_null)
    """
    pattern = re.compile(r"^tt\d{7,8}$")       # Mẫu hợp lệ cho IMDb id dạng chuỗi
    is_null = df["imdbId_tt"].isna()
    # Với giá trị không null: kiểm tra khớp regex
    non_null = df.loc[~is_null, "imdbId_tt"].astype(str)
    bad_mask = ~non_null.str.match(pattern)
    count_bad = int(bad_mask.sum())
    count_null = int(is_null.sum())
    is_ok = count_bad == 0
    msg = "✅ imdbId_tt format OK" if is_ok else f"⚠️ {count_bad} imdbId_tt invalid format; nulls: {count_null}"
    return is_ok, msg, count_bad, count_null

# --------- Hàm tính coverage (tỉ lệ thiếu) ---------
def coverage_missing(series, label):
    """
    Tính số lượng & tỉ lệ phần trăm giá trị null trên 1 cột.
    Trả về (count_null, ratio)
    """
    total = len(series)
    count_null = int(series.isna().sum())
    ratio = (count_null / total * 100.0) if total > 0 else 0.0
    return count_null, ratio

# --------- Hàm chính chạy sanity ---------
def main():
    # Mở file báo cáo để ghi kết quả
    with open(REPORT_PATH, "w", encoding="utf-8") as rep:
        writeln(rep, "===== Sanity Check Report =====")
        writeln(rep, "")

        # ---------- 1) Kiểm tra ratings.cleaned ----------
        ratings_path = INTERMEDIATE / "ratings.cleaned.parquet"
        if ratings_path.exists():
            df_ratings = pd.read_parquet(ratings_path)   # Đọc parquet vào DataFrame
            writeln(rep, "[ratings.cleaned]")
            # Kiểm tra duplicates theo (userId, movieId)
            ok_dup, msg_dup, _ = check_duplicates_ratings(df_ratings)
            writeln(rep, msg_dup)
            # Kiểm tra range rating
            ok_rng, msg_rng, _ = check_rating_range(df_ratings)
            writeln(rep, msg_rng)
            # Kiểm tra timestamp có cột hay không (không bắt buộc – chỉ log số null nếu có)
            if "timestamp" in df_ratings.columns:
                n_null_ts, ratio_ts = coverage_missing(df_ratings["timestamp"], "timestamp")
                writeln(rep, f"ℹ️ timestamp nulls: {n_null_ts} ({ratio_ts:.2f}%)")
            writeln(rep, "")
        else:
            writeln(rep, "[ratings.cleaned] ❌ Missing file")
            writeln(rep, "")

        # ---------- 2) Kiểm tra movies.cleaned ----------
        movies_path = INTERMEDIATE / "movies.cleaned.parquet"
        if movies_path.exists():
            df_movies = pd.read_parquet(movies_path)
            writeln(rep, "[movies.cleaned]")
            # movieId phải unique
            ok_unique, msg_unique, _ = check_unique(df_movies["movieId"], "movieId")
            writeln(rep, msg_unique)
            # Kiểm tra range year và số lượng null
            ok_year, msg_year, _, count_null_year = check_year_range(df_movies)
            writeln(rep, msg_year)
            # Coverage: tỉ lệ year bị null
            cnt_year_null, ratio_year_null = coverage_missing(df_movies["year"], "year")
            writeln(rep, f"ℹ️ year nulls: {cnt_year_null} ({ratio_year_null:.2f}%)")
            writeln(rep, "")
        else:
            writeln(rep, "[movies.cleaned] ❌ Missing file")
            writeln(rep, "")

        # ---------- 3) Kiểm tra links.cleaned ----------
        links_path = INTERMEDIATE / "links.cleaned.parquet"
        if links_path.exists():
            df_links = pd.read_parquet(links_path)
            writeln(rep, "[links.cleaned]")
            # movieId phải unique
            ok_unique_l, msg_unique_l, _ = check_unique(df_links["movieId"], "movieId")
            writeln(rep, msg_unique_l)
            # imdbId_tt null hoặc đúng regex
            ok_imdb, msg_imdb, _, count_null_imdb = check_imdb_format(df_links)
            writeln(rep, msg_imdb)
            # Coverage: % tmdbId bị null
            cnt_tmdb_null, ratio_tmdb_null = coverage_missing(df_links["tmdbId"], "tmdbId")
            writeln(rep, f"ℹ️ tmdbId nulls: {cnt_tmdb_null} ({ratio_tmdb_null:.2f}%)")
            writeln(rep, "")
        else:
            writeln(rep, "[links.cleaned] ❌ Missing file")
            writeln(rep, "")

        # ---------- 4) Kết luận tổng hợp ----------
        writeln(rep, "Summary:")
        # Đếm số cảnh báo trong toàn bộ file report bằng cách đơn giản: đọc lại nội dung vừa ghi
    # Mở lại file để đếm số dòng có ký hiệu cảnh báo "⚠️"
    text = REPORT_PATH.read_text(encoding="utf-8")
    warnings = text.count("⚠️")
    with open(REPORT_PATH, "a", encoding="utf-8") as rep:
        writeln(rep, f"{'No warnings' if warnings==0 else f'{warnings} warning(s) detected'}")
        writeln(rep, "-" * 36)
        writeln(rep, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    # In ra màn hình vị trí file báo cáo + số cảnh báo
    print(f"[sanity] Wrote report -> {REPORT_PATH}")
    print(f"[sanity] Warnings: {warnings}")

# Điểm vào chương trình
if __name__ == "__main__":
    main()
