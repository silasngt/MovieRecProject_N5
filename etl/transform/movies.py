import re
import pandas as pd
from pathlib import Path


def clean_movies():
    """
    Làm sạch dữ liệu movies.csv
    - Tách title và year
    - Chuyển genres thành danh sách
    """
    # Đường dẫn file
    raw_path = "etl/raw/movies.csv"
    out_path = "etl/intermediate/movies.cleaned.parquet"
    
    # Đọc dữ liệu
    print(f"Đọc dữ liệu từ: {raw_path}")
    df = pd.read_csv(raw_path)
    print(f"Số dòng ban đầu: {len(df)}")
    
    # Loại bỏ duplicate movieId
    duplicates = df.duplicated(subset=['movieId']).sum()
    if duplicates > 0:
        print(f"  Tìm thấy {duplicates} movieId trùng lặp, giữ lại dòng đầu tiên")
        df = df.drop_duplicates(subset=['movieId'], keep='first')
    
    # Tách title và year bằng regex
    print("Đang tách title và year...")
    pattern = r'^(.*)\s\((\d{4})\)$'
    
    def parse_title_year(title):
        match = re.match(pattern, title)
        if match:
            title_clean = match.group(1).strip()
            year = int(match.group(2))
            # Validate year >= 1900
            if year >= 1900:
                return title_clean, year
        # Không khớp pattern hoặc year < 1900
        return title, None
    
    parsed = df['title'].apply(parse_title_year)
    df['title_clean'] = parsed.apply(lambda x: x[0])
    df['year'] = parsed.apply(lambda x: x[1])
    
    # Chuyển genres thành list
    print("Đang chuyển genres thành danh sách...")
    
    def parse_genres(genres):
        if genres == "(no genres listed)":
            return []
        # Split bởi | và loại bỏ chuỗi rỗng
        return [g.strip() for g in genres.split('|') if g.strip()]
    
    df['genres_list'] = df['genres'].apply(parse_genres)
    
    # Chuyển đổi kiểu dữ liệu
    df['movieId'] = df['movieId'].astype(int)
    df['year'] = df['year'].astype('Int64')  # Int64 cho phép null
    
    # Chọn các cột cần thiết
    df_clean = df[['movieId', 'title_clean', 'year', 'genres_list']].copy()
    
    # Thống kê
    null_years = df_clean['year'].isnull().sum()
    null_year_pct = (null_years / len(df_clean)) * 100
    
    print(f"\n Kết quả:")
    print(f"  - Số dòng đầu ra: {len(df_clean)}")
    print(f"  - Số phim không có year: {null_years} ({null_year_pct:.2f}%)")
    
    if null_year_pct >= 5:
        print(f"    Cảnh báo: Tỷ lệ thiếu year >= 5%")
    
    # Tạo thư mục output nếu chưa có
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Lưu file Parquet
    print(f"\n Lưu dữ liệu vào: {out_path}")
    df_clean.to_parquet(out_path, index=False)
    print(" Hoàn thành!")
    
    return df_clean


if __name__ == "__main__":
    clean_movies()