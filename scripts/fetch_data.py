# scripts/fetch_data.py
import os, zipfile, io, sys, urllib.request, pathlib

# ✅ URL Kaggle Dataset (Movie Lens Small Latest)
URL = "https://www.kaggle.com/api/v1/datasets/download/shubhammehta21/movie-lens-small-latest-dataset"

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "etl" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

def main():
    print("📥 Đang tải Movie Lens Small Latest Dataset từ Kaggle...")

    print("⚠️ Lưu ý: Script này yêu cầu bạn đã đăng nhập Kaggle CLI.")
    print("   (Nếu chưa cài đặt, chạy: pip install kaggle)")
    print("   Sau đó tạo file kaggle.json trong ~/.kaggle/ hoặc C:\\Users\\<user>\\.kaggle\\)")

    try:
        os.system(f"kaggle datasets download -d shubhammehta21/movie-lens-small-latest-dataset -p {RAW} --unzip")
    except Exception as e:
        print("❌ Lỗi khi tải từ Kaggle:", e)
        sys.exit(1)

    print("✅ Đã tải xong dataset. Kiểm tra thư mục etl/raw/ để thấy các file CSV.")

if __name__ == "__main__":
    sys.exit(main())
