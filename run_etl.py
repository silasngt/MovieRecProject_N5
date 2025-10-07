# run_etl.py
"""
Chạy pipeline ETL tuần 1 theo thứ tự:
1) transform/movies.py       -> clean_movies()
2) transform/ratings.py      -> clean_ratings()
3) transform/links.py        -> clean_links()
4) schemas/validate_and_profile.py -> validate_and_profile()
5) load/load_to_mongo.py     -> load_to_mongo()
"""

def main():
    # Import tại thời điểm chạy để tránh lỗi khi file skeleton chưa có nội dung
    # from transform.movies import clean_movies
    # from transform.ratings import clean_ratings
    # from transform.links import clean_links
    # from schemas.validate_and_profile import validate_and_profile
    # from load.load_to_mongo import load_to_mongo

    print("▶ 1/5 Transform movies")
    # clean_movies()

    print("▶ 2/5 Transform ratings")
    # clean_ratings()

    print("▶ 3/5 Transform links")
    # clean_links()

    print("▶ 4/5 Validate & profile")
    # validate_and_profile()

    print("▶ 5/5 Load to MongoDB")
    # load_to_mongo()

    print("✅ Pipeline ETL hoàn tất.")


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as e:
        print("❌ Không tìm thấy module. Kiểm tra lại cấu trúc thư mục:")
        print("   transform/, schemas/, load/ phải cùng cấp với run_etl.py")
        print("Chi tiết lỗi:", e)
    except Exception as e:
        print("❌ ETL lỗi. Chi tiết:")
        print(e)
