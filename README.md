MovieRecProject/
│
├── etl/ # Khu vực xử lý dữ liệu (ETL)
│ ├── raw/ # CSV gốc (movies.csv, ratings.csv, links.csv)
│ ├── intermediate/ # file cleaned (parquet, JSON tạm)
│ ├── reports/ # báo cáo profiling / summary
│ ├── schemas/ # schema mô tả dữ liệu
│ ├── load/ # script nạp MongoDB
│ ├── sanity/ # kiểm thử truy vấn cơ bản
│ └── transform/ # scripts từng người phụ trách cleaning
│
├── .vscode/ # settings chung VSCode
├── .gitignore
├── requirements.txt
├── README.md
└── run_etl.py # script tổng gọi lần lượt các bước

# 1️⃣ Clone repo

git clone https://github.com/silasngt/MovieRecProject_N5.git
cd MovieRecProject_N5

# 2️⃣ Tạo môi trường ảo & cài đặt thư viện cơ bản

python -m venv .venv
.\.venv\Scripts\activate # Windows

# source .venv/bin/activate # macOS/Linux

pip install pandas kaggle

# 3️⃣ Tải dataset MovieLens Small Latest (tự động)

# Script này sẽ tự động tải từ Kaggle và giải nén vào etl/raw/

python scripts/fetch_data.py

# 4️⃣ Kiểm tra lại các file đã được tải

dir etl\raw # Windows

# hoặc

ls etl/raw # macOS/Linux
