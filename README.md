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
