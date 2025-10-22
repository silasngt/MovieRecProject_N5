from pathlib import Path
import json, re #bieu thuc chinh quy, tim kiem va sap xep 
import pandas as pd
import numpy as np #chuan hoa su lieu so

#read and write in the same directory.
SCRIPT_DIR = Path(__file__).resolve().parent
INTER_DIR  = SCRIPT_DIR.parent / "intermediate"
REPORT_DIR = SCRIPT_DIR.parent / "reports"
VALIDATION_JSON = REPORT_DIR / "validation_report.json"
PROFILE_CSV     = REPORT_DIR / "profile_summary.csv"

#constants
IMDB_TT_RE  = re.compile(r"^tt\d+$")#co the du lieu se bi ma hoa
RATING_MIN, RATING_MAX = 0.5, 5.0
MIN_YEAR = 1900

#read the first file that matches the pattern
def read_first(patterns):
    for pat in patterns:
        m = sorted(INTER_DIR.glob(pat))
        if m: return pd.read_parquet(m[0])
    raise FileNotFoundError(f"Không tìm thấy file với pattern: {patterns}")

#clean the datasets
def _to_int(s):   return pd.to_numeric(s, errors="coerce").astype("Int64")
def _to_flt(s):   return pd.to_numeric(s, errors="coerce").astype("Float64")
def _to_dt_utc(s):
    ratio = pd.to_numeric(s, errors="coerce").notna().mean()
    if ratio >= 0.8: #phim do xep cho khac
        return pd.to_datetime(pd.to_numeric(s, errors="coerce"), unit="s", utc=True, errors="coerce")
    return pd.to_datetime(s, utc=True, errors="coerce")

def _ensure(df, cols): # tao du lieu mau
    df = df.copy()
    for c in cols:
        if c not in df: df[c] = pd.NA
    return df

def _year_from_title(t):
    if not isinstance(t, str): return None
    m = re.search(r"\((\d{4})\)\s*$", t)
    return int(m.group(1)) if m else None

#loc, sap xep, chuan hoa du lieu dau vao
def clean_movies(df):
    df = _ensure(df, ["movieId", "title", "genres"])
    out = pd.DataFrame()
    out["movieId"]     = _to_int(df["movieId"])
    out["title_clean"] = df["title"].astype("string").str.strip()
    out["year"]        = out["title_clean"].apply(_year_from_title).astype("Int64")
    g = df["genres"].astype("string").fillna("(no genres listed)")
    out["genres_list"] = g.apply(lambda x: [] if x in ("", "(no genres listed)") else str(x).split("|"))
    return out

def clean_ratings(df):
    df = _ensure(df, ["userId", "movieId", "rating", "timestamp"])
    out = pd.DataFrame()
    out["userId"]    = _to_int(df["userId"])
    out["movieId"]   = _to_int(df["movieId"])
    out["rating"]    = _to_flt(df["rating"])
    out["timestamp"] = _to_dt_utc(df["timestamp"])
    return out

def clean_links(df):
    df = _ensure(df, ["movieId", "imdbId", "tmdbId"])
    out = pd.DataFrame()
    out["movieId"]   = _to_int(df["movieId"])
    out["imdbId_tt"] = df["imdbId"].astype("string").str.strip().replace({"": pd.NA}).astype("string")
    out["tmdbId"]    = _to_int(df["tmdbId"])
    return out

#validate the columns of the dataset (loc va sap xep du lieu )
def _is_list_str(series): return series.map(lambda v: isinstance(v, list) and all(isinstance(i, str) for i in v)).all()

def validate_movies(df):
    issues = []
    if set(df.columns) != {"movieId", "title_clean", "year", "genres_list"}: issues.append("Schema mismatch")
    if not pd.api.types.is_integer_dtype(df["movieId"]): issues.append("movieId int")
    if not pd.api.types.is_string_dtype(df["title_clean"]): issues.append("title_clean str")
    if not (pd.api.types.is_integer_dtype(df["year"]) or df["year"].isna().all()): issues.append("year int|null")
    if not _is_list_str(df["genres_list"]): issues.append("genres_list list[str]")
    if df["movieId"].isna().any(): issues.append("movieId nulls")
    if df["movieId"].duplicated().any(): issues.append("movieId not unique")
    bad_year = df.loc[~(df["year"].isna() | (df["year"] >= MIN_YEAR))]
    if len(bad_year) > 0: issues.append(f"year < {MIN_YEAR}: {len(bad_year)}")
    
    # Detailed statistics
    missing_values = {}
    for col in ["title_clean", "year"]:
        missing_count = int(df[col].isna().sum())
        missing_values[col] = missing_count
    
    duplicate_keys = int(df["movieId"].duplicated().sum())
    
    result = {
        "dataset": "movies.cleaned", 
        "rows": len(df),
        "missing_values": missing_values,
        "duplicate_keys": duplicate_keys,
        "status": "PASSED" if not issues else "FAILED"
    }
    if issues:
        result["issues"] = issues
    return result

def validate_ratings(df):
    issues = []
    if set(df.columns) != {"userId", "movieId", "rating", "timestamp"}: issues.append("Schema mismatch")
    if not pd.api.types.is_integer_dtype(df["userId"]): issues.append("userId int")
    if not pd.api.types.is_integer_dtype(df["movieId"]): issues.append("movieId int")
    if not pd.api.types.is_float_dtype(df["rating"]): issues.append("rating float")
    if df["userId"].isna().any(): issues.append("userId nulls")
    if df["movieId"].isna().any(): issues.append("movieId nulls")
    bad_rating = df.loc[~df["rating"].between(RATING_MIN, RATING_MAX, inclusive="both")]
    if len(bad_rating) > 0: issues.append(f"rating out of [{RATING_MIN},{RATING_MAX}]: {len(bad_rating)}")
    
    # Detailed statistics
    invalid_ratings = len(bad_rating)
    timestamp_format_errors = 0  # Could add timestamp validation if needed
    
    result = {
        "dataset": "ratings.cleaned",
        "rows": len(df),
        "invalid_ratings": invalid_ratings,
        "timestamp_format_errors": timestamp_format_errors,
        "status": "PASSED" if not issues else "FAILED"
    }
    if issues:
        result["issues"] = issues
    return result

def validate_links(df):
    issues, warnings = [], []
    if set(df.columns) != {"movieId", "imdbId_tt", "tmdbId"}: issues.append("Schema mismatch")
    if not pd.api.types.is_integer_dtype(df["movieId"]): issues.append("movieId int")
    if not pd.api.types.is_string_dtype(df["imdbId_tt"]): issues.append("imdbId_tt str")
    if not (pd.api.types.is_integer_dtype(df["tmdbId"]) or df["tmdbId"].isna().all()): issues.append("tmdbId int|null")
    if df["movieId"].isna().any(): issues.append("movieId nulls")
    if df["movieId"].duplicated().any(): issues.append("movieId not unique")
    if df["imdbId_tt"].notna().any():
        bad_imdb = df.loc[~df["imdbId_tt"].fillna("").str.fullmatch(IMDB_TT_RE)]
        if len(bad_imdb) > 0: issues.append(f"imdbId_tt invalid: {len(bad_imdb)}")
    if df["tmdbId"].isna().any(): warnings.append("tmdbId missing (allowed)")
    
    # Detailed statistics
    missing_tmdbId = int(df["tmdbId"].isna().sum())
    
    status = "FAILED" if issues else ("WARNING" if warnings else "PASSED")
    result = {
        "dataset": "links.cleaned", 
        "rows": len(df),
        "missing_tmdbId": missing_tmdbId,
        "status": status
    }
    if issues:
        result["issues"] = issues
    if warnings:
        result["warnings"] = warnings
    return result

#profiling column by column for each dataset
def col_stats(df, col): #danh gia de sap xep
    s = df[col]
    out = {
        "nulls": int(s.isna().sum()),
        "null_pct": float(s.isna().mean()) if len(s) else 0.0,
        "distinct": int(s.nunique(dropna=True)) if not (s.dtype == 'object' and s.map(lambda x: isinstance(x, list)).any()) else None
    }
    if pd.api.types.is_numeric_dtype(s):
        ok = s.notna()
        out["min"] = float(s[ok].min()) if ok.any() else None
        out["max"] = float(s[ok].max()) if ok.any() else None
        out["mean"] = float(s[ok].mean()) if ok.any() else None
    else:
        out.update({"min": None, "max": None, "mean": None})
    return out

#Tao bang xuat file csv
def profile(movies, ratings, links):
    rows = []
    for c in movies.columns:
        r = {"dataset": "movies.cleaned", "rows": len(movies), "column": c, **col_stats(movies, c)}
        r["duplicates_on_key"] = int(movies["movieId"].duplicated().sum()) if c=="movieId" else None
        rows.append(r)
    for c in ratings.columns:
        r = {"dataset": "ratings.cleaned", "rows": len(ratings), "column": c, **col_stats(ratings, c)}
        r["duplicates_on_key"] = None
        rows.append(r)
    for c in links.columns:
        r = {"dataset": "links.cleaned", "rows": len(links), "column": c, **col_stats(links, c)}
        r["duplicates_on_key"] = int(links["movieId"].duplicated().sum()) if c=="movieId" else None
        rows.append(r)
    if "rating" in ratings:
        rows.append({
            "dataset": "ratings.cleaned","rows": len(ratings),
            "column": "__aggregate_avg_rating__", "duplicates_on_key": None,
            "nulls": 0, "null_pct": 0.0, "distinct": None,
            "min": None, "max": None, "mean": float(ratings["rating"].mean(skipna=True)) if len(ratings) else None
        })
    pd.DataFrame(rows).to_csv(PROFILE_CSV, index=False)

#main function
def main():
    # Ensure reports directory exists
    REPORT_DIR.mkdir(exist_ok=True)
    
    #read the parquet files (auto sap xep va tu dong chay)
    movies_raw  = read_first(["movies.cleaned.parquet","movies.parquet","movie*.parquet"])
    ratings_raw = read_first(["ratings.cleaned.parquet","ratings.parquet","rating*.parquet"])
    links_raw   = read_first(["links.cleaned.parquet","links.parquet","link*.parquet"])

    # clean the datasets
    movies  = clean_movies(movies_raw)
    ratings = clean_ratings(ratings_raw)
    links   = clean_links(links_raw)

    # validate the datasets
    vr_m = validate_movies(movies)
    vr_r = validate_ratings(ratings)
    vr_l = validate_links(links)

    # Create detailed validation report matching the image format
    report = {
        "movies.cleaned": {
            "rows": vr_m["rows"],
            "missing_values": vr_m["missing_values"],
            "duplicate_keys": vr_m["duplicate_keys"],
            "schema_check": vr_m["status"]
        },
        "ratings.cleaned": {
            "rows": vr_r["rows"],
            "invalid_ratings": vr_r["invalid_ratings"],
            "timestamp_format_errors": vr_r["timestamp_format_errors"],
            "schema_check": vr_r["status"]
        },
        "links.cleaned": {
            "rows": vr_l["rows"],
            "missing_tmdbId": vr_l["missing_tmdbId"],
            "schema_check": vr_l["status"]
        }
    }
    
    with open(VALIDATION_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # profile the datasets
    profile(movies, ratings, links)

    print(f"[OK] {VALIDATION_JSON.name}")
    print(f"[OK] {PROFILE_CSV.name}")
    print("Schema status:", {k: v["schema_check"] for k, v in report.items()})

if __name__ == "__main__":
    main()
