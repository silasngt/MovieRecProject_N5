"""
03_classification.py

Run 10-fold stratified CV with GridSearch for 3 classifiers and compare F1 (macro).

Inputs:
 - etl/datasets/movie_features.csv

Outputs:
 - etl/reports/model_comparison.csv

Usage (from repo root):
 python notebooks/03_classification.py

This script is written to be conservative (small grids) so it runs reasonably fast for interactive use.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import json
import warnings

warnings.filterwarnings("ignore")


def load_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}\nRun `python scripts/fetch_data.py` and `python run_etl.py` (or the individual transforms) to create it.")
    df = pd.read_csv(path)
    return df


def prepare_features(df: pd.DataFrame):
    # Features observed in movie_features.csv: avg_rating, rating_count, rating_std, year, tmdbId
    features = ["avg_rating", "rating_count", "rating_std", "year", "tmdbId"]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    # Label
    if "label_genre" not in df.columns:
        raise ValueError("Missing label column 'label_genre' in dataset")

    X = df[features].copy()
    y = df["label_genre"].copy()

    # Drop rows with missing label
    mask = y.notna()
    X = X[mask]
    y = y[mask]

    # For simplicity drop rows with any missing numeric feature
    X = X.dropna()
    y = y.loc[X.index]

    # Convert types
    X["rating_count"] = X["rating_count"].astype(float)
    X["year"] = X["year"].astype(float)
    X["tmdbId"] = pd.to_numeric(X["tmdbId"], errors="coerce")
    X = X.fillna(0)

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    return X.values, y_enc, le, X.columns.tolist()


def run_model_selection(X, y, feature_names, out_report: Path):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score, average="macro")

    candidates = [
        (
            "KNN",
            KNeighborsClassifier(),
            {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
        ),
        (
            "RandomForest",
            RandomForestClassifier(random_state=42),
            {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
        ),
        (
            "SVM",
            SVC(random_state=42),
            {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]},
        ),
    ]

    results = []

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for name, model, grid in candidates:
        print(f"\n>>> Tuning {name} ...")
        gs = GridSearchCV(
            estimator=model,
            param_grid=grid,
            scoring=scorer,
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        gs.fit(X_scaled, y)

        best = gs.best_estimator_
        best_params = gs.best_params_
        # Re-run cross_val_score on the best estimator to get mean/std
        scores = cross_val_score(best, X_scaled, y, cv=cv, scoring=scorer, n_jobs=-1)

        row = {
            "model": name,
            "mean_cv_f1_macro": float(scores.mean()),
            "std_cv_f1_macro": float(scores.std()),
            "best_params": json.dumps(best_params, ensure_ascii=False),
            "n_features": len(feature_names),
            "features": ",".join(feature_names),
        }
        results.append(row)

        print(f"{name} best params: {best_params}")
        print(f"{name} CV f1_macro: {scores.mean():.4f} ± {scores.std():.4f}")

    df_res = pd.DataFrame(results).sort_values("mean_cv_f1_macro", ascending=False)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_report, index=False, encoding="utf-8")
    print(f"\nSaved model comparison report -> {out_report}")
    return df_res


def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "etl" / "datasets" / "movie_features.csv"
    out_report = root / "etl" / "reports" / "model_comparison.csv"

    print("Loading dataset:", data_path)
    df = load_data(data_path)
    print(f"Dataset shape: {df.shape}")

    X, y, le, feature_names = prepare_features(df)
    print(f"Prepared X shape: {X.shape}, y shape: {y.shape}, n_classes: {len(le.classes_)}")

    results = run_model_selection(X, y, feature_names, out_report)
    print("\nSummary:\n", results)


if __name__ == "__main__":
    main()
