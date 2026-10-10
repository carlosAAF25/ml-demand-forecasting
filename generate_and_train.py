import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

def generate_dataset(start_date="2023-01-01", num_weeks=104, random_seed=42):
    """Generates synthetic weekly demand dataset for agrochemical products with strict seed."""
    np.random.seed(random_seed)
    branches = ["Sucursal_Norte", "Sucursal_Sur", "Sucursal_Central"]
    products = [
        {"id": "PROD_GLIFO", "name": "Glifosato 480 SL", "category": "Herbicida", "base_demand": 120, "peak_week": 44},
        {"id": "PROD_PARAQUAT", "name": "Paraquat 200 SL", "category": "Herbicida", "base_demand": 80, "peak_week": 42},
        {"id": "PROD_ATRAZINA", "name": "Atrazina 500 SC", "category": "Herbicida", "base_demand": 95, "peak_week": 46},
        {"id": "PROD_CLORPI", "name": "Clorpirifos 48 EC", "category": "Insecticida", "base_demand": 60, "peak_week": 6},
        {"id": "PROD_MANCO", "name": "Mancozeb 80 WP", "category": "Fungicida", "base_demand": 110, "peak_week": 8},
        {"id": "PROD_UREA_FOL", "name": "Fertilizante Foliar NPK", "category": "Fertilizante", "base_demand": 150, "peak_week": 48}
    ]

    base_date = datetime.strptime(start_date, "%Y-%m-%d")
    records = []

    for week_idx in range(num_weeks):
        current_date = base_date + timedelta(weeks=week_idx)
        year, week_num, _ = current_date.isocalendar()
        is_rainy_season = 1 if (week_num >= 44 or week_num <= 14) else 0

        for branch in branches:
            branch_factor = 1.3 if branch == "Sucursal_Norte" else (1.0 if branch == "Sucursal_Sur" else 0.8)
            for prod in products:
                diff = abs(week_num - prod["peak_week"])
                week_dist = min(diff, 52 - diff)
                seasonality_factor = np.exp(-0.5 * (week_dist / 4.0) ** 2) * 1.5 + 0.3
                noise = np.random.normal(0, 8)
                promo = np.random.choice([0, 1], p=[0.85, 0.15])
                promo_factor = 1.25 if promo else 1.0

                demand = (prod["base_demand"] * branch_factor * seasonality_factor * promo_factor) + noise
                demand = max(0, int(round(demand)))

                records.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "year": year,
                    "week_of_year": week_num,
                    "branch_id": branch,
                    "product_id": prod["id"],
                    "product_category": prod["category"],
                    "is_rainy_season": is_rainy_season,
                    "promo_applied": promo,
                    "demand_liters": demand
                })

    df = pd.DataFrame(records)
    df = df.sort_values(by=["branch_id", "product_id", "year", "week_of_year"]).reset_index(drop=True)
    df["lag_1_demand"] = df.groupby(["branch_id", "product_id"])["demand_liters"].shift(1)
    df["lag_2_demand"] = df.groupby(["branch_id", "product_id"])["demand_liters"].shift(2)
    df["lag_4_demand"] = df.groupby(["branch_id", "product_id"])["demand_liters"].shift(4)
    df["rolling_mean_4w"] = df.groupby(["branch_id", "product_id"])["demand_liters"].transform(
        lambda x: x.shift(1).rolling(window=4, min_periods=1).mean()
    )
    clean_df = df.dropna().reset_index(drop=True)
    return clean_df

def main():
    os.makedirs("data", exist_ok=True)
    dataset_path = os.path.join("data", "agrochemical_weekly_demand.csv")
    
    print("1. Generating synthetic dataset...")
    df = generate_dataset()
    df.to_csv(dataset_path, index=False)
    print(f"Dataset saved to {dataset_path} with {len(df)} samples.")

    # Temporal split: Last 12 weeks reserved for test set (strictly chronological)
    test_weeks_cutoff = df["week_of_year"].max() - 12
    max_year = df["year"].max()
    test_mask = (df["year"] == max_year) & (df["week_of_year"] > test_weeks_cutoff)
    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()

    feature_cols = [
        "branch_id", "product_id", "product_category",
        "week_of_year", "is_rainy_season", "promo_applied",
        "lag_1_demand", "lag_2_demand", "lag_4_demand", "rolling_mean_4w"
    ]
    target_col = "demand_liters"

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    cat_features = ["branch_id", "product_id", "product_category"]
    num_features = ["week_of_year", "is_rainy_season", "promo_applied", "lag_1_demand", "lag_2_demand", "lag_4_demand", "rolling_mean_4w"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
    ])

    # 1. Baseline Evaluation (Naive Lag-1 and Rolling Mean)
    train_naive_mae = mean_absolute_error(train_df["demand_liters"], train_df["lag_1_demand"])
    test_naive_mae = mean_absolute_error(y_test, test_df["lag_1_demand"])
    test_naive_mase = test_naive_mae / train_naive_mae

    test_rolling_mae = mean_absolute_error(y_test, test_df["rolling_mean_4w"])
    test_rolling_mase = test_rolling_mae / train_naive_mae

    print(f"\n--- BASELINE (Naive Lag-1) ---")
    print(f"MAE: {test_naive_mae:.2f} L | MASE: {test_naive_mase:.3f}")
    print(f"--- BASELINE (Rolling Mean 4w) ---")
    print(f"MAE: {test_rolling_mae:.2f} L | MASE: {test_rolling_mase:.3f}")

    # 2. KNN Regressor
    knn_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", KNeighborsRegressor(n_neighbors=5, weights="distance"))
    ])
    knn_pipeline.fit(X_train, y_train)
    knn_preds = knn_pipeline.predict(X_test)
    knn_mae = mean_absolute_error(y_test, knn_preds)
    knn_mase = knn_mae / train_naive_mae

    print(f"\n--- MODEL A (KNN Regressor) ---")
    print(f"MAE: {knn_mae:.2f} L | MASE: {knn_mase:.3f}")

    # 3. Random Forest Regressor
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
    ])
    rf_pipeline.fit(X_train, y_train)
    rf_preds = rf_pipeline.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_mase = rf_mae / train_naive_mae

    print(f"\n--- MODEL B (Random Forest Regressor) ---")
    print(f"MAE: {rf_mae:.2f} L | MASE: {rf_mase:.3f}")

    # Select best model and save artifact
    best_pipeline = rf_pipeline if rf_mae <= knn_mae else knn_pipeline
    best_name = "RandomForest" if rf_mae <= knn_mae else "KNN"

    model_bundle = {
        "model_name": best_name,
        "pipeline": best_pipeline,
        "train_naive_mae": train_naive_mae,
        "feature_cols": feature_cols
    }
    joblib.dump(model_bundle, "demand_forecaster.joblib")
    print(f"\nSaved best model bundle ({best_name}) to demand_forecaster.joblib")

if __name__ == "__main__":
    main()