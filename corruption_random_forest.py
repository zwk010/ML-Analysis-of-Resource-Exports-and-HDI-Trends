import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
from sklearn.inspection import PartialDependenceDisplay
from sklearn.tree import plot_tree
import seaborn as sns

# it appears the defense has very minimal correlation
BUDGET_COLS = [
    "defense", "economic", "education", "environment",
    "services", "housing", "health", "safety",
    "recreation", "social"
]

CSV_PATH = "data/government_expenditures/bugetary_data.csv"

def load_and_preprocess(path=CSV_PATH):
    df = pd.read_csv(path)

    # Ensure numeric values
    for c in BUDGET_COLS + ["revenue", "bci"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows missing any required values
    required = BUDGET_COLS + ["revenue", "bci"]
    df_clean = df.dropna(subset=required).copy()

    # Drop invalid rows (zero or negative revenue)
    df_clean = df_clean[df_clean["revenue"] > 0].copy()

    # Create ratio features
    ratio_cols = []
    for col in BUDGET_COLS:
        rc = f"{col}_ratio"
        df_clean[rc] = df_clean[col] / df_clean["revenue"]
        ratio_cols.append(rc)

    return df_clean, ratio_cols

def train_and_evaluate(df, feature_cols, test_size=0.2, random_state=42):
    X = df[feature_cols]
    y = df["bci"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Random Forest Regressor
    model = RandomForestRegressor(
        n_estimators=200,    # number of trees
        max_depth=None,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"R^2: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
    depths = [estimator.get_depth() for estimator in model.estimators_]
    print("Max depth among trees:", max(depths))
    print("Average depth:", sum(depths) / len(depths))

    # Feature importance plot
    fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    plt.figure(figsize=(8,5))
    fi.plot(kind="bar")
    plt.title("Random Forest Feature Importances")
    plt.ylabel("Importance")

    y_pred = model.predict(X_test)
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel("Actual BCI")
    plt.ylabel("Predicted BCI")
    plt.title("Random Forest Predictions vs Actual")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")  # diagonal line
    plt.show()

    return model, (X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    df, ratio_cols = load_and_preprocess(CSV_PATH)
    if df.empty:
        raise SystemExit("No rows with complete data.")
    model, splits = train_and_evaluate(df, ratio_cols)

    joblib.dump(model, "models/bci_random_forest.joblib")