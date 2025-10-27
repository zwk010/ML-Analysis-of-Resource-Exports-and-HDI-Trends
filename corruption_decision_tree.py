import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import joblib

# it appears the defense has very minimal correlation
BUDGET_COLS = [
    "defense", "economic", "education", "environment",
    "services", "housing", "health", "safety",
    "recreation", "social"
]

CSV_PATH = "data/government_expenditures/bugetary_data.csv"

def load_and_preprocess(path=CSV_PATH):
    df = pd.read_csv(path)

    for c in BUDGET_COLS + ["revenue", "bci"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            raise KeyError(f"Column '{c}' not found in CSV")

    # Drop any row that is missing any of the budget labels, revenue, or bci
    required = BUDGET_COLS + ["revenue", "bci"]
    before = len(df)
    df_clean = df.dropna(subset=required).copy()
    after_dropna = len(df_clean)

    df_clean = df_clean[df_clean["revenue"] != 0].copy()
    after_nonzero = len(df_clean)

    ratio_cols = []
    for col in BUDGET_COLS:
        rc = f"{col}_ratio"
        df_clean[rc] = df_clean[col] / df_clean["revenue"]
        ratio_cols.append(rc)

    print(f"Rows total: {before}, after dropna: {after_dropna}, after removing zero revenue: {after_nonzero}")
    return df_clean, ratio_cols

def train_and_evaluate(df, feature_cols, test_size=0.2, random_state=42, max_depth=5):
    X = df[feature_cols]
    y = df["bci"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = DecisionTreeRegressor(random_state=random_state, max_depth=max_depth)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"R^2: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")

    fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("Feature importances:")
    print(fi)

    plt.figure(figsize=(12,6))
    plot_tree(model, feature_names=feature_cols, filled=True, rounded=True, fontsize=8)
    plt.tight_layout()
    plt.show()

    return model, (X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    df, ratio_cols = load_and_preprocess(CSV_PATH)
    if df.empty:
        raise SystemExit("No rows remaining after dropping missing label/revenue/bci values.")
    model, splits = train_and_evaluate(df, ratio_cols, test_size=0.2, random_state=42, max_depth=5)

    joblib.dump(model, "models/bci_decision_tree.joblib")