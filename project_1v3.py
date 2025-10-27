import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, silhouette_score
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim


def load_data():
    """
    Loads export datasets and stores them in a list.

    Returns:
        list: A list containing pandas DataFrames for cereals, inorganic,
              mineral, ores, and wood exports.
    """
    cereals = pd.read_csv("data/Exports Per Capita/Cereals_capita.csv")
    inorganic = pd.read_csv("data/Exports Per Capita/Inorganic_capita.csv")
    mineral = pd.read_csv("data/Exports Per Capita/Mineral_capita.csv")
    ores = pd.read_csv("data/Exports Per Capita/Ores_capita.csv")
    wood = pd.read_csv("data/Exports Per Capita/Wood_capita.csv")
    return [cereals, inorganic, mineral, ores, wood]

def filter_years(df, years_to_keep):
    """
    Filter a DataFrame to keep only rows where the 'year' column is in years_to_keep.

    Args:
        df (pd.DataFrame): Input DataFrame with a 'year' column
        years_to_keep (list or array-like): Years to retain

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    return df[df['year'].isin(years_to_keep)].copy()

def clustering(df, n_clusters, xcol, ycol, random_state=42):
    """
    Perform KMeans clustering on given columns of df and returns with a 'cluster' column.
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    df = df.copy()
    data = df[[xcol, ycol]].to_numpy()
    df['cluster'] = km.fit_predict(data)
    return df, km


def auto_clustering(df, xcol, ycol, k_min=2, k_max=10, random_state=42):
    """
    Perform automated KMeans clustering and cluster selection on given columns of df. Returns with a 'cluster' column.
    """
    X = df[[xcol, ycol]].to_numpy()
    best_k, best_score, best_km = None, -1, None

    print("X shape for clustering:", X.shape)
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state)
        labels = km.fit_predict(X)
        try:
            score = silhouette_score(X, labels)
        except Exception as e:
            print(f"k={k} failed silhouette_score: {e}")
            continue
        print(f"k={k}, silhouette={score}")
        if score > best_score:
            best_score = score
            best_k = k
            best_km = km

    print(f"Auto-selected {best_k} clusters (silhouette={best_score:.3f})")
    df = df.copy()
    df['cluster'] = best_km.predict(X)
    return df, best_km, best_k


def nonlinear_regression(x, y, model_func, lr=1e-4, epochs=5000, p0=None):
    """
    Fit a nonlinear model y = model_func(x, *params) using gradient descent.

    Args:
        x, y: numpy arrays
        model_func: callable f(x, *params)
        lr: learning rate
        epochs: number of iterations
        p0: initial parameter guess (list or numpy array)

    Returns:
        params: optimized parameters
        r2: R^2 of final fit
    """
    x = np.asarray(x)
    y = np.asarray(y)
    params = np.array(p0, dtype=float)

    for _ in range(epochs):
        y_pred = model_func(x, *params)
        error = y_pred - y
        grad = np.zeros_like(params)

        eps = 1e-6
        for i in range(len(params)):
            params_eps = params.copy()
            params_eps[i] += eps
            y_eps = model_func(x, *params_eps)
            grad[i] = np.sum((y_eps - y_pred) * error) / (len(y) * eps)

        params -= lr * grad

    y_final = model_func(x, *params)
    r2 = r2_score(y, y_final)
    return params, r2


def train_nn_regression(x_scaled, y_scaled, lr=1e-3, epochs=500, hidden_units=10):
    """
    Train a PyTorch neural network on scaled data.

    Returns:
        model: trained PyTorch model
        r2: R² on training data
    """
    X_tensor = torch.from_numpy(x_scaled.reshape(-1, 1).astype(np.float32))
    Y_tensor = torch.from_numpy(y_scaled.reshape(-1, 1).astype(np.float32))

    model = nn.Sequential(
        nn.Linear(1, hidden_units),
        nn.Tanh(),
        nn.Linear(hidden_units, 1)
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_tensor)
        loss = loss_fn(y_pred, Y_tensor)
        loss.backward()
        optimizer.step()

    # R² computation
    y_train_pred = model(X_tensor).detach().numpy().flatten()
    ss_res = np.sum((y_scaled.flatten() - y_train_pred) ** 2)
    ss_tot = np.sum((y_scaled.flatten() - np.mean(y_scaled.flatten())) ** 2)
    r2 = 1 - ss_res / ss_tot

    return model, r2

def combined_regression_clustering(df, name, xcol, ycol,
                                   filteringBounds,
                                   model_func, p0,
                                   n_clusters=None,
                                   cluster_filtered_dfs=None,
                                   lr=1e-4, epochs=5000,
                                   figsize=(8, 5),
                                   random_state=42):
    """
    Perform KMeans clustering, nonlinear regression fits, and plotting,
    using scaled data for analysis but original values for plotting.
    Uses the nonlinear_regression() function (gradient descent parameterized model).
    """
    df = df.copy()

    # Step 0: filtering
    min_val, max_val, y_scalar = filteringBounds
    df = df.dropna(subset=[xcol, ycol])
    if min_val is not None:
        df = df[df[xcol] >= min_val]
    if max_val is not None:
        df = df[df[xcol] <= max_val]

    # Keep original values for plotting
    df['_x_orig'] = df[xcol].copy()
    df['_y_orig'] = df[ycol].copy()

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    df['_x_scaled'] = scaler_x.fit_transform(df[[xcol]])
    df['_y_scaled'] = scaler_y.fit_transform(df[[ycol]]) * y_scalar

    print(df[['_x_scaled', '_y_scaled']].describe())

    # Step 1: clustering on scaled values
    if n_clusters is None:
        df, km, n_clusters = auto_clustering(df, '_x_scaled', '_y_scaled', random_state)
    else:
        df, km = clustering(df, n_clusters, '_x_scaled', '_y_scaled', random_state)

    # Step 2: plot clusters using original values
    palette = sns.color_palette("tab10", n_colors=n_clusters)
    fig, ax = plt.subplots(figsize=figsize)
    for ci in range(n_clusters):
        subset = df[df['cluster'] == ci]
        if subset.empty:
            continue
        ax.scatter(subset['_x_orig'], subset['_y_orig'], label=f"Cluster {ci}",
                   alpha=0.6, s=40, edgecolors='none', color=palette[ci])

    # Step 3: Nonlinear regression using model_func
    if cluster_filtered_dfs is None:
        cluster_filtered_dfs = [([], 'black')]

    for removed_clusters, fit_color in cluster_filtered_dfs:
        df_used = df[~df['cluster'].isin(removed_clusters)]
        x_scaled = df_used['_x_scaled'].to_numpy().flatten()
        y_scaled = df_used['_y_scaled'].to_numpy().flatten()

        params, r2 = nonlinear_regression(
            x_scaled, y_scaled,
            model_func=model_func,
            lr=lr, epochs=epochs, p0=p0
        )

        # Generate fit curve
        x_fit_scaled = np.linspace(x_scaled.min(), x_scaled.max(), 300)
        y_fit_scaled = model_func(x_fit_scaled, *params)

        # Map back to original values
        x_fit_orig = scaler_x.inverse_transform(x_fit_scaled.reshape(-1, 1))
        y_fit_orig = scaler_y.inverse_transform((y_fit_scaled / y_scalar).reshape(-1, 1)).flatten()

        removed_str = ",".join(map(str, removed_clusters)) if removed_clusters else "none"
        ax.plot(x_fit_orig, y_fit_orig, color=fit_color or None, linewidth=2,
                label=f"Fit (removed: {removed_str})")
        ax.annotate(
            rf"$R^2$={r2:.3f}",
            xy=(x_fit_orig[-1], y_fit_orig[-1]),
            xytext=(5, 10),
            textcoords="offset points",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", alpha=0.3),
            arrowprops=dict(arrowstyle="->", lw=0.5)
        )

    # Step 4: Final formatting
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(f"Clustering + Nonlinear Regression for {name}\n"
                 f"Filtered between {min_val}, {max_val}, {y_scalar}:1 scaling")

    ax.legend(frameon=True, fontsize=8, loc='lower right')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    return df, km

def combined_NN_clustering(df, name, xcol, ycol,
                                   filteringBounds,
                                   n_clusters=None,
                                   cluster_filtered_dfs=None,
                                   lr=1e-3, epochs=500,
                                   figsize=(8, 5),
                                   random_state=42):
    """
    Perform KMeans clustering, nonlinear regression fits, and plotting,
    using scaled data for analysis but original values for plotting.
    Replaces the old parameterized nonlinear regression with a small PyTorch model
    to avoid needing p0.
    """
    df = df.copy()

    # Step 0: filtering
    min_val, max_val, y_scalar = filteringBounds
    df = df.dropna(subset=[xcol, ycol])
    if min_val is not None:
        df = df[df[xcol] >= min_val]
    if max_val is not None:
        df = df[df[xcol] <= max_val]

    # Keep original values for plotting
    df['_x_orig'] = df[xcol].copy()
    df['_y_orig'] = df[ycol].copy()

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    df['_x_scaled'] = scaler_x.fit_transform(df[[xcol]])
    df['_y_scaled'] = scaler_y.fit_transform(df[[ycol]]) * y_scalar

    print(df[['_x_scaled', '_y_scaled']].describe())

    # Step 1: clustering on scaled values
    if n_clusters is None:
        df, km, n_clusters = auto_clustering(df, '_x_scaled', '_y_scaled', random_state)
    else:
        df, km = clustering(df, n_clusters, '_x_scaled', '_y_scaled', random_state)

    # Step 2: plot clusters using original values
    palette = sns.color_palette("tab10", n_colors=n_clusters)
    fig, ax = plt.subplots(figsize=figsize)
    for ci in range(n_clusters):
        subset = df[df['cluster'] == ci]
        if subset.empty:
            continue
        ax.scatter(subset['_x_orig'], subset['_y_orig'], label=f"Cluster {ci}",
                   alpha=0.6, s=40, edgecolors='none', color=palette[ci])

    # Step 3: Neural network regression using helper
    if cluster_filtered_dfs is None:
        cluster_filtered_dfs = [([], 'black')]

    for removed_clusters, fit_color in cluster_filtered_dfs:
        df_used = df[~df['cluster'].isin(removed_clusters)]
        x_scaled = df_used['_x_scaled'].to_numpy()
        y_scaled = df_used['_y_scaled'].to_numpy()

        model, r2 = train_nn_regression(x_scaled, y_scaled, lr=lr, epochs=epochs)

        x_fit_scaled = np.linspace(x_scaled.min(), x_scaled.max(), 300)
        x_fit_torch = torch.from_numpy(x_fit_scaled.reshape(-1, 1).astype(np.float32))
        y_fit_scaled = model(x_fit_torch).detach().numpy().flatten()

        x_fit_orig = scaler_x.inverse_transform(x_fit_scaled.reshape(-1, 1))
        y_fit_orig = scaler_y.inverse_transform((y_fit_scaled / y_scalar).reshape(-1, 1)).flatten()

        removed_str = ",".join(map(str, removed_clusters)) if removed_clusters else "none"
        ax.plot(x_fit_orig, y_fit_orig, color=fit_color or None, linewidth=2,
                label=f"NN Fit (removed: {removed_str})")
        ax.annotate(
            rf"$R^2$={r2:.3f}",
            xy=(x_fit_orig[-1], y_fit_orig[-1]),
            xytext=(5, 10),
            textcoords="offset points",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", alpha=0.3),
            arrowprops=dict(arrowstyle="->", lw=0.5)
        )

    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(f"Clustering + Neural Network Regression for {name}\n"
                 f"Filtered between {min_val}, {max_val}, {y_scalar}:1 scaling")

    ax.legend(frameon=True, fontsize=8, loc='lower right')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    return df, km

def save_clustering_results(df, km, name):
    """
    Saves the given DataFrame and KMeans model to disk.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        km (KMeans): The fitted KMeans model to save.
        name (str): Base name for output files.
        folder (str): Directory to save files into. Default: 'data/auto_clustering'.
    """
    data_path = os.path.join("data/auto_clustering", f"{name}_data.csv")
    model_path = os.path.join("models", f"{name}_clustering.joblib")

    df.to_csv(data_path, index=False)
    joblib.dump(km, model_path)

    print(f"Saved DataFrame to {data_path}")
    print(f"Saved clustering model to {model_path}")
