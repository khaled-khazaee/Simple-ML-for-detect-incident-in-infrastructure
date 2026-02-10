import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# -----------------------------
# 1) Generate synthetic metrics
# -----------------------------
def generate_synthetic_metrics(n_points=2000, seed=42):
    rng = np.random.default_rng(seed)

    t = np.arange(n_points)

    # Base "normal" signals + noise
    cpu = 25 + 5*np.sin(t/50) + rng.normal(0, 2.0, n_points)
    ram = 40 + 3*np.sin(t/60) + rng.normal(0, 1.5, n_points)
    disk = 55 + 0.005*t + rng.normal(0, 0.5, n_points)  # slowly increasing
    err5xx = rng.poisson(0.2, n_points).astype(float)    # usually near 0
    latency = 80 + 10*np.sin(t/45) + rng.normal(0, 4.0, n_points)

    df = pd.DataFrame({
        "t": t,
        "cpu": cpu,
        "ram": ram,
        "disk": disk,
        "err5xx": err5xx,
        "latency_ms": latency
    })

    return df

# -----------------------------------
# 2) Inject incidents (anomalies)
# -----------------------------------
def inject_incidents(df):
    df = df.copy()

    # Incident A: traffic spike / backend issues -> latency + 5xx jump
    a_start, a_end = 700, 820
    df.loc[a_start:a_end, "latency_ms"] += 120
    df.loc[a_start:a_end, "err5xx"] += np.random.poisson(10, a_end-a_start+1)

    # Incident B: resource leak -> cpu/ram jump
    b_start, b_end = 1200, 1350
    df.loc[b_start:b_end, "cpu"] += 40
    df.loc[b_start:b_end, "ram"] += 30

    # Incident C: disk nearly full -> disk goes high
    c_start, c_end = 1700, 1850
    df.loc[c_start:c_end, "disk"] += 20

    return df

# -----------------------------
# 3) Train Isolation Forest
# -----------------------------
def train_model(train_df, feature_cols):
    model = IsolationForest(
        n_estimators=200,
        contamination=0.02,  # expected anomaly ratio (2%)
        random_state=42
    )
    model.fit(train_df[feature_cols])
    return model

# -----------------------------
# 4) Score & flag anomalies
# -----------------------------
def score_anomalies(model, df, feature_cols):
    df = df.copy()

    # decision_function: higher = more normal
    # We'll convert it so higher = more anomalous for easier reading
    normal_score = model.decision_function(df[feature_cols])
    anomaly_score = -normal_score

    # predict: -1 anomaly, +1 normal
    pred = model.predict(df[feature_cols])

    df["anomaly_score"] = anomaly_score
    df["is_anomaly"] = (pred == -1)
    return df

# -----------------------------
# 5) Plot results
# -----------------------------
def plot_results(df):
    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(df["t"], df["latency_ms"], label="latency_ms")
    ax1.set_xlabel("time step")
    ax1.set_ylabel("latency (ms)")

    # mark anomalies
    anomalies = df[df["is_anomaly"]]
    ax1.scatter(anomalies["t"], anomalies["latency_ms"], marker="x", label="anomaly points")

    plt.title("Synthetic Metrics + IsolationForest Anomaly Detection")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    feature_cols = ["cpu", "ram", "disk", "err5xx", "latency_ms"]

    # Create normal data
    df_normal = generate_synthetic_metrics(n_points=2000, seed=42)

    # Use first part as "training normal" (before incidents)
    train_df = df_normal.iloc[:600].copy()

    # Create test data with incidents
    df_test = inject_incidents(df_normal)

    # Train
    model = train_model(train_df, feature_cols)

    # Score
    scored = score_anomalies(model, df_test, feature_cols)

    # Print a quick summary
    total = len(scored)
    n_anom = int(scored["is_anomaly"].sum())
    print(f"Total points: {total}")
    print(f"Anomalies flagged: {n_anom} ({n_anom/total:.2%})")
    print("\nTop 10 most anomalous points:")
    print(scored.sort_values("anomaly_score", ascending=False).head(10)[["t","cpu","ram","disk","err5xx","latency_ms","anomaly_score"]])

    # Save results
    scored.to_csv("scored_metrics.csv", index=False)
    print("\nSaved: scored_metrics.csv")

    # Plot
    plot_results(scored)

if __name__ == "__main__":
    main()
