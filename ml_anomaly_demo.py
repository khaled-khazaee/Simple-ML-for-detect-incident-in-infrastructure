import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


# -----------------------------
# Synthetic data generation
# -----------------------------
def generate_synthetic_metrics(n_points: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Create synthetic "normal" system metrics for a demo.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_points)

    cpu = 25 + 5 * np.sin(t / 50) + rng.normal(0, 2.0, n_points)
    ram = 40 + 3 * np.sin(t / 60) + rng.normal(0, 1.5, n_points)
    disk = 55 + 0.005 * t + rng.normal(0, 0.5, n_points)
    err5xx = rng.poisson(0.2, n_points).astype(float)
    latency = 80 + 10 * np.sin(t / 45) + rng.normal(0, 4.0, n_points)

    return pd.DataFrame(
        {
            "t": t,
            "cpu": cpu,
            "ram": ram,
            "disk": disk,
            "err5xx": err5xx,
            "latency_ms": latency,
        }
    )


def inject_incidents(df: pd.DataFrame, seed: int = 123) -> tuple[pd.DataFrame, list[tuple[int, int, str]]]:
    """
    Inject a few incident windows into the normal metrics.
    Returns:
      - updated dataframe
      - list of (start, end, label) for ground-truth incident windows
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    windows: list[tuple[int, int, str]] = []

    # Incident A: backend trouble -> latency + 5xx jump
    a_start, a_end = 700, 820
    df.loc[a_start:a_end, "latency_ms"] += 120
    df.loc[a_start:a_end, "err5xx"] += rng.poisson(10, a_end - a_start + 1)
    windows.append((a_start, a_end, "A: latency+5xx spike"))

    # Incident B: resource leak -> cpu/ram jump
    b_start, b_end = 1200, 1350
    df.loc[b_start:b_end, "cpu"] += 40
    df.loc[b_start:b_end, "ram"] += 30
    windows.append((b_start, b_end, "B: cpu+ram leak"))

    # Incident C: disk pressure -> disk higher than normal
    c_start, c_end = 1700, 1850
    df.loc[c_start:c_end, "disk"] += 20
    windows.append((c_start, c_end, "C: disk pressure"))

    return df, windows


# -----------------------------
# Model training & scoring
# -----------------------------
def train_isolation_forest(train_df: pd.DataFrame, feature_cols: list[str]) -> IsolationForest:
    """
    Train an IsolationForest using only "normal" data.
    """
    model = IsolationForest(
        n_estimators=300,
        contamination=0.01,  # lower sensitivity than the very first version
        random_state=42,
    )
    model.fit(train_df[feature_cols])
    return model


def score_points(model: IsolationForest, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Score each point. Higher anomaly_score means "more suspicious".
    """
    df = df.copy()

    normal_score = model.decision_function(df[feature_cols])  # higher == more normal
    df["anomaly_score"] = -normal_score                       # higher == more anomalous
    df["is_anomaly_raw"] = (model.predict(df[feature_cols]) == -1)

    return df


# -----------------------------
# Incident confirmation logic (Option A)
# -----------------------------
def confirm_incidents(
    df: pd.DataFrame,
    min_consecutive: int = 5,
    min_score: float = 0.05,
) -> pd.DataFrame:
    """
    Convert noisy point anomalies into "confirmed incident points".

    A point becomes a confirmed incident point only if:
      1) The model flags it as anomalous
      2) The anomaly score is strong enough (min_score)
      3) The condition persists for at least min_consecutive points
    """
    df = df.copy()

    strong = df["is_anomaly_raw"] & (df["anomaly_score"] >= min_score)
    df["is_anomaly_strong"] = strong

    run_id = (strong != strong.shift(1, fill_value=False)).cumsum()
    run_lengths = strong.groupby(run_id).transform("sum")

    df["is_incident_point"] = strong & (run_lengths >= min_consecutive)
    return df


def extract_incident_windows(df: pd.DataFrame) -> list[tuple[int, int]]:
    """
    Convert incident points (True/False per time step) to a list of (start, end) windows.
    """
    incident = df["is_incident_point"].to_numpy()
    t = df["t"].to_numpy()

    windows: list[tuple[int, int]] = []
    in_run = False
    start_t = 0

    for i, flag in enumerate(incident):
        if flag and not in_run:
            in_run = True
            start_t = int(t[i])
        elif not flag and in_run:
            end_t = int(t[i - 1])
            windows.append((start_t, end_t))
            in_run = False

    if in_run:
        windows.append((start_t, int(t[-1])))

    return windows


# -----------------------------
# Evidence collection (demo)
# -----------------------------
def collect_evidence_for_window(df: pd.DataFrame, start: int, end: int) -> dict:
    """
    Simulate evidence collection using metrics around the incident window.
    In a real system, this would run commands and fetch logs/metrics.
    """
    slice_df = df[(df["t"] >= start) & (df["t"] <= end)].copy()

    return {
        "window": {"start": start, "end": end, "duration_points": int(end - start + 1)},
        "summary": {
            "cpu_max": float(slice_df["cpu"].max()),
            "ram_max": float(slice_df["ram"].max()),
            "disk_max": float(slice_df["disk"].max()),
            "err5xx_sum": float(slice_df["err5xx"].sum()),
            "latency_max_ms": float(slice_df["latency_ms"].max()),
            "latency_mean_ms": float(slice_df["latency_ms"].mean()),
            "anomaly_score_max": float(slice_df["anomaly_score"].max()),
        },
        "recommended_next_steps": [
            "Fetch service logs around the window (e.g., nginx/app logs).",
            "Check recent deployments or configuration changes.",
            "Inspect resource usage per process and open file descriptors.",
            "Run a non-destructive health check to confirm impact before remediation.",
        ],
    }


def write_incident_events_jsonl(events: list[dict], path: str = "incident_events.jsonl") -> None:
    """
    Write incident events to a JSONL file (one JSON object per line).
    """
    with open(path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


# -----------------------------
# Plotting
# -----------------------------
def plot_dashboard(
    df: pd.DataFrame,
    ground_truth_windows: list[tuple[int, int, str]],
    detected_windows: list[tuple[int, int]],
) -> None:
    """
    Display a clearer interactive dashboard:
      - Separate charts for CPU, latency, and 5xx errors
      - Ground-truth incidents shaded lightly
      - Detected incidents shaded more prominently
      - Confirmed incident points marked with X on the latency chart
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(df["t"], df["cpu"], label="cpu (%)")
    axes[0].set_ylabel("CPU (%)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["t"], df["latency_ms"], label="latency (ms)")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df["t"], df["err5xx"], label="5xx errors")
    axes[2].set_ylabel("5xx")
    axes[2].set_xlabel("time step")
    axes[2].grid(True, alpha=0.3)

    for start, end, label in ground_truth_windows:
        for ax in axes:
            ax.axvspan(start, end, alpha=0.10)
        axes[0].text(start, axes[0].get_ylim()[1] * 0.95, label, fontsize=9, va="top")

    for start, end in detected_windows:
        for ax in axes:
            ax.axvspan(start, end, alpha=0.25)

    incident_points = df[df["is_incident_point"]]
    axes[1].scatter(
        incident_points["t"],
        incident_points["latency_ms"],
        marker="x",
        label="confirmed incident points",
    )
    axes[1].legend(loc="upper right")

    fig.suptitle("Synthetic Monitoring Dashboard (Ground Truth vs Detected Incidents)")
    plt.tight_layout()
    plt.show()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    feature_cols = ["cpu", "ram", "disk", "err5xx", "latency_ms"]

    df_normal = generate_synthetic_metrics(n_points=2000, seed=42)
    df_test, gt_windows = inject_incidents(df_normal, seed=123)

    train_df = df_test.iloc[:600].copy()
    model = train_isolation_forest(train_df, feature_cols)

    scored = score_points(model, df_test, feature_cols)

    scored = confirm_incidents(
        scored,
        min_consecutive=5,  # Option A: softer duration requirement
        min_score=0.05,     # Option A: softer score threshold
    )

    detected_windows = extract_incident_windows(scored)

    scored.to_csv("scored_metrics_v3.csv", index=False)

    events: list[dict] = []
    for i, (start, end) in enumerate(detected_windows, start=1):
        evidence = collect_evidence_for_window(scored, start, end)
        events.append(
            {
                "incident_id": f"INC-{i:04d}",
                "type": "anomaly_detected",
                "evidence": evidence,
            }
        )

    write_incident_events_jsonl(events, path="incident_events.jsonl")

    print(f"Detected incident windows: {len(detected_windows)}")
    for start, end in detected_windows:
        print(f"  - start={start}, end={end}, duration_points={end - start + 1}")

    print("\nSaved files:")
    print("  - scored_metrics_v3.csv")
    print("  - incident_events.jsonl")

    plot_dashboard(scored, gt_windows, detected_windows)


if __name__ == "__main__":
    main()
