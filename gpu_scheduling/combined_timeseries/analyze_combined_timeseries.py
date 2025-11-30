import argparse
import os
import numpy as np
import pandas as pd


def jain_fairness(values):
    vals = np.array(values, dtype=float)
    if len(vals) == 0:
        return 0.0
    print("numerator", vals.sum() ** 2)
    print("denominator", (len(vals) * (vals ** 2).sum()))
    return (vals.sum() ** 2) / (len(vals) * (vals ** 2).sum())


def compute_job_times(timestamps, mem_col):
    """
    Compute job metrics assuming:
        - job arrives at timestamps[0]
        - job finishes at last nonzero mem row
        - working_time = sum of dt where mem > 0
        - waiting_time = response_time - working_time
    """

    active = mem_col.values > 0
    if not active.any():
        return None  # job never ran

    arrival = timestamps[0]
    finish_idx = len(active) - 1 - np.argmax(active[::-1])
    finish = timestamps[finish_idx]

    # Compute dt
    deltas = np.diff(timestamps)
    deltas = np.append(deltas, deltas[-1])

    working_time = deltas[active].sum()

    response_time = finish - arrival
    waiting_time = response_time - working_time
    if waiting_time < 0:
        waiting_time = 0.0  # numerical jitter clamp

    return {
        "arrival": float(arrival),
        "finish": float(finish),
        "working_time": float(working_time),
        "waiting_time": float(waiting_time),
        "response_time": float(response_time),
    }


def analyze_single_csv(csv_path):
    df = pd.read_csv(csv_path)

    # Remove relative_time if present
    df = df.drop(columns=[c for c in df.columns if c == "relative_time"], errors="ignore")

    if "timestamp" not in df:
        raise ValueError(f"{csv_path}: missing timestamp column")

    timestamps = df["timestamp"].values.astype(float)

    job_cols = [c for c in df.columns if c != "timestamp"]
    if not job_cols:
        raise ValueError(f"{csv_path}: no job columns found")

    results = {}
    for job in job_cols:
        job_res = compute_job_times(timestamps, df[job])
        if job_res is not None:
            results[job] = job_res

    # Global metrics
    makespan = timestamps[-1] - timestamps[0]
    throughput = len(results) / makespan if makespan > 0 else 0.0

    # Fairness on time-share
    shares = [
        results[j]["working_time"] / makespan
        for j in results
    ]
    print("shares", shares)
    fairness = jain_fairness(shares)

    # Output
    print(f"\n==== Results for {csv_path} ====\n")

    print("Working Time per Job (seconds):")
    for job, r in results.items():
        print(f"  {job}: {r['working_time']:.2f}")
    print()

    print("Waiting Time per Job (seconds):")
    for job, r in results.items():
        print(f"  {job}: {r['waiting_time']:.2f}")
    print()

    print(f"Makespan: {makespan:.2f} seconds")
    print(f"Throughput (#jobs/makespan): {throughput:.6f}\n")

    print("Response Time per Job (seconds):")
    for job, r in results.items():
        print(f"  {job}: {r['response_time']:.2f}")
    print()

    print(f"Fairness Index (Jain over shares): {fairness:.4f}")

    return {
        "jobs": results,
        "fairness": fairness,
        "makespan": makespan,
        "throughput": throughput,
        "shares": shares,
    }


def analyze_directory(directory_path):
    csv_files = sorted([
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if f.lower().endswith(".csv")
    ])

    if not csv_files:
        raise ValueError(f"No CSV files found in {directory_path}")

    print(f"Found {len(csv_files)} CSV files.\n")

    return {
        csv: analyze_single_csv(csv)
        for csv in csv_files
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze GPU scheduling logs with fixed arrival at t0 and fairness based on timeshare."
    )
    parser.add_argument("directory", help="Directory containing scheduling CSVs")
    args = parser.parse_args()

    analyze_directory(args.directory)
