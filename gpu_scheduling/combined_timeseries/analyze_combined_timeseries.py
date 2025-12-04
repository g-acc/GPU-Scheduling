import argparse
import os
import numpy as np
import pandas as pd


def jain_fairness(values):
    vals = np.array(values, dtype=float)
    if len(vals) == 0:
        return 0.0
    return (vals.sum() ** 2) / (len(vals) * (vals ** 2).sum())


def compute_job_times(timestamps, mem_col, is_small_job):
    """
    Compute job metrics assuming:
        - job arrives at timestamps[0]
        - job finishes at last nonzero mem row
        - working_time = sum of time where mem > 0
        - total_active_time = finish - arrival
        - waiting_time = total_active_time - working_time
    """
    combined = list(zip(timestamps, mem_col))
    working_time = 0
    waiting_time = 0
    has_job_started = False
    first_active_index = -1
    last_active_index = -1
    for i in range(1, len(combined)):
            ts, mem = combined[i]
            prev_ts, _ = combined[i-1]
            time_increment = (ts - prev_ts)
            if mem > 0:
                if first_active_index == -1:
                    first_active_index = i
                working_time += time_increment
                last_active_index = i
    arrival_time = timestamps[0]
    finish_time = timestamps[last_active_index]
    total_time = finish_time - arrival_time
    waiting_time = total_time - working_time
    slowdown = total_time / (70 if is_small_job else 406)

    return {
        "working_time": float(working_time),
        "waiting_time": float(waiting_time),
        "total_time": float(total_time),
        "slowdown" : float(slowdown)
    }


def analyze_single_csv(csv_path):
    df = pd.read_csv(csv_path)

    # (HARDCODED) Remove relative_time if present
    df = df.drop(columns=[c for c in df.columns if c == "relative_time"], errors="ignore")

    if "timestamp" not in df:
        raise ValueError(f"{csv_path}: missing timestamp column")

    timestamps = df["timestamp"].values.astype(float)

    job_cols = [c for c in df.columns if c != "timestamp"]
    if not job_cols:
        raise ValueError(f"{csv_path}: no job columns found")

    results = {}
    for job in job_cols:
        job_res = compute_job_times(timestamps, df[job], "small" in job)
        if job_res is not None:
            results[job] = job_res

    # Global metrics
    makespan = timestamps[-1] - timestamps[0]
    throughput = len(results) / makespan if makespan > 0 else 0.0

    # Fairness on slowdown
    slowdowns = [
        1 / results[j]["slowdown"] for j in results
    ]
    fairness = jain_fairness(slowdowns)

    # Output
    print(f"\n==== Results for {csv_path} ====\n")

    print("Working Time per Job (sec):")
    for job, r in results.items():
        print(f"- {job}: {r['working_time']:.2f}")
    print()

    print("Waiting Time per Job (sec):")
    for job, r in results.items():
        print(f"- {job}: {r['waiting_time']:.2f}")
    print()

    print(f"Makespan: {makespan:.2f} sec")
    print(f"Throughput (#jobs/makespan): {throughput:.6f}\n")

    print("Response Time per Job (seconds):")
    for job, r in results.items():
        print(f"- {job}: {r['total_time']:.2f}")
    print()

    print("Slowdown per Job:")
    for job, r in results.items():
        print(f"- {job}: {r['slowdown']:.2f}")
    print()

    print(f"Jain Fairness Index over slowdown: {fairness:.4f}")


def analyze_directory(directory_path):
    csv_files = sorted([
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if f.lower().endswith(".csv")
    ])

    if not csv_files:
        raise ValueError(f"No CSV files found in {directory_path}")

    print(f"Found {len(csv_files)} CSV files.\n")
    for csv_file in csv_files:
        analyze_single_csv(csv_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze GPU scheduling logs with fixed arrival at t0 and fairness based on timeshare."
    )
    parser.add_argument("directory", help="Directory containing scheduling CSVs")
    args = parser.parse_args()

    analyze_directory(args.directory)
