from gpu_scheduling import workqueue as wq
import random
from pathlib import Path

"""
Priority scheduling based on memory usage. Equal quanta, lottery priority.
One big job, one small job.
"""

# Organized output directory structure
OUTPUT_DIR = Path("results/single_gpu/lottery_memory_proportional")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CSV_DIR = OUTPUT_DIR / "csvs"

# Create directories if they don't exist
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

jobs = [
    wq.Job(
        name=str("gpt2-small"),
        cmd=["python", 
             "gpu_scheduling/model_training_scripts/train_gpt2.py", 
             "--checkpoint_dir", str(CHECKPOINT_DIR / "small"),
             "--csv_file", str(CSV_DIR / "gpt2_small.csv")]
    ),
    wq.Job(
        name=str("gpt2"),
        cmd=["python", 
             "gpu_scheduling/model_training_scripts/train_gpt2.py",  
             "--checkpoint_dir", str(CHECKPOINT_DIR / "big"),
             "--csv_file", str(CSV_DIR / "gpt2.csv"),
             "--model_name", "gpt2"]
    )
]

def get_next_job(jobs: list[wq.Job]):
    # Assign lottery tickets based on memory usage.
    # For the first lottery, split it equally (since we don't have memory usage yet)
    tickets = [-2, 2] if (jobs[0].memory_usage_bytes == 0 or jobs[1].memory_usage_bytes == 0) \
        else [-1 * jobs[0].memory_usage_bytes, jobs[1].memory_usage_bytes]
    lotto = random.randrange(tickets[0], tickets[1])
    print("Lottery tickets", tickets, "winning number", lotto)
    if lotto >= 0:
        return 1 
    return 0

if __name__ == "__main__":
    round_robin_equal_time_scheduler = wq.Scheduler(get_next_job_fn=get_next_job, get_working_time_fn=lambda _: 120)
    exp = wq.WorkQueue(jobs, round_robin_equal_time_scheduler, str(OUTPUT_DIR))
    exp.manage_schedule()