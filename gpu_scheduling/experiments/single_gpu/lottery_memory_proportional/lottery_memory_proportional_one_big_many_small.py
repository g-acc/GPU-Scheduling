from gpu_scheduling import workqueue as wq
import random
from pathlib import Path

"""
Priority scheduling based on memory usage. Equal quanta, lottery priority.
One big job, three small jobs.
"""

# Organized output directory structure
OUTPUT_DIR = Path("results/single_gpu/lottery_memory_proportional_one_big_many_small")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CSV_DIR = OUTPUT_DIR / "csvs"

# Create directories if they don't exist
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

jobs = [
    wq.Job(
        name=str("first gpt2-small"),
        cmd=["python", 
             "gpu_scheduling/model_training_scripts/train_gpt2.py", 
             "--checkpoint_dir", str(CHECKPOINT_DIR / "small_first"),
             "--csv_file", str(CSV_DIR / "gpt2_small.csv")]
    )
    wq.Job(
        name=str("second gpt2-small"),
        cmd=["python", 
             "gpu_scheduling/model_training_scripts/train_gpt2.py", 
             "--checkpoint_dir", str(CHECKPOINT_DIR / "small_second"),
             "--csv_file", str(CSV_DIR / "gpt2_small.csv")]
    ),
    wq.Job(
        name=str("third gpt2-small"),
        cmd=["python", 
             "gpu_scheduling/model_training_scripts/train_gpt2.py", 
             "--checkpoint_dir", str(CHECKPOINT_DIR / "small_third"),
             "--csv_file", str(CSV_DIR / "gpt2_small.csv")]
    ),
    wq.Job(
        name=str("gpt2-large"),
        cmd=["python", 
             "gpu_scheduling/model_training_scripts/train_gpt2.py",  
             "--checkpoint_dir", str(CHECKPOINT_DIR / "big"),
             "--csv_file", str(CSV_DIR / "gpt2.csv"),
             "--model_name", "gpt2"]
    )
]

def get_next_job(jobs: list[wq.Job]):
    # Assign lottery tickets based on memory usage.
    # For the first lottery, give every job an equal priority
    probabilities = [0.25, 0.25, 0.25, 0.25] 
    if jobs[0].memory_usage_bytes != 0 and jobs[1].memory_usage_bytes != 0 and jobs[2].memory_usage_bytes != 0 and jobs[3].memory_usage_bytes != 0:
        total_mem_usage = sum(job.memory_usage_bytes for job in jobs)
        probabilities = [job.memory_usage_bytes / total_mem_usage for job in jobs] 
    choices = [0,1,2,3]   # items
    pick = random.choices(choices, weights=probabilities, k=1)[0]

    print("Lottery tickets", probabilities, "winning number", pick)
    return pick

if __name__ == "__main__":
    round_robin_equal_time_scheduler = wq.Scheduler(get_next_job_fn=get_next_job, get_working_time_fn=lambda _: 120)
    exp = wq.WorkQueue(jobs, round_robin_equal_time_scheduler, str(OUTPUT_DIR))
    exp.manage_schedule()