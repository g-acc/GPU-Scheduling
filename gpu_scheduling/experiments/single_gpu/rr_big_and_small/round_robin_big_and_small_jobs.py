from gpu_scheduling import workqueue as wq

jobs = [
            wq.Job(
                name=str("gpt2-small"),
                cmd=["python", 
                     "gpu_scheduling/model_training_scripts/train_gpt2.py", 
                     "--checkpoint_dir", "gpu_scheduling/experiments/single_gpu/rr_big_and_small/small"]
            ),
            wq.Job(
                name=str("gpt2-large"),
                cmd=["python", 
                     "gpu_scheduling/model_training_scripts/train_gpt2.py",  
                     "--checkpoint_dir", "gpu_scheduling/experiments/single_gpu/rr_big_and_small/big",
                     "--model_name", "gpt2-large"]
            )
        ]

if __name__ == "__main__":
    round_robin_equal_time_scheduler = wq.Scheduler(get_next_job_fn=lambda _: 0, get_working_time_fn=lambda _: 120)
    exp = wq.WorkQueue(jobs, round_robin_equal_time_scheduler, "gpu_scheduling/experiments/single_gpu/rr_big_and_small/")
    exp.manage_schedule()