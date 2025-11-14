# gpu-checkpointing
Applications of GPU checkpointing to scheduling

## Installation

Install dependencies using uv (this will automatically create a virtual environment if needed):

```bash
# Install the package and dependencies (creates .venv automatically)
uv sync
```

Or manually create the venv first:

```bash
# Create virtual environment
uv venv

# Install the package and dependencies
uv sync
```

## Running Experiments

After installation, you can run experiments:

```bash
# Activate the virtual environment first
source .venv/bin/activate

# Run experiments
python -m gpu_scheduling.experiments.single_gpu.rr_equal.round_robin_equal_jobs
```

Or use uv directly:

```bash
uv run python -m gpu_scheduling.experiments.single_gpu.rr_equal.round_robin_equal_jobs
```