"""Preprocess, then run a parallel grid search of training jobs.

Phase 1: the preprocess Job must complete before any training starts.
Phase 2: one training Job per hyperparameter combination, all running
concurrently in the cluster.
"""
from pathlib import Path

from mlk8s import (
    JobGroup,
    PreprocessingJob,
    TrainingError,
    TrainingJob,
    build_image,
    content_tag,
)

TRAINING_DIR = Path(__file__).parent / "training"
PREPROCESSING_DIR = Path(__file__).parent / "preprocessing"
ALPHAS = [0.001, 0.01, 0.1, 1.0]

# Content-addressed tags: each workload directory gets its own image, and
# any change inside it produces a new tag (no stale node caches).
preprocess_image = build_image(
    "preprocess", content_tag(PREPROCESSING_DIR), context=PREPROCESSING_DIR
)
train_image = build_image(
    "sklearn-regression", content_tag(TRAINING_DIR), context=TRAINING_DIR
)
print(f"built images: {preprocess_image}, {train_image}")

# Phase 1: preprocess (must finish before training starts)
preprocess = PreprocessingJob(name="preprocess", image=preprocess_image)
print(f"submitting job {preprocess.name!r}...")
preprocess.submit(replace=True)
for line in preprocess.stream_logs():
    print(f"  [preprocess] {line}")
preprocess.wait()
print("preprocess succeeded")

# Phase 2: grid search — one job per alpha, all in parallel
group = JobGroup(
    [
        TrainingJob(
            name=f"train-alpha-{a}",
            image=train_image,
            env={"ALPHA": str(a)},
            gpus=0,  # default: CPU-only. For accelerators: gpus=1 (NVIDIA
            # default), or e.g. gpus=1, accelerator="amd.com/gpu".
        )
        for a in ALPHAS
    ]
)
print(f"submitting {len(group.jobs)} training jobs in parallel...")
group.submit_all()
try:
    group.wait_all()
except TrainingError as e:
    print(f"ERROR: {e}")
    preprocess.delete()
    group.delete()
    raise SystemExit(1)

for name, line in group.stream_logs():
    print(f"  [{name}] {line}")

print("grid search succeeded, cleaning up...")
preprocess.delete()
group.delete()
