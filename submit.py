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
    delete_namespace,
    ensure_namespace,
)

TRAINING_DIR = Path(__file__).parent / "training"
PREPROCESSING_DIR = Path(__file__).parent / "preprocessing"
NAMESPACE = "ml-pipeline"
ALPHAS = [0.001, 0.01, 0.1, 1.0]
# Set False to keep finished jobs/pods in the namespace for post-run
# inspection (logs, metrics comparison). They still expire automatically
# via ttl_seconds_after_finished â€” raise TTL_SECONDS to keep them longer.
CLEANUP = False
TTL_SECONDS = 3600

# The pipeline lives in its own namespace: isolated from other workloads,
# and teardown at the end is a single namespace delete.
ensure_namespace(NAMESPACE)

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
preprocess = PreprocessingJob(
    name="preprocess",
    image=preprocess_image,
    namespace=NAMESPACE,
    ttl_seconds_after_finished=TTL_SECONDS,
)
print(f"submitting job {preprocess.name!r}...")
preprocess.submit(replace=True)
for line in preprocess.stream_logs():
    print(f"  [preprocess] {line}")
preprocess.wait()
print("preprocess succeeded")

# Phase 2: grid search â€” one job per alpha, all in parallel
group = JobGroup(
    [
        TrainingJob(
            name=f"train-alpha-{a}",
            image=train_image,
            namespace=NAMESPACE,
            env={"ALPHA": str(a)},
            gpus=0,  # default: CPU-only. For accelerators: gpus=1 (NVIDIA
            # default), or e.g. gpus=1, accelerator="amd.com/gpu".
            ttl_seconds_after_finished=TTL_SECONDS,
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
    if CLEANUP:
        delete_namespace(NAMESPACE)
    else:
        print(f"resources kept in namespace {NAMESPACE!r} for debugging")
    raise SystemExit(1)

for name, line in group.stream_logs():
    print(f"  [{name}] {line}")

if CLEANUP:
    print("grid search succeeded, tearing down namespace...")
    delete_namespace(NAMESPACE)
else:
    print(
        f"grid search succeeded â€” runs kept in namespace {NAMESPACE!r} "
        f"(expire {TTL_SECONDS}s after finishing). Inspect with:"
    )
    print(f"  kubectl logs -n {NAMESPACE} job/train-alpha-0.1")


