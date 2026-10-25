"""Preprocessing jobs: run-to-completion data prep before training."""
from mlk8s.jobs import Job


class PreprocessingJob(Job):
    """A batch Job that prepares data before training starts.

    Same lifecycle as TrainingJob; distinguished by its job-type label
    so you can list/cleanup preprocessing runs independently.
    """

    job_type = "preprocessing"
