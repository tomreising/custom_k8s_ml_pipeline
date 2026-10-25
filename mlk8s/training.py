"""Training jobs: run-to-completion model training."""
from mlk8s.jobs import Job, TrainingError

__all__ = ["TrainingJob", "TrainingError"]


class TrainingJob(Job):
    """A batch Job that trains a model to completion.

    Same lifecycle as any Job; distinguished by its job-type label so
    training runs can be listed/cleaned up independently.
    """

    job_type = "training"
