"""Parallel fan-out: submit and manage a set of jobs as one unit.

The primary use case is grid search — one MLJob per hyperparameter
combination, all running concurrently in the cluster.
"""
from collections.abc import Iterator

from mlk8s.jobs import MLJob, TrainingError


class JobGroup:
    def __init__(self, jobs: list[MLJob]):
        if not jobs:
            raise ValueError("JobGroup requires at least one job")
        names = [j.name for j in jobs]
        if len(set(names)) != len(names):
            raise ValueError(f"job names must be unique, got: {names}")
        self.jobs = jobs

    def submit_all(self, replace: bool = True) -> None:
        """Submit every job; they run concurrently in the cluster."""
        for job in self.jobs:
            job.submit(replace=replace)

    def wait_all(self, timeout_seconds: int = 600) -> None:
        """Block until all jobs complete; raise listing every failure."""
        failures: dict[str, TrainingError] = {}
        for job in self.jobs:
            try:
                job.wait(timeout_seconds)
            except TrainingError as e:
                failures[job.name] = e
        if failures:
            summary = "; ".join(f"{n}: {e}" for n, e in failures.items())
            raise TrainingError(
                f"{len(failures)}/{len(self.jobs)} jobs failed: {summary}"
            )

    def stream_logs(self) -> Iterator[tuple[str, str]]:
        """Yield (job_name, log_line) pairs, job by job.

        Jobs already run in parallel; logs are collected sequentially,
        so call after wait_all() for a clean per-job dump.
        """
        for job in self.jobs:
            for line in job.stream_logs():
                yield job.name, line

    def delete(self) -> None:
        for job in self.jobs:
            job.delete()
