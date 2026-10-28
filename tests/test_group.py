"""Offline tests for JobGroup validation."""
import pytest

from mlk8s import JobGroup, MLJob


def make_job(name: str) -> MLJob:
    return MLJob(name=name, image="img:tag")


class TestJobGroup:
    def test_empty_group_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            JobGroup([])

    def test_duplicate_names_rejected(self):
        with pytest.raises(ValueError, match="unique"):
            JobGroup([make_job("a"), make_job("a")])

    def test_jobs_stored_in_order(self):
        jobs = [make_job("a"), make_job("b"), make_job("c")]
        assert JobGroup(jobs).jobs == jobs
