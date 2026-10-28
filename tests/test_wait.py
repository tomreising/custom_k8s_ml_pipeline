"""Offline tests for wait()'s fail-fast behavior, using stubbed K8s clients.

Clients connect lazily, so tests can replace job.batch / job.core with fakes
and drive every scheduling/failure path without a cluster.
"""
from types import SimpleNamespace as NS

import pytest

from mlk8s import MLJob, TrainingError


class FakeBatch:
    def __init__(self, conditions):
        self._job = NS(status=NS(conditions=conditions))

    def read_namespaced_job(self, name, namespace):
        return self._job


class FakeCore:
    """pods may be a list, or a callable returning a list (to simulate
    cluster state changing between polls)."""

    def __init__(self, pods):
        self._pods = pods

    def list_namespaced_pod(self, namespace, label_selector):
        pods = self._pods() if callable(self._pods) else self._pods
        return NS(items=pods)


def job_with(conditions=None, pods=None):
    job = MLJob(name="j", image="img:tag")
    job.batch = FakeBatch(conditions)
    job.core = FakeCore(pods if pods is not None else [])
    return job


def pending_pod(message="0/3 nodes are available: 2 Insufficient nvidia.com/gpu."):
    return NS(
        metadata=NS(name="j-abc"),
        status=NS(
            conditions=[NS(type="PodScheduled", status="False", message=message)],
            container_statuses=[],
        ),
    )


def running_pod():
    return NS(
        metadata=NS(name="j-abc"),
        status=NS(conditions=[NS(type="PodScheduled", status="True")], container_statuses=[]),
    )


def waiting_pod(reason, message=""):
    container = NS(state=NS(waiting=NS(reason=reason, message=message)))
    return NS(metadata=NS(name="j-abc"), status=NS(conditions=[], container_statuses=[container]))


COMPLETE = [NS(type="Complete", status="True")]
FAILED = [NS(type="Failed", status="True", reason="DeadlineExceeded", message="too slow")]

FAST = {"poll_interval": 0.01}


class TestWait:
    def test_complete_returns(self):
        job_with(conditions=COMPLETE).wait(**FAST)  # no exception

    def test_failed_condition_raises_with_reason(self):
        with pytest.raises(TrainingError, match="DeadlineExceeded - too slow"):
            job_with(conditions=FAILED).wait(**FAST)

    def test_persistent_unschedulable_fails_with_scheduler_reason(self):
        job = job_with(conditions=None, pods=[pending_pod()])
        with pytest.raises(TrainingError, match="Insufficient nvidia.com/gpu"):
            job.wait(schedule_timeout_seconds=0.1, **FAST)

    def test_transient_unschedulable_is_tolerated(self):
        # Busy nodes for the first few polls, then capacity frees and the
        # job schedules and completes.
        state = {"pods": [pending_pod("0/3 nodes: Insufficient cpu.")]}
        calls = {"n": 0}

        def pods():
            calls["n"] += 1
            if calls["n"] > 3:
                state["pods"] = [running_pod()]
            return state["pods"]

        class FlippingBatch:
            def read_namespaced_job(self, name, namespace):
                done = calls["n"] > 5
                return NS(status=NS(conditions=COMPLETE if done else None))

        job = MLJob(name="j", image="img:tag")
        job.batch = FlippingBatch()
        job.core = FakeCore(pods)
        job.wait(schedule_timeout_seconds=60, **FAST)  # no exception

    def test_image_pull_failure_fails_immediately(self):
        job = job_with(
            conditions=None,
            pods=[waiting_pod("ImagePullBackOff", "Back-off pulling image")],
        )
        with pytest.raises(TrainingError, match="ImagePullBackOff"):
            job.wait(schedule_timeout_seconds=600, **FAST)  # never waits that long

    def test_invalid_image_name_fails_immediately(self):
        job = job_with(conditions=None, pods=[waiting_pod("InvalidImageName")])
        with pytest.raises(TrainingError, match="InvalidImageName"):
            job.wait(**FAST)

    def test_overall_timeout_includes_pending_reason(self):
        job = job_with(conditions=None, pods=[pending_pod()])
        with pytest.raises(TrainingError, match="still unschedulable.*Insufficient"):
            job.wait(timeout_seconds=0.1, schedule_timeout_seconds=None, **FAST)

    def test_schedule_timeout_none_disables_early_failure(self):
        job = job_with(conditions=None, pods=[pending_pod()])
        with pytest.raises(TrainingError, match="did not finish within"):
            job.wait(timeout_seconds=0.1, schedule_timeout_seconds=None, **FAST)
