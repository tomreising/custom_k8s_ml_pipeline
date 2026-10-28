"""MLJob: a run-to-completion K8s batch Job, no YAML required.

One class covers every ML pipeline stage — training, preprocessing,
evaluation, and custom stages — with the stage recorded as the
mlk8s/job-type label via the job_type parameter.
"""
import re
import time
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path

from kubernetes import client, watch
from kubernetes.client.exceptions import ApiException

from mlk8s.client import load_clients


class TrainingError(Exception):
    """Raised when a job fails or cannot be submitted."""


class MLJobType(StrEnum):
    """Built-in pipeline stages. Custom DNS-safe strings are also accepted."""

    TRAINING = "training"
    PREPROCESSING = "preprocessing"
    POSTPROCESSING = "postprocessing"
    EVALUATION = "evaluation"


# job_type becomes both a label value and the container name, so it must
# satisfy the stricter of the two: a DNS-1123 label.
_DNS_LABEL = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")


class MLJob:
    """A K8s batch Job that runs a container to completion.

    The manifest is built entirely in Python; nothing is read from YAML.
    job_type sets the mlk8s/job-type label and the container name — pass an
    MLJobType member or any DNS-safe string for custom stages.
    """

    def __init__(
        self,
        name: str,
        image: str,
        job_type: MLJobType | str = MLJobType.TRAINING,
        namespace: str = "default",
        image_pull_policy: str = "IfNotPresent",
        image_pull_secrets: list[str] | None = None,
        command: list[str] | None = None,
        env: dict[str, str] | None = None,
        gpus: int = 0,
        accelerator: str = "nvidia.com/gpu",
        cpu_request: str = "250m",
        cpu_limit: str = "1",
        memory_request: str = "512Mi",
        memory_limit: str = "1Gi",
        backoff_limit: int = 1,
        active_deadline_seconds: int | None = None,
        ttl_seconds_after_finished: int = 300,
    ):
        job_type_str = str(job_type)
        if not _DNS_LABEL.fullmatch(job_type_str):
            raise ValueError(
                f"job_type {job_type_str!r} must be DNS-safe "
                "(lowercase alphanumeric and '-', 1-63 chars)"
            )
        self.job_type = job_type_str
        self.name = name
        self.image = image
        self.namespace = namespace
        self.image_pull_policy = image_pull_policy
        # Names of docker-registry Secrets for pulling private images.
        # Optional: clusters often attach pull credentials to the default
        # ServiceAccount instead, in which case leave this empty.
        self.image_pull_secrets = image_pull_secrets or []
        self.command = command
        self.env = env or {}
        self.gpus = gpus
        # Resource name advertised by the vendor's device plugin, e.g.
        # nvidia.com/gpu, amd.com/gpu, gpu.intel.com/i915, google.com/tpu.
        self.accelerator = accelerator
        if gpus == 0 and accelerator != "nvidia.com/gpu":
            raise ValueError("accelerator has no effect when gpus=0")
        self.cpu_request = cpu_request
        self.cpu_limit = cpu_limit
        self.memory_request = memory_request
        self.memory_limit = memory_limit
        self.backoff_limit = backoff_limit
        # Hard wall-clock cap measured from pod creation, INCLUDING time spent
        # Pending — the guard against unschedulable jobs (e.g. requesting GPUs
        # on a GPU-less cluster) hanging forever. The job fails with
        # reason=DeadlineExceeded instead of pending indefinitely.
        self.active_deadline_seconds = active_deadline_seconds
        self.ttl_seconds_after_finished = ttl_seconds_after_finished
        # Clients connect lazily on first API use, so constructing a job and
        # rendering its manifest never require a cluster connection.
        self.batch: client.BatchV1Api | None = None
        self.core: client.CoreV1Api | None = None

    def _ensure_clients(self) -> None:
        if self.batch is None or self.core is None:
            self.batch, self.core = load_clients()

    def build_manifest(self) -> dict:
        """Render the Job spec as a plain dict (dry-run/debug friendly)."""
        limits: dict[str, str] = {"cpu": self.cpu_limit, "memory": self.memory_limit}
        if self.gpus > 0:
            # K8s requires accelerators only under limits; requests are set equal implicitly.
            limits[self.accelerator] = str(self.gpus)
        container = {
            "name": self.job_type,
            "image": self.image,
            "imagePullPolicy": self.image_pull_policy,
            "resources": {
                "requests": {"cpu": self.cpu_request, "memory": self.memory_request},
                "limits": limits,
            },
        }
        if self.command:
            container["command"] = self.command
        if self.env:
            container["env"] = [{"name": k, "value": v} for k, v in self.env.items()]

        labels = {"app.kubernetes.io/managed-by": "mlk8s", "mlk8s/job-type": self.job_type}
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": self.name, "labels": labels},
            "spec": {
                "backoffLimit": self.backoff_limit,
                **(
                    {"activeDeadlineSeconds": self.active_deadline_seconds}
                    if self.active_deadline_seconds is not None
                    else {}
                ),
                "ttlSecondsAfterFinished": self.ttl_seconds_after_finished,
                "template": {
                    "metadata": {"labels": {**labels, "job-name": self.name}},
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [container],
                        **(
                            {"imagePullSecrets": [{"name": s} for s in self.image_pull_secrets]}
                            if self.image_pull_secrets
                            else {}
                        ),
                    },
                },
            },
        }

    def exists(self) -> bool:
        self._ensure_clients()
        try:
            self.batch.read_namespaced_job(self.name, self.namespace)
            return True
        except ApiException as e:
            if e.status == 404:
                return False
            raise

    def submit(self, replace: bool = False) -> None:
        """Create the Job. With replace=True, delete any existing job first."""
        self._ensure_clients()
        if self.exists():
            if not replace:
                raise TrainingError(f"job {self.name!r} already exists in {self.namespace!r}")
            self.delete()
        try:
            self.batch.create_namespaced_job(self.namespace, self.build_manifest())
        except ApiException as e:
            raise TrainingError(f"failed to create job: {e.reason}") from e

    def wait(
        self,
        timeout_seconds: int = 600,
        schedule_timeout_seconds: int | None = 120,
        poll_interval: float = 2.0,
    ) -> None:
        """Block until the Job completes; raise TrainingError on failure.

        Fail-fast behavior:
        - Fatal container states that retries cannot fix (bad image name,
          ImagePullBackOff, missing ConfigMap/Secret refs) raise immediately.
        - If any pod stays unschedulable longer than schedule_timeout_seconds,
          raise with the scheduler's reason (e.g. "Insufficient nvidia.com/gpu").
          The timer resets when scheduling succeeds, so transient queuing —
          busy nodes, autoscaler scale-up — is tolerated. Set
          schedule_timeout_seconds=None to disable, or raise it on
          autoscaling clusters where node provisioning takes minutes.
        """
        self._ensure_clients()
        deadline = time.monotonic() + timeout_seconds
        unschedulable_since: float | None = None
        pending_reason: str | None = None
        while time.monotonic() < deadline:
            job = self.batch.read_namespaced_job(self.name, self.namespace)
            for condition in job.status.conditions or []:
                if condition.type == "Complete" and condition.status == "True":
                    return
                if condition.type == "Failed" and condition.status == "True":
                    raise TrainingError(f"job failed: {condition.reason} - {condition.message}")

            pending_reason = self._check_pods()
            if pending_reason is None:
                unschedulable_since = None
            else:
                unschedulable_since = unschedulable_since or time.monotonic()
                if (
                    schedule_timeout_seconds is not None
                    and time.monotonic() - unschedulable_since > schedule_timeout_seconds
                ):
                    raise TrainingError(
                        f"job unschedulable for >{schedule_timeout_seconds}s: "
                        f"{pending_reason}"
                    )
            time.sleep(poll_interval)

        suffix = f"; still unschedulable: {pending_reason}" if pending_reason else ""
        raise TrainingError(f"job did not finish within {timeout_seconds}s{suffix}")

    # Container waiting states that retries cannot fix — fail immediately.
    _FATAL_WAITING_REASONS = {
        "ErrImagePull",
        "ImagePullBackOff",
        "InvalidImageName",
        "CreateContainerConfigError",
    }

    def _check_pods(self) -> str | None:
        """Raise on fatal container errors; else return the unschedulable
        reason if any pod can't be scheduled, else None."""
        pods = self.core.list_namespaced_pod(
            self.namespace, label_selector=f"job-name={self.name}"
        )
        unschedulable_reason: str | None = None
        for pod in pods.items:
            for cs in pod.status.container_statuses or []:
                waiting = cs.state.waiting
                if waiting and waiting.reason in self._FATAL_WAITING_REASONS:
                    raise TrainingError(
                        f"pod {pod.metadata.name} cannot start: "
                        f"{waiting.reason} - {waiting.message}"
                    )
            for condition in pod.status.conditions or []:
                if condition.type == "PodScheduled" and condition.status == "False":
                    unschedulable_reason = condition.message
        return unschedulable_reason

    def _pod_names(self) -> list[str]:
        self._ensure_clients()
        pods = self.core.list_namespaced_pod(
            self.namespace, label_selector=f"job-name={self.name}"
        )
        return [p.metadata.name for p in pods.items]

    def _wait_for_pod_start(self, pod_name: str, timeout_seconds: int = 300) -> None:
        """Block until the pod's container has started (or already finished).

        The log API returns 400 while a container is still in ContainerCreating,
        so streaming must wait for the pod to leave Pending first.
        """
        self._ensure_clients()
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            pod = self.core.read_namespaced_pod(pod_name, self.namespace)
            if pod.status.phase in ("Running", "Succeeded", "Failed"):
                return
            time.sleep(1)
        raise TrainingError(f"pod {pod_name!r} did not start within {timeout_seconds}s")

    def stream_logs(self) -> Iterator[str]:
        """Yield log lines from the job's pod(s), following until they exit."""
        deadline = time.monotonic() + 120
        pod_names: list[str] = []
        while not pod_names and time.monotonic() < deadline:
            pod_names = self._pod_names()
            time.sleep(1)
        if not pod_names:
            raise TrainingError(f"no pods found for job {self.name!r}")

        w = watch.Watch()
        for pod in pod_names:
            self._wait_for_pod_start(pod)
            for line in w.stream(
                self.core.read_namespaced_pod_log,
                name=pod,
                namespace=self.namespace,
                follow=True,
            ):
                yield line

    def create_pull_secret(
        self, secret_name: str, server: str, namespace: str | None = None
    ) -> None:
        """Create a docker-registry pull secret from the local docker login.

        Reuses the credentials already in ~/.docker/config.json (works with
        credential helpers), so no password is passed through here. The secret
        lands in the job's namespace (or the one given) — secrets are
        namespace-scoped, so this is per-namespace by design.
        """
        import json
        from base64 import b64encode

        self._ensure_clients()
        config_path = Path.home() / ".docker" / "config.json"
        docker_config = json.loads(config_path.read_text())
        encoded = b64encode(
            json.dumps(docker_config).encode()
        ).decode()
        secret = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": secret_name, "namespace": namespace or self.namespace},
            "type": "kubernetes.io/dockerconfigjson",
            "data": {".dockerconfigjson": encoded},
        }
        try:
            self.core.create_namespaced_secret(namespace or self.namespace, secret)
        except ApiException as e:
            if e.status == 409:  # already exists — refresh it
                self.core.replace_namespaced_secret(
                    secret_name, namespace or self.namespace, secret
                )
            else:
                raise

    def delete(self) -> None:
        """Delete the Job and its pods. Safe to call when it doesn't exist."""
        self._ensure_clients()
        try:
            self.batch.delete_namespaced_job(
                self.name,
                self.namespace,
                propagation_policy="Foreground",
            )
        except ApiException as e:
            if e.status != 404:
                raise
        deadline = time.monotonic() + 60
        while self.exists() and time.monotonic() < deadline:
            time.sleep(1)
