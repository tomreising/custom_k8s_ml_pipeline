"""Base class for run-to-completion K8s batch Jobs, no YAML required.

TrainingJob and PreprocessingJob are thin specializations of this;
anything that is "run a container until it exits" belongs here.
"""
import time
from collections.abc import Iterator
from pathlib import Path

from kubernetes import client, watch
from kubernetes.client.exceptions import ApiException

from mlk8s.client import load_clients


class TrainingError(Exception):
    """Raised when a job fails or cannot be submitted."""


class Job:
    """A K8s batch Job that runs a container to completion.

    The manifest is built entirely in Python; nothing is read from YAML.
    Subclasses set job_type, which drives the mlk8s/job-type label and
    the container name.
    """

    job_type = "job"

    def __init__(
        self,
        name: str,
        image: str,
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
        ttl_seconds_after_finished: int = 300,
    ):
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
        self.ttl_seconds_after_finished = ttl_seconds_after_finished
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
        try:
            self.batch.read_namespaced_job(self.name, self.namespace)
            return True
        except ApiException as e:
            if e.status == 404:
                return False
            raise

    def submit(self, replace: bool = False) -> None:
        """Create the Job. With replace=True, delete any existing job first."""
        if self.exists():
            if not replace:
                raise TrainingError(f"job {self.name!r} already exists in {self.namespace!r}")
            self.delete()
        try:
            self.batch.create_namespaced_job(self.namespace, self.build_manifest())
        except ApiException as e:
            raise TrainingError(f"failed to create job: {e.reason}") from e

    def wait(self, timeout_seconds: int = 600, poll_interval: float = 2.0) -> None:
        """Block until the Job completes; raise TrainingError on failure/timeout."""
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            job = self.batch.read_namespaced_job(self.name, self.namespace)
            for condition in job.status.conditions or []:
                if condition.type == "Complete" and condition.status == "True":
                    return
                if condition.type == "Failed" and condition.status == "True":
                    raise TrainingError(f"job failed: {condition.reason} - {condition.message}")
            time.sleep(poll_interval)
        raise TrainingError(f"job did not finish within {timeout_seconds}s")

    def _pod_names(self) -> list[str]:
        pods = self.core.list_namespaced_pod(
            self.namespace, label_selector=f"job-name={self.name}"
        )
        return [p.metadata.name for p in pods.items]

    def _wait_for_pod_start(self, pod_name: str, timeout_seconds: int = 300) -> None:
        """Block until the pod's container has started (or already finished).

        The log API returns 400 while a container is still in ContainerCreating,
        so streaming must wait for the pod to leave Pending first.
        """
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
