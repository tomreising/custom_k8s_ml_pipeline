from kubernetes import client, config
from kubernetes.client.exceptions import ApiException


def load_clients() -> tuple[client.BatchV1Api, client.CoreV1Api]:
    """Load kube credentials (in-cluster first, then local kubeconfig)."""
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    return client.BatchV1Api(), client.CoreV1Api()


def ensure_namespace(name: str, labels: dict[str, str] | None = None) -> str:
    """Create a namespace if it doesn't exist; return its name.

    Idempotent — safe to call at the top of every pipeline. Namespace names
    must be DNS-safe (lowercase alphanumeric and '-'), same as job names.
    """
    _, core = load_clients()
    body = {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {
            "name": name,
            "labels": {"app.kubernetes.io/managed-by": "mlk8s", **(labels or {})},
        },
    }
    try:
        core.create_namespace(body)
    except ApiException as e:
        if e.status == 409:  # already exists — that's fine
            return name
        raise
    return name


def delete_namespace(name: str, wait: bool = True, timeout_seconds: int = 120) -> None:
    """Delete a namespace and everything in it. Safe if it doesn't exist.

    Namespaces terminate asynchronously (K8s finalizes all contained
    resources first), so this optionally blocks until it's actually gone.
    """
    import time

    _, core = load_clients()
    try:
        core.delete_namespace(name)
    except ApiException as e:
        if e.status == 404:
            return
        raise
    if not wait:
        return
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            core.read_namespace(name)
        except ApiException as e:
            if e.status == 404:
                return
            raise
        time.sleep(2)
    raise TimeoutError(f"namespace {name!r} still terminating after {timeout_seconds}s")
