from kubernetes import client, config


def load_clients() -> tuple[client.BatchV1Api, client.CoreV1Api]:
    """Load kube credentials (in-cluster first, then local kubeconfig)."""
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    return client.BatchV1Api(), client.CoreV1Api()
