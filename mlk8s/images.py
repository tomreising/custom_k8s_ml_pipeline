"""Image build/push helpers with content-addressed tags.

Tagging images by a hash of the code that goes into them means a rebuild
after a code change always produces a NEW tag, so nodes can never serve a
stale cached image under an old tag (the IfNotPresent trap).

Local prototyping needs no registry: build images and rely on the node's
shared image store (Docker Desktop) or `kind load`. For remote clusters,
pass a Registry to build_image() to prefix, and optionally push, images.
"""
import hashlib
import json
import subprocess
from collections.abc import Iterator
from pathlib import Path

_IGNORED_DIRS = {"__pycache__", ".git", ".venv", "node_modules"}


def _iter_files(path: Path) -> Iterator[Path]:
    """Yield files under a directory recursively (skipping junk), or the file itself."""
    if path.is_file():
        yield path
        return
    for child in sorted(path.rglob("*")):
        if child.is_file() and not any(part in _IGNORED_DIRS for part in child.parts):
            yield child


def content_tag(*paths: str | Path, length: int = 12) -> str:
    """Short sha256 over the given files/directories.

    Directories are hashed recursively: each file contributes its path
    relative to the directory root plus its contents, so renames and
    moves within the tree also change the tag.
    """
    h = hashlib.sha256()
    for path in sorted(Path(p) for p in paths):
        if not path.exists():
            raise FileNotFoundError(f"cannot hash missing path: {path}")
        for f in _iter_files(path):
            h.update(str(f.relative_to(path if path.is_dir() else path.parent)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:length]


class Registry:
    """An optional container registry for remote clusters.

    Works with any OCI registry — Docker Hub, Harbor, Nexus, Artifactory,
    GHCR, GitLab, ECR, GCR, ACR, self-hosted. There is no vendor-specific
    code; `login` is passed straight to `docker login`.

    - prefix:       registry host (+ optional namespace) prepended to image
                    names, e.g. "harbor.corp.example.com/team-ml" or
                    "docker.io/myuser". Images are built with the prefix so
                    `docker push` targets the right place.
    - push:         push images after building. Default False — safe for
                    local clusters where nodes share the daemon's image store.
    - login:        (username, server) for registries with stable credentials
                    (Docker Hub PAT, Harbor robot account, GHCR/GitLab token).
                    Runs `docker login <server> -u <user> --password-stdin`
                    once, lazily, before the first push; the password comes
                    from the env var named by password_env (never hardcode it).
    - password_env: env var holding the registry password/token.

    Skip `login` when docker is already authenticated — e.g. cloud credential
    helpers (amazon-ecr-credential-helper, gcloud, az) refresh tokens
    transparently. Prefer helpers for ECR specifically: its login tokens
    expire every 12h, which a one-shot `login` cannot renew.
    """

    def __init__(
        self,
        prefix: str,
        push: bool = False,
        login: tuple[str, str] | None = None,
        password_env: str = "MLK8S_REGISTRY_PASSWORD",
    ):
        self.prefix = prefix.rstrip("/")
        self.push = push
        self._login = login
        self._password_env = password_env
        self._logged_in = False

    @classmethod
    def from_env(cls, push: bool = True) -> "Registry":
        """Build from env vars — keeps credentials and endpoints out of code.

        Required:
            MLK8S_REGISTRY           prefix, e.g. "harbor.corp.example.com/ml"
        Optional login (omit when docker is already authenticated):
            MLK8S_REGISTRY_USER      registry username
            MLK8S_REGISTRY_SERVER    defaults to the host part of MLK8S_REGISTRY
            MLK8S_REGISTRY_PASSWORD  password/token for `docker login`

        Ideal for CI: the pipeline sets env vars, the code stays unchanged.
        """
        import os

        prefix = os.environ.get("MLK8S_REGISTRY")
        if not prefix:
            raise RuntimeError(
                "Registry.from_env() requires $MLK8S_REGISTRY "
                '(e.g. "harbor.corp.example.com/team-ml")'
            )
        user = os.environ.get("MLK8S_REGISTRY_USER")
        login = None
        if user:
            server = os.environ.get("MLK8S_REGISTRY_SERVER") or prefix.split("/")[0]
            login = (user, server)
        return cls(prefix, push=push, login=login)

    def qualify(self, name: str) -> str:
        return f"{self.prefix}/{name}"

    def ensure_login(self) -> None:
        """Log in once, lazily — only when a push actually happens."""
        if self._logged_in or self._login is None:
            return
        import os

        user, server = self._login
        password = os.environ.get(self._password_env)
        if not password:
            raise RuntimeError(
                f"registry login requested but ${self._password_env} is not set"
            )
        subprocess.run(
            ["docker", "login", server, "-u", user, "--password-stdin"],
            input=password.encode(),
            check=True,
            capture_output=True,  # keep the password out of build output
        )
        self._logged_in = True

    def is_logged_in(self, server: str) -> bool:
        """Check docker's config for existing credentials for this server."""
        config_path = Path.home() / ".docker" / "config.json"
        if not config_path.exists():
            return False
        auths = json.loads(config_path.read_text()).get("auths", {})
        return any(server in entry for entry in auths)


def build_image(
    name: str,
    tag: str,
    context: str | Path = ".",
    dockerfile: str | Path = "Dockerfile",
    registry: Registry | None = None,
) -> str:
    """docker build and return the full image reference.

    With a registry, the reference is prefixed (and pushed if the registry
    has push=True). Without one, returns 'name:tag' as before.
    """
    qualified = registry.qualify(name) if registry else name
    image = f"{qualified}:{tag}"
    dockerfile = Path(dockerfile)
    if not dockerfile.is_absolute():
        dockerfile = Path(context) / dockerfile
    subprocess.run(
        ["docker", "build", "-t", image, "-f", str(dockerfile), str(context)],
        check=True,
    )
    if registry and registry.push:
        registry.ensure_login()
        subprocess.run(["docker", "push", image], check=True)
    return image
