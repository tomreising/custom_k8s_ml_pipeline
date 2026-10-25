# custom-ml-k8s

A lightweight Python SDK for running ML workloads on Kubernetes — training jobs,
preprocessing steps, and parallel hyperparameter search — with **no YAML and no
CRDs**. Everything is a plain `batch/v1` Job built and driven through the
official [`kubernetes`](https://github.com/kubernetes-client/python) Python client.

## Features

- **Run-to-completion jobs as Python objects** — submit, stream logs, wait, delete
- **Typed workload classes** — `TrainingJob` and `PreprocessingJob` share a common
  `Job` base; the `mlk8s/job-type` label keeps them distinguishable in the cluster
- **Parallel grid search** — `JobGroup` fans out one Job per hyperparameter
  combination and waits on them as a unit
- **Content-addressed image tags** — tags are a hash of the workload directory,
  so a code change always produces a new tag and nodes can never serve a stale
  cached image under an old tag
- **Vendor-agnostic accelerators** — NVIDIA (default), AMD, Intel, TPU via a
  single `accelerator` parameter
- **No YAML anywhere** — manifests are Python dicts; `build_manifest()` doubles
  as a dry-run/debug view

## Project layout

```text
custom_ml_k8s/
|-- mlk8s/                  # the SDK
|   |-- jobs.py             #   Job base class: build/submit/wait/logs/delete
|   |-- training.py         #   TrainingJob(Job)
|   |-- preprocessing.py    #   PreprocessingJob(Job)
|   |-- group.py            #   JobGroup: parallel fan-out (grid search)
|   |-- images.py           #   content_tag / build_image helpers
|   |-- client.py           #   kubeconfig / in-cluster auth
|-- training/               # training workload = its own Docker context
|   |-- Dockerfile          #   python:3.12-slim + scikit-learn
|   |-- train.py            #   Ridge regression, hyperparams via env vars
|-- preprocessing/          # preprocessing workload = its own Docker context
|   |-- Dockerfile          #   python:3.12-slim + numpy
|   |-- preprocess.py       #   data loading/validation (no persistence yet)
|-- submit.py               # example pipeline: preprocess -> grid search
```

Each workload directory is self-contained: its Dockerfile is the image contract
(the `CMD` is the entrypoint), and the directory contents define the image tag.

## Prerequisites

- Python >= 3.14 with [uv](https://docs.astral.sh/uv/)
- Docker Desktop (or any Docker daemon)
- A Kubernetes cluster and a current kubeconfig context. Developed against
  Docker Desktop's built-in multi-node cluster (kind-based) on Windows 11;
  any cluster works, but see "Image delivery" below.

Install dependencies:

```powershell
uv sync
```

## Quickstart

```powershell
uv run python submit.py
```

This one command:

1. Hashes `preprocessing/` and `training/` and builds both images with
   content-addressed tags (`preprocess:<hash>`, `sklearn-regression:<hash>`)
2. Submits a `PreprocessingJob` and **blocks until it completes**
3. Fans out 4 `TrainingJob`s in parallel, one per value in `ALPHAS`
4. Streams per-job logs, then cleans up every job it created

## SDK usage

### A single training job

```python
from mlk8s import TrainingJob

job = TrainingJob(
    name="my-run",
    image="my-image:tag",
    env={"ALPHA": "0.1"},          # hyperparameters as env vars
    gpus=0,                        # CPU by default
)
job.submit(replace=True)           # replace=True deletes a stale run first
for line in job.stream_logs():     # live log stream, follows until exit
    print(line)
job.wait()                         # raises TrainingError on failure/timeout
job.delete()
```

### A preprocessing step that must finish first

```python
from mlk8s import PreprocessingJob

preprocess = PreprocessingJob(name="preprocess", image="preprocess:tag")
preprocess.submit(replace=True)
preprocess.wait()                  # ordering guarantee: training starts after
```

### Parallel grid search

```python
from mlk8s import JobGroup, TrainingJob

group = JobGroup([
    TrainingJob(name=f"train-alpha-{a}", image="img:tag", env={"ALPHA": str(a)})
    for a in [0.001, 0.01, 0.1, 1.0]
])
group.submit_all()                 # all jobs run concurrently in the cluster
group.wait_all()                   # one error listing every failure, if any
for name, line in group.stream_logs():
    print(f"[{name}] {line}")
group.delete()
```

### GPUs and other accelerators

```python
TrainingJob(..., gpus=2)                                   # nvidia.com/gpu (default)
TrainingJob(..., gpus=1, accelerator="amd.com/gpu")        # AMD ROCm nodes
TrainingJob(..., gpus=4, accelerator="google.com/tpu")     # TPU slice on GKE
```

The accelerator resource only appears under `limits` when `gpus > 0`
(Kubernetes rejects GPU requests outside of limits; requests are implicitly
set equal). Passing `accelerator` with `gpus=0` raises `ValueError` instead of
silently running CPU-only. The target cluster needs the vendor's device plugin
installed, or the job will stay `Pending`.

### Building images with content-addressed tags

```python
from mlk8s import build_image, content_tag

tag = content_tag("training")              # recursive hash of the directory
image = build_image("sklearn-regression", tag, context="training")
```

`content_tag` hashes every file under the directory (path + contents), skipping
`__pycache__`, `.venv`, `.git`, `node_modules`. Any edit, rename, addition, or
deletion produces a new tag.

## How it works

- **Manifests are dicts.** `Job.build_manifest()` renders the `batch/v1` Job as
  a plain Python dict — print it for a free dry-run. No Jinja, no YAML files.
- **Labels are the SDK's memory.** Every resource is labeled
  `app.kubernetes.io/managed-by=mlk8s` and `mlk8s/job-type=<type>`, and pods get
  the standard `job-name` label. Log streaming and cleanup find pods by label
  selector, not by tracking names.
- **Failure mapping.** `wait()` polls Job status conditions and raises
  `TrainingError` with the cluster's failure reason instead of hanging.
  Log streaming waits for the pod to leave `Pending` first, since the log API
  returns HTTP 400 while a container is still in `ContainerCreating`.

### Image delivery

The examples use `imagePullPolicy: IfNotPresent` with locally built images.
This works on Docker Desktop's Kubernetes (the nodes share the local containerd
image store) and on kind after `kind load docker-image <image>`. On remote
clusters, push to a registry instead — the content-addressed tag scheme works
unchanged (`registry.example.com/sklearn-regression:<hash>`).

**Do not** rely on mutable tags like `latest` with `IfNotPresent`: once a node
has cached a tag, rebuilds under the same tag are never picked up.

### Registries (remote clusters)

`Registry` works with any OCI registry — Docker Hub, Harbor, Nexus, Artifactory,
GHCR, GitLab, ECR, GCR, ACR, self-hosted. There is no vendor-specific code; it
drives the standard `docker login` / `docker push` flow. Login is **optional** —
local prototyping needs none of this.

```python
from mlk8s import Registry, build_image, content_tag

# Private registry (Harbor, self-hosted, etc.) with a robot account:
registry = Registry(
    "harbor.corp.example.com/team-ml",                # host + optional namespace
    push=True,
    login=("svc-robot", "harbor.corp.example.com"),   # (username, server)
)
image = build_image("train", content_tag("training"), context="training",
                    registry=registry)
# -> builds and pushes harbor.corp.example.com/team-ml/train:<hash>
```

The password comes from `$MLK8S_REGISTRY_PASSWORD` (or a name you choose via
`password_env`) — it is passed through stdin and never appears in code, logs,
or command lines. What the "password" actually is depends on the registry:

| Registry | Use as password |
|---|---|
| Docker Hub | A personal access token (not your account password) |
| GHCR | A GitHub PAT with `read:packages, write:packages` |
| GitLab | A deploy token or PAT with `read_registry, write_registry` |
| Harbor / private | Robot account secret or user password |
| ECR | `aws ecr get-login-password` output — **expires every 12h** |

For short-lived tokens (ECR especially), skip `login=` and configure docker's
credential helper instead (`amazon-ecr-credential-helper`, `gcloud`, `az`);
docker then refreshes tokens transparently and `Registry(..., push=True)` just
works.

For CI, keep everything in env vars with `Registry.from_env()`:

```powershell
$env:MLK8S_REGISTRY = "harbor.corp.example.com/team-ml"   # required
$env:MLK8S_REGISTRY_USER = "svc-robot"                    # optional; omit if
$env:MLK8S_REGISTRY_PASSWORD = "..."                      #   already logged in
# MLK8S_REGISTRY_SERVER defaults to the host part of MLK8S_REGISTRY
```

```python
registry = Registry.from_env()   # no credentials or endpoints in source
```

Private images also need pull credentials *in the cluster*. Either attach them
to the namespace's default ServiceAccount (ops-side, no SDK code), or per job:

```python
job.create_pull_secret("registry-creds", server="harbor.corp.example.com")
# reuses your local docker login; idempotent; secrets are namespace-scoped

TrainingJob(..., image=image, image_pull_secrets=["registry-creds"])
```

## Configuration

`TrainingJob` / `PreprocessingJob` accept the same parameters (inherited from `Job`):

| Parameter | Default | Purpose |
|---|---|---|
| `name` | (required) | Job name (DNS-safe; embed hyperparams e.g. `train-alpha-0.1`) |
| `image` | (required) | Full image reference |
| `namespace` | `"default"` | Target namespace |
| `command` | image `CMD` | Override container entrypoint |
| `env` | `{}` | Env vars (hyperparameters, secrets) |
| `gpus` | `0` | Accelerator count; `0` = CPU-only |
| `accelerator` | `"nvidia.com/gpu"` | Device-plugin resource name |
| `cpu_request` / `cpu_limit` | `250m` / `1` | CPU resources |
| `memory_request` / `memory_limit` | `512Mi` / `1Gi` | Memory resources |
| `backoff_limit` | `1` | Pod retries before the Job fails |
| `ttl_seconds_after_finished` | `300` | Auto-cleanup of finished Jobs (seconds) |

## Current limitations / roadmap

- No persistence between pipeline steps yet (preprocessing writes nothing that
  training reads) — PVC and S3 artifact support is the next milestone
- No serving API yet (`Deployment` + `Service` for trained models)
- No multi-node distributed training yet (Indexed Job + headless Service +
  `torchrun` env injection is the planned approach)
- Single namespace per job object; no quota/RBAC management
- Log streaming is per-pod sequential; `JobGroup` collects logs after completion
  rather than interleaving them live
