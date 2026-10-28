"""Offline tests for MLJob manifest building and validation.

None of these touch a cluster: clients connect lazily, so constructing
jobs and rendering manifests works with no kubeconfig at all.
"""
import pytest

from mlk8s import MLJob, MLJobType


def make_job(**kwargs) -> MLJob:
    defaults = {"name": "test-job", "image": "img:tag"}
    return MLJob(**(defaults | kwargs))


def manifest(job: MLJob) -> dict:
    return job.build_manifest()["spec"]["template"]["spec"]


class TestResources:
    def test_default_is_cpu_only(self):
        resources = manifest(make_job())["containers"][0]["resources"]
        assert resources == {
            "requests": {"cpu": "250m", "memory": "512Mi"},
            "limits": {"cpu": "1", "memory": "1Gi"},
        }

    def test_gpus_add_accelerator_to_limits_only(self):
        resources = manifest(make_job(gpus=2))["containers"][0]["resources"]
        assert resources["limits"]["nvidia.com/gpu"] == "2"
        assert "nvidia.com/gpu" not in resources["requests"]

    def test_custom_accelerator(self):
        limits = manifest(make_job(gpus=1, accelerator="amd.com/gpu"))["containers"][0][
            "resources"
        ]["limits"]
        assert limits["amd.com/gpu"] == "1"
        assert "nvidia.com/gpu" not in limits

    def test_accelerator_without_gpus_raises(self):
        with pytest.raises(ValueError, match="no effect"):
            make_job(accelerator="amd.com/gpu")


class TestJobType:
    def test_default_is_training(self):
        job = make_job()
        assert job.build_manifest()["metadata"]["labels"]["mlk8s/job-type"] == "training"

    def test_enum_sets_label_and_container_name(self):
        job = make_job(job_type=MLJobType.PREPROCESSING)
        m = job.build_manifest()
        assert m["metadata"]["labels"]["mlk8s/job-type"] == "preprocessing"
        assert m["spec"]["template"]["spec"]["containers"][0]["name"] == "preprocessing"

    def test_custom_string_type_accepted(self):
        job = make_job(job_type="distillation")
        assert job.build_manifest()["metadata"]["labels"]["mlk8s/job-type"] == "distillation"

    @pytest.mark.parametrize("bad", ["Not_DNS", "training!", "-leading", "trailing-", ""])
    def test_invalid_type_rejected(self, bad):
        with pytest.raises(ValueError, match="DNS-safe"):
            make_job(job_type=bad)


class TestManifest:
    def test_structure(self):
        m = make_job().build_manifest()
        assert m["apiVersion"] == "batch/v1"
        assert m["kind"] == "Job"
        assert m["metadata"]["name"] == "test-job"
        assert m["metadata"]["labels"]["app.kubernetes.io/managed-by"] == "mlk8s"
        assert m["spec"]["backoffLimit"] == 1
        assert m["spec"]["ttlSecondsAfterFinished"] == 300
        pod = m["spec"]["template"]
        assert pod["metadata"]["labels"]["job-name"] == "test-job"
        assert pod["spec"]["restartPolicy"] == "Never"

    def test_env_rendered_as_name_value_list(self):
        env = manifest(make_job(env={"ALPHA": "0.1", "EPOCHS": "10"}))["containers"][0]["env"]
        assert {"name": "ALPHA", "value": "0.1"} in env
        assert {"name": "EPOCHS", "value": "10"} in env

    def test_no_env_key_when_empty(self):
        assert "env" not in manifest(make_job())["containers"][0]

    def test_command_override(self):
        container = manifest(make_job(command=["python", "eval.py"]))["containers"][0]
        assert container["command"] == ["python", "eval.py"]

    def test_no_command_key_by_default(self):
        assert "command" not in manifest(make_job())["containers"][0]

    def test_pull_secrets_only_when_set(self):
        assert "imagePullSecrets" not in manifest(make_job())
        spec = manifest(make_job(image_pull_secrets=["ecr-creds"]))
        assert spec["imagePullSecrets"] == [{"name": "ecr-creds"}]

    def test_active_deadline_only_when_set(self):
        assert "activeDeadlineSeconds" not in make_job().build_manifest()["spec"]
        assert (
            make_job(active_deadline_seconds=600).build_manifest()["spec"][
                "activeDeadlineSeconds"
            ]
            == 600
        )

    def test_custom_resources(self):
        resources = manifest(
            make_job(cpu_request="2", cpu_limit="4", memory_request="4Gi", memory_limit="8Gi")
        )["containers"][0]["resources"]
        assert resources["requests"] == {"cpu": "2", "memory": "4Gi"}
        assert resources["limits"] == {"cpu": "4", "memory": "8Gi"}
