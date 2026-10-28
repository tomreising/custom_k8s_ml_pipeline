"""Offline tests for content hashing, image building, and registries."""
import pytest

from mlk8s import Registry, build_image, content_tag


class TestContentTag:
    def test_stable_for_same_content(self, tmp_path):
        (tmp_path / "a.py").write_text("print('hi')")
        assert content_tag(tmp_path) == content_tag(tmp_path)

    def test_changes_when_file_edited(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("v1")
        before = content_tag(tmp_path)
        f.write_text("v2")
        assert content_tag(tmp_path) != before

    def test_changes_when_file_added(self, tmp_path):
        (tmp_path / "a.py").write_text("x")
        before = content_tag(tmp_path)
        (tmp_path / "b.py").write_text("x")
        assert content_tag(tmp_path) != before

    def test_changes_when_file_renamed(self, tmp_path):
        (tmp_path / "a.py").write_text("x")
        before = content_tag(tmp_path)
        (tmp_path / "a.py").rename(tmp_path / "renamed.py")
        assert content_tag(tmp_path) != before

    def test_skips_ignored_dirs(self, tmp_path):
        (tmp_path / "a.py").write_text("x")
        before = content_tag(tmp_path)
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "a.pyc").write_text("junk")
        assert content_tag(tmp_path) == before

    def test_hashes_subdirectory_contents(self, tmp_path):
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "mod.py").write_text("v1")
        before = content_tag(tmp_path)
        (sub / "mod.py").write_text("v2")
        assert content_tag(tmp_path) != before

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            content_tag(tmp_path / "does-not-exist")


class FakeRun:
    def __init__(self):
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append(args)


class TestBuildImage:
    def test_local_build(self, monkeypatch):
        from pathlib import Path

        run = FakeRun()
        monkeypatch.setattr("mlk8s.images.subprocess.run", run)
        image = build_image("train", "abc123", context="training")
        assert image == "train:abc123"
        assert run.calls == [
            [
                "docker", "build", "-t", "train:abc123",
                "-f", str(Path("training") / "Dockerfile"), "training",
            ]
        ]

    def test_registry_prefix_and_push(self, monkeypatch):
        run = FakeRun()
        monkeypatch.setattr("mlk8s.images.subprocess.run", run)
        registry = Registry("harbor.corp.example.com/ml", push=True)
        image = build_image("train", "abc123", registry=registry)
        assert image == "harbor.corp.example.com/ml/train:abc123"
        assert run.calls[-1] == ["docker", "push", image]

    def test_no_push_by_default(self, monkeypatch):
        run = FakeRun()
        monkeypatch.setattr("mlk8s.images.subprocess.run", run)
        build_image("train", "abc123", registry=Registry("harbor.corp.example.com/ml"))
        assert all("push" not in call for call in run.calls)


class TestRegistry:
    def test_qualify_strips_trailing_slash(self):
        assert Registry("docker.io/me/").qualify("train") == "docker.io/me/train"

    def test_login_requires_password_env(self, monkeypatch):
        monkeypatch.delenv("MLK8S_REGISTRY_PASSWORD", raising=False)
        registry = Registry("docker.io/me", push=True, login=("me", "docker.io"))
        with pytest.raises(RuntimeError, match="MLK8S_REGISTRY_PASSWORD"):
            registry.ensure_login()

    def test_from_env_requires_registry(self, monkeypatch):
        monkeypatch.delenv("MLK8S_REGISTRY", raising=False)
        with pytest.raises(RuntimeError, match="MLK8S_REGISTRY"):
            Registry.from_env()

    def test_from_env_full_config(self, monkeypatch):
        monkeypatch.setenv("MLK8S_REGISTRY", "harbor.corp.example.com/ml")
        monkeypatch.setenv("MLK8S_REGISTRY_USER", "svc-robot")
        monkeypatch.delenv("MLK8S_REGISTRY_SERVER", raising=False)
        registry = Registry.from_env()
        assert registry.prefix == "harbor.corp.example.com/ml"
        assert registry.push is True
        assert registry._login == ("svc-robot", "harbor.corp.example.com")

    def test_from_env_without_user_skips_login(self, monkeypatch):
        monkeypatch.setenv("MLK8S_REGISTRY", "123.dkr.ecr.us-east-1.amazonaws.com/ml")
        monkeypatch.delenv("MLK8S_REGISTRY_USER", raising=False)
        registry = Registry.from_env()
        assert registry._login is None
