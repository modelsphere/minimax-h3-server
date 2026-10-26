import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
WEIGHT = "minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors"


@pytest.fixture
def shell_env(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text(f"#!{sys.executable}\n" + '''import json, os, subprocess, sys
args = sys.argv[1:]
with open(os.environ["DOCKER_CALLS"], "a") as output:
    output.write(json.dumps(args) + "\\n")
if "--entrypoint" in args:
    result = subprocess.run([sys.executable, "-c", args[-1]])
    sys.exit(result.returncode)
if args[:2] == ["network", "inspect"] and os.getenv("DOCKER_NETWORK_MISSING"):
    sys.exit(1)
if args[:2] == ["network", "create"] and os.getenv("DOCKER_NETWORK_CREATE_FAIL"):
    sys.exit(1)
''')
    docker.chmod(0o755)
    mkdir = fake_bin / "mkdir"
    mkdir.write_text("#!/bin/sh\nexit 0\n")  # Never touch actual /mnt directories.
    mkdir.chmod(0o755)
    models = tmp_path / "models with spaces"
    models.mkdir()
    (models / WEIGHT).touch()
    env = {key: value for key, value in os.environ.items() if not key.startswith("H3_")}
    return {**env, "PATH": f"{fake_bin}:{env['PATH']}", "PYTHONPATH": str(REPO),
            "DOCKER_CALLS": str(tmp_path / "calls.jsonl"), "H3_REF2VA_LORA_DIR": str(models),
            "H3_SERVICE_CAPABILITY": "ref2va", "H3_SERVER_IMAGE": "h3-server:test"}


def launch(script, env, tmp_path):
    script_path = REPO / script if "/" in script else REPO / "h3_service" / script
    result = subprocess.run(["bash", str(script_path)], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=10)
    calls_file = Path(env["DOCKER_CALLS"])
    calls = [json.loads(line) for line in calls_file.read_text().splitlines()] if calls_file.exists() else []
    return result, calls


@pytest.mark.parametrize("enabled", [False, True])
def test_api_guard_precedes_container_removal_and_sampling_follows_switch(tmp_path, shell_env, enabled):
    shell_env["H3_REF2VA_TURBO_ENABLED"] = str(enabled).lower()
    result, calls = launch("docker.sh", shell_env, tmp_path)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 4
    guard, network, remove, run = calls
    assert network == ["network", "inspect", "minimax-h3"]
    assert guard[:6] == ["run", "--rm", "--network", "none", "--entrypoint", "python"]
    assert guard[6] == run[-1] == "h3-server:test"
    assert remove == ["rm", "-f", "minimax-h3-api"]
    sampling = [item for item in run if item.startswith("H3_REF2VA_")]
    assert sampling == (["H3_REF2VA_NUM_INFERENCE_STEPS=9", "H3_REF2VA_FLOW_SHIFT=6.0",
                         "H3_REF2VA_AUDIO_FLOW_SHIFT=3.0"] if enabled else [])


@pytest.mark.parametrize("enabled", [False, True])
def test_incompatible_image_never_removes_existing_api(tmp_path, shell_env, enabled):
    package = tmp_path / "h3_service"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "config.py").write_text("# Legacy image without optional sampling support\n")
    shell_env.update(PYTHONPATH=str(tmp_path), H3_REF2VA_TURBO_ENABLED=str(enabled).lower())
    result, calls = launch("docker.sh", shell_env, tmp_path)
    assert result.returncode != 0
    assert "Incompatible H3 Server image" in result.stderr
    assert len(calls) == 1 and "--entrypoint" in calls[0]


@pytest.mark.parametrize("enabled", [False, True])
def test_sglang_profile_keeps_warmup_and_toggles_all_lora_args(tmp_path, shell_env, enabled):
    shell_env["H3_REF2VA_TURBO_ENABLED"] = str(enabled).lower()
    result, calls = launch("sglang_ref2va.sh", shell_env, tmp_path)
    assert result.returncode == 0, result.stderr
    network, remove, run = calls
    assert network == ["network", "inspect", "minimax-h3"]
    assert remove == ["rm", "-f", "minimax-h3-sglang-ref2va"]
    assert run[run.index("--warmup-steps") + 1] == "2"
    assert run[run.index("--warmup-mode") + 1] == "request"
    flags = [item for item in run if item.startswith("--lora-")]
    assert len(flags) == (6 if enabled else 0)
    assert any(":/lora:ro" in item for item in run) is enabled
    if enabled:
        assert f"{shell_env['H3_REF2VA_LORA_DIR']}:/lora:ro" in run
        assert run[run.index("--lora-weight-name") + 1] == WEIGHT


def test_missing_weight_fails_before_replacing_sglang(tmp_path, shell_env):
    shell_env["H3_REF2VA_LORA_DIR"] = str(tmp_path / "missing")
    result, calls = launch("sglang_ref2va.sh", shell_env, tmp_path)
    assert result.returncode != 0
    assert "Missing Ref2VA Turbo LoRA" in result.stderr
    assert calls == []
    shell_env["H3_REF2VA_TURBO_ENABLED"] = "false"
    result, calls = launch("sglang_ref2va.sh", shell_env, tmp_path)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 3


def test_fl2va_ignores_ref_profile_and_has_no_guard_or_sampling(tmp_path, shell_env):
    shell_env.update(H3_SERVICE_CAPABILITY="fl2va", H3_REF2VA_TURBO_ENABLED="bad",
                     H3_REF2VA_NUM_INFERENCE_STEPS="bad")
    result, calls = launch("docker.sh", shell_env, tmp_path)
    assert result.returncode == 0, result.stderr
    network, remove, run = calls
    assert network == ["network", "inspect", "minimax-h3"]
    assert remove[0] == "rm"
    assert not any(item.startswith("H3_REF2VA_") for item in run)


@pytest.mark.parametrize("script", ["docker.sh", "sglang_ref2va.sh"])
def test_invalid_turbo_switch_does_not_replace_containers(tmp_path, shell_env, script):
    shell_env["H3_REF2VA_TURBO_ENABLED"] = "invalid"
    result, calls = launch(script, shell_env, tmp_path)
    assert result.returncode != 0
    assert "H3_REF2VA_TURBO_ENABLED" in result.stderr
    assert calls == []


@pytest.mark.parametrize("network_name", [None, "h3-custom"])
@pytest.mark.parametrize("missing", [False, True])
def test_single_host_launchers_use_shared_network_and_container_addresses(tmp_path, shell_env, network_name, missing):
    from urllib.parse import urlparse

    shell_env.update(H3_SERVICE_CAPABILITY="fl2va", H3_REF2VA_TURBO_ENABLED="false")
    if network_name:
        shell_env["H3_DOCKER_NETWORK"] = network_name
    if missing:
        shell_env["DOCKER_NETWORK_MISSING"] = "1"
    expected_network = network_name or "minimax-h3"
    launches = {}
    for index, script in enumerate([
        "postgresql_demo/docker.sh", "h3_service/sglang.sh", "h3_service/docker.sh",
        "h3_router/docker.sh", "h3_service/sglang_ref2va.sh",
    ]):
        shell_env["DOCKER_CALLS"] = str(tmp_path / f"calls-{index}.jsonl")
        result, calls = launch(script, shell_env, tmp_path)
        assert result.returncode == 0, result.stderr
        assert calls[0] == ["network", "inspect", expected_network]
        if missing:
            assert calls[1] == ["network", "create", expected_network]
        else:
            assert not any(call[:2] == ["network", "create"] for call in calls)
        run = calls[-1]
        assert run[run.index("--network") + 1] == expected_network
        launches[script] = run

    def name(script):
        run = launches[script]
        return run[run.index("--name") + 1]

    api = dict(item.split("=", 1) for item in launches["h3_service/docker.sh"] if item.startswith("H3_"))
    router = dict(item.split("=", 1) for item in launches["h3_router/docker.sh"] if item.startswith("H3_"))
    assert urlparse(api["H3_DATABASE_URL"]).hostname == name("postgresql_demo/docker.sh")
    assert urlparse(api["H3_SGLANG_URL"]).hostname == name("h3_service/sglang.sh")
    assert urlparse(router["H3_FL2VA_SERVICE_URLS"]).hostname == name("h3_service/docker.sh")

    shell_env.update(H3_SERVICE_CAPABILITY="ref2va", DOCKER_CALLS=str(tmp_path / "ref-api.jsonl"))
    result, calls = launch("docker.sh", shell_env, tmp_path)
    assert result.returncode == 0, result.stderr
    ref_api = dict(item.split("=", 1) for item in calls[-1] if item.startswith("H3_"))
    assert urlparse(ref_api["H3_SGLANG_URL"]).hostname == name("h3_service/sglang_ref2va.sh")


def test_remote_backend_overrides_are_preserved(tmp_path, shell_env):
    shell_env.update(H3_SERVICE_CAPABILITY="fl2va", H3_DATABASE_URL="postgresql://h3:pw@db.example:5432/h3",
                     H3_SGLANG_URL="http://gpu.example:30010", H3_FL2VA_SERVICE_URLS="http://api.example:8000",
                     H3_REF2VA_SERVICE_URLS="http://ref.example:8000")
    for script, variables in [
        ("h3_service/docker.sh", ["H3_DATABASE_URL", "H3_SGLANG_URL"]),
        ("h3_router/docker.sh", ["H3_FL2VA_SERVICE_URLS", "H3_REF2VA_SERVICE_URLS"]),
    ]:
        result, calls = launch(script, shell_env, tmp_path)
        assert result.returncode == 0, result.stderr
        for key in variables:
            assert f"{key}={shell_env[key]}" in calls[-1]


def test_network_creation_failure_preserves_existing_api(tmp_path, shell_env):
    shell_env.update(H3_SERVICE_CAPABILITY="fl2va", DOCKER_NETWORK_MISSING="1", DOCKER_NETWORK_CREATE_FAIL="1")
    result, calls = launch("docker.sh", shell_env, tmp_path)
    assert result.returncode != 0
    assert calls == [["network", "inspect", "minimax-h3"], ["network", "create", "minimax-h3"]]


@pytest.mark.parametrize("script", ["h3_service/build.sh", "h3_router/build.sh", "h3_service/build_ref2va.sh"])
def test_build_scripts_forward_arguments_without_splitting(tmp_path, shell_env, script):
    build_args = ["--build-arg", "PIP_INDEX_URL=https://mirror.example/simple", "--label", "description=H3 local build"]
    result = subprocess.run(["bash", str(REPO / script), *build_args], cwd=tmp_path,
                            env=shell_env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in Path(shell_env["DOCKER_CALLS"]).read_text().splitlines()]
    assert len(calls) == 1
    assert calls[0][0] == "build"
    assert calls[0][3:] == [*build_args, "."]
