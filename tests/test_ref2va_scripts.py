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
    result = subprocess.run(["bash", str(REPO / "h3_service" / script)], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=10)
    calls_file = Path(env["DOCKER_CALLS"])
    calls = [json.loads(line) for line in calls_file.read_text().splitlines()] if calls_file.exists() else []
    return result, calls


@pytest.mark.parametrize("enabled", [False, True])
def test_api_guard_precedes_container_removal_and_sampling_follows_switch(tmp_path, shell_env, enabled):
    shell_env["H3_REF2VA_TURBO_ENABLED"] = str(enabled).lower()
    result, calls = launch("docker.sh", shell_env, tmp_path)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 3
    guard, remove, run = calls
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
    remove, run = calls
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
    assert len(calls) == 2


def test_fl2va_ignores_ref_profile_and_has_no_guard_or_sampling(tmp_path, shell_env):
    shell_env.update(H3_SERVICE_CAPABILITY="fl2va", H3_REF2VA_TURBO_ENABLED="bad",
                     H3_REF2VA_NUM_INFERENCE_STEPS="bad")
    result, calls = launch("docker.sh", shell_env, tmp_path)
    assert result.returncode == 0, result.stderr
    remove, run = calls
    assert remove[0] == "rm"
    assert not any(item.startswith("H3_REF2VA_") for item in run)


@pytest.mark.parametrize("script", ["docker.sh", "sglang_ref2va.sh"])
def test_invalid_turbo_switch_does_not_replace_containers(tmp_path, shell_env, script):
    shell_env["H3_REF2VA_TURBO_ENABLED"] = "invalid"
    result, calls = launch(script, shell_env, tmp_path)
    assert result.returncode != 0
    assert "H3_REF2VA_TURBO_ENABLED" in result.stderr
    assert calls == []
